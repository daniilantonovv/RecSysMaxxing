"""Learning from logged responses alone: no reward model and no generated lists.

Both methods start from the base recommender, read the train period window by window like next-item training,
and keep the epoch with the lowest next-item loss on the validation period:

reward-weighted fine-tuning  next-item training in which each logged item weighs exp(r / temperature), r its logged
                             response. Under the anchored objective this is the closed-form optimum fitted to the
                             log (exponential reward-weighted SFT, Chidambaram et al. 2026; AWR, Peng et al. 2019).
                             Without a response every item weighs one: plain continued training, the control for
                             what the extra epoch at a lower learning rate does by itself.
within-user DPO              direct preference optimisation (Rafailov et al., 2023) on pairs of items that one user
                             consumed after the same history: one taken to more than is usual for it, one to less.
                             Neither item is preferred for having been started, since both were, nor for being short,
                             since each is compared with its own usual response: only this user's taste remains.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gptrec.data.batches import EventBatch
from gptrec.data.log import PAD_ITEM, InteractionLog
from gptrec.models.policy import NextItemPolicy
from gptrec.models.sequence_recommender import SequenceRecommender
from gptrec.tensors import TensorGroup
from gptrec.training.precision import seed_everything
from gptrec.training.supervised import Metrics, OptimizationConfig, fit_on_train_period, next_item_loss, predicts_target

LOGGED_RESPONSES = ("relative_ratio", "played_ratio")  # what reward-weighted fine-tuning can weigh an item by
USUAL_QUANTILE = 0.5  # the relative ratio of a response that is usual for its item


@dataclass(frozen=True)
class RewardWeightedConfig:
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(epochs=1, learning_rate=1e-4))
    response: str | None = "relative_ratio"  # one of LOGGED_RESPONSES; None weighs every item the same (continued training)
    temperature: float = 0.3  # the smaller, the more the best-received items dominate


@dataclass(frozen=True)
class LoggedPreferenceConfig:
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(epochs=1, learning_rate=1e-4))
    beta: float = 0.1  # DPO's inverse temperature: the larger, the nearer to the base recommender the policy stays
    margin: float = 0.1  # preferred: relative ratio at least 0.5 + margin; rejected: at most 0.5 - margin
    lookahead: int = 20  # the user's next events that a pair is drawn from


@dataclass
class PreferencePairs(TensorGroup):
    """Pairs of items one user consumed after the same moment of a window, preferred first."""

    window: Tensor  # long [P], the window each pair comes from
    moment: Tensor  # long [P], position of the last event before both items
    preferred: Tensor  # long [P], item taken to more than is usual for it
    rejected: Tensor  # long [P], item taken to less than is usual for it


def finetune_with_reward_weights(
    pretrained_policy: NextItemPolicy, log: InteractionLog, config: RewardWeightedConfig, window_stride: int, device: torch.device
) -> tuple[NextItemPolicy, list[Metrics]]:
    """Continues next-item training on a copy of the policy, each target weighted by exp(its logged response / temperature).

    With no response every target weighs one, and the same epoch is plain continued next-item training.
    """
    if config.response is not None and config.response not in LOGGED_RESPONSES:
        raise ValueError(f"unknown logged response {config.response!r}: expected one of {LOGGED_RESPONSES}, or None for equal weights")
    seed_everything(config.optimization.seed)
    policy = copy.deepcopy(pretrained_policy).to(device).requires_grad_(True)
    loss = partial(reward_weighted_loss, response=config.response, temperature=config.temperature)
    window_length = policy.config.transformer.max_positions
    history = fit_on_train_period(policy, log, loss, predicts_target, window_length, window_stride, config.optimization, device, validation_loss=next_item_loss)
    return policy.eval(), history


def reward_weighted_loss(model: SequenceRecommender, batch: EventBatch, response: str | None, temperature: float) -> Tensor:
    """Cross-entropy of every target event weighted by exp(response / temperature), the weights normalised to a mean of one per batch.

    Without a response every weight is one, which is the plain next-item loss.
    """
    positions = predicts_target(batch)
    states = model.hidden_states(batch.items)[:, :-1][positions]
    if len(states) == 0:
        return states.sum()
    cross_entropy = F.cross_entropy(model.item_logits(states), batch.items[:, 1:][positions], reduction="none")
    if response is None:
        return cross_entropy.mean()
    responses = getattr(batch, response)[:, 1:][positions].float()
    weights = torch.softmax(responses / temperature, dim=0) * len(responses)
    return (weights * cross_entropy).mean()


def finetune_on_logged_preferences(
    pretrained_policy: NextItemPolicy, log: InteractionLog, config: LoggedPreferenceConfig, window_stride: int, device: torch.device
) -> tuple[NextItemPolicy, list[Metrics]]:
    """DPO from the base recommender on within-user pairs (see `within_user_pairs`); the base recommender is the reference."""
    seed_everything(config.optimization.seed)
    reference = pretrained_policy.to(device).eval().requires_grad_(False)
    policy = _without_dropout(copy.deepcopy(reference)).requires_grad_(True)
    loss = partial(logged_preference_loss, reference=reference, config=config)
    window_length = policy.config.transformer.max_positions
    history = fit_on_train_period(policy, log, loss, predicts_target, window_length, window_stride, config.optimization, device, validation_loss=next_item_loss)
    return policy.eval(), history


def logged_preference_loss(policy: NextItemPolicy, batch: EventBatch, reference: NextItemPolicy, config: LoggedPreferenceConfig) -> Tensor:
    """The DPO loss over one pair per window; windows without a pair are left out."""
    pairs = within_user_pairs(batch, config.margin, config.lookahead)
    if len(pairs) == 0:
        return policy.item_embedding.weight.sum() * 0.0
    margin = _preference_margin(policy, batch, pairs)
    with torch.no_grad():
        reference_margin = _preference_margin(reference, batch, pairs)
    return -F.logsigmoid(config.beta * (margin - reference_margin)).mean()


def within_user_pairs(batch: EventBatch, margin: float, lookahead: int) -> PreferencePairs:
    """At most one pair per window: a random moment, then one event above and one below the usual among the next `lookahead` events.

    Only target events form pairs, so the context of a held-out window never does.
    """
    positions = torch.arange(batch.items.shape[1], device=batch.items.device)
    comes_after = (positions[None, :] > positions[:, None]) & (positions[None, :] <= positions[:, None] + lookahead)
    above = batch.is_target & (batch.relative_ratio >= USUAL_QUANTILE + margin)
    below = batch.is_target & (batch.relative_ratio <= USUAL_QUANTILE - margin)
    above_after = above[:, None, :] & comes_after[None]
    below_after = below[:, None, :] & comes_after[None]
    has_pair = above_after.any(dim=2) & below_after.any(dim=2) & (batch.items != PAD_ITEM)
    window = has_pair.any(dim=1).nonzero().squeeze(1)
    if len(window) == 0:
        nothing = torch.zeros(0, dtype=torch.long, device=batch.items.device)
        return PreferencePairs(nothing, nothing, nothing, nothing)
    moment = torch.multinomial(has_pair[window].float(), num_samples=1).squeeze(1)
    preferred = torch.multinomial(above_after[window, moment].float(), num_samples=1).squeeze(1)
    rejected = torch.multinomial(below_after[window, moment].float(), num_samples=1).squeeze(1)
    return PreferencePairs(window, moment, batch.items[window, preferred], batch.items[window, rejected])


def _preference_margin(model: NextItemPolicy, batch: EventBatch, pairs: PreferencePairs) -> Tensor:
    """log pi(preferred | history) - log pi(rejected | history) for every pair. [P]"""
    rows = torch.arange(len(pairs), device=pairs.window.device)
    states = model.hidden_states(batch.items[pairs.window])[rows, pairs.moment]
    log_probs = F.log_softmax(model.item_logits(states), dim=-1)
    return log_probs[rows, pairs.preferred] - log_probs[rows, pairs.rejected]


def _without_dropout(model: nn.Module) -> nn.Module:
    """`model` with every dropout switched off, so that its log-probabilities and the reference's differ only by what it learned."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
    return model
