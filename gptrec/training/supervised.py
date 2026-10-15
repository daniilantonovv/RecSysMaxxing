"""Supervised training: next-item prediction for the recommender, feedback prediction for the reward model."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from tqdm.auto import tqdm

from gptrec.data.batches import EventBatch, EventWindows, swap_one_target_item
from gptrec.data.log import PAD_ITEM, InteractionLog
from gptrec.evaluation.reward_model_metrics import choice_quality, predict_heldout_feedback, prediction_quality, retention_quality
from gptrec.models.backbone import TransformerConfig
from gptrec.models.policy import NextItemPolicy, PolicyConfig
from gptrec.models.reward_model import UNKNOWN_FEEDBACK, RewardModel, RewardModelConfig
from gptrec.models.sequence_recommender import SequenceRecommender
from gptrec.training.precision import autocast, clipped_optimizer_step, gradient_scaler, seed_everything

LossFunction = Callable[[nn.Module, EventBatch], Tensor]
Validation = Callable[[nn.Module], dict[str, float]]
Metrics = dict[str, float]

PROGRESS_REFRESH_SECONDS = 5.0  # keeps the logs of long unattended runs short


@dataclass(frozen=True)
class OptimizationConfig:
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2  # epochs without improvement of the validation loss before stopping
    mixed_precision: bool = True
    seed: int = 0


@dataclass(frozen=True)
class PolicyTrainingConfig:
    """Training windows are `transformer.max_positions` events long, so every position the model can read gets trained."""

    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    window_stride: int = 64  # distance between the starts of consecutive training windows of one user


@dataclass(frozen=True)
class FeedbackLossWeights:
    """Weight of each feedback signal in the reward model's loss."""

    played_ratio: float = 1.0
    session_continues: float = 0.25
    returns: float = 0.25
    is_chosen: float = 0.25
    relative_ratio: float = 0.5
    # Learned only where the log records them; zero leaves a reward model as it was before these signals existed.
    liked: float = 0.0
    disliked: float = 0.0
    adopts: float = 0.0


@dataclass(frozen=True)
class RewardModelTrainingConfig:
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    window_stride: int = 64
    num_ratio_buckets: int = 10
    loss_weights: FeedbackLossWeights = field(default_factory=FeedbackLossWeights)
    # Hidden responses: in this share of training windows the responses to the last 1..max_hidden_responses events
    # are replaced by "unknown", so the model learns to predict after items whose response nobody saw, as after
    # generated ones. Zero trains a model without the "unknown" input.
    hidden_response_share: float = 0.0
    max_hidden_responses: int = 10


def train_policy(log: InteractionLog, config: PolicyTrainingConfig, device: torch.device, extra_validation: Validation | None = None) -> tuple[NextItemPolicy, list[Metrics]]:
    """Trains the recommender on the train period; the validation period decides when to stop.

    `extra_validation` adds metrics (e.g. NDCG@10) to the per-epoch history without affecting training.
    """
    seed_everything(config.optimization.seed)
    policy = NextItemPolicy(PolicyConfig(log.num_items, config.transformer)).to(device)
    window_length = config.transformer.max_positions
    history = fit_on_train_period(
        policy, log, next_item_loss, predicts_target, window_length, config.window_stride, config.optimization, device, extra_validation=extra_validation
    )
    return policy.eval(), history


def finetune_on_liked_items(
    pretrained_policy: NextItemPolicy,
    log: InteractionLog,
    config: OptimizationConfig,
    window_stride: int,
    liked_threshold: float,
    device: torch.device,
) -> tuple[NextItemPolicy, list[Metrics]]:
    """Continues next-item training on a copy of the policy, but only towards items the user played at least `liked_threshold` of.

    The supervised alternative to RL: it learns from the logged feedback directly and needs no reward model.
    """
    seed_everything(config.seed)
    policy = copy.deepcopy(pretrained_policy).to(device).requires_grad_(True)
    loss = partial(liked_next_item_loss, liked_threshold=liked_threshold)
    scored_positions = partial(predicts_liked_target, liked_threshold=liked_threshold)
    window_length = policy.config.transformer.max_positions
    history = fit_on_train_period(policy, log, loss, scored_positions, window_length, window_stride, config, device)
    return policy.eval(), history


def train_reward_model(log: InteractionLog, config: RewardModelTrainingConfig, device: torch.device, users: np.ndarray | None = None) -> tuple[RewardModel, list[Metrics]]:
    """Trains a reward model on the train period of `users` (default: everyone)."""
    seed_everything(config.optimization.seed)
    learns_unknown_responses = config.hidden_response_share > 0
    reward_model = RewardModel(RewardModelConfig(log.num_items, config.transformer, config.num_ratio_buckets, learns_unknown_responses)).to(device)
    loss = partial(feedback_loss, weights=config.loss_weights, hidden_response_share=config.hidden_response_share, max_hidden_responses=config.max_hidden_responses)
    # Early stopping looks at the responses as they were observed, without random hiding.
    validation_loss = partial(feedback_loss, weights=config.loss_weights)
    window_length = config.transformer.max_positions
    validation_windows = EventWindows.for_heldout(log, "validation", window_length, users)

    def prediction_quality_on_validation(trained: nn.Module) -> Metrics:
        """The validation loss in readable terms: how well each signal of validation-period events is predicted."""
        predicted = predict_heldout_feedback(trained, validation_windows, config.optimization.batch_size, device, config.optimization.mixed_precision)
        played_ratio = prediction_quality(predicted.played_ratio, predicted.predicted_played_ratio)
        return {
            "validation_auc": played_ratio["auc"],
            "validation_mae": played_ratio["mae"],
            "validation_relative_auc": prediction_quality(predicted.relative_ratio, predicted.predicted_relative_ratio)["auc"],
            "validation_returns_auc": retention_quality(predicted).loc["returns", "auc"],
            "validation_choice_auc": choice_quality(predicted)["auc"],
        }

    history = fit_on_train_period(
        reward_model,
        log,
        loss,
        lambda batch: batch.is_target,
        window_length,
        config.window_stride,
        config.optimization,
        device,
        validation_loss=validation_loss,
        users=users,
        extra_validation=prediction_quality_on_validation,
    )
    return reward_model.eval().requires_grad_(False), history


def fit_on_train_period(
    model: nn.Module,
    log: InteractionLog,
    loss_function: LossFunction,
    scored_positions: Callable[[EventBatch], Tensor],
    window_length: int,
    window_stride: int,
    config: OptimizationConfig,
    device: torch.device,
    validation_loss: LossFunction | None = None,
    users: np.ndarray | None = None,
    extra_validation: Validation | None = None,
) -> list[Metrics]:
    """Trains `model` on windows of the train period and keeps the weights with the lowest validation-period loss.

    `scored_positions` marks, for a batch, the positions the validation loss averages over, so that batches
    with different numbers of targets are weighted correctly. The validation loss defaults to the training loss.
    `extra_validation` adds readable quality metrics to every epoch's record; they do not affect training.
    """
    train_windows = EventWindows.for_training(log, window_length, window_stride, users)
    validation_windows = EventWindows.for_heldout(log, "validation", window_length, users)

    def validate(trained: nn.Module) -> Metrics:
        loss = mean_heldout_loss(trained, validation_windows, validation_loss or loss_function, scored_positions, config, device)
        return {"validation_loss": loss} | (extra_validation(trained) if extra_validation else {})

    return fit(model, train_windows, loss_function, validate, config, device)


def next_item_loss(model: SequenceRecommender, batch: EventBatch) -> Tensor:
    """Cross-entropy of every target event given the events before it in its window."""
    return _next_item_cross_entropy(model, batch, predicts_target(batch))


def liked_next_item_loss(model: SequenceRecommender, batch: EventBatch, liked_threshold: float) -> Tensor:
    """Cross-entropy of the target events the user played at least `liked_threshold` of; skipped events are not imitated."""
    return _next_item_cross_entropy(model, batch, predicts_liked_target(batch, liked_threshold))


def predicts_target(batch: EventBatch) -> Tensor:
    """bool [B, L - 1]: positions whose next event is a target. The first event of a window has no context, so it is never one."""
    return batch.is_target[:, 1:]


def predicts_liked_target(batch: EventBatch, liked_threshold: float) -> Tensor:
    """bool [B, L - 1]: positions whose next event is a target that the user played at least `liked_threshold` of."""
    return batch.is_target[:, 1:] & (batch.played_ratio[:, 1:] >= liked_threshold)


def _next_item_cross_entropy(model: SequenceRecommender, batch: EventBatch, positions: Tensor) -> Tensor:
    """Full-softmax cross-entropy of the next item at the given positions (bool [B, L - 1])."""
    states = model.hidden_states(batch.items)[:, :-1][positions]
    if len(states) == 0:
        return states.sum()  # nothing to predict in this batch: a zero that still backpropagates
    return F.cross_entropy(model.item_logits(states), batch.items[:, 1:][positions])


def feedback_loss(
    reward_model: RewardModel, batch: EventBatch, weights: FeedbackLossWeights, hidden_response_share: float = 0.0, max_hidden_responses: int = 1
) -> Tensor:
    """Weighted binary cross-entropy of every feedback signal on target events.

    The played ratio and its quantile inside the item are soft labels in [0, 1]. The retention signals, likes,
    dislikes and adoption are learned only from events whose outcome the log knows. Whether an item is chosen
    is learned by contrast, see `choice_loss`. With `hidden_response_share` the model reads some responses as
    unknown (see `hide_last_responses`); the labels it is trained on are always the observed ones.
    """
    played_ratio_read = hide_last_responses(batch, hidden_response_share, max_hidden_responses) if hidden_response_share > 0 else batch.played_ratio
    logits = reward_model(batch.items, played_ratio_read)
    reactions_known = batch.is_target & batch.reactions_known
    return (
        weights.played_ratio * _mean_binary_cross_entropy(logits.played_ratio, batch.played_ratio, batch.is_target)
        + weights.session_continues * _mean_binary_cross_entropy(logits.session_continues, batch.session_continues, batch.is_target & batch.session_continues_known)
        + weights.returns * _mean_binary_cross_entropy(logits.returns, batch.returns, batch.is_target & batch.returns_known)
        + weights.is_chosen * choice_loss(reward_model, batch, logits.is_chosen, played_ratio_read)
        + weights.relative_ratio * _mean_binary_cross_entropy(logits.relative_ratio, batch.relative_ratio, batch.is_target)
        + weights.liked * _mean_binary_cross_entropy(logits.liked, batch.liked, reactions_known)
        + weights.disliked * _mean_binary_cross_entropy(logits.disliked, batch.disliked, reactions_known)
        + weights.adopts * _mean_binary_cross_entropy(logits.adopts, batch.adopts, batch.is_target & batch.adopts_known)
    )


def hide_last_responses(batch: EventBatch, share: float, max_hidden: int) -> Tensor:
    """The played ratios a reward model reads, with the responses to the last 1..max_hidden events of a random `share` of windows unknown. [B, L]

    This is the situation of a generated list: the history's responses are known, the generated items' are not.
    The first event of a window always keeps its response.
    """
    num_events = (batch.items != PAD_ITEM).sum(dim=1)
    num_hidden = torch.randint(1, max_hidden + 1, num_events.shape, device=num_events.device)
    first_hidden = (num_events - num_hidden).clamp_min(1)
    is_hiding = torch.rand(num_events.shape, device=num_events.device) < share
    position = torch.arange(batch.items.shape[1], device=num_events.device)
    hidden = is_hiding[:, None] & (position[None, :] >= first_hidden[:, None]) & (position[None, :] < num_events[:, None])
    return batch.played_ratio.masked_fill(hidden, UNKNOWN_FEEDBACK)


def choice_loss(reward_model: RewardModel, batch: EventBatch, logits_of_logged_items: Tensor, played_ratio_read: Tensor | None = None) -> Tensor:
    """Binary cross-entropy of telling the item a user chose from an item they did not; the two kinds weigh the same.

    Logged target events are the chosen items. For the others, one target event per window is scored again
    with another item of the batch in its place (see `swap_one_target_item`). Such replacements are as popular
    as real choices, so the head has to use the history, not popularity. logits_of_logged_items: [B, L].
    played_ratio_read: the responses the model reads, if not the observed ones.
    """
    swapped = swap_one_target_item(batch)
    played_ratio_read = batch.played_ratio if played_ratio_read is None else played_ratio_read
    logits_of_swapped_items = reward_model(swapped.items, played_ratio_read).is_chosen.gather(1, swapped.position[:, None]).squeeze(1)
    chosen_loss = _mean_binary_cross_entropy(logits_of_logged_items, torch.ones_like(logits_of_logged_items), batch.is_target)
    not_chosen_loss = _mean_binary_cross_entropy(logits_of_swapped_items, torch.zeros_like(logits_of_swapped_items), swapped.is_different)
    return 0.5 * (chosen_loss + not_chosen_loss)


def _mean_binary_cross_entropy(logits: Tensor, labels: Tensor, is_selected: Tensor) -> Tensor:
    """Mean over the selected entries; a zero that still backpropagates when nothing is selected."""
    if not is_selected.any():
        return logits.sum() * 0.0
    return F.binary_cross_entropy_with_logits(logits[is_selected], labels[is_selected])


def fit(model: nn.Module, train_windows: EventWindows, loss_function: LossFunction, validate: Validation, config: OptimizationConfig, device: torch.device) -> list[Metrics]:
    """Minimises `loss_function` with AdamW and returns per-epoch metrics.

    `validate` must report "validation_loss". The model ends with the weights of its best epoch.
    """
    steps_per_epoch = train_windows.num_batches(config.batch_size, drop_last=True)
    if steps_per_epoch == 0:
        raise ValueError(f"{len(train_windows)} training windows do not fill one batch of {config.batch_size}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _warmup_then_cosine(config.warmup_steps, config.epochs * steps_per_epoch))
    scaler = gradient_scaler(device, config.mixed_precision)

    history: list[Metrics] = []
    best_loss, best_weights, epochs_without_improvement = math.inf, copy.deepcopy(model.state_dict()), 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = 0.0
        batches = train_windows.batches(config.batch_size, shuffle_seed=config.seed + epoch, drop_last=True)
        progress = tqdm(batches, total=steps_per_epoch, desc=f"epoch {epoch}", mininterval=PROGRESS_REFRESH_SECONDS)
        for step, batch in enumerate(progress, start=1):
            with autocast(device, config.mixed_precision):
                loss = loss_function(model, batch.to(device))
            clipped_optimizer_step(loss, optimizer, scaler, model.parameters(), config.max_grad_norm)
            scheduler.step()
            loss_sum += loss.item()
            progress.set_postfix(train_loss=loss_sum / step)

        metrics = {"epoch": epoch, "train_loss": loss_sum / steps_per_epoch} | validate(model)
        history.append(metrics)
        print({name: round(value, 5) for name, value in metrics.items()})
        if not math.isfinite(metrics["validation_loss"]):
            raise FloatingPointError(f"validation loss is {metrics['validation_loss']} after epoch {epoch}: training diverged")

        if metrics["validation_loss"] < best_loss:
            best_loss, best_weights, epochs_without_improvement = metrics["validation_loss"], copy.deepcopy(model.state_dict()), 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    model.load_state_dict(best_weights)
    return history


def _warmup_then_cosine(warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    def learning_rate_factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return learning_rate_factor


@torch.no_grad()
def mean_heldout_loss(model: nn.Module, windows: EventWindows, loss_function: LossFunction, scored_positions: Callable[[EventBatch], Tensor], config: OptimizationConfig, device: torch.device) -> float:
    """Average over all windows of a loss that is a mean over each batch's scored positions."""
    model.eval()
    loss_sum, num_targets = 0.0, 0
    for batch in windows.batches(config.batch_size):
        batch = batch.to(device)
        batch_targets = int(scored_positions(batch).sum())
        if batch_targets == 0:
            continue
        with autocast(device, config.mixed_precision):
            loss_sum += loss_function(model, batch).item() * batch_targets
        num_targets += batch_targets
    return loss_sum / max(1, num_targets)
