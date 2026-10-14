"""Off-policy REINFORCE on logged sessions, after Matveev et al. (arXiv:2607.02818).

Nothing is generated and no reward model is asked. The recommender re-reads what users really did, and the
log-probability of every logged item is raised in proportion to the reward that followed it in the log:

    loss = - sum over logged actions of  discount^t * weight_t * return_t * log pi(a_t | x_t)  /  sum of discount^t * weight_t

    return_t   discounted sum of the logged rewards from action t to the end of its session
    weight_t   min(cap, product of pi / pi_behaviour over the next `importance_weight_factors` actions of the session)
    t          how far into its session the action is

The behaviour policy is the frozen pre-trained recommender, as in the paper, where the target policy also
starts from the behaviour model. Two things differ from the paper. Every logged event is an action here;
the paper uses only the events a recommender served, which KION does not mark. And a session that
started before the training window is read from the window's first event.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor

from gptrec.data.batches import EventBatch
from gptrec.data.log import InteractionLog
from gptrec.models.policy import NextItemPolicy
from gptrec.models.reward_model import Feedback
from gptrec.rollout import RewardFunction, RewardSpec
from gptrec.training.precision import seed_everything
from gptrec.training.supervised import Metrics, OptimizationConfig, fit_on_train_period, next_item_loss, predicts_target


@dataclass(frozen=True)
class LoggedReturnConfig:
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(epochs=1, learning_rate=1e-4))
    reward: RewardSpec = field(default_factory=RewardSpec)  # applied to logged feedback, so it cannot weigh by choice or logged relevance
    discount: float = 0.9
    importance_weight_factors: int = 10  # policy ratios multiplied into one importance weight
    importance_weight_cap: float = 0.5  # the larger, the further the policy may move from the behaviour policy


def finetune_on_logged_returns(
    pretrained_policy: NextItemPolicy,
    log: InteractionLog,
    config: LoggedReturnConfig,
    window_stride: int,
    device: torch.device,
) -> tuple[NextItemPolicy, list[Metrics]]:
    """Fine-tunes a copy of the recommender with off-policy REINFORCE; the original is the behaviour policy.

    The validation loss, which picks the best epoch, is the plain next-item loss.
    """
    reward = config.reward
    if reward.weigh_by_choice or reward.needs_logged_future or reward.needs_other_user or reward.depends_on_the_list:
        raise ValueError("a reward read from the log has one logged response per item: it cannot weigh by choice, by logged relevance, against another user or by the rest of a list")
    seed_everything(config.optimization.seed)
    behaviour_policy = pretrained_policy.to(device).eval().requires_grad_(False)
    policy = copy.deepcopy(behaviour_policy).requires_grad_(True)
    reward_function = RewardFunction(config.reward, log.items, device)
    loss = partial(logged_return_loss, behaviour_policy=behaviour_policy, reward_function=reward_function, config=config)
    window_length = policy.config.transformer.max_positions
    history = fit_on_train_period(policy, log, loss, predicts_target, window_length, window_stride, config.optimization, device, validation_loss=next_item_loss)
    return policy.eval(), history


def logged_return_loss(policy: NextItemPolicy, batch: EventBatch, behaviour_policy: NextItemPolicy, reward_function: RewardFunction, config: LoggedReturnConfig) -> Tensor:
    """The loss of the module docstring for one batch of windows. Action a is the window's event a + 1, chosen after events 0..a."""
    is_action = predicts_target(batch)
    stays_in_session = (batch.session_continues[:, 1:] > 0.5) & _ahead(is_action, 1)  # the next action belongs to the same session
    log_probs = _logged_item_log_probs(policy, batch)
    with torch.no_grad():
        log_policy_ratio = (log_probs - _logged_item_log_probs(behaviour_policy, batch)).masked_fill(~is_action, 0.0)
        importance_weight = _capped_importance_weights(log_policy_ratio, stays_in_session, config.importance_weight_factors, config.importance_weight_cap)
        returns = _session_returns(_logged_rewards(batch, reward_function)[:, 1:], stays_in_session, config.discount)
        action_weight = config.discount ** _steps_into_session(stays_in_session) * importance_weight * is_action
    return -(action_weight * returns * log_probs).sum() / action_weight.sum().clamp_min(1e-8)


def _logged_item_log_probs(policy: NextItemPolicy, batch: EventBatch) -> Tensor:
    """log pi(event a + 1 | events 0..a) for every action of the window. [B, L - 1]"""
    log_distributions = F.log_softmax(policy.item_logits(policy.hidden_states(batch.items)[:, :-1]), dim=-1)
    return log_distributions.gather(2, batch.items[:, 1:, None]).squeeze(2)


def _logged_rewards(batch: EventBatch, reward_function: RewardFunction) -> Tensor:
    """The reward of every logged event; a retention outcome the log does not know counts as absent. [B, L]"""
    logged = Feedback(
        played_ratio=batch.played_ratio,
        session_continues=batch.session_continues * batch.session_continues_known,
        returns=batch.returns * batch.returns_known,
        is_chosen=torch.ones_like(batch.played_ratio),
        relative_ratio=batch.relative_ratio,
    )
    return reward_function(logged, batch.items)


def _session_returns(rewards: Tensor, stays_in_session: Tensor, discount: float) -> Tensor:
    """returns[:, a] = rewards[:, a] + discount * returns[:, a + 1] while the session lasts. [B, A]"""
    returns = torch.zeros_like(rewards)
    return_from_next_action = torch.zeros_like(rewards[:, 0])
    for action in reversed(range(rewards.shape[1])):
        return_from_next_action = rewards[:, action] + discount * stays_in_session[:, action] * return_from_next_action
        returns[:, action] = return_from_next_action
    return returns


def _capped_importance_weights(log_policy_ratio: Tensor, stays_in_session: Tensor, num_factors: int, cap: float) -> Tensor:
    """min(cap, product of the policy ratios of an action and of the up to `num_factors` - 1 actions after it in its session). [B, A]"""
    log_weight = log_policy_ratio.clone()
    session_lasts = torch.ones_like(stays_in_session)
    for actions_ahead in range(1, num_factors):
        session_lasts = session_lasts & _ahead(stays_in_session, actions_ahead - 1)
        log_weight = log_weight + _ahead(log_policy_ratio, actions_ahead) * session_lasts
    return log_weight.exp().clamp(max=cap)


def _steps_into_session(stays_in_session: Tensor) -> Tensor:
    """0 for the first action of a session (or of the window), then 1, 2, ... [B, A]"""
    steps = torch.zeros(stays_in_session.shape, device=stays_in_session.device)
    for action in range(1, stays_in_session.shape[1]):
        steps[:, action] = (steps[:, action - 1] + 1) * stays_in_session[:, action - 1]
    return steps


def _ahead(values: Tensor, steps: int) -> Tensor:
    """values[:, a + steps] in column a; zero (False) where that is beyond the last column. [B, A]"""
    if steps == 0:
        return values
    return torch.cat([values[:, steps:], values.new_zeros(values.shape[0], min(steps, values.shape[1]))], dim=1)
