"""Steering the base recommender at inference with the closed form of the anchored objective: no fine-tuning.

For a reward r and the anchor beta * KL(pi || pi_0), the best policy at one position of a list is

    pi*(a | s) = pi_0(a | s) * exp(Q(s, a) / beta) / Z(s),        Q(s, a) = r(s, a) + discount * U(s, a),

where U(s, a) is the discounted reward of the items that follow a. Re-ranking by r + beta * log pi_0 takes the top
of pi* with U = 0 and every candidate judged once, after the real history. `guided_lists` decides one position at
a time, after the items already chosen, and can look ahead with a U learned from the base recommender's own lists:
the reference policy's value, as in controlled decoding (Mudgal et al., 2024) and as the first step of Q-sharp
(Zhou et al., 2025), which iterates it towards the value of pi* itself. The recommender's weights never change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from gptrec.data.batches import HistoryBatch, TrainingHistorySampler
from gptrec.data.log import PAD_ITEM, InteractionLog
from gptrec.models.policy import NextItemPolicy
from gptrec.models.reward_model import RewardModel
from gptrec.models.value_model import ContinuationValueModel
from gptrec.rollout import (
    GenerationConfig,
    RewardFunction,
    RewardSpec,
    SimulatedFeedback,
    append_actions,
    constrain_logits,
    generate_actions,
    simulate_feedback,
)
from gptrec.training.precision import autocast, clipped_optimizer_step, gradient_scaler, seed_everything
from gptrec.training.rl import PROGRESS_REFRESH_SECONDS, Metrics, discounted_returns


@dataclass(frozen=True)
class ContinuationTrainingConfig:
    """How U is learned: by regression on lists the base recommender samples for random moments of the train period."""

    num_steps: int = 1000
    histories_per_step: int = 512
    history_len: int = 118  # plus the horizon of `generation`, this must fit the models' max_positions
    min_history: int = 8
    max_future: int = 20
    generation: GenerationConfig = field(default_factory=GenerationConfig)  # sampled, as in RL, so U is the value of pi_0
    simulated_feedback: SimulatedFeedback = SimulatedFeedback.EXPECTED
    discount: float = 0.9
    learning_rate: float = 1e-4
    max_grad_norm: float = 1.0
    mixed_precision: bool = False
    seed: int = 0


@dataclass(frozen=True)
class GuidanceConfig:
    """How a list is steered at inference."""

    strength: float = 1.0  # beta: the weight of log pi_0 against Q. Large values return the base recommender's lists, small ones follow Q
    discount: float = 0.0  # 0: each item is judged by its own reward; above 0: also by the learned worth of what follows it
    num_candidates: int = 50  # the base recommender's most likely next items that each position chooses from


def fit_continuation_values(
    reference: NextItemPolicy,
    reward_model: RewardModel,
    log: InteractionLog,
    reward: RewardSpec,
    config: ContinuationTrainingConfig,
    device: torch.device,
) -> tuple[ContinuationValueModel, list[Metrics]]:
    """Learns U for `reward`: the discounted reward of the items the base recommender samples after each item, as the reward model scores them."""
    require_guidable(reward)
    seed_everything(config.seed)
    reference = reference.to(device).eval().requires_grad_(False)
    reward_model = reward_model.to(device).eval()
    reward_function = RewardFunction(reward, log.items, device)
    value_model = ContinuationValueModel.from_policy(reference, config.generation.horizon).to(device).eval()
    optimizer = torch.optim.AdamW(value_model.parameters(), lr=config.learning_rate, weight_decay=0.0)
    scaler = gradient_scaler(device, config.mixed_precision)
    moments = TrainingHistorySampler(log, config.history_len, config.max_future, config.min_history, seed=config.seed)

    records: list[Metrics] = []
    for step in tqdm(range(1, config.num_steps + 1), desc="continuation values", mininterval=PROGRESS_REFRESH_SECONDS):
        history = moments.sample(config.histories_per_step).history.to(device)
        with torch.no_grad(), autocast(device, config.mixed_precision):
            actions = generate_actions(reference, history, config.generation).actions
            feedback = simulate_feedback(reward_model, history, actions, config.simulated_feedback)
            targets = continuation_targets(reward_function(feedback, actions), config.discount)
        with autocast(device, config.mixed_precision):
            predicted = continuation_values(value_model, history, actions)
            loss = F.mse_loss(predicted, targets)
        clipped_optimizer_step(loss, optimizer, scaler, value_model.parameters(), config.max_grad_norm)
        unexplained = (targets - predicted.detach()).var() / targets.var().clamp_min(1e-8)
        records.append({"step": step, "loss": loss.item(), "explained_variance": 1.0 - unexplained.item(), "mean_target": targets.mean().item()})
    return value_model.eval().requires_grad_(False), records


def continuation_targets(rewards: Tensor, discount: float) -> Tensor:
    """U after every item of the lists: the discounted reward of the items after it, 0 after the last. [B, K]"""
    returns = discounted_returns(rewards, discount)
    return torch.cat([returns[:, 1:], torch.zeros_like(returns[:, :1])], dim=1)


def continuation_values(value_model: ContinuationValueModel, history: HistoryBatch, actions: Tensor) -> Tensor:
    """The value model's U after every item of the lists. actions: long [B, K] -> [B, K]"""
    horizon = actions.shape[1]
    positions = history.lengths[:, None] + torch.arange(horizon, device=actions.device)[None, :]
    items = append_actions(history, actions)
    list_ranks = torch.arange(1, horizon + 1, device=actions.device).repeat(len(actions), 1)
    list_positions = torch.zeros_like(items).scatter(1, positions, list_ranks)
    return value_model(items, list_positions).gather(1, positions)


@torch.no_grad()
def guided_lists(
    reference: NextItemPolicy,
    reward_model: RewardModel,
    reward_function: RewardFunction,
    value_model: ContinuationValueModel | None,
    history: HistoryBatch,
    guidance: GuidanceConfig,
    generation: GenerationConfig,
) -> Tensor:
    """Lists built one position at a time, each taking the candidate with the highest log pi_0 + Q / strength. long [B, K]

    Q is the candidate's reward as the reward model predicts it after the history and the items already chosen,
    whose expected responses are fed back as in `simulate_feedback`, plus `guidance.discount` times U from
    `value_model`. The item constraints are those of `generation`; its sampling settings play no part.
    """
    require_guidable(reward_function.spec)
    if guidance.discount > 0 and value_model is None:
        raise ValueError("looking ahead needs a value model of what follows each item")
    horizon, num_candidates = generation.horizon, guidance.num_candidates
    num_users, width = history.items.shape
    users = torch.arange(num_users, device=history.items.device)
    trials = torch.arange(num_users * num_candidates, device=history.items.device)
    items = F.pad(history.items, (0, horizon), value=PAD_ITEM)
    played_ratio = F.pad(history.played_ratio, (0, horizon), value=0.0)
    list_positions = torch.zeros_like(items)
    chosen = history.items[:, :0]

    for step in range(horizon):
        position = history.lengths + step
        visible = width + step + 1
        state = reference.hidden_states(items[:, : width + step])[users, position - 1]
        logits = constrain_logits(reference.item_logits(state), history.items, chosen, generation)
        candidate_log_probs, candidates = F.log_softmax(logits.float(), dim=-1).topk(num_candidates, dim=-1)

        trial_position = position.repeat_interleave(num_candidates)
        trial_items = items[:, :visible].repeat_interleave(num_candidates, dim=0)
        trial_items[trials, trial_position] = candidates.reshape(-1)
        predicted = reward_model.predict(trial_items, played_ratio[:, :visible].repeat_interleave(num_candidates, dim=0))
        feedback = predicted.map(lambda values: values[trials, trial_position].reshape(num_users, num_candidates))
        q_value = reward_function.item_reward(feedback, candidates)
        if step == horizon - 1 and reward_function.spec.rewards_the_end_of_the_list:
            q_value = q_value + reward_function.spec.final_returns_weight * feedback.returns
        if guidance.discount > 0:
            trial_list_positions = list_positions[:, :visible].repeat_interleave(num_candidates, dim=0)
            trial_list_positions[trials, trial_position] = step + 1
            continuation = value_model(trial_items, trial_list_positions)[trials, trial_position].reshape(num_users, num_candidates)
            q_value = q_value + guidance.discount * continuation

        best = (candidate_log_probs + q_value.float() / guidance.strength).argmax(dim=1)
        action = candidates[users, best]
        items[users, position] = action
        list_positions[users, position] = step + 1
        played_ratio[users, position] = feedback.played_ratio[users, best].to(played_ratio.dtype)
        chosen = torch.cat([chosen, action[:, None]], dim=1)
    return chosen


def require_guidable(reward: RewardSpec) -> None:
    """Guided decoding judges each candidate for its own user, from predictions alone, one position at a time."""
    if reward.needs_other_user or reward.needs_logged_future or reward.depends_on_the_list:
        raise ValueError("a reward measured against another user, read from the logged future, or of the list as a whole cannot guide decoding")
