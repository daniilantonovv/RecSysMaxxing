"""RL fine-tuning of the recommender against a reward model, with PPO.

One iteration:
  1. sample recommendation lists from the current policy, for random moments of train-period histories;
  2. let the reward model predict the user's response to every item of every list: the rewards;
  3. estimate advantages - how much better each item did than its state promised;
  4. take several gradient steps on PPO's clipped objective, reusing the sampled lists.

A value model (the critic) learns what return a state promises and supplies the baseline of step 3
through generalised advantage estimation. A KL penalty towards the pre-trained policy keeps the
fine-tuned one close to observed behaviour. Unless the reward itself asks for relevance (see
`RewardSpec`), that penalty is the only thing that does, and its weight is the main knob of the
relevance / reward trade-off.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

from gptrec.data.batches import EvalExamples, HistoryBatch, TrainingHistorySampler
from gptrec.data.log import InteractionLog
from gptrec.evaluation.relevance import contains
from gptrec.models.policy import NextItemPolicy
from gptrec.models.reward_model import RewardModel
from gptrec.models.value_model import ValueModel
from gptrec.rollout import (
    GeneratedActions,
    GenerationConfig,
    RewardFunction,
    RewardSpec,
    SimulatedFeedback,
    action_log_distributions,
    generate_actions,
    histories_of_other_users,
    simulate_feedback,
    state_values,
)
from gptrec.training.precision import autocast, clipped_optimizer_step, gradient_scaler, seed_everything

Metrics = dict[str, float]

PROGRESS_REFRESH_SECONDS = 5.0  # keeps the logs of long unattended runs short


class AdvantageEstimator(str, Enum):
    """Where the baseline comes from that turns returns into advantages."""

    GAE = "gae"  # a learned value model with generalised advantage estimation: PPO as it is usually run
    LEAVE_ONE_OUT = "leave_one_out"  # the mean return of the other lists sampled in the same iteration; no value model


@dataclass(frozen=True)
class RLConfig:
    num_steps: int = 2000  # iterations; each samples new lists and takes `ppo_epochs` gradient steps on them
    histories_per_step: int = 512
    rollouts_per_history: int = 1  # lists sampled for the same history
    history_len: int = 118  # plus the generation horizon, this must fit the models' max_positions
    min_history: int = 8  # train events a user needs before a recommendation moment is sampled from their timeline
    max_future: int = 20  # next train events of the user that a logged-relevance reward counts as relevant
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    simulated_feedback: SimulatedFeedback = SimulatedFeedback.EXPECTED
    reward: RewardSpec = field(default_factory=RewardSpec)

    discount: float = 0.99
    advantage_estimator: AdvantageEstimator = AdvantageEstimator.GAE
    gae_lambda: float = 0.9  # 0 trusts the value model fully (one-step advantages), 1 ignores its bootstrap (full returns)
    normalize_advantages: bool = True  # zero-mean, unit-variance advantages keep the KL weight comparable across rewards

    ppo_epochs: int = 4  # gradient steps on each batch of sampled lists
    ppo_clip: float = 0.2  # how far the probability of a sampled item may move, as a ratio, before its gradient is cut
    # Penalty on KL(policy || pre-trained policy), per generated item. It is the only thing holding the policy near
    # the data: with REINFORCE on Yambda the policy collapsed at 0.1 and held at 0.3.
    reference_kl_weight: float = 0.3
    entropy_weight: float = 0.0

    learning_rate: float = 1e-5
    value_learning_rate: float = 1e-4
    value_warmup_steps: int = 50  # first iterations in which only the value model learns, so the policy never follows an untrained critic
    max_grad_norm: float = 1.0
    evaluate_every: int = 200
    mixed_precision: bool = False
    seed: int = 0


def finetune_with_rl(
    pretrained_policy: NextItemPolicy,
    reward_model: RewardModel,
    log: InteractionLog,
    config: RLConfig,
    device: torch.device,
    evaluate: Callable[[NextItemPolicy], Metrics] | None = None,
) -> tuple[NextItemPolicy, list[Metrics]]:
    """Fine-tunes a copy of `pretrained_policy`; the original stays frozen as the KL reference.

    Returns the fine-tuned policy and one metrics record per iteration. Every `evaluate_every` iterations
    (and before the first one) the record also holds `evaluate(policy)` under "validation/" keys.
    """
    seed_everything(config.seed)
    reference = pretrained_policy.to(device).eval().requires_grad_(False)
    # Dropout stays off: the log-probabilities recorded while sampling must equal those recomputed for the update.
    policy = copy.deepcopy(reference).eval().requires_grad_(True)
    reward_model = reward_model.to(device).eval()
    reward_function = RewardFunction(config.reward, log.items, device)
    moments = TrainingHistorySampler(log, config.history_len, config.max_future, config.min_history, seed=config.seed)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate, weight_decay=0.0)
    scaler = gradient_scaler(device, config.mixed_precision)
    uses_critic = config.advantage_estimator is AdvantageEstimator.GAE
    critic = _Critic(reference, config, device) if uses_critic else None
    sampling_statistics = _SamplingStatistics(log, device)

    def validation_metrics() -> Metrics:
        return {f"validation/{name}": value for name, value in evaluate(policy).items()} if evaluate else {}

    records: list[Metrics] = [{"step": 0} | validation_metrics()]
    for step in tqdm(range(1, config.num_steps + 1), desc="RL fine-tuning", mininterval=PROGRESS_REFRESH_SECONDS):
        sampled = moments.sample(config.histories_per_step).to(device).map(lambda tensor: tensor.repeat_interleave(config.rollouts_per_history, dim=0))
        history = sampled.history
        with torch.no_grad(), autocast(device, config.mixed_precision):
            generated = generate_actions(policy, history, config.generation)
            feedback = simulate_feedback(reward_model, history, generated.actions, config.simulated_feedback)
            feedback_for_other_user = None
            if config.reward.needs_other_user:
                other_users = histories_of_other_users(history, shift=config.rollouts_per_history)
                feedback_for_other_user = simulate_feedback(reward_model, other_users, generated.actions, config.simulated_feedback)
            rewards = reward_function(feedback, generated.actions, sampled.future_items, feedback_for_other_user)
            returns = discounted_returns(rewards, config.discount)
            if critic is not None:
                advantages = generalised_advantages(rewards, critic.values(history, generated.actions), config.discount, config.gae_lambda)
            else:
                advantages = leave_one_out_advantages(returns, config.rollouts_per_history)
            if config.normalize_advantages:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            reference_log_distributions = action_log_distributions(reference, history, generated.actions, config.generation)

        record: Metrics = {"step": step, "reward": rewards.mean().item()}
        record |= {f"predicted_{signal}": values.mean().item() for signal, values in vars(feedback).items()}
        if feedback_for_other_user is not None:
            record["predicted_personal_played_ratio"] = (feedback.played_ratio - feedback_for_other_user.played_ratio).mean().item()
        record |= sampling_statistics(sampled, generated)
        policy_is_learning = critic is None or step > config.value_warmup_steps
        for _ in range(config.ppo_epochs):
            if policy_is_learning:
                with autocast(device, config.mixed_precision):
                    log_distributions = action_log_distributions(policy, history, generated.actions, config.generation)
                    loss, loss_terms = policy_improvement_loss(log_distributions, reference_log_distributions, generated, advantages, config)
                gradient_norm = clipped_optimizer_step(loss, optimizer, scaler, policy.parameters(), config.max_grad_norm)
                record |= loss_terms | {"gradient_norm": gradient_norm.item()}
            if critic is not None:
                record |= critic.learn(history, generated.actions, returns)

        if step % config.evaluate_every == 0 or step == config.num_steps:
            record |= validation_metrics()
        records.append(record)

    return policy, records


def discounted_returns(rewards: Tensor, discount: float) -> Tensor:
    """returns[:, k] = rewards[:, k] + discount * rewards[:, k + 1] + ...  for rewards of shape [B, K]."""
    returns = torch.zeros_like(rewards)
    return_from_next_step = torch.zeros_like(rewards[:, 0])
    for step in reversed(range(rewards.shape[1])):
        return_from_next_step = rewards[:, step] + discount * return_from_next_step
        returns[:, step] = return_from_next_step
    return returns


def generalised_advantages(rewards: Tensor, values: Tensor, discount: float, gae_lambda: float) -> Tensor:
    """Generalised advantage estimation (Schulman et al., 2016) for lists that end after their last item.

    rewards, values: [B, K]; values[:, k] is the value of the state in which item k was chosen.
    advantage[:, k] = sum over l of (discount * gae_lambda)^l * residual[:, k + l], where
    residual[:, k] = rewards[:, k] + discount * values[:, k + 1] - values[:, k] and the value after the last item is 0.
    """
    value_of_next_state = torch.cat([values[:, 1:], torch.zeros_like(values[:, :1])], dim=1)
    residuals = rewards + discount * value_of_next_state - values
    advantages = torch.zeros_like(residuals)
    advantage_of_next_step = torch.zeros_like(residuals[:, 0])
    for step in reversed(range(residuals.shape[1])):
        advantage_of_next_step = residuals[:, step] + discount * gae_lambda * advantage_of_next_step
        advantages[:, step] = advantage_of_next_step
    return advantages


def leave_one_out_advantages(returns: Tensor, rollouts_per_history: int) -> Tensor:
    """Each return minus the mean return, at the same step, of the other rollouts that share its history.

    The baseline of a rollout never depends on that rollout's own actions, so the policy gradient stays
    unbiased. With one rollout per history, all the other rollouts of the batch serve as the baseline.

    returns: [B * rollouts_per_history, K] with the rollouts of one history in consecutive rows.
    """
    horizon = returns.shape[1]
    grouped = returns.reshape(-1, rollouts_per_history, horizon) if rollouts_per_history > 1 else returns.reshape(1, -1, horizon)
    group_size = grouped.shape[1]
    if group_size < 2:
        raise ValueError("a leave-one-out baseline needs at least two rollouts")
    mean_of_others = (grouped.sum(dim=1, keepdim=True) - grouped) / (group_size - 1)
    return (grouped - mean_of_others).reshape(returns.shape)


def policy_improvement_loss(
    log_distributions: Tensor,
    reference_log_distributions: Tensor,
    generated: GeneratedActions,
    advantages: Tensor,
    config: RLConfig,
) -> tuple[Tensor, Metrics]:
    """PPO's clipped surrogate, plus the KL penalty towards the reference policy and the entropy bonus.

    log_distributions, reference_log_distributions: [B, K, V+1]; advantages: [B, K].
    """
    action_log_probs = log_distributions.gather(2, generated.actions[:, :, None]).squeeze(2)
    probability_ratio = (action_log_probs - generated.log_probs).exp()
    clipped_ratio = probability_ratio.clamp(1.0 - config.ppo_clip, 1.0 + config.ppo_clip)
    surrogate = torch.minimum(probability_ratio * advantages, clipped_ratio * advantages).mean()

    probabilities = log_distributions.exp()
    kl_to_reference = (probabilities * (log_distributions - reference_log_distributions)).sum(dim=-1).mean()
    entropy = -(probabilities * log_distributions).sum(dim=-1).mean()

    loss = -surrogate + config.reference_kl_weight * kl_to_reference - config.entropy_weight * entropy
    was_clipped = (probability_ratio - 1.0).abs() > config.ppo_clip
    return loss, {
        "loss": loss.item(),
        "kl_to_reference": kl_to_reference.item(),
        "entropy": entropy.item(),
        "clip_fraction": was_clipped.float().mean().item(),
    }


class _Critic:
    """PPO's value model together with its optimiser. It starts from the pre-trained recommender's transformer."""

    def __init__(self, pretrained_policy: NextItemPolicy, config: RLConfig, device: torch.device) -> None:
        self._config = config
        self._device = device
        # No dropout, like the policy: the values used for the advantages and those being fitted come from the same function.
        self._value_model = ValueModel.from_policy(pretrained_policy).to(device).eval()
        self._optimizer = torch.optim.AdamW(self._value_model.parameters(), lr=config.value_learning_rate, weight_decay=0.0)
        self._scaler = gradient_scaler(device, config.mixed_precision)

    @torch.no_grad()
    def values(self, history: HistoryBatch, actions: Tensor) -> Tensor:
        """Expected return from the state in which each item of the list was chosen. [B, K]"""
        return state_values(self._value_model, history, actions)

    def learn(self, history: HistoryBatch, actions: Tensor, returns: Tensor) -> Metrics:
        """One gradient step towards the returns that were actually collected."""
        with autocast(self._device, self._config.mixed_precision):
            values = state_values(self._value_model, history, actions)
            loss = F.mse_loss(values, returns)
        clipped_optimizer_step(loss, self._optimizer, self._scaler, self._value_model.parameters(), self._config.max_grad_norm)
        unexplained_variance = (returns - values.detach()).var() / returns.var().clamp_min(1e-8)
        return {"value_loss": loss.item(), "value_explained_variance": 1.0 - unexplained_variance.item()}


class _SamplingStatistics:
    """Cheap per-iteration indicators of what the policy currently samples, to watch for collapse while training."""

    def __init__(self, log: InteractionLog, device: torch.device) -> None:
        self._popularity = torch.as_tensor(log.items.train_popularity, dtype=torch.float32, device=device)
        self._duration = torch.as_tensor(log.items.duration_seconds, dtype=torch.float32, device=device)

    def __call__(self, sampled: EvalExamples, generated: GeneratedActions) -> Metrics:
        actions = generated.actions
        return {
            "distinct_action_share": actions.unique().numel() / actions.numel(),
            "logged_hit_rate": contains(actions, sampled.future_items).float().mean().item(),  # relevance, measured while training
            "history_repeat_rate": contains(actions, sampled.history.items).float().mean().item(),
            "mean_action_popularity": self._popularity[actions].mean().item(),
            "mean_action_duration_seconds": self._duration[actions].mean().item(),
        }
