"""Generating recommendations item by item, and simulating the user's response to them.

The recommender only reads item tokens, so its choices do not depend on the simulated feedback.
Generation and simulation are therefore separate steps, and any reward model can score any list.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor

from gptrec.data.batches import HistoryBatch
from gptrec.data.log import PAD_ITEM, ItemTable
from gptrec.evaluation.behavior import pairwise_item_distances
from gptrec.evaluation.relevance import contains
from gptrec.models.policy import NextItemPolicy
from gptrec.models.sequence_recommender import MASKED_LOGIT, NextItemScorer
from gptrec.models.value_model import ValueModel
from gptrec.models.reward_model import Feedback, RewardModel
from gptrec.tensors import TensorGroup


@dataclass(frozen=True)
class GenerationConfig:
    horizon: int = 10  # number of items generated one after another
    greedy: bool = False  # take the most likely item instead of sampling
    temperature: float = 1.0
    block_generated_items: bool = True  # a generated list never contains an item twice
    block_history_items: bool = False  # never recommend what is already in the visible history
    sample_from_top: int | None = None  # choose only among this many most likely items; None: among all of them


class RatioBaseline(str, Enum):
    """What the predicted played ratio of an item is measured against before it counts as reward."""

    NONE = "none"  # nothing: an item everybody finishes earns the most, whoever it is shown to
    ITEM_MEAN = "item_mean"  # the item's mean played ratio in the train period
    OTHER_USER = "other_user"  # the ratio the same reward model predicts when the same list is shown to another user


class SimulatedFeedback(str, Enum):
    """What the reward model is told about the user's response to an already generated item."""

    EXPECTED = "expected"  # its own expected played ratio
    SAMPLED = "sampled"  # a draw: fully played with probability equal to the expected ratio, else skipped
    ASSUME_COMPLETED = "assume_completed"  # the user played everything


@dataclass
class GeneratedActions(TensorGroup):
    actions: Tensor  # long [B, K], generated items in order
    log_probs: Tensor  # float [B, K], log-probability of each action when it was chosen
    entropy: Tensor  # float [B, K], entropy of the distribution each action was chosen from


def constrain_logits(logits: Tensor, history_items: Tensor, generated_items: Tensor, config: GenerationConfig) -> Tensor:
    """Applies temperature and removes the items that may not be recommended next.

    logits: [B, V+1], history_items: long [B, H], generated_items: long [B, k] -> [B, V+1]
    """
    blocked = [torch.full_like(history_items[:, :1], PAD_ITEM)]
    if config.block_generated_items:
        blocked.append(generated_items)
    if config.block_history_items:
        blocked.append(history_items)
    allowed = (logits / config.temperature).scatter(1, torch.cat(blocked, dim=1), MASKED_LOGIT)
    if config.sample_from_top is not None:
        lowest_kept_logit = allowed.topk(config.sample_from_top, dim=-1).values[:, -1:]
        allowed = allowed.masked_fill(allowed < lowest_kept_logit, MASKED_LOGIT)
    return allowed


def histories_of_other_users(history: HistoryBatch, shift: int = 1) -> HistoryBatch:
    """Row b gets the history of row b - shift, so that every list in a batch meets a user it was not made for.

    The pairing is arbitrary, not random. `shift` must not be a multiple of the batch size, and where each
    history is repeated several times in a row it must be that number of repeats.
    """
    return history.map(lambda tensor: tensor.roll(shift, dims=0))


def append_actions(history: HistoryBatch, actions: Tensor) -> Tensor:
    """Item tokens of each history followed directly by its generated items. long [B, H + K]"""
    horizon = actions.shape[1]
    items = F.pad(history.items, (0, horizon), value=PAD_ITEM)
    return items.scatter(1, _action_positions(history, horizon), actions)


def _action_positions(history: HistoryBatch, horizon: int) -> Tensor:
    """Column of the k-th generated item in the extended sequence. long [B, K]"""
    return history.lengths[:, None] + torch.arange(horizon, device=history.lengths.device)[None, :]


@torch.no_grad()
def recommend_top_k(scorer: NextItemScorer, history: HistoryBatch, k: int, config: GenerationConfig) -> Tensor:
    """Score-and-rank: the k best next items, none of them fed back to the model. long [B, k], best first."""
    nothing_generated = history.items[:, :0]
    logits = scorer.next_item_logits(history.items, history.lengths)
    return constrain_logits(logits, history.items, nothing_generated, config).topk(k, dim=-1).indices


@torch.no_grad()
def generate_actions(policy: NextItemPolicy, history: HistoryBatch, config: GenerationConfig) -> GeneratedActions:
    """Extends each history with `config.horizon` items, feeding every generated item back as input."""
    batch_index = torch.arange(len(history), device=history.items.device)
    history_width = history.items.shape[1]
    items = F.pad(history.items, (0, config.horizon), value=PAD_ITEM)
    actions = torch.zeros((len(history), 0), dtype=torch.long, device=items.device)
    log_probs, entropies = [], []

    for step in range(config.horizon):
        last_position = history.lengths - 1 + step
        state = policy.hidden_states(items[:, : history_width + step])[batch_index, last_position]
        step_log_probs = F.log_softmax(constrain_logits(policy.item_logits(state), history.items, actions, config), dim=-1)
        action = step_log_probs.argmax(dim=-1) if config.greedy else torch.multinomial(step_log_probs.exp(), num_samples=1).squeeze(1)

        items[batch_index, last_position + 1] = action
        actions = torch.cat([actions, action[:, None]], dim=1)
        log_probs.append(step_log_probs[batch_index, action])
        entropies.append(_entropy(step_log_probs))

    return GeneratedActions(actions, torch.stack(log_probs, dim=1), torch.stack(entropies, dim=1))


def action_log_distributions(policy: NextItemPolicy, history: HistoryBatch, actions: Tensor, config: GenerationConfig) -> Tensor:
    """log pi(. | history, actions[:, :k]) for every step k under the constraints used while generating.

    Differentiable counterpart of `generate_actions`: one pass over the already generated sequence. [B, K, V+1]
    """
    horizon = actions.shape[1]
    hidden = policy.hidden_states(append_actions(history, actions))
    state_positions = _action_positions(history, horizon) - 1
    states = hidden.gather(1, state_positions[:, :, None].expand(-1, -1, hidden.shape[-1]))
    logits = policy.item_logits(states)
    constrained = [constrain_logits(logits[:, step], history.items, actions[:, :step], config) for step in range(horizon)]
    return F.log_softmax(torch.stack(constrained, dim=1), dim=-1)


def state_values(value_model: ValueModel, history: HistoryBatch, actions: Tensor) -> Tensor:
    """The value model's estimate of the return still to come, in each state in which an item of the list was chosen. [B, K]"""
    values = value_model(append_actions(history, actions))
    return values.gather(1, _action_positions(history, actions.shape[1]) - 1)


@torch.no_grad()
def simulate_feedback(reward_model: RewardModel, history: HistoryBatch, actions: Tensor, mode: SimulatedFeedback) -> Feedback:
    """The reward model's expected response to every generated item, played out step by step. Two [B, K] tensors.

    The model sees the real feedback on the history; for generated items it sees the feedback chosen by `mode`.
    """
    mode = SimulatedFeedback(mode)
    batch_index = torch.arange(len(history), device=actions.device)
    history_width, horizon = history.items.shape[1], actions.shape[1]
    items = append_actions(history, actions)
    played_ratio = F.pad(history.played_ratio, (0, horizon), value=0.0)
    steps = []

    for step in range(horizon):
        position = history.lengths + step
        visible = history_width + step + 1
        prediction = reward_model.predict(items[:, :visible], played_ratio[:, :visible])
        expected = prediction.map(lambda values: values[batch_index, position])
        played_ratio[batch_index, position] = _fed_back_played_ratio(expected.played_ratio, mode)
        steps.append(expected)

    return Feedback.stack(steps, dim=1)


def _fed_back_played_ratio(expected_played_ratio: Tensor, mode: SimulatedFeedback) -> Tensor:
    if mode is SimulatedFeedback.EXPECTED:
        return expected_played_ratio
    if mode is SimulatedFeedback.SAMPLED:
        return torch.bernoulli(expected_played_ratio)
    if mode is SimulatedFeedback.ASSUME_COMPLETED:
        return torch.ones_like(expected_played_ratio)
    raise ValueError(f"unknown simulated feedback mode: {mode}")


def _entropy(log_probs: Tensor) -> Tensor:
    return -(log_probs.exp() * log_probs).sum(dim=-1)


@dataclass(frozen=True)
class RewardSpec:
    """How the response to one recommended item becomes a scalar reward.

    reward = P(chosen) * (played_ratio_weight * ratio term
                          + relative_ratio_weight * relative played ratio
                          + session_continues_weight * P(session continues)
                          + returns_weight * P(user returns))
             + logged_relevance_weight * [the user really consumed the item next] / log2(position in the list + 1)
             + diversity_weight * distance of the item to the items before it in the list

    P(chosen) is left out unless `weigh_by_choice` is set. The first line is predicted by a reward model
    for any item. The second is read from the log and exists only for moments whose future is known:
    it is the immediate reward of NDCG in Petrov & Macdonald (arXiv:2403.04875). The third depends on the
    rest of the list, so it is the one term that scoring items one at a time cannot maximise.

    The relative played ratio is the played ratio as a quantile among everyone who consumed the item. Its
    mean is 0.5 for every item, so no item earns it by being short or easy to finish.

    The ratio term is the predicted played ratio minus `ratio_baseline`. Without a baseline a policy earns
    the most by showing everybody the items everybody finishes. With one it earns only what depends on who
    receives the item: measured against another user, a list that ignores its user earns zero on average.
    """

    played_ratio_weight: float = 1.0
    ratio_baseline: RatioBaseline = RatioBaseline.NONE
    relative_ratio_weight: float = 0.0
    session_continues_weight: float = 0.0
    returns_weight: float = 0.0
    # Reward the time spent rather than the fraction played: ratio * (item duration / median duration).
    # Removes the incentive to recommend short items, which are easy to finish.
    weigh_ratio_by_duration: bool = False
    # Count the predicted response only as far as the user is predicted to take the item up at all. Without it
    # an item nobody like this user ever picked earns whatever those who did pick it gave it.
    weigh_by_choice: bool = False
    logged_relevance_weight: float = 0.0
    # Cosine distance, in the base recommender's item embeddings, to the items already in the list, summed and divided
    # by K - 1. Over a list of K items these terms add up to K / 2 times its mean pairwise distance.
    diversity_weight: float = 0.0

    @property
    def needs_logged_future(self) -> bool:
        return self.logged_relevance_weight != 0.0

    @property
    def needs_other_user(self) -> bool:
        return RatioBaseline(self.ratio_baseline) is RatioBaseline.OTHER_USER

    @property
    def depends_on_the_list(self) -> bool:
        return self.diversity_weight != 0.0


class RewardFunction:
    """Computes the reward of `RewardSpec` for lists of items."""

    def __init__(self, spec: RewardSpec, items: ItemTable, device: torch.device | str, item_embeddings: Tensor | None = None) -> None:
        """item_embeddings: [V+1, d], the space in which the distance between two items is measured; needed only for the diversity term."""
        self.spec = spec
        duration = torch.as_tensor(items.duration_seconds, dtype=torch.float32, device=device)
        self._relative_duration = duration / duration[1:].median().clamp_min(1e-6)
        self._item_mean_played_ratio = torch.as_tensor(items.mean_played_ratio, dtype=torch.float32, device=device)
        if spec.depends_on_the_list and item_embeddings is None:
            raise ValueError("this reward has a diversity term, which needs item embeddings to measure distances in")
        self.unit_item_embeddings = None if item_embeddings is None else F.normalize(item_embeddings.detach().float().to(device), dim=-1)

    def __call__(self, feedback: Feedback, lists: Tensor, future_items: Tensor | None = None, feedback_for_other_user: Feedback | None = None) -> Tensor:
        """Reward of every item of every list. [B, K]

        feedback: the response to each item, predicted or logged, with the shape of `lists`.
        lists: long [B, K], column k is position k + 1 of a recommendation list.
        future_items: long [B, M], what each user really consumed next; needed only for logged relevance.
        feedback_for_other_user: the predicted response of another user to the same lists; needed only for that baseline.
        """
        reward = self.item_reward(feedback, lists, future_items, feedback_for_other_user)
        if self.spec.depends_on_the_list:
            reward = reward + self.spec.diversity_weight * distance_to_earlier_items(lists, self.unit_item_embeddings)
        return reward

    def item_reward(self, feedback: Feedback, lists: Tensor, future_items: Tensor | None = None, feedback_for_other_user: Feedback | None = None) -> Tensor:
        """The part of the reward an item earns whatever else is in its list. Arguments and shape as in `__call__`."""
        spec = self.spec
        ratio_term = feedback.played_ratio - self._ratio_baseline(lists, feedback_for_other_user)
        if spec.weigh_ratio_by_duration:
            ratio_term = ratio_term * self._relative_duration[lists]
        reward = (
            spec.played_ratio_weight * ratio_term
            + spec.relative_ratio_weight * feedback.relative_ratio
            + spec.session_continues_weight * feedback.session_continues
            + spec.returns_weight * feedback.returns
        )
        if spec.weigh_by_choice:
            reward = reward * feedback.is_chosen
        if spec.needs_logged_future:
            if future_items is None:
                raise ValueError("this reward has a logged-relevance term, which needs the items the user really consumed next")
            reward = reward + spec.logged_relevance_weight * discounted_logged_relevance(lists, future_items)
        return reward

    def _ratio_baseline(self, lists: Tensor, feedback_for_other_user: Feedback | None) -> Tensor | float:
        baseline = RatioBaseline(self.spec.ratio_baseline)
        if baseline is RatioBaseline.ITEM_MEAN:
            return self._item_mean_played_ratio[lists]
        if baseline is RatioBaseline.OTHER_USER:
            if feedback_for_other_user is None:
                raise ValueError("this reward is measured against another user, whose predicted response is missing")
            return feedback_for_other_user.played_ratio
        return 0.0


def discounted_logged_relevance(lists: Tensor, future_items: Tensor) -> Tensor:
    """1 / log2(position + 1) where the listed item is among the user's real next items, else 0: the terms of DCG. [B, K]"""
    position = torch.arange(1, lists.shape[1] + 1, device=lists.device)
    return contains(lists, future_items).float() / torch.log2(position + 1.0)


def distance_to_earlier_items(lists: Tensor, unit_item_embeddings: Tensor) -> Tensor:
    """For each listed item, its cosine distance to the items before it in the list, summed and divided by K - 1. [B, K]

    The first item earns nothing. Over a list the terms add up to K / 2 times its mean pairwise distance, so
    what an item earns depends on its neighbours: this is a reward for the list as a whole.
    """
    list_length = lists.shape[1]
    comes_earlier = torch.ones(list_length, list_length, dtype=torch.bool, device=lists.device).tril(diagonal=-1)
    return (pairwise_item_distances(lists, unit_item_embeddings) * comes_earlier).sum(dim=2) / max(1, list_length - 1)


MAX_SEQUENCES_PER_PASS = 4096  # bounds the memory of scoring many candidate items or lists per user
USERS_PER_GREEDY_PASS = 1024  # bounds the memory of comparing every candidate of a user with the items already picked


@dataclass
class ScoredCandidates(TensorGroup):
    """The recommender's most likely next items for each user, with the response a reward model predicts to each."""

    items: Tensor  # long [U, N], most likely first
    log_probs: Tensor  # float [U, N], the recommender's log-probability of each as the next item
    feedback: Feedback  # predicted response if the item is recommended next, [U, N]
    feedback_for_other_user: Feedback  # predicted response if the same item is recommended next to another user, [U, N]


@torch.no_grad()
def score_candidates(scorer: NextItemScorer, reward_model: RewardModel, history: HistoryBatch, num_candidates: int, config: GenerationConfig) -> ScoredCandidates:
    """Lets the reward model judge each of the recommender's `num_candidates` most likely next items on its own.

    Every candidate is judged twice: for the user it is a candidate for, and for another user of the batch.
    """
    nothing_generated = history.items[:, :0]
    logits = constrain_logits(scorer.next_item_logits(history.items, history.lengths), history.items, nothing_generated, config)
    log_probs, candidates = F.log_softmax(logits, dim=-1).topk(num_candidates, dim=-1)
    users_per_pass = max(1, MAX_SEQUENCES_PER_PASS // num_candidates)

    def judged_for(recipients: HistoryBatch) -> Feedback:
        feedback = []
        for first_user in range(0, len(recipients), users_per_pass):
            users = slice(first_user, first_user + users_per_pass)
            one_item_lists = candidates[users].reshape(-1, 1)
            history_per_candidate = recipients[users].repeat_interleave(num_candidates)
            predicted = simulate_feedback(reward_model, history_per_candidate, one_item_lists, SimulatedFeedback.EXPECTED)
            feedback.append(predicted.map(lambda values: values.reshape(-1, num_candidates)))
        return Feedback.concatenate(feedback)

    return ScoredCandidates(candidates, log_probs, judged_for(history), judged_for(histories_of_other_users(history)))


def rerank(candidates: ScoredCandidates, reward_function: RewardFunction, list_length: int, relevance_weight: float) -> Tensor:
    """The greedy alternative to RL: the `list_length` candidates with the highest `predicted reward + relevance_weight * log-probability`.

    Weight 0 ranks by reward alone; a large weight returns the recommender's own top list. A reward that depends
    on the list is maximised one position at a time, see `pick_greedily`. long [U, list_length], best first.
    """
    reward = reward_function.item_reward(candidates.feedback, candidates.items, feedback_for_other_user=candidates.feedback_for_other_user)
    score = reward + relevance_weight * candidates.log_probs
    if not reward_function.spec.depends_on_the_list:
        return candidates.items.gather(1, score.topk(list_length, dim=-1).indices)
    return pick_greedily(candidates.items, score, reward_function.unit_item_embeddings, reward_function.spec.diversity_weight, list_length)


def pick_greedily(items: Tensor, scores: Tensor, unit_item_embeddings: Tensor, diversity_weight: float, list_length: int) -> Tensor:
    """Maximal marginal relevance: fills a list one position at a time with the best candidate given the items already picked.

    A candidate is worth its score plus `diversity_weight` times what `distance_to_earlier_items` would pay it
    at that position. This is the greedy maximiser of the list reward RL is trained on, and needs no training.
    items: long [U, N] candidates; scores: float [U, N] -> long [U, list_length]
    """
    picked_lists = []
    for first_user in range(0, len(items), USERS_PER_GREEDY_PASS):
        users = slice(first_user, first_user + USERS_PER_GREEDY_PASS)
        candidates, remaining_scores = items[users], scores[users].clone()
        embedded = unit_item_embeddings[candidates]
        user_index = torch.arange(len(candidates), device=candidates.device)
        similarity_to_picked = torch.zeros_like(remaining_scores)
        picked = []
        for position in range(list_length):
            distance_to_picked = (position - similarity_to_picked) / max(1, list_length - 1)
            best = (remaining_scores + diversity_weight * distance_to_picked).argmax(dim=1)
            picked.append(candidates[user_index, best])
            similarity_to_picked = similarity_to_picked + (embedded @ embedded[user_index, best][:, :, None]).squeeze(2).float()
            remaining_scores[user_index, best] = MASKED_LOGIT
        picked_lists.append(torch.stack(picked, dim=1))
    return torch.cat(picked_lists)


@torch.no_grad()
def best_of_sampled_lists(
    policy: NextItemPolicy,
    reward_model: RewardModel,
    reward_function: RewardFunction,
    history: HistoryBatch,
    num_lists: int,
    discount: float,
    config: GenerationConfig,
    simulated_feedback: SimulatedFeedback,
) -> Tensor:
    """Samples `num_lists` lists per user and keeps the one with the highest predicted discounted return. long [B, K]

    The test-time scaling of Matveev et al. (arXiv:2607.02818), which serves the first item of the best
    roll-out; here the whole roll-out is the recommendation list. Nothing is trained.
    """
    sampling = replace(config, greedy=False)
    discounts = discount ** torch.arange(config.horizon, device=history.items.device)
    users_per_pass = max(1, MAX_SEQUENCES_PER_PASS // num_lists)
    best_lists = []
    for first_user in range(0, len(history), users_per_pass):
        repeated_history = history[first_user : first_user + users_per_pass].repeat_interleave(num_lists)
        lists = generate_actions(policy, repeated_history, sampling).actions
        rewards = reward_function(simulate_feedback(reward_model, repeated_history, lists, simulated_feedback), lists)
        predicted_return = (rewards * discounts).sum(dim=1).reshape(-1, num_lists)
        lists = lists.reshape(-1, num_lists, config.horizon)
        best_lists.append(lists[torch.arange(len(lists), device=lists.device), predicted_return.argmax(dim=1)])
    return torch.cat(best_lists)
