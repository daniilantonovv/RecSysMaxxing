"""Everything measured about one recommender on held-out users: relevance, predicted reward, behaviour."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable

import pandas as pd
import torch
from torch import Tensor

from gptrec.data.batches import EvalExamples, HistoryBatch
from gptrec.data.log import PAD_ITEM, ItemTable
from gptrec.evaluation.behavior import recommendation_profile
from gptrec.evaluation.logged_feedback import logged_reception, logged_reward_per_user
from gptrec.evaluation.relevance import contains, per_user_ranking_metrics, ranking_metrics
from gptrec.evaluation.significance import paired_bootstrap
from gptrec.models.policy import NextItemPolicy
from gptrec.models.reward_model import Feedback, RewardModel
from gptrec.rollout import (
    GeneratedActions,
    GenerationConfig,
    SimulatedFeedback,
    generate_actions,
    histories_of_other_users,
    recommend_top_k,
    simulate_feedback,
)
from gptrec.training.precision import autocast


@dataclass(frozen=True)
class EvaluationConfig:
    ks: tuple[int, ...] = (1, 5, 10)
    generation: GenerationConfig = field(default_factory=lambda: GenerationConfig(greedy=True))
    simulated_feedback: SimulatedFeedback = SimulatedFeedback.EXPECTED
    liked_threshold: float = 0.5  # a future item counts as liked when at least this fraction of it was played
    batch_size: int = 256
    mixed_precision: bool = False


@dataclass
class PolicyEvaluation:
    """Scalar metrics plus the lists they were computed from, for further analysis. Tensors are on the CPU."""

    metrics: dict[str, float]
    top_k: Tensor  # long [U, max(ks)], score-and-rank recommendations
    generated: GeneratedActions  # [U, horizon], recommendations generated item by item
    feedback: dict[str, Feedback]  # reward model name -> its predicted response to `generated.actions`, [U, horizon]
    feedback_for_other_user: dict[str, Feedback]  # the same, had each list been shown to another user

    def generated_lists(self) -> ListEvaluation:
        """The item-by-item generated lists in the form shared with lists from any other source."""
        metrics = {name: value for name, value in list_metrics(self.metrics, "next_k").items() if "/" in name}
        metrics |= {name: value for name, value in self.metrics.items() if name.startswith(("predicted/", "training/"))}
        return ListEvaluation(metrics, self.generated.actions, self.feedback, self.feedback_for_other_user)


@dataclass
class ListEvaluation:
    """What is measured about one list per user, however the lists were produced. Tensors are on the CPU.

    Metric names: <target set>/<metric>, profile/<statistic>, logged/<statistic>, predicted/<reward model>/<signal>.
    """

    metrics: dict[str, float]
    lists: Tensor  # long [U, K]
    feedback: dict[str, Feedback]  # reward model name -> its predicted response to `lists`, [U, K]
    feedback_for_other_user: dict[str, Feedback]  # the same, had each list been shown to another user


@torch.no_grad()
def evaluate_policy(
    policy: NextItemPolicy,
    examples: EvalExamples,
    items: ItemTable,
    reward_models: dict[str, RewardModel],
    config: EvaluationConfig,
    device: torch.device,
) -> PolicyEvaluation:
    """Recommends to every user in `examples` in two ways and scores both.

    "top_k" ranks all items by next-item probability. "next_k" generates the list item by item, feeding
    each choice back - the mode RL fine-tuning optimises. Metric names read <list>/<target set or "profile">/<metric>.
    """
    policy.eval()
    top_k_parts, generated_parts = [], []
    reception = _ReceptionOfBatches(reward_models, config)
    for batch in examples.batches(config.batch_size):
        history = batch.history.to(device)
        with autocast(device, config.mixed_precision):
            top_k_parts.append(recommend_top_k(policy, history, max(config.ks), config.generation).cpu())
            generated = generate_actions(policy, history, config.generation)
            generated_parts.append(generated.cpu())
            reception.add(history, generated.actions)

    top_k = torch.cat(top_k_parts)
    generated = GeneratedActions.concatenate(generated_parts)
    feedback, feedback_for_other_user = reception.of_all_batches()

    metrics = _prefixed("top_k", list_quality(top_k, examples, items, config))
    metrics |= _prefixed("next_k", list_quality(generated.actions, examples, items, config))
    metrics["next_k/policy_entropy"] = generated.entropy.mean().item()
    metrics |= predicted_reception(feedback, feedback_for_other_user, generated.actions, items)
    return PolicyEvaluation(metrics, top_k, generated, feedback, feedback_for_other_user)


@torch.no_grad()
def collect_lists(make_lists: Callable[[HistoryBatch], Tensor], examples: EvalExamples, batch_size: int, device: torch.device) -> Tensor:
    """One list per user of `examples` from any function of a batch of histories (given on `device`). long [U, K], on the CPU."""
    return torch.cat([make_lists(batch.history.to(device)).cpu() for batch in examples.batches(batch_size)])


@torch.no_grad()
def evaluate_lists(
    lists: Tensor,
    examples: EvalExamples,
    items: ItemTable,
    reward_models: dict[str, RewardModel],
    config: EvaluationConfig,
    device: torch.device,
) -> ListEvaluation:
    """Relevance, behaviour profile and predicted reception of given lists. lists: long [U, K], row u is for examples[u]."""
    lists = lists.cpu()
    reception = _ReceptionOfBatches(reward_models, config)
    for batch_lists, batch in zip(lists.split(config.batch_size), examples.batches(config.batch_size)):
        with autocast(device, config.mixed_precision):
            reception.add(batch.history.to(device), batch_lists.to(device))
    feedback, feedback_for_other_user = reception.of_all_batches()
    metrics = list_quality(lists, examples, items, config) | predicted_reception(feedback, feedback_for_other_user, lists, items)
    return ListEvaluation(metrics, lists, feedback, feedback_for_other_user)


class _ReceptionOfBatches:
    """Collects, batch by batch, what every reward model predicts for lists: from their own users and from other users."""

    def __init__(self, reward_models: dict[str, RewardModel], config: EvaluationConfig) -> None:
        self._reward_models = reward_models
        self._simulated_feedback = config.simulated_feedback
        self._own: dict[str, list[Feedback]] = {name: [] for name in reward_models}
        self._other: dict[str, list[Feedback]] = {name: [] for name in reward_models}

    def add(self, history: HistoryBatch, lists: Tensor) -> None:
        """history and lists of one batch, on the device of the reward models."""
        other_users = histories_of_other_users(history)
        for name, reward_model in self._reward_models.items():
            self._own[name].append(simulate_feedback(reward_model, history, lists, self._simulated_feedback).cpu())
            self._other[name].append(simulate_feedback(reward_model, other_users, lists, self._simulated_feedback).cpu())

    def of_all_batches(self) -> tuple[dict[str, Feedback], dict[str, Feedback]]:
        """(response of the users the lists were made for, response of other users), each: reward model name -> [U, K]."""
        own = {name: Feedback.concatenate(parts) for name, parts in self._own.items()}
        other = {name: Feedback.concatenate(parts) for name, parts in self._other.items()}
        return own, other


def list_quality(lists: Tensor, examples: EvalExamples, items: ItemTable, config: EvaluationConfig) -> dict[str, float]:
    """Everything about the lists that needs no reward model.

    Relevance against each target set, the behaviour profile, and the logged response to the listed items the
    users went on to consume. Names: <target set>/<metric>, profile/<statistic>, logged/<statistic>.
    """
    reachable_ks = [k for k in config.ks if k <= lists.shape[1]]
    quality: dict[str, float] = {}
    for target_name, relevant in relevance_targets(examples, items, config.liked_threshold).items():
        quality |= _prefixed(target_name, ranking_metrics(lists, relevant, reachable_ks))
    quality |= _prefixed("profile", recommendation_profile(lists, examples.history.items, items))
    return quality | _prefixed("logged", logged_reception(lists, examples, config.liked_threshold))


def predicted_reception(feedback: dict[str, Feedback], feedback_for_other_user: dict[str, Feedback], lists: Tensor, items: ItemTable) -> dict[str, float]:
    """The mean response every reward model predicts to the lists. Names: predicted/<reward model>/<signal>."""
    duration = torch.as_tensor(items.duration_seconds)
    reception = {}
    for name, predicted in feedback.items():
        signals = per_list_reception(predicted, feedback_for_other_user[name]) | {"played_seconds": (predicted.played_ratio * duration[lists]).mean(dim=1)}
        reception |= {f"predicted/{name}/{signal}": values.mean().item() for signal, values in signals.items()}
    return reception


def per_list_reception(predicted: Feedback, predicted_for_other_user: Feedback) -> dict[str, Tensor]:
    """The predicted response to each list, averaged over its items: signal name -> float [U].

    `personal_played_ratio` is what the list earns with its own user beyond what it would earn with another
    one: zero on average for lists that do not depend on the user. `chosen_played_ratio` counts an item the
    user is predicted not to take up as zero. `relative_ratio` is the played ratio as a quantile among
    everyone who consumed the item, which no item earns by being easy to finish. `returns_after_list` is the
    probability of a return after the last item, the end of the session the list would make.
    """
    played_ratio, played_ratio_for_other_user = predicted.played_ratio.mean(dim=1), predicted_for_other_user.played_ratio.mean(dim=1)
    relative_ratio, relative_ratio_for_other_user = predicted.relative_ratio.mean(dim=1), predicted_for_other_user.relative_ratio.mean(dim=1)
    return {
        "played_ratio": played_ratio,
        "played_ratio_for_other_user": played_ratio_for_other_user,
        "personal_played_ratio": played_ratio - played_ratio_for_other_user,
        "relative_ratio": relative_ratio,
        "personal_relative_ratio": relative_ratio - relative_ratio_for_other_user,
        "session_continues": predicted.session_continues.mean(dim=1),
        "returns": predicted.returns.mean(dim=1),
        "is_chosen": predicted.is_chosen.mean(dim=1),
        "chosen_played_ratio": (predicted.is_chosen * predicted.played_ratio).mean(dim=1),
        "liked": predicted.liked.mean(dim=1),
        "disliked": predicted.disliked.mean(dim=1),
        "adopts": predicted.adopts.mean(dim=1),
        "returns_after_list": predicted.returns[:, -1],
    }


def paired_comparison(before: ListEvaluation, after: ListEvaluation, examples: EvalExamples, items: ItemTable, config: EvaluationConfig) -> pd.DataFrame:
    """Per-user change from `before` to `after`, with bootstrap intervals: relevance, logged and predicted reception. One row per quantity."""
    k = max(k for k in config.ks if k <= before.lists.shape[1])
    targets = relevance_targets(examples, items, config.liked_threshold)
    compared = (
        ("all", f"ndcg@{k}"),
        ("liked", f"ndcg@{k}"),
        ("skipped", f"recall@{k}"),
        ("above_usual", f"ndcg@{k}"),
        ("below_usual", f"recall@{k}"),
        ("new", f"ndcg@{k}"),
        ("repeat", f"ndcg@{k}"),
    )
    rows = {}
    for target_name, metric in compared:
        per_user_before = per_user_ranking_metrics(before.lists, targets[target_name], [k])[metric]
        per_user_after = per_user_ranking_metrics(after.lists, targets[target_name], [k])[metric]
        rows[f"{target_name}/{metric}"] = paired_bootstrap(per_user_before, per_user_after)
    logged_before = logged_reward_per_user(before.lists, examples, config.liked_threshold)
    logged_after = logged_reward_per_user(after.lists, examples, config.liked_threshold)
    for quantity in logged_before:
        rows[f"logged/{quantity}"] = paired_bootstrap(logged_before[quantity], logged_after[quantity])
    for name in before.feedback:
        reception_before = per_list_reception(before.feedback[name], before.feedback_for_other_user[name])
        reception_after = per_list_reception(after.feedback[name], after.feedback_for_other_user[name])
        for signal in reception_before:
            rows[f"predicted/{name}/{signal}"] = paired_bootstrap(reception_before[signal], reception_after[signal])
    return pd.DataFrame(rows).T


def relevance_targets(examples: EvalExamples, items: ItemTable, liked_threshold: float) -> dict[str, Tensor]:
    """The items each user consumed next, split by how they responded. Each value is long [U, M], PAD_ITEM-padded.

    "liked" and "skipped" split at a fixed played ratio: a recommender that learned to avoid skips should keep
    the first and lose the second. That split favours items that are easy to finish. "above_usual" and
    "below_usual" split at the item's own mean played ratio in the train period, so every item can fall on
    either side: they tell whether the user took to the item more than people usually do. "repeat" and "new"
    split by whether the item is in the visible history: where people return to what they know, replaying
    the history already finds the first, and only the second asks for a recommender.
    """
    future_items = examples.future_items.cpu()
    played_ratio = examples.future_played_ratio.cpu()
    usual_played_ratio = torch.as_tensor(items.mean_played_ratio)[future_items]
    is_in_history = contains(future_items, examples.history.items.cpu())
    no_item = torch.full_like(future_items, PAD_ITEM)
    return {
        "all": future_items,
        "liked": future_items.where(played_ratio >= liked_threshold, no_item),
        "skipped": future_items.where(played_ratio < liked_threshold, no_item),
        "above_usual": future_items.where(played_ratio >= usual_played_ratio, no_item),
        "below_usual": future_items.where(played_ratio < usual_played_ratio, no_item),
        "repeat": future_items.where(is_in_history, no_item),
        "new": future_items.where(~is_in_history, no_item),
    }


def per_step_table(evaluation: PolicyEvaluation, examples: EvalExamples, items: ItemTable) -> pd.DataFrame:
    """One row per generation step: how the generated item and its predicted reception change along the list.

    Later steps condition on more generated - not observed - items, so drift shows up as a trend over rows.
    """
    actions = evaluation.generated.actions
    columns = {
        "policy_entropy": evaluation.generated.entropy.mean(0),
        "in_true_continuation": contains(actions, examples.future_items.cpu()).float().mean(0),
        "in_history": contains(actions, examples.history.items.cpu()).float().mean(0),
        "popularity": torch.as_tensor(items.train_popularity, dtype=torch.float32)[actions].mean(0),
        "duration_seconds": torch.as_tensor(items.duration_seconds)[actions].mean(0),
    }
    for name, predicted in evaluation.feedback.items():
        for signal, values in vars(predicted).items():
            columns[f"predicted_{signal}/{name}"] = values.mean(0)
    for first, second in combinations(evaluation.feedback, 2):
        disagreement = (evaluation.feedback[first].played_ratio - evaluation.feedback[second].played_ratio).abs()
        columns[f"played_ratio_disagreement/{first}_vs_{second}"] = disagreement.mean(0)
    step = pd.RangeIndex(1, actions.shape[1] + 1, name="step")
    return pd.DataFrame({name: values.numpy() for name, values in columns.items()}, index=step)


def popularity_baseline_metrics(examples: EvalExamples, items: ItemTable, config: EvaluationConfig) -> dict[str, float]:
    """Relevance and profile of showing every user the most popular items: the floor a recommender must beat."""
    most_popular = torch.as_tensor(items.train_popularity).topk(max(config.ks)).indices
    lists = most_popular.repeat(len(examples), 1)
    metrics: dict[str, float] = {}
    for target_name, relevant in relevance_targets(examples, items, config.liked_threshold).items():
        metrics |= _prefixed(f"top_k/{target_name}", ranking_metrics(lists, relevant, config.ks))
    return metrics | _prefixed("top_k/profile", recommendation_profile(lists, examples.history.items, items))


def list_metrics(metrics: dict[str, float], list_name: str) -> dict[str, float]:
    """The metrics of one kind of list ("top_k" or "next_k") with that prefix removed, so both kinds can share rows."""
    prefix = f"{list_name}/"
    return {name.removeprefix(prefix): value for name, value in metrics.items() if name.startswith(prefix)}


def metrics_table(evaluations: dict[str, PolicyEvaluation]) -> pd.DataFrame:
    """Metrics of several recommenders side by side: one column per recommender."""
    return pd.DataFrame({name: evaluation.metrics for name, evaluation in evaluations.items()})


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{name}": value for name, value in metrics.items()}
