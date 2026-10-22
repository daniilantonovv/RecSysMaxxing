"""Analyses of fine-tuned recommenders that only evaluate: along the list, across users, across items.

Each answers one question about a policy that the headline metrics leave open: what an item the user never
consumed does to the rest of a generated list; whether a simulation can trust its assumption about such items;
who loses relevance; whether the policy learned something about users or one preference per item, and whether
the reward predicted which; how far a policy moved along its own lists; and what a retention label records.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor

from gptrec.data.batches import EvalExamples
from gptrec.data.log import PAD_ITEM, InteractionLog, ItemTable
from gptrec.evaluation.logged_feedback import logged_reward_per_user
from gptrec.evaluation.relevance import contains, per_user_ranking_metrics
from gptrec.evaluation.reward_model_metrics import COMPLETED_THRESHOLD, roc_auc
from gptrec.models.policy import NextItemPolicy
from gptrec.models.reward_model import RewardModel
from gptrec.rollout import GenerationConfig, SimulatedFeedback, action_log_distributions, simulate_feedback
from gptrec.training.precision import autocast


def position_hits(lists: dict[str, Tensor], future_items: Tensor) -> pd.DataFrame:
    """Per position of each list: how often the item is among the user's next items, overall, after a miss and after a hit.

    lists: name -> long [U, K]. In a generated list, a miss at position k - 1 means that the item at position k was
    chosen after an item the user never consumed. A ranked list has no such context, so the gap between the two
    after a miss is the cost of continuing from a context that did not happen. Columns: <list>/<statistic>.
    """
    columns = {}
    future_items = future_items.cpu()
    for name, items in lists.items():
        hit = contains(items.cpu(), future_items).float()
        before, now = hit[:, :-1], hit[:, 1:]
        no_previous_position = torch.tensor([float("nan")])
        columns[f"{name}/hit_rate"] = hit.mean(dim=0)
        columns[f"{name}/hit_rate_after_miss"] = torch.cat([no_previous_position, _conditional_rate(now, 1.0 - before)])
        columns[f"{name}/hit_rate_after_hit"] = torch.cat([no_previous_position, _conditional_rate(now, before)])
    position = pd.RangeIndex(1, next(iter(lists.values())).shape[1] + 1, name="position")
    return pd.DataFrame({name: values.numpy() for name, values in columns.items()}, index=position)


def _conditional_rate(event: Tensor, condition: Tensor) -> Tensor:
    """Per column, the mean of `event` over the rows where `condition` holds; NaN where it never does."""
    num_conditioned = condition.sum(dim=0)
    rate = (event * condition).sum(dim=0) / num_conditioned.clamp_min(1)
    return rate.masked_fill(num_conditioned == 0, float("nan"))


@torch.no_grad()
def feedback_assumptions_on_real_sequences(
    reward_model: RewardModel,
    examples: EvalExamples,
    modes: Sequence[SimulatedFeedback],
    steps: int,
    batch_size: int,
    device: torch.device,
    mixed_precision: bool = False,
) -> pd.DataFrame:
    """How well the reward model predicts the response to a user's real next items when the earlier responses are assumed.

    For every request with at least `steps` next events, the list is the user's real next `steps` items in their
    order. The model predicts the response to each one from the history and the items before it, fed back as
    `mode` says: as logged, or as one of the assumptions a simulation of generated items has to make. Where an
    assumption's predictions at later steps agree with what users did, a simulation can rely on it.
    Rows: step. Columns: logged/mean_played_ratio, and <mode>/<mean_predicted, bias, mae, auc>.
    """
    has_enough_future = (examples.future_items[:, steps - 1] != PAD_ITEM).nonzero().squeeze(1)
    if len(has_enough_future) == 0:
        return pd.DataFrame(index=pd.RangeIndex(1, steps + 1, name="step"))
    requests = examples[has_enough_future]
    predicted: dict[SimulatedFeedback, list[Tensor]] = {SimulatedFeedback(mode): [] for mode in modes}
    for batch in requests.batches(batch_size):
        history = batch.history.to(device)
        next_items, next_played_ratio = batch.future_items[:, :steps].to(device), batch.future_played_ratio[:, :steps].to(device)
        with autocast(device, mixed_precision):
            for mode, parts in predicted.items():
                parts.append(simulate_feedback(reward_model, history, next_items, mode, next_played_ratio).played_ratio.float().cpu())
    logged = requests.future_played_ratio[:, :steps].float()

    columns = {"logged/mean_played_ratio": logged.mean(dim=0).numpy(), "logged/num_requests": np.full(steps, float(len(requests)))}
    for mode, parts in predicted.items():
        values = torch.cat(parts)
        columns[f"{mode.value}/mean_predicted"] = values.mean(dim=0).numpy()
        columns[f"{mode.value}/bias"] = (values - logged).mean(dim=0).numpy()
        columns[f"{mode.value}/mae"] = (values - logged).abs().mean(dim=0).numpy()
        columns[f"{mode.value}/auc"] = np.array([roc_auc(logged[:, step].numpy() >= COMPLETED_THRESHOLD, values[:, step].numpy()) for step in range(steps)])
    return pd.DataFrame(columns, index=pd.RangeIndex(1, steps + 1, name="step"))


def relevance_by_history_length(
    lists: dict[str, Tensor], examples: EvalExamples, prior_events: np.ndarray, k: int, liked_threshold: float, num_groups: int = 4
) -> pd.DataFrame:
    """NDCG@k and the logged share per recommended item of each list, for users grouped by how much history they had.

    prior_events: [U], each user's number of events before the request; the groups are its quantiles.
    Rows: group. Columns: num_users and <list>/<metric>.
    """
    edges = np.unique(np.quantile(prior_events, np.linspace(0.0, 1.0, num_groups + 1)))
    group = np.clip(np.searchsorted(edges, prior_events, side="right") - 1, 0, len(edges) - 2)
    per_list = {
        name: (
            per_user_ranking_metrics(items.cpu(), examples.future_items.cpu(), [k])[f"ndcg@{k}"].numpy(),
            logged_reward_per_user(items, examples, liked_threshold)["played_ratio_per_item"].numpy(),
        )
        for name, items in lists.items()
    }
    rows = {}
    for index in range(len(edges) - 1):
        members = group == index
        if not members.any():
            continue
        row = {"num_users": float(members.sum())}
        for name, (ndcg, logged_share) in per_list.items():
            row[f"{name}/ndcg@{k}"] = float(np.nanmean(ndcg[members]))
            row[f"{name}/logged_played_ratio_per_item"] = float(logged_share[members].mean())
        rows[f"{int(edges[index])} to {int(edges[index + 1])} events"] = row
    return pd.DataFrame(rows).T


@torch.no_grad()
def logit_shift_by_item(
    policy: NextItemPolicy, reference: NextItemPolicy, examples: EvalExamples, items: ItemTable, batch_size: int, device: torch.device, max_requests: int = 2048
) -> dict[str, float]:
    """How much of what fine-tuning changed in the next-item distribution is one constant per item, the same for every user.

    For each history the shift of item j is log pi(j | h) - log pi0(j | h). If the policy learned "show more of
    item j" regardless of the user, every history shifts j alike, and the mean shift per item explains all of
    it. `r2_per_item` is that share of the variance, weighted by pi0 so that items no user would be shown do not
    count; `r2_per_item_unweighted` gives every item the same weight. The correlations relate the per-item shift
    to the item's mean played ratio, log popularity and duration, across items weighted by pi0's mass.
    """
    num_items = items.num_items
    weighted = {name: torch.zeros(num_items, dtype=torch.float64, device=device) for name in ("weight", "shift", "squared_shift")}
    unweighted = {name: torch.zeros(num_items, dtype=torch.float64, device=device) for name in ("count", "shift", "squared_shift")}
    for batch in examples[:max_requests].batches(batch_size):
        history = batch.history.to(device)
        log_probs = F.log_softmax(policy.next_item_logits(history.items, history.lengths)[:, 1:].float(), dim=-1).double()
        reference_log_probs = F.log_softmax(reference.next_item_logits(history.items, history.lengths)[:, 1:].float(), dim=-1).double()
        shift, weight = log_probs - reference_log_probs, reference_log_probs.exp()
        weighted["weight"] += weight.sum(dim=0)
        weighted["shift"] += (weight * shift).sum(dim=0)
        weighted["squared_shift"] += (weight * shift**2).sum(dim=0)
        unweighted["count"] += len(shift)
        unweighted["shift"] += shift.sum(dim=0)
        unweighted["squared_shift"] += (shift**2).sum(dim=0)

    def share_explained_by_item(weight: Tensor, shift: Tensor, squared_shift: Tensor) -> float:
        total = squared_shift.sum() - shift.sum() ** 2 / weight.sum()
        within_items = (squared_shift - shift**2 / weight.clamp_min(1e-300)).sum()
        return float(1.0 - within_items / total)

    mass = weighted["weight"].cpu().numpy()
    shift_per_item = (weighted["shift"] / weighted["weight"].clamp_min(1e-300)).cpu().numpy()
    attributes = {
        "mean_played_ratio": items.mean_played_ratio[1:].astype(np.float64),
        "log_popularity": np.log1p(items.train_popularity[1:].astype(np.float64)),
        "duration_seconds": items.duration_seconds[1:].astype(np.float64),
    }
    return {
        "r2_per_item": share_explained_by_item(weighted["weight"], weighted["shift"], weighted["squared_shift"]),
        "r2_per_item_unweighted": share_explained_by_item(unweighted["count"], unweighted["shift"], unweighted["squared_shift"]),
        "mean_shift": float((weighted["shift"].sum() / weighted["weight"].sum()).item()),
    } | {f"correlation_with_{name}": _weighted_correlation(shift_per_item, values, mass) for name, values in attributes.items()}


def _weighted_correlation(first: np.ndarray, second: np.ndarray, weight: np.ndarray) -> float:
    weight = weight / weight.sum()
    first_centred, second_centred = first - (weight * first).sum(), second - (weight * second).sum()
    covariance = (weight * first_centred * second_centred).sum()
    return float(covariance / np.sqrt((weight * first_centred**2).sum() * (weight * second_centred**2).sum()))


def exposure_lorenz(lists: dict[str, Tensor], num_items: int, num_points: int = 101) -> pd.DataFrame:
    """Lorenz curves of impressions over the catalogue: the share of all impressions that goes to the least recommended share of items.

    A diagonal means every item is shown equally often; a curve that stays at zero and jumps at the end means
    a few items take everything. Rows: share of items. Columns: lists.
    """
    share_of_items = np.linspace(0.0, 1.0, num_points)
    at_share = np.round(share_of_items * num_items).astype(int)
    curves = {}
    for name, items in lists.items():
        impressions = torch.bincount(items.flatten().cpu(), minlength=num_items + 1)[1:].double().sort().values
        cumulative = torch.cat([torch.zeros(1, dtype=torch.float64), impressions.cumsum(0)]) / impressions.sum().clamp_min(1)
        curves[name] = cumulative.numpy()[at_share]
    return pd.DataFrame(curves, index=pd.Index(share_of_items, name="share of items"))


def impressions_by_attribute(lists: dict[str, Tensor], attribute_of_item: np.ndarray) -> pd.DataFrame:
    """Share of impressions per value of an item attribute, such as film or series. attribute_of_item: [V+1]. Columns: lists."""
    shares = {}
    for name, items in lists.items():
        values = pd.Series(attribute_of_item[items.flatten().cpu().numpy()])
        shares[name] = values.value_counts(normalize=True, dropna=False)
    return pd.DataFrame(shares).fillna(0.0)


def largest_impression_changes(before: Tensor, after: Tensor, item_table: pd.DataFrame, num_items: int = 10) -> pd.DataFrame:
    """The items whose impressions grew most and shrank most from the lists `before` to the lists `after`.

    item_table: one row per item token (index) with whatever describes the items. Both lists: long [U, K].
    """
    size = len(item_table)
    impressions_before = torch.bincount(before.flatten().cpu(), minlength=size)[:size].numpy()
    impressions_after = torch.bincount(after.flatten().cpu(), minlength=size)[:size].numpy()
    table = item_table.assign(impressions_before=impressions_before, impressions_after=impressions_after)
    table = table.assign(change=table["impressions_after"] - table["impressions_before"]).drop(index=PAD_ITEM, errors="ignore")
    gained, lost = table.nlargest(num_items, "change"), table.nsmallest(num_items, "change")
    return pd.concat({"gained": gained, "lost": lost}, names=["direction", "item_token"])


@torch.no_grad()
def mean_logit_shift_per_item(
    policy: NextItemPolicy, reference: NextItemPolicy, examples: EvalExamples, batch_size: int, device: torch.device, max_requests: int = 4096
) -> Tensor:
    """Per item, the mean over histories of log pi(j | h) - log pi_0(j | h), weighted by pi_0(j | h). float [V+1], 0 for padding, on the CPU.

    Adding it to the base recommender's scores gives the part of a fine-tuning that is the same for every user;
    subtracting it from the fine-tuned recommender's scores leaves the part that depends on who receives the list.
    """
    weight_sum = shift_sum = 0.0
    for batch in examples[:max_requests].batches(batch_size):
        history = batch.history.to(device)
        log_probs = F.log_softmax(policy.next_item_logits(history.items, history.lengths).float(), dim=-1).double()
        reference_log_probs = F.log_softmax(reference.next_item_logits(history.items, history.lengths).float(), dim=-1).double()
        weight = reference_log_probs.exp()
        weight_sum = weight_sum + weight.sum(dim=0)
        shift_sum = shift_sum + (weight * (log_probs - reference_log_probs)).sum(dim=0)
    offsets = (shift_sum / weight_sum.clamp_min(1e-300)).float().cpu()
    offsets[PAD_ITEM] = 0.0
    return offsets


def item_effect_share(values: Tensor, items: Tensor, weights: Tensor) -> float:
    """Share of the within-history variation of `values` that one effect per item explains. values, items, weights: [U, N].

    values[u, n] belongs to item items[u, n] for history u, and weights[u, n] says how much the pair counts, for
    instance the base recommender's probability of the item. Values are centred within each history, so a constant
    per history, such as the normaliser of a softmax, explains nothing; the effect of an item is the weighted mean
    of its centred values. 1: every history orders the items alike; 0: an item's worth depends on the history alone.
    """
    values, weights = values.double().cpu(), weights.double().cpu()
    items = items.cpu()
    history_mean = (weights * values).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True).clamp_min(1e-300)
    centred = values - history_mean
    size = int(items.max()) + 1
    weight_per_item = torch.zeros(size, dtype=torch.float64).index_add_(0, items.flatten(), weights.flatten())
    effect = torch.zeros(size, dtype=torch.float64).index_add_(0, items.flatten(), (weights * centred).flatten()) / weight_per_item.clamp_min(1e-300)
    residual = centred - effect[items]
    return float(1.0 - (weights * residual**2).sum() / (weights * centred**2).sum().clamp_min(1e-300))


@torch.no_grad()
def candidate_log_probs(policy: NextItemPolicy, examples: EvalExamples, candidates: Tensor, batch_size: int, device: torch.device) -> Tensor:
    """log pi(item | history) of each user's candidate items, before any list constraint. candidates: long [U, N] -> float [U, N], on the CPU."""
    parts = []
    for batch, batch_candidates in zip(examples.batches(batch_size), candidates.split(batch_size)):
        history = batch.history.to(device)
        log_probs = F.log_softmax(policy.next_item_logits(history.items, history.lengths).float(), dim=-1)
        parts.append(log_probs.gather(1, batch_candidates.to(device)).cpu())
    return torch.cat(parts)


@torch.no_grad()
def kl_along_lists(
    policy: NextItemPolicy,
    reference: NextItemPolicy,
    examples: EvalExamples,
    lists: Tensor,
    generation: GenerationConfig,
    batch_size: int,
    device: torch.device,
    max_requests: int = 4096,
) -> float:
    """Mean KL(pi || pi_0) per item over the states in which the given lists were built, under the list constraints. lists: long [U, K]

    One yardstick for every recommender, however it was obtained: a fine-tuning run reports its KL on the lists it
    sampled while training, a recommender made by interpolating weights has no such record.
    """
    total, count = 0.0, 0
    for batch, batch_lists in zip(examples[:max_requests].batches(batch_size), lists[:max_requests].split(batch_size)):
        history, actions = batch.history.to(device), batch_lists.to(device)
        log_distributions = action_log_distributions(policy, history, actions, generation)
        reference_log_distributions = action_log_distributions(reference, history, actions, generation)
        kl = (log_distributions.exp() * (log_distributions - reference_log_distributions)).sum(dim=-1)
        total += kl.sum().item()
        count += kl.numel()
    return total / max(1, count)


def return_rate_by_duration(log: InteractionLog, num_groups: int = 10) -> pd.DataFrame:
    """How often a user returns after an event, by the duration of the event's own item and by that of the next item the user consumed.

    Training-period events with a known return outcome and a later event of the same user. Where a log keeps one row
    per user and item, dated by its last view, a user who moves on to a long series leaves no row until the series is
    finished, and the event before it looks as if the user stayed away. If the label follows the next item's duration
    more than the event's own, a reward for returns pays for that bookkeeping. Rows: duration groups, shortest first.
    """
    user_starts, user_ends = np.asarray(log.user_offsets[:-1]), np.asarray(log.user_offsets[1:])
    train_starts, train_ends = log.period_rows("train")
    boundaries = np.zeros(log.num_events + 1, dtype=np.int64)
    np.add.at(boundaries, train_starts, 1)
    np.add.at(boundaries, train_ends, -1)
    in_train = np.cumsum(boundaries)[:-1] > 0
    has_next_event = np.ones(log.num_events, dtype=bool)
    has_next_event[user_ends[user_ends > user_starts] - 1] = False
    events = np.flatnonzero(in_train & has_next_event & np.asarray(log.returns_known))
    duration = np.asarray(log.items.duration_seconds, dtype=np.float64)
    item = np.asarray(log.item)
    returned = np.asarray(log.returns)[events].astype(np.float64)
    columns, correlation = {}, {}
    for name, item_of_event in (("own item", item[events]), ("next item", item[events + 1])):
        hours = duration[item_of_event] / 3600
        edges = np.unique(np.quantile(hours, np.linspace(0.0, 1.0, num_groups + 1)))
        group = np.clip(np.searchsorted(edges, hours, side="right") - 1, 0, len(edges) - 2)
        groups = np.arange(len(edges) - 1)
        columns[f"{name}: mean hours"] = pd.Series([hours[group == g].mean() for g in groups], index=groups + 1)
        columns[f"{name}: return rate"] = pd.Series([returned[group == g].mean() for g in groups], index=groups + 1)
        correlation[f"{name}: return rate"] = np.corrcoef(np.log1p(hours), returned)[0, 1]
    table = pd.DataFrame(columns)
    table.loc["correlation of log hours with return"] = pd.Series(correlation)
    return table.rename_axis("duration group")
