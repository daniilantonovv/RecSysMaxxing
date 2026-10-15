"""How well a reward model predicts held-out feedback, and whether it beats history-free baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from gptrec.data.batches import EventBatch, EventWindows, swap_one_target_item
from gptrec.data.log import ItemTable
from gptrec.evaluation.relevance import contains
from gptrec.models.reward_model import RewardModel
from gptrec.rollout import ScoredCandidates
from gptrec.training.precision import autocast

COMPLETED_THRESHOLD = 0.5  # an item counts as "played" when at least half of it was consumed


@dataclass
class HeldoutFeedback:
    """Observed and predicted feedback for held-out events. All arrays are aligned, one entry per event."""

    item: np.ndarray  # item token
    played_ratio: np.ndarray
    relative_ratio: np.ndarray  # the played ratio as a quantile among the item's train-period events
    session_continues: np.ndarray
    returns: np.ndarray
    session_continues_known: np.ndarray  # False where the observed data ends too soon to tell
    returns_known: np.ndarray
    is_organic: np.ndarray  # the user picked the item themselves (it was not recommended)
    liked: np.ndarray
    disliked: np.ndarray
    reactions_known: np.ndarray  # False where the dataset records no reactions or the log ends too soon to tell
    adopts: np.ndarray
    adopts_known: np.ndarray  # True only for first events of an item whose window ends inside the observed data
    predicted_played_ratio: np.ndarray
    predicted_relative_ratio: np.ndarray
    predicted_session_continues: np.ndarray
    predicted_returns: np.ndarray
    predicted_is_chosen: np.ndarray  # for the item the user chose
    predicted_is_chosen_if_swapped: np.ndarray  # for another item put in its place; NaN for events that were not probed
    predicted_liked: np.ndarray
    predicted_disliked: np.ndarray
    predicted_adopts: np.ndarray
    recent_played_ratio: np.ndarray  # mean played ratio over the user's preceding events in the window; NaN if none
    recent_relative_ratio: np.ndarray  # the same mean of the relative ratio


@torch.no_grad()
def predict_heldout_feedback(reward_model: RewardModel, windows: EventWindows, batch_size: int, device: torch.device, mixed_precision: bool = True) -> HeldoutFeedback:
    """Teacher-forced predictions for every target event of `windows`: the model sees the true feedback on all earlier events.

    One target event per window is also scored with another item in its place, to see whether the model can tell.
    """
    reward_model.eval()
    collected: dict[str, list[np.ndarray]] = {name: [] for name in HeldoutFeedback.__dataclass_fields__}
    for batch in windows.batches(batch_size):
        batch = batch.to(device)
        with autocast(device, mixed_precision):
            predicted = reward_model.predict(batch.items, batch.played_ratio)
            is_chosen_if_swapped = _is_chosen_if_swapped(reward_model, batch)
        per_event = {
            "item": batch.items,
            "played_ratio": batch.played_ratio,
            "relative_ratio": batch.relative_ratio,
            "session_continues": batch.session_continues,
            "returns": batch.returns,
            "session_continues_known": batch.session_continues_known,
            "returns_known": batch.returns_known,
            "is_organic": batch.is_organic,
            "liked": batch.liked,
            "disliked": batch.disliked,
            "reactions_known": batch.reactions_known,
            "adopts": batch.adopts,
            "adopts_known": batch.adopts_known,
            "predicted_played_ratio": predicted.played_ratio,
            "predicted_relative_ratio": predicted.relative_ratio,
            "predicted_session_continues": predicted.session_continues,
            "predicted_returns": predicted.returns,
            "predicted_is_chosen": predicted.is_chosen,
            "predicted_is_chosen_if_swapped": is_chosen_if_swapped,
            "predicted_liked": predicted.liked,
            "predicted_disliked": predicted.disliked,
            "predicted_adopts": predicted.adopts,
            "recent_played_ratio": _mean_of_preceding(batch.played_ratio),
            "recent_relative_ratio": _mean_of_preceding(batch.relative_ratio),
        }
        for name, values in per_event.items():
            collected[name].append(values[batch.is_target].cpu().numpy())
    return HeldoutFeedback(**{name: np.concatenate(chunks) for name, chunks in collected.items()})


def _is_chosen_if_swapped(reward_model: RewardModel, batch: EventBatch) -> torch.Tensor:
    """Predicted probability of being chosen for one swapped-in item per window, NaN at every other position. [B, L]"""
    swapped = swap_one_target_item(batch)
    predicted = reward_model.predict(swapped.items, batch.played_ratio).is_chosen.gather(1, swapped.position[:, None])
    predicted = predicted.masked_fill(~swapped.is_different[:, None], float("nan"))
    return torch.full_like(batch.played_ratio, float("nan")).scatter(1, swapped.position[:, None], predicted.to(batch.played_ratio.dtype))


def _mean_of_preceding(values: torch.Tensor) -> torch.Tensor:
    """Mean of the values before each position of its window. NaN at the first position. [B, L] -> [B, L]"""
    num_preceding = torch.arange(values.shape[1], device=values.device)
    sum_of_preceding = values.cumsum(dim=1) - values
    return sum_of_preceding / num_preceding


def reward_model_report(heldout: HeldoutFeedback, items: ItemTable) -> pd.DataFrame:
    """Played-ratio prediction quality of the model next to baselines that ignore the user or the item."""
    train_mean = items.train_mean_played_ratio
    predictors = {
        "reward model": heldout.predicted_played_ratio,
        "train mean": np.full_like(heldout.played_ratio, train_mean),
        "item mean": items.mean_played_ratio[heldout.item],
        "user recent mean": np.nan_to_num(heldout.recent_played_ratio, nan=train_mean),
    }
    return pd.DataFrame({name: prediction_quality(heldout.played_ratio, predicted) for name, predicted in predictors.items()}).T


def relative_ratio_report(heldout: HeldoutFeedback) -> pd.DataFrame:
    """Prediction quality of the relative played ratio next to predictors that know nothing about the item.

    `auc` is for the event "the user played more of the item than people usually do". No item statistic can
    predict it, because every item's mean relative ratio is 0.5 by construction.
    """
    usual_for_the_item = 0.5
    predictors = {
        "reward model": heldout.predicted_relative_ratio,
        "usual for the item": np.full_like(heldout.relative_ratio, usual_for_the_item),
        "user recent mean": np.nan_to_num(heldout.recent_relative_ratio, nan=usual_for_the_item),
    }
    return pd.DataFrame({name: prediction_quality(heldout.relative_ratio, predicted) for name, predicted in predictors.items()}).T


def quality_by_origin(heldout: HeldoutFeedback) -> pd.DataFrame:
    """Played-ratio prediction quality separately for events the user chose and events a recommender served.

    RL asks the reward model about recommendations, so its quality on recommended events is the one that matters.
    """
    origins = {"chosen by the user": heldout.is_organic, "recommended": ~heldout.is_organic}
    rows = {}
    for origin, has_origin in origins.items():
        if has_origin.any():
            quality = prediction_quality(heldout.played_ratio[has_origin], heldout.predicted_played_ratio[has_origin])
            rows[origin] = quality | {"mean_played_ratio": float(heldout.played_ratio[has_origin].mean()), "num_events": float(has_origin.sum())}
    return pd.DataFrame(rows).T


RETENTION_SIGNALS = ("session_continues", "returns")


def retention_quality(heldout: HeldoutFeedback) -> pd.DataFrame:
    """Prediction quality of each retention signal on the events whose outcome the log knows, with how often it is true."""
    rows = {}
    for signal in RETENTION_SIGNALS:
        is_known = getattr(heldout, f"{signal}_known")
        observed, predicted = getattr(heldout, signal)[is_known], getattr(heldout, f"predicted_{signal}")[is_known]
        quality = prediction_quality(observed, predicted) if is_known.any() else dict.fromkeys(("mae", "rmse", "cross_entropy", "auc"), float("nan"))
        rows[signal] = quality | {"base_rate": float(observed.mean()) if is_known.any() else float("nan"), "num_events": float(is_known.sum())}
    return pd.DataFrame(rows).T


REACTION_SIGNALS = {"liked": "reactions_known", "disliked": "reactions_known", "adopts": "adopts_known"}


def reaction_quality(heldout: HeldoutFeedback) -> pd.DataFrame:
    """Prediction quality of likes, dislikes and adoption on the events whose outcome the log knows, with how often each happens."""
    rows = {}
    for signal, known in REACTION_SIGNALS.items():
        is_known = getattr(heldout, known)
        observed, predicted = getattr(heldout, signal)[is_known], getattr(heldout, f"predicted_{signal}")[is_known]
        quality = prediction_quality(observed, predicted) if is_known.any() else dict.fromkeys(("mae", "rmse", "cross_entropy", "auc"), float("nan"))
        rows[signal] = quality | {"base_rate": float(observed.mean()) if is_known.any() else float("nan"), "num_events": float(is_known.sum())}
    return pd.DataFrame(rows).T


def choice_quality(heldout: HeldoutFeedback) -> dict[str, float]:
    """Whether the model tells the item a user chose from another item put in its place.

    `auc` is the probability that a chosen item is scored above a swapped-in one; 0.5 means the model
    cannot tell and its reward says nothing about relevance.
    """
    was_probed = ~np.isnan(heldout.predicted_is_chosen_if_swapped)
    of_chosen_items, of_swapped_items = heldout.predicted_is_chosen, heldout.predicted_is_chosen_if_swapped[was_probed]
    is_chosen_item = np.concatenate([np.ones(len(of_chosen_items), dtype=bool), np.zeros(len(of_swapped_items), dtype=bool)])
    return {
        "auc": roc_auc(is_chosen_item, np.concatenate([of_chosen_items, of_swapped_items])),
        "mean_probability_of_chosen_items": float(of_chosen_items.mean()),
        "mean_probability_of_swapped_items": float(of_swapped_items.mean()) if was_probed.any() else float("nan"),
        "num_swapped_items": float(was_probed.sum()),
    }


def prediction_quality(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Error of predictions in [0, 1] against observations in [0, 1]."""
    observed, predicted = observed.astype(np.float64), predicted.astype(np.float64)
    clipped = np.clip(predicted, 1e-6, 1 - 1e-6)
    return {
        "mae": float(np.abs(observed - predicted).mean()),
        "rmse": float(np.sqrt(((observed - predicted) ** 2).mean())),
        "cross_entropy": float(-(observed * np.log(clipped) + (1 - observed) * np.log(1 - clipped)).mean()),
        "auc": roc_auc(observed >= COMPLETED_THRESHOLD, predicted),
    }


def roc_auc(is_positive: np.ndarray, score: np.ndarray) -> float:
    """Probability that a random positive is scored above a random negative; ties count as half."""
    num_positive, num_negative = int(is_positive.sum()), int((~is_positive).sum())
    if num_positive == 0 or num_negative == 0:
        return float("nan")
    _, tie_group, group_size = np.unique(score, return_inverse=True, return_counts=True)
    last_rank_in_group = np.cumsum(group_size)
    mean_rank = (last_rank_in_group - (group_size - 1) / 2)[tie_group]
    rank_sum_of_positives = mean_rank[is_positive].sum()
    return float((rank_sum_of_positives - num_positive * (num_positive + 1) / 2) / (num_positive * num_negative))


def calibration_table(observed: np.ndarray, predicted: np.ndarray, num_bins: int = 10) -> pd.DataFrame:
    """Mean prediction against mean observation inside equal-width prediction bins; a calibrated model has them equal."""
    bin_index = np.clip((predicted * num_bins).astype(int), 0, num_bins - 1)
    table = pd.DataFrame({"bin": bin_index, "predicted": predicted, "observed": observed})
    return table.groupby("bin").agg(mean_predicted=("predicted", "mean"), mean_observed=("observed", "mean"), num_events=("observed", "size"))


def error_by_group(heldout: HeldoutFeedback, group_of_item: np.ndarray) -> pd.DataFrame:
    """Played-ratio bias and absolute error per item group, e.g. popularity or duration buckets.

    group_of_item: [V+1], the group label of every item token.
    """
    error = heldout.predicted_played_ratio.astype(np.float64) - heldout.played_ratio
    table = pd.DataFrame({"group": group_of_item[heldout.item], "error": error, "absolute_error": np.abs(error)})
    return table.groupby("group").agg(bias=("error", "mean"), mae=("absolute_error", "mean"), num_events=("error", "size"))


def signal_spread_over_candidates(candidates: ScoredCandidates, future_items: torch.Tensor) -> pd.DataFrame:
    """What each predicted signal offers a recommender that chooses among its own candidates. One row per signal.

    spread_between_candidates: standard deviation over one user's candidates, averaged over users. A signal
        that hardly differs between the items a user could be shown gives RL and re-ranking nothing to choose by.
    spread_between_users: standard deviation over users of the mean over their candidates.
    relevance_auc: probability that a candidate the user really consumed next has a higher value than one the
        user did not; at 0.5 the signal knows nothing about relevance. The recommender's own log-probability
        is the yardstick.

    candidates: on the CPU; future_items: long [U, M], what each user really consumed next.
    """
    is_relevant = contains(candidates.items, future_items).flatten().numpy()
    predicted = candidates.feedback
    signals = vars(predicted) | {"chosen_played_ratio": predicted.is_chosen * predicted.played_ratio, "recommender_log_prob": candidates.log_probs}
    rows = {}
    for name, values in signals.items():
        values = values.float()
        rows[name] = {
            "spread_between_candidates": values.std(dim=1).mean().item(),
            "spread_between_users": values.mean(dim=1).std().item(),
            "relevance_auc": roc_auc(is_relevant, values.flatten().numpy()),
        }
    return pd.DataFrame(rows).T
