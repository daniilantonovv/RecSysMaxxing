"""What users really did with the recommended items they went on to consume.

Every other reward in the study is a prediction of a reward model. A recommended item that is among the
user's next events - a hit - comes with the response the log recorded, and no model stands between that
response and the recommender. Hits are a selected sample of the recommendations, so these numbers say how
the items the user did reach were received, not how the whole list would have been.
"""

from __future__ import annotations

import torch
from torch import Tensor

from gptrec.data.batches import EvalExamples
from gptrec.data.log import PAD_ITEM
from gptrec.evaluation.relevance import first_occurrence

USUAL_RELATIVE_RATIO = 0.5  # the relative played ratio of an item received exactly as items usually are


def logged_reception(lists: Tensor, examples: EvalExamples, liked_threshold: float) -> dict[str, float]:
    """The logged response to the hits of all lists together. lists: long [U, K], row u is for examples[u].

    hits_per_list: listed items that are among the user's next events.
    played_ratio_of_hits, liked_share_of_hits: how much of them the user consumed, and how often at least `liked_threshold`.
    relative_ratio_of_hits: the same response as a quantile inside the item; above 0.5 the user took to the
        recommended items more than people usually take to those items, however easy the items are to finish.
    played_ratio_per_item, relative_surplus_per_item: the same two per recommended item, an item that was not hit counting as nothing.
    served_hits_per_list, played_ratio_of_served_hits: only the next events that a recommender served, where the log says which.
    """
    per_user = hits_per_user(lists, examples, liked_threshold)
    total = {name: values.sum().item() for name, values in per_user.items()}
    reward = _reward_per_item(per_user, list_length=lists.shape[1])

    def mean_over(hits: str, summed: str) -> float:
        return total[summed] / total[hits] if total[hits] > 0 else float("nan")

    return {
        "hits_per_list": total["hits"] / len(lists),
        "played_ratio_of_hits": mean_over("hits", "played_ratio"),
        "liked_share_of_hits": mean_over("hits", "liked_hits"),
        "relative_ratio_of_hits": mean_over("hits", "relative_ratio"),
        "played_ratio_per_item": reward["played_ratio_per_item"].mean().item(),
        "relative_surplus_per_item": reward["relative_surplus_per_item"].mean().item(),
        "served_hits_per_list": total["served_hits"] / len(lists),
        "played_ratio_of_served_hits": mean_over("served_hits", "served_played_ratio"),
    }


def logged_reward_per_user(lists: Tensor, examples: EvalExamples, liked_threshold: float) -> dict[str, Tensor]:
    """What each user's list earned in logged feedback per recommended item, an item that was not hit earning nothing. float [U] each.

    Defined for every user, so two recommenders can be compared user by user.
    """
    return _reward_per_item(hits_per_user(lists, examples, liked_threshold), list_length=lists.shape[1])


def _reward_per_item(per_user: dict[str, Tensor], list_length: int) -> dict[str, Tensor]:
    return {
        "played_ratio_per_item": per_user["played_ratio"] / list_length,
        "relative_surplus_per_item": (per_user["relative_ratio"] - USUAL_RELATIVE_RATIO * per_user["hits"]) / list_length,
    }


def hits_per_user(lists: Tensor, examples: EvalExamples, liked_threshold: float) -> dict[str, Tensor]:
    """Per user, the number of hits and the sums of the logged responses over them. float [U] each.

    An item the user consumed several times counts once, with the response to its first next occurrence.
    """
    lists = lists.cpu()
    future_items, played_ratio, relative_ratio = examples.future_items.cpu(), examples.future_played_ratio.cpu(), examples.future_relative_ratio.cpu()
    served_items = future_items.where(~examples.future_is_organic.cpu(), torch.full_like(future_items, PAD_ITEM))

    is_hit, position = _first_occurrence_in_pool(lists, future_items)
    is_served_hit, served_position = _first_occurrence_in_pool(lists, served_items)
    played_ratio_of_hit = played_ratio.gather(1, position)
    return {
        "hits": is_hit.sum(dim=1),
        "played_ratio": (is_hit * played_ratio_of_hit).sum(dim=1),
        "liked_hits": (is_hit * (played_ratio_of_hit >= liked_threshold)).sum(dim=1),
        "relative_ratio": (is_hit * relative_ratio.gather(1, position)).sum(dim=1),
        "served_hits": is_served_hit.sum(dim=1),
        "served_played_ratio": (is_served_hit * played_ratio.gather(1, served_position)).sum(dim=1),
    }


def _first_occurrence_in_pool(lists: Tensor, pools: Tensor) -> tuple[Tensor, Tensor]:
    """Where each listed item first occurs in its user's pool.

    lists: long [U, K], pools: long [U, M] padded with PAD_ITEM. Returns (float [U, K], 1 where the item occurs
    and has not appeared earlier in the list; long [U, K], the column of its first occurrence, 0 where there is none).
    """
    occurs_at = (lists[:, :, None] == pools[:, None, :]) & (lists[:, :, None] != PAD_ITEM)
    is_hit = occurs_at.any(dim=2) & first_occurrence(lists)
    return is_hit.float(), occurs_at.int().argmax(dim=2)
