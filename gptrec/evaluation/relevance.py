"""Ranking quality of recommendation lists against the items users actually consumed next."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from gptrec.data.log import PAD_ITEM


def ranking_metrics(recommended: Tensor, relevant: Tensor, ks: Sequence[int]) -> dict[str, float]:
    """Recall@K, NDCG@K, MAP@K and HitRate@K, averaged over the users who have at least one relevant item.

    recommended: long [U, N], best first. A repeated item scores only at its first position.
    relevant: long [U, M], in any order, padded with PAD_ITEM. Repeats count as one relevant item.
    """
    per_user = per_user_ranking_metrics(recommended, relevant, ks)
    has_relevant = ~next(iter(per_user.values())).isnan()
    metrics = {"num_users": float(has_relevant.sum())}
    return metrics | {name: values[has_relevant].mean().item() if has_relevant.any() else float("nan") for name, values in per_user.items()}


def per_user_ranking_metrics(recommended: Tensor, relevant: Tensor, ks: Sequence[int]) -> dict[str, Tensor]:
    """The same metrics for every user separately: metric name -> float [U], NaN for users without relevant items."""
    if max(ks) > recommended.shape[1]:
        raise ValueError(f"cannot compute metrics @{max(ks)} from lists of {recommended.shape[1]} items")
    hits = (contains(recommended, relevant) & first_occurrence(recommended)).float()
    num_relevant = _count_distinct_items(relevant)
    undefined = num_relevant == 0
    num_relevant = num_relevant.clamp(min=1)  # keeps the divisions below finite; these users are set to NaN at the end

    per_user = {}
    for k in ks:
        hits_at_k = hits[:, :k]
        rank = torch.arange(1, k + 1, device=hits.device)
        discount = 1.0 / torch.log2(rank + 1.0)
        best_possible_hits = num_relevant.clamp(max=k)
        ideal_dcg = discount.cumsum(0)[best_possible_hits - 1]
        precision_at_rank = hits_at_k.cumsum(1) / rank

        per_user[f"recall@{k}"] = hits_at_k.sum(1) / num_relevant
        per_user[f"ndcg@{k}"] = (hits_at_k * discount).sum(1) / ideal_dcg
        per_user[f"map@{k}"] = (precision_at_rank * hits_at_k).sum(1) / best_possible_hits
        per_user[f"hit_rate@{k}"] = hits_at_k.amax(1)
    return {name: values.masked_fill(undefined, float("nan")) for name, values in per_user.items()}


def first_occurrence(lists: Tensor) -> Tensor:
    """bool [U, N]: False where the item already appeared earlier in the same list."""
    length = lists.shape[1]
    same_item = lists[:, :, None] == lists[:, None, :]
    earlier_position = torch.ones(length, length, dtype=torch.bool, device=lists.device).tril(diagonal=-1)
    return ~(same_item & earlier_position).any(dim=2)


def contains(lists: Tensor, pools: Tensor) -> Tensor:
    """bool [U, N]: whether lists[u, i] occurs anywhere in pools[u]. PAD_ITEM never counts as contained."""
    return ((lists[:, :, None] == pools[:, None, :]).any(dim=2)) & (lists != PAD_ITEM)


def _count_distinct_items(items: Tensor) -> Tensor:
    """Number of distinct non-padding items in each row. long [U]"""
    ordered = items.sort(dim=1).values
    starts_new_run = torch.ones_like(ordered, dtype=torch.bool)
    starts_new_run[:, 1:] = ordered[:, 1:] != ordered[:, :-1]
    return (starts_new_run & (ordered != PAD_ITEM)).sum(dim=1)
