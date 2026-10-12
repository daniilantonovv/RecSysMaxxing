"""What a recommender recommends, independent of whether it is relevant.

These statistics expose the usual ways of gaming a completion-rate reward: collapsing onto a few
items, drifting to popular or short items, and re-recommending what the user already consumed.
"""

from __future__ import annotations

import torch
from torch import Tensor

from gptrec.data.log import ItemTable
from gptrec.evaluation.relevance import contains, first_occurrence

HEAD_FRACTION = 0.01  # the "head" of the catalogue: its most popular 1% of items


def recommendation_profile(recommended: Tensor, history_items: Tensor, items: ItemTable) -> dict[str, float]:
    """Summary statistics of the lists shown to all users.

    recommended: long [U, N], history_items: long [U, H] padded with PAD_ITEM.
    """
    recommended, history_items = recommended.cpu(), history_items.cpu()
    num_users, list_length = recommended.shape
    is_first = first_occurrence(recommended)
    users_per_item = torch.bincount(recommended[is_first], minlength=items.num_items + 1)[1:].double()

    popularity = torch.as_tensor(items.train_popularity, dtype=torch.double)
    head_size = max(1, round(HEAD_FRACTION * items.num_items))
    head_threshold = popularity[1:].topk(head_size).values.min()

    return {
        "catalog_coverage": (users_per_item > 0).double().mean().item(),
        "item_gini": _gini(users_per_item),
        "top_item_user_share": (users_per_item.max() / num_users).item(),
        "mean_user_pair_overlap": _mean_user_pair_overlap(users_per_item, num_users, list_length),
        "duplicate_rate": 1.0 - is_first.double().mean().item(),
        "history_repeat_rate": contains(recommended, history_items).double().mean().item(),
        "mean_popularity": popularity[recommended].mean().item(),
        "head_item_share": (popularity[recommended] >= head_threshold).double().mean().item(),
        "mean_duration_seconds": torch.as_tensor(items.duration_seconds)[recommended].double().mean().item(),
        "mean_item_played_ratio": torch.as_tensor(items.mean_played_ratio)[recommended].double().mean().item(),
    }


def list_overlap(lists: Tensor, other_lists: Tensor) -> float:
    """Average share of a user's list that also appears in the same user's other list."""
    return contains(lists.cpu(), other_lists.cpu()).double().mean().item()


def _gini(counts: Tensor) -> float:
    """Inequality of how often each item is recommended: 0 = all items equally often, 1 = a single item."""
    ascending = counts.sort().values
    num_items = len(ascending)
    rank = torch.arange(1, num_items + 1, dtype=torch.double)
    return (((2 * rank - num_items - 1) * ascending).sum() / (num_items * ascending.sum())).item()


def _mean_user_pair_overlap(users_per_item: Tensor, num_users: int, list_length: int) -> float:
    """Expected share of items two random users have in common: 0 = fully personalised, 1 = everyone gets the same list."""
    if num_users < 2:
        return float("nan")
    user_pairs_sharing_item = users_per_item * (users_per_item - 1) / 2
    num_user_pairs = num_users * (num_users - 1) / 2
    return (user_pairs_sharing_item.sum() / (num_user_pairs * list_length)).item()
