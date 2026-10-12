"""Baselines that only count: item popularity, the user's own replays, and which item tends to follow which."""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from gptrec.data.log import PAD_ITEM, InteractionLog, ItemTable
from gptrec.models.sequence_recommender import MASKED_LOGIT


class PopularityRecommender:
    """Scores every item by its train-period popularity, the same for every user."""

    def __init__(self, items: ItemTable, device: torch.device) -> None:
        self._logits = torch.as_tensor(items.train_popularity, dtype=torch.float32, device=device).log1p()
        self._logits[PAD_ITEM] = MASKED_LOGIT

    def next_item_logits(self, items: Tensor, lengths: Tensor) -> Tensor:
        """[B, V+1], identical rows."""
        return self._logits.expand(len(items), -1)


class ReplayRecommender:
    """Recommends what the user already plays: items of the visible history, most played first, the more recent first among equals.

    No model is involved, so this is the floor for any recommender in a domain where people return to what
    they know. Items outside the history follow in order of popularity. Where history items may not be
    recommended at all, this reduces to popularity.
    """

    def __init__(self, items: ItemTable, device: torch.device) -> None:
        popularity = torch.as_tensor(items.train_popularity, dtype=torch.float32, device=device)
        self._popularity_share = popularity / popularity.sum()  # below 1: only orders items the user has not played

    def next_item_logits(self, items: Tensor, lengths: Tensor) -> Tensor:
        """[B, V+1]: number of plays in the history, plus a recency term below 1 that orders equally played items."""
        num_users, width = items.shape
        recency = (torch.arange(1, width + 1, dtype=torch.float32, device=items.device) / (width + 1)).repeat(num_users, 1)
        plays = torch.zeros(num_users, len(self._popularity_share), device=items.device).scatter_add_(1, items, torch.ones_like(recency))
        last_played = torch.zeros_like(plays).scatter_reduce(1, items, recency, reduce="amax")
        logits = plays + last_played + self._popularity_share
        logits[:, PAD_ITEM] = MASKED_LOGIT
        return logits


class MarkovChainRecommender:
    """First-order Markov chain: scores an item by how often it directly followed the user's last item in the train period.

    Items never observed after the last item are ranked by popularity, below every observed successor.
    """

    def __init__(self, log: InteractionLog, device: torch.device, max_successors: int = 200) -> None:
        successors, transition_counts = _most_frequent_successors(log, max_successors)
        self._successors = torch.as_tensor(successors, device=device)  # long [V+1, N], PAD_ITEM where an item has fewer successors
        self._transition_counts = torch.as_tensor(transition_counts, device=device)  # float [V+1, N], at least 1 for real successors
        popularity = torch.as_tensor(log.items.train_popularity, dtype=torch.float32, device=device)
        self._popularity_share = popularity / popularity.sum()  # below 1, so it only orders items without an observed transition

    def next_item_logits(self, items: Tensor, lengths: Tensor) -> Tensor:
        """[B, V+1]"""
        last_item = items[torch.arange(len(items), device=items.device), lengths - 1]
        logits = self._popularity_share.repeat(len(items), 1)
        logits.scatter_add_(1, self._successors[last_item], self._transition_counts[last_item])
        logits[:, PAD_ITEM] = MASKED_LOGIT
        return logits


def _most_frequent_successors(log: InteractionLog, max_successors: int) -> tuple[np.ndarray, np.ndarray]:
    """For every item token, the items that most often came right after it in the train period, and how often.

    Returns (successors int64 [V+1, max_successors], counts float32 [V+1, max_successors]), most frequent first.
    """
    item = np.asarray(log.item).astype(np.int64)
    follows_same_user = np.ones(log.num_events, dtype=bool)
    follows_same_user[np.asarray(log.user_offsets[:-1])] = False
    in_train_period = np.asarray(log.timestamp) < log.split.validation_start
    is_transition = follows_same_user[1:] & in_train_period[1:]
    previous_item, next_item = item[:-1][is_transition], item[1:][is_transition]

    vocabulary_size = log.num_items + 1
    pair_code, pair_count = np.unique(previous_item * vocabulary_size + next_item, return_counts=True)
    previous_item, next_item = pair_code // vocabulary_size, pair_code % vocabulary_size
    by_item_then_frequency = np.lexsort((-pair_count, previous_item))
    previous_item, next_item, pair_count = previous_item[by_item_then_frequency], next_item[by_item_then_frequency], pair_count[by_item_then_frequency]
    rank_among_successors = np.arange(len(previous_item)) - np.searchsorted(previous_item, previous_item, side="left")
    kept = rank_among_successors < max_successors

    successors = np.full((vocabulary_size, max_successors), PAD_ITEM, dtype=np.int64)
    counts = np.zeros((vocabulary_size, max_successors), dtype=np.float32)
    successors[previous_item[kept], rank_among_successors[kept]] = next_item[kept]
    counts[previous_item[kept], rank_among_successors[kept]] = pair_count[kept]
    return successors, counts
