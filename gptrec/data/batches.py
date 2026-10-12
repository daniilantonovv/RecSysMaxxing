"""Turning the interaction log into tensors.

Every sequence tensor is right-padded with PAD_ITEM. The models attend causally, so a real event never
sees the padding behind it and no attention mask is needed anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import torch
from torch import Tensor

from gptrec.data.log import PAD_ITEM, InteractionLog, Period
from gptrec.tensors import TensorGroup


@dataclass
class HistoryBatch(TensorGroup):
    """What is known about each user at the moment a recommendation is requested.

    Row b holds the user's most recent events in chronological order in columns 0 .. lengths[b] - 1.
    """

    items: Tensor  # long [B, H]
    played_ratio: Tensor  # float [B, H]
    lengths: Tensor  # long [B], at least 1

    def repeat_interleave(self, repeats: int) -> HistoryBatch:
        """Each history `repeats` times in a row: rows b * repeats .. (b + 1) * repeats - 1 are copies of b."""
        return self.map(lambda tensor: tensor.repeat_interleave(repeats, dim=0))


@dataclass
class EvalExamples(TensorGroup):
    """Recommendation requests whose outcome is known: a user's history up to some moment and what the user did next."""

    history: HistoryBatch
    future_items: Tensor  # long [U, M], the user's next events in order; PAD_ITEM where there are fewer than M
    future_played_ratio: Tensor  # float [U, M]
    user: Tensor  # long [U], user index in the interaction log

    def batches(self, batch_size: int) -> Iterator[EvalExamples]:
        for start in range(0, len(self), batch_size):
            yield self[start : start + batch_size]


@dataclass
class EventBatch(TensorGroup):
    """Windows of consecutive events for teacher-forced training and scoring."""

    items: Tensor  # long [B, L]
    played_ratio: Tensor  # float [B, L]
    session_continues: Tensor  # float [B, L]
    returns: Tensor  # float [B, L]
    session_continues_known: Tensor  # bool [B, L], False where the observed data ends too soon to tell
    returns_known: Tensor  # bool [B, L]
    is_organic: Tensor  # bool [B, L], the user picked the item themselves (it was not recommended)
    is_target: Tensor  # bool [B, L], events to learn from or score; the others only provide context


@dataclass
class SwappedItems(TensorGroup):
    """Windows in which one target event each had its item replaced by an item the user did not choose then."""

    items: Tensor  # long [B, L]
    position: Tensor  # long [B], the column of the replaced item
    is_different: Tensor  # bool [B], False where the replacement happens to be the item it replaced


def swap_one_target_item(batch: EventBatch) -> SwappedItems:
    """In every window, a random target event gets the item of a random target event of the batch.

    Replacements are therefore as popular as the items users really chose, and the events before the
    replaced one are untouched. Every window needs at least one target event.
    """
    position = torch.multinomial(batch.is_target.float(), num_samples=1)
    items_of_targets = batch.items[batch.is_target]
    replacement = items_of_targets[torch.randint(len(items_of_targets), position.shape, device=position.device)]
    is_different = (replacement != batch.items.gather(1, position)).squeeze(1)
    return SwappedItems(batch.items.scatter(1, position, replacement), position.squeeze(1), is_different)


def gather_right_padded(values: np.ndarray, starts: np.ndarray, ends: np.ndarray, width: int, pad_value) -> np.ndarray:
    """Stacks the slices values[starts[i] : ends[i]] into an [N, width] array, padding short slices on the right."""
    if len(starts) and (ends - starts).max() > width:
        raise ValueError(f"a slice is longer than the requested width {width}")
    positions = starts[:, None] + np.arange(width)[None, :]
    is_real = positions < ends[:, None]
    gathered = np.asarray(values)[np.minimum(positions, len(values) - 1)]
    return np.where(is_real, gathered, pad_value)


def _history_batch(log: InteractionLog, history_ends: np.ndarray, user_starts: np.ndarray, history_len: int) -> HistoryBatch:
    history_starts = np.maximum(user_starts, history_ends - history_len)
    return HistoryBatch(
        items=torch.from_numpy(gather_right_padded(log.item, history_starts, history_ends, history_len, PAD_ITEM)).long(),
        played_ratio=torch.from_numpy(gather_right_padded(log.played_ratio, history_starts, history_ends, history_len, 0.0)).float(),
        lengths=torch.from_numpy(history_ends - history_starts).long(),
    )


def _examples(log: InteractionLog, users: np.ndarray, future_starts: np.ndarray, future_limits: np.ndarray, history_len: int, max_future: int) -> EvalExamples:
    """For each user, the history before row `future_starts` and up to `max_future` events from that row on, none at or after `future_limits`."""
    user_starts = np.asarray(log.user_offsets[:-1])[users]
    future_ends = np.minimum(future_limits, future_starts + max_future)
    return EvalExamples(
        history=_history_batch(log, future_starts, user_starts, history_len),
        future_items=torch.from_numpy(gather_right_padded(log.item, future_starts, future_ends, max_future, PAD_ITEM)).long(),
        future_played_ratio=torch.from_numpy(gather_right_padded(log.played_ratio, future_starts, future_ends, max_future, 0.0)).float(),
        user=torch.from_numpy(users).long(),
    )


def build_eval_examples(log: InteractionLog, period: Period, history_len: int, max_future: int) -> EvalExamples:
    """A request at the start of `period` for every user who has events both before and inside it."""
    user_starts = np.asarray(log.user_offsets[:-1])
    period_starts, period_ends = log.period_rows(period)
    users = np.flatnonzero((period_starts > user_starts) & (period_ends > period_starts))
    return _examples(log, users, period_starts[users], period_ends[users], history_len, max_future)


class TrainingHistorySampler:
    """Draws random recommendation moments from the train period: a random user at a random point of their timeline.

    Each comes with what the user did next inside the train period, which may be nothing.
    """

    def __init__(self, log: InteractionLog, history_len: int, max_future: int, min_history: int = 8, seed: int = 0) -> None:
        self._log = log
        self._history_len = history_len
        self._max_future = max_future
        self._min_history = min_history
        self._random = np.random.default_rng(seed)
        self._user_starts = np.asarray(log.user_offsets[:-1])
        _, self._train_ends = log.period_rows("train")
        self._eligible_users = np.flatnonzero(self._train_ends - self._user_starts >= min_history)
        if len(self._eligible_users) == 0:
            raise ValueError(f"no user has {min_history} train events")

    def sample(self, batch_size: int) -> EvalExamples:
        users = self._random.choice(self._eligible_users, size=batch_size)
        earliest_end = self._user_starts[users] + self._min_history
        history_ends = self._random.integers(earliest_end, self._train_ends[users], endpoint=True)
        return _examples(self._log, users, history_ends, self._train_ends[users], self._history_len, self._max_future)


class EventWindows:
    """Slices of user timelines, each a run of consecutive events of one user.

    Events at or after a window's target start are targets; the events before it are context only.
    """

    def __init__(self, log: InteractionLog, window_starts: np.ndarray, window_ends: np.ndarray, target_starts: np.ndarray, seq_len: int) -> None:
        self._log = log
        self._window_starts = window_starts
        self._window_ends = window_ends
        self._target_starts = target_starts
        self.seq_len = seq_len

    @classmethod
    def for_training(cls, log: InteractionLog, seq_len: int, stride: int | None = None, users: np.ndarray | None = None) -> EventWindows:
        """Windows covering every user's train period; consecutive windows start at most `stride` events apart. All events are targets.

        The last window of a user is moved back to end at the user's last train event, so no window is a short leftover.
        """
        stride = stride or seq_len
        if not 0 < stride <= seq_len:
            raise ValueError(f"stride {stride} must be between 1 and seq_len={seq_len}, otherwise events fall between windows")
        period_starts, period_ends = (rows[_selection(users)] for rows in log.period_rows("train"))
        num_events = period_ends - period_starts
        windows_per_user = np.where(num_events >= 2, 1 + _ceil_div(np.maximum(num_events - seq_len, 0), stride), 0)
        last_window_starts = np.maximum(period_ends - seq_len, period_starts)
        window_starts = np.repeat(period_starts, windows_per_user) + stride * _index_within_group(windows_per_user)
        window_starts = np.minimum(window_starts, np.repeat(last_window_starts, windows_per_user))
        window_ends = np.minimum(window_starts + seq_len, np.repeat(period_ends, windows_per_user))
        return cls(log, window_starts, window_ends, target_starts=window_starts, seq_len=seq_len)

    @classmethod
    def for_heldout(cls, log: InteractionLog, period: Period, seq_len: int, users: np.ndarray | None = None) -> EventWindows:
        """Windows in which every event of `period` is a target exactly once, preceded by at least seq_len // 2 events of context."""
        targets_per_window = seq_len // 2
        user_starts = np.asarray(log.user_offsets[:-1])[_selection(users)]
        period_starts, period_ends = (rows[_selection(users)] for rows in log.period_rows(period))
        windows_per_user = _ceil_div(period_ends - period_starts, targets_per_window)
        target_starts = np.repeat(period_starts, windows_per_user) + targets_per_window * _index_within_group(windows_per_user)
        window_ends = np.minimum(target_starts + targets_per_window, np.repeat(period_ends, windows_per_user))
        window_starts = np.maximum(np.repeat(user_starts, windows_per_user), window_ends - seq_len)
        return cls(log, window_starts, window_ends, target_starts, seq_len)

    def __len__(self) -> int:
        return len(self._window_starts)

    def num_batches(self, batch_size: int, drop_last: bool = False) -> int:
        return len(self) // batch_size if drop_last else _ceil_div(len(self), batch_size)

    def batches(self, batch_size: int, shuffle_seed: int | None = None, drop_last: bool = False) -> Iterator[EventBatch]:
        """Iterates over all windows once; in random order when `shuffle_seed` is given."""
        order = np.arange(len(self)) if shuffle_seed is None else np.random.default_rng(shuffle_seed).permutation(len(self))
        for batch in range(self.num_batches(batch_size, drop_last)):
            yield self._batch(np.sort(order[batch * batch_size : (batch + 1) * batch_size]))

    def _batch(self, windows: np.ndarray) -> EventBatch:
        starts, ends = self._window_starts[windows], self._window_ends[windows]
        rows = starts[:, None] + np.arange(self.seq_len)[None, :]
        is_target = (rows >= self._target_starts[windows][:, None]) & (rows < ends[:, None])

        def gather(values: np.ndarray, pad_value) -> Tensor:
            return torch.from_numpy(gather_right_padded(values, starts, ends, self.seq_len, pad_value))

        return EventBatch(
            items=gather(self._log.item, PAD_ITEM).long(),
            played_ratio=gather(self._log.played_ratio, 0.0).float(),
            session_continues=gather(self._log.session_continues, False).float(),
            returns=gather(self._log.returns, False).float(),
            session_continues_known=gather(self._log.session_continues_known, False),
            returns_known=gather(self._log.returns_known, False),
            is_organic=gather(self._log.is_organic, False),
            is_target=torch.from_numpy(is_target),
        )


def _selection(users: np.ndarray | None):
    """Index that keeps the given users, or everyone when no subset is requested."""
    return slice(None) if users is None else users


def _ceil_div(numerator, denominator):
    return -(-numerator // denominator)


def _index_within_group(group_sizes: np.ndarray) -> np.ndarray:
    """[0, 1, .., n0 - 1, 0, 1, .., n1 - 1, ...] for group sizes n0, n1, ..."""
    group_first_index = np.cumsum(group_sizes) - group_sizes
    return np.arange(group_sizes.sum()) - np.repeat(group_first_index, group_sizes)
