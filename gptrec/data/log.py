"""The preprocessed interaction log: its in-memory layout and its on-disk format."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

PAD_ITEM = 0

Period = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class TemporalSplit:
    """Global time boundaries, on the same clock as event timestamps.

    train: t < validation_start;  validation: validation_start <= t < test_start;  test: t >= test_start.
    """

    validation_start: int
    test_start: int

    def bounds(self, period: Period) -> tuple[float, float]:
        return {
            "train": (-np.inf, self.validation_start),
            "validation": (self.validation_start, self.test_start),
            "test": (self.test_start, np.inf),
        }[period]


@dataclass(frozen=True)
class ItemTable:
    """Per-item attributes indexed by item token. Row 0 belongs to PAD_ITEM and holds zeros.

    Tokens are assigned by train-period popularity: token 1 is the most popular item.
    """

    raw_id: np.ndarray  # int64 [V+1], the item id used by the source dataset
    train_popularity: np.ndarray  # int64 [V+1], number of train-period events
    duration_seconds: np.ndarray  # float32 [V+1]
    mean_played_ratio: np.ndarray  # float32 [V+1], train-period mean

    @property
    def num_items(self) -> int:
        return len(self.raw_id) - 1

    @property
    def train_mean_played_ratio(self) -> float:
        """Mean played ratio over all train-period events."""
        return float(np.average(self.mean_played_ratio, weights=self.train_popularity))


@dataclass(frozen=True)
class InteractionLog:
    """Events of all users, sorted by (user, time). Event arrays are aligned: index i is one event.

    The events of user u are rows user_offsets[u] : user_offsets[u + 1].
    """

    item: np.ndarray  # int32 [E], item tokens in 1..V
    played_ratio: np.ndarray  # float32 [E], fraction of the item consumed, in [0, 1]
    relative_ratio: np.ndarray  # float32 [E], the played ratio as a quantile among the item's train-period events: 0.5 is usual for the item
    session_continues: np.ndarray  # bool [E], the user's next event followed within the session gap
    returns: np.ndarray  # bool [E], the user started another session within the return window
    session_continues_known: np.ndarray  # bool [E], False where the observed data ends before the session gap has passed
    returns_known: np.ndarray  # bool [E], False where the observed data ends before the return window has passed
    liked: np.ndarray  # bool [E], the user liked the item after this event, before consuming it again
    disliked: np.ndarray  # bool [E], the same for a dislike
    reactions_known: np.ndarray  # bool [E], False where the dataset records no reactions or the log ends too soon to tell
    adopts: np.ndarray  # bool [E], the user's first event with the item, and the user came back to the item later
    adopts_known: np.ndarray  # bool [E], False for later events of an item, and where the log ends too soon to tell
    is_organic: np.ndarray  # bool [E], the user picked the item themselves (it was not recommended)
    timestamp: np.ndarray  # int64 [E], seconds
    user_offsets: np.ndarray  # int64 [U+1]
    items: ItemTable
    split: TemporalSplit
    directory: Path | None = None

    @property
    def num_users(self) -> int:
        return len(self.user_offsets) - 1

    @property
    def num_events(self) -> int:
        return len(self.item)

    @property
    def num_items(self) -> int:
        return self.items.num_items

    def period_rows(self, period: Period) -> tuple[np.ndarray, np.ndarray]:
        """Per-user row range [start, end) of the events that fall into `period`. Both arrays are int64 [U]."""
        period_start, period_end = self.split.bounds(period)
        return self._first_row_at_or_after(period_start), self._first_row_at_or_after(period_end)

    def _first_row_at_or_after(self, time: float) -> np.ndarray:
        user_starts, user_ends = np.asarray(self.user_offsets[:-1]), np.asarray(self.user_offsets[1:])
        earlier_events = np.concatenate([[0], np.cumsum(np.asarray(self.timestamp) < time)])
        return user_starts + earlier_events[user_ends] - earlier_events[user_starts]

    def save(self, directory: str | Path) -> InteractionLog:
        """Writes the log to `directory` and returns it re-opened from disk (memory-mapped)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name in _EVENT_ARRAYS:
            np.save(directory / f"{name}.npy", getattr(self, name))
        for name in _ITEM_ARRAYS:
            np.save(directory / f"item_{name}.npy", getattr(self.items, name))
        (directory / "split.json").write_text(json.dumps(asdict(self.split)))
        return InteractionLog.load(directory)

    @classmethod
    def load(cls, directory: str | Path) -> InteractionLog:
        """Opens a saved log. Event arrays are memory-mapped, so opening is instant and cheap to repeat."""
        directory = Path(directory)
        events = {name: np.load(directory / f"{name}.npy", mmap_mode="r") for name in _EVENT_ARRAYS}
        items = ItemTable(**{name: np.load(directory / f"item_{name}.npy") for name in _ITEM_ARRAYS})
        split = TemporalSplit(**json.loads((directory / "split.json").read_text()))
        return cls(**events, items=items, split=split, directory=directory)


_EVENT_ARRAYS = (
    "item",
    "played_ratio",
    "relative_ratio",
    "session_continues",
    "returns",
    "session_continues_known",
    "returns_known",
    "liked",
    "disliked",
    "reactions_known",
    "adopts",
    "adopts_known",
    "is_organic",
    "timestamp",
    "user_offsets",
)
_ITEM_ARRAYS = ("raw_id", "train_popularity", "duration_seconds", "mean_played_ratio")
