"""From a raw event table to an InteractionLog: sessions, temporal split, item vocabulary, user filter.

Every dataset adapter (see `gptrec.data.yambda`) produces the same raw event table, a polars
DataFrame with one row per interaction and the columns listed in RAW_EVENT_COLUMNS. A dataset that
records likes and dislikes also produces a reaction table with the columns of REACTION_COLUMNS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from gptrec.data.log import InteractionLog, ItemTable, TemporalSplit

RAW_EVENT_COLUMNS = {
    "user_id": "any integer id",
    "item_id": "any integer id",
    "timestamp": "seconds, any origin",
    "played_ratio": "fraction of the item consumed, in [0, 1]",
    "duration_seconds": "length of the item; null where unknown",
    "is_organic": "True when the user picked the item themselves, False when it was recommended",
}
REACTION_COLUMNS = {
    "user_id": "as in the raw events",
    "item_id": "as in the raw events",
    "timestamp": "seconds, on the clock of the raw events",
    "liked": "True for a like, False for a dislike",
}

SECONDS_PER_DAY = 86_400


@dataclass(frozen=True)
class PreprocessingConfig:
    max_items: int = 30_000  # catalogue: the most popular items of the train period
    min_item_events: int = 20  # train-period events an item needs to enter the catalogue
    min_user_events: int = 50  # train-period events (on catalogue items) a user needs to be kept
    validation_days: float = 7.0
    test_days: float = 7.0  # the last days of the log
    session_gap_seconds: int = 30 * 60  # a longer pause between two events of a user ends the session
    return_window_seconds: int = SECONDS_PER_DAY  # the user "returns" when their next session starts within this long
    reaction_window_seconds: int = SECONDS_PER_DAY  # a like or dislike counts for the user's last event with the item at most this long before it
    adoption_window_seconds: int | None = None  # a new item is "adopted" when the user comes back to it within this long; None: not measured


def build_interaction_log(raw_events: pl.DataFrame, config: PreprocessingConfig, reactions: pl.DataFrame | None = None) -> InteractionLog:
    """Builds the in-memory log. Call `.save(directory)` on the result to persist and memory-map it.

    `reactions`: likes and dislikes with the columns of REACTION_COLUMNS; without them every reaction label is unknown.
    """
    events = _in_user_time_order(raw_events.select(list(RAW_EVENT_COLUMNS)).with_row_index("source_row"))
    split = _temporal_split(events, config)
    # Sessions, returns, reactions and adoption are properties of everything the user did, so they are derived before any event is dropped.
    events = events.with_columns(_retention_signals(events, split, config))
    events = events.with_columns(_reaction_labels(events, reactions, split, config))
    events = events.with_columns(_adoption_labels(events, split, config))
    in_train_period = pl.col("timestamp") < split.validation_start

    catalogue = _item_catalogue(events.filter(in_train_period), config)
    events = events.join(catalogue.select("item_id", "token"), on="item_id", how="inner")
    events = events.with_columns(_relative_ratio(events, split))
    train_events_per_user = events.filter(in_train_period).group_by("user_id").agg(pl.len().alias("num_train_events"))
    active_users = train_events_per_user.filter(pl.col("num_train_events") >= config.min_user_events)
    events = _in_user_time_order(events.join(active_users.select("user_id"), on="user_id", how="semi"))

    events_per_user = events.group_by("user_id", maintain_order=True).agg(pl.len().alias("num_events"))
    user_offsets = np.concatenate([[0], np.cumsum(events_per_user["num_events"].to_numpy())]).astype(np.int64)
    print(
        f"kept {len(events):,} of {len(raw_events):,} events ({len(events) / len(raw_events):.1%}), "
        f"{len(events_per_user):,} users, {len(catalogue):,} items"
    )
    return InteractionLog(
        item=events["token"].to_numpy().astype(np.int32),
        played_ratio=events["played_ratio"].to_numpy().astype(np.float32),
        relative_ratio=events["relative_ratio"].to_numpy().astype(np.float32),
        session_continues=events["session_continues"].to_numpy().astype(bool),
        returns=events["returns"].to_numpy().astype(bool),
        session_continues_known=events["session_continues_known"].to_numpy().astype(bool),
        returns_known=events["returns_known"].to_numpy().astype(bool),
        liked=events["liked"].to_numpy().astype(bool),
        disliked=events["disliked"].to_numpy().astype(bool),
        reactions_known=events["reactions_known"].to_numpy().astype(bool),
        adopts=events["adopts"].to_numpy().astype(bool),
        adopts_known=events["adopts_known"].to_numpy().astype(bool),
        is_organic=events["is_organic"].to_numpy().astype(bool),
        timestamp=events["timestamp"].to_numpy().astype(np.int64),
        user_offsets=user_offsets,
        items=_item_table(catalogue),
        split=split,
    )


def _in_user_time_order(events: pl.DataFrame) -> pl.DataFrame:
    """Sorted by user, then time; events with equal timestamps keep the order of the source table."""
    return events.sort(["user_id", "timestamp", "source_row"])


def _retention_signals(events: pl.DataFrame, split: TemporalSplit, config: PreprocessingConfig) -> list[pl.Series]:
    """For every event, whether the user kept going and whether they came back. Expects user-time order.

    session_continues: the user's next event follows within the session gap.
    returns: the user starts another session within the return window, i.e. has a later event that is
             more than a session gap but at most a return window away.

    That nothing followed is only a fact once the waiting time has passed inside the observed data, so each
    signal comes with a `_known` flag. Train-period events are observed until the train period ends: their
    labels never depend on held-out events. Held-out events are observed until the log ends.
    """
    timestamp, user_id = events["timestamp"].to_numpy(), events["user_id"].to_numpy()
    clock = _clock_shared_by_all_users(timestamp, user_id, longest_wait=config.session_gap_seconds + config.return_window_seconds)

    seconds_to_next_event = np.diff(clock, append=np.iinfo(np.int64).max)
    first_event_after_session = np.searchsorted(clock, clock + config.session_gap_seconds, side="right")
    has_event_after_session = first_event_after_session < len(clock)
    seconds_to_next_session = clock[np.minimum(first_event_after_session, len(clock) - 1)] - clock

    last_observed_second = _last_observed_second(timestamp, split)
    return [
        pl.Series("session_continues", seconds_to_next_event <= config.session_gap_seconds),
        pl.Series("returns", has_event_after_session & (seconds_to_next_session <= config.return_window_seconds)),
        pl.Series("session_continues_known", timestamp + config.session_gap_seconds <= last_observed_second),
        pl.Series("returns_known", timestamp + config.return_window_seconds <= last_observed_second),
    ]


def _last_observed_second(timestamp: np.ndarray, split: TemporalSplit) -> np.ndarray:
    """The last moment whose events may decide the label of each event: train-period labels never depend on held-out events. int64 [E]"""
    return np.where(timestamp < split.validation_start, split.validation_start - 1, timestamp.max())


def _reaction_labels(events: pl.DataFrame, reactions: pl.DataFrame | None, split: TemporalSplit, config: PreprocessingConfig) -> list[pl.Series]:
    """For every event, whether the user liked or disliked the item afterwards, and whether the log can tell.

    A reaction counts for the user's latest event with the same item at most `reaction_window_seconds` before
    it, so a like given during or soon after a listen labels that listen and no earlier one. A label is known
    once its window has passed inside the observed data; without a reaction table no label is known.
    Expects user-time order and the raw item ids.
    """
    timestamp = events["timestamp"].to_numpy()
    last_observed_second = _last_observed_second(timestamp, split)
    nothing = np.zeros(len(events), dtype=bool)
    if reactions is None:
        return [pl.Series("liked", nothing), pl.Series("disliked", nothing), pl.Series("reactions_known", nothing)]

    listens = events.select(
        "source_row", "user_id", "item_id", listen_timestamp=pl.col("timestamp"), last_observed_second=pl.Series(last_observed_second)
    ).sort("listen_timestamp")
    attributed = (
        reactions.select(list(REACTION_COLUMNS))
        .sort("timestamp")
        .join_asof(
            listens,
            left_on="timestamp",
            right_on="listen_timestamp",
            by=["user_id", "item_id"],
            strategy="backward",
            tolerance=config.reaction_window_seconds,
        )
        .filter(pl.col("source_row").is_not_null() & (pl.col("timestamp") <= pl.col("last_observed_second")))
        .group_by("source_row")
        .agg(liked=pl.col("liked").any(), disliked=(~pl.col("liked")).any())
    )
    source_row = events["source_row"].to_numpy().astype(np.int64)
    attributed_row = attributed["source_row"].to_numpy().astype(np.int64)
    liked, disliked = np.zeros(source_row.max() + 1, dtype=bool), np.zeros(source_row.max() + 1, dtype=bool)
    liked[attributed_row] = attributed["liked"].to_numpy().astype(bool)
    disliked[attributed_row] = attributed["disliked"].to_numpy().astype(bool)
    return [
        pl.Series("liked", liked[source_row]),
        pl.Series("disliked", disliked[source_row]),
        pl.Series("reactions_known", timestamp + config.reaction_window_seconds <= last_observed_second),
    ]


def _adoption_labels(events: pl.DataFrame, split: TemporalSplit, config: PreprocessingConfig) -> list[pl.Series]:
    """For every event, whether it is the user's first with the item and the user came back to the item later.

    The item counts as adopted when the user consumed it again more than a session gap and at most
    `adoption_window_seconds` after the first time. The label is known for first events whose window ends
    inside the observed data, and unknown for every later event and when no window is set. A replay inside
    the first session does not count: it says more about the session than about the item. Expects user-time order.
    """
    nothing = np.zeros(len(events), dtype=bool)
    if config.adoption_window_seconds is None:
        return [pl.Series("adopts", nothing), pl.Series("adopts_known", nothing)]
    by_pair = ["user_id", "item_id"]
    pairs = events.select(*by_pair, "timestamp").with_columns(
        first_timestamp=pl.col("timestamp").min().over(by_pair),
        is_first=pl.int_range(pl.len()).over(by_pair) == 0,
    )
    is_after_first_session = pl.col("timestamp") > pl.col("first_timestamp") + config.session_gap_seconds
    comeback = pairs.select(pl.when(is_after_first_session).then(pl.col("timestamp")).min().over(by_pair)).to_series()
    timestamp = events["timestamp"].to_numpy()
    is_first = pairs["is_first"].to_numpy().astype(bool)
    seconds_to_comeback = comeback.fill_null(np.iinfo(np.int64).max).to_numpy() - timestamp
    return [
        pl.Series("adopts", is_first & (seconds_to_comeback <= config.adoption_window_seconds)),
        pl.Series("adopts_known", is_first & (timestamp + config.adoption_window_seconds <= _last_observed_second(timestamp, split))),
    ]


def _relative_ratio(events: pl.DataFrame, split: TemporalSplit) -> pl.Series:
    """For every event, the share of the item's train-period events that were played less, equally played ones counting half.

    A quantile inside the item: 0.5 is what is usual for it, whatever its length or kind, and the train-period
    mean of every item is exactly 0.5. A long series that hardly anybody finishes and a film that most people
    finish are thereby put on one scale. Held-out events are ranked against the train period only.
    Expects the `token` column of catalogue items.
    """
    token = events["token"].to_numpy().astype(np.float64)
    played_ratio = events["played_ratio"].to_numpy().astype(np.float64)
    # Played ratios lie in [0, 1], so the events of item `token` occupy [2 * token, 2 * token + 1] and no other item's do.
    sort_key = 2.0 * token + played_ratio
    train_keys = np.sort(sort_key[events["timestamp"].to_numpy() < split.validation_start])
    first_event_of_item = np.searchsorted(train_keys, 2.0 * token, side="left")
    num_train_events = np.searchsorted(train_keys, 2.0 * token + 1.0, side="right") - first_event_of_item
    played_less = np.searchsorted(train_keys, sort_key, side="left") - first_event_of_item
    played_at_most_as_much = np.searchsorted(train_keys, sort_key, side="right") - first_event_of_item
    return pl.Series("relative_ratio", ((played_less + played_at_most_as_much) / (2.0 * num_train_events)).astype(np.float32))


def _clock_shared_by_all_users(timestamp: np.ndarray, user_id: np.ndarray, longest_wait: int) -> np.ndarray:
    """Increasing event times in which two users are never within `longest_wait` seconds of each other.

    Events are in user-time order, so one binary search over this clock finds a user's next event
    after any moment without ever returning an event of somebody else as a near one. int64 [E]
    """
    seconds_since_first_event = timestamp - timestamp.min()
    starts_new_user = np.concatenate([[False], user_id[1:] != user_id[:-1]])
    seconds_between_users = int(seconds_since_first_event.max()) + longest_wait + 1
    return np.cumsum(starts_new_user).astype(np.int64) * seconds_between_users + seconds_since_first_event


def _temporal_split(events: pl.DataFrame, config: PreprocessingConfig) -> TemporalSplit:
    last_timestamp = int(events["timestamp"].max())
    test_start = last_timestamp - round(config.test_days * SECONDS_PER_DAY)
    validation_start = test_start - round(config.validation_days * SECONDS_PER_DAY)
    return TemporalSplit(validation_start=validation_start, test_start=test_start)


def _item_catalogue(train_events: pl.DataFrame, config: PreprocessingConfig) -> pl.DataFrame:
    """One row per kept item, most popular first; `token` is its 1-based popularity rank."""
    return (
        train_events.group_by("item_id")
        .agg(
            train_popularity=pl.len(),
            duration_seconds=pl.col("duration_seconds").median(),
            mean_played_ratio=pl.col("played_ratio").mean(),
        )
        .filter(pl.col("train_popularity") >= config.min_item_events)
        .sort(["train_popularity", "item_id"], descending=[True, False])
        .head(config.max_items)
        .with_columns(pl.col("duration_seconds").fill_null(pl.col("duration_seconds").median()).fill_null(0.0))
        .with_row_index("token", offset=1)
    )


def _item_table(catalogue: pl.DataFrame) -> ItemTable:
    def with_padding_row(column: str, dtype: type) -> np.ndarray:
        values = catalogue[column].to_numpy().astype(dtype)
        return np.concatenate([np.zeros(1, dtype), values])

    return ItemTable(
        raw_id=with_padding_row("item_id", np.int64),
        train_popularity=with_padding_row("train_popularity", np.int64),
        duration_seconds=with_padding_row("duration_seconds", np.float32),
        mean_played_ratio=with_padding_row("mean_played_ratio", np.float32),
    )
