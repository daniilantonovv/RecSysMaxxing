"""Adapter for Yambda, the Yandex Music listening log (https://huggingface.co/datasets/yandex/yambda).

The listen events are the interactions: each has the played share of the track, the track length and a flag
telling whether the user chose the track themselves or a recommender served it. Likes and dislikes are
separate events; they become labels of the listens they follow (see `gptrec.data.preprocess`).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl
from huggingface_hub import hf_hub_download

from gptrec.paths import RAW_DATA_DIR

REPOSITORY = "yandex/yambda"
SIZES = ("50m", "500m", "5b")  # approximate number of events; 10 thousand, 100 thousand and 1 million users
REACTIONS = ("likes", "dislikes")


def download_listens(size: str = "50m", directory: Path = RAW_DATA_DIR) -> Path:
    """Path of the listens table of the given dataset size, downloaded from the Hugging Face Hub on first use."""
    return _downloaded("listens", size, directory)


def download_reactions(size: str = "50m", directory: Path = RAW_DATA_DIR) -> dict[str, Path]:
    """Paths of the likes and the dislikes table of the given dataset size, downloaded on first use."""
    return {reaction: _downloaded(reaction, size, directory) for reaction in REACTIONS}


def _downloaded(table: str, size: str, directory: Path) -> Path:
    if size not in SIZES:
        raise ValueError(f"unknown Yambda size {size!r}; expected one of {SIZES}")
    return Path(hf_hub_download(repo_id=REPOSITORY, repo_type="dataset", filename=f"flat/{size}/{table}.parquet", local_dir=directory / "yambda"))


def load_raw_events(listens_path: str | Path) -> pl.DataFrame:
    """The listens table in the raw-event format of `gptrec.data.preprocess`.

    Yambda timestamps are seconds since the first event of the dataset, rounded to 5 seconds.
    `played_ratio_pct` can exceed 100 when a track is replayed, so the ratio is capped at 1.
    `is_organic` is 1 for the user's own choices and 0 for the personalised feed and playlists.
    """
    return pl.read_parquet(listens_path).drop_nulls().select(
        user_id=pl.col("uid").cast(pl.Int64),
        item_id=pl.col("item_id").cast(pl.Int64),
        timestamp=pl.col("timestamp").cast(pl.Int64),
        played_ratio=(pl.col("played_ratio_pct").cast(pl.Float32) / 100.0).clip(0.0, 1.0),
        duration_seconds=pl.col("track_length_seconds").cast(pl.Float32),
        is_organic=pl.col("is_organic") == 1,
    )


def load_reactions(reaction_paths: dict[str, str | Path]) -> pl.DataFrame:
    """Likes and dislikes in the reaction format of `gptrec.data.preprocess`, on the clock of the listens."""
    return pl.concat([
        pl.read_parquet(reaction_paths[reaction]).drop_nulls(["uid", "item_id", "timestamp"]).select(
            user_id=pl.col("uid").cast(pl.Int64),
            item_id=pl.col("item_id").cast(pl.Int64),
            timestamp=pl.col("timestamp").cast(pl.Int64),
            liked=pl.lit(reaction == "likes"),
        )
        for reaction in REACTIONS
    ])
