"""Adapter for KION, the MTS online-cinema dataset (https://github.com/irsafilo/KION_DATASET).

One row per user-item pair: the date the user last watched the film or series, the seconds watched
in total, and the share of the item watched.
"""

from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

import polars as pl

from gptrec.data.preprocess import SECONDS_PER_DAY
from gptrec.paths import RAW_DATA_DIR

# The repository also has a small interactions.csv.zip at its root: it holds only the first 50 MB of the table.
ARCHIVE_URL = "https://github.com/irsafilo/KION_DATASET/raw/f69775be31fa5779907cf0a92ddedb70037fb5ae/data_en.zip"
INTERACTIONS_FILE = "interactions.csv"  # user_id, item_id, last_watch_dt (YYYY-MM-DD), total_dur (seconds), watched_pct (0-100)


def download_interactions(directory: Path = RAW_DATA_DIR) -> Path:
    """Path of the interactions table, downloaded and unpacked on first use."""
    interactions_path = directory / "kion" / INTERACTIONS_FILE
    if not interactions_path.exists():
        interactions_path.parent.mkdir(parents=True, exist_ok=True)
        archive_path, _ = urllib.request.urlretrieve(ARCHIVE_URL)
        with zipfile.ZipFile(archive_path) as archive:
            member = next(name for name in archive.namelist() if name.endswith(INTERACTIONS_FILE) and not name.startswith("__MACOSX"))
            with archive.open(member) as packed, open(interactions_path, "wb") as unpacked:
                shutil.copyfileobj(packed, unpacked)
        Path(archive_path).unlink()
    return interactions_path


def load_raw_events(interactions_path: str | Path) -> pl.DataFrame:
    """The interactions table in the raw-event format of `gptrec.data.preprocess`.

    What KION does not record is filled in as follows. Timestamps are dates, so the order of a user's
    events within one day is the order of the file. The item's length is estimated from each viewing as
    seconds watched / share watched. Nothing says whether a viewing came from a recommendation, so
    every event counts as organic.
    """
    watched_share = pl.col("watched_pct") / 100.0
    return (
        pl.read_csv(interactions_path)
        .drop_nulls(["last_watch_dt", "watched_pct"])
        .select(
            user_id=pl.col("user_id").cast(pl.Int64),
            item_id=pl.col("item_id").cast(pl.Int64),
            timestamp=pl.col("last_watch_dt").str.to_date().to_physical().cast(pl.Int64) * SECONDS_PER_DAY,
            played_ratio=watched_share.clip(0.0, 1.0).cast(pl.Float32),
            duration_seconds=pl.when(pl.col("watched_pct") > 0).then(pl.col("total_dur") / watched_share).cast(pl.Float32),
            is_organic=pl.lit(True),
        )
    )
