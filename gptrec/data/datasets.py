"""The datasets the pipeline can run on, by name."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from gptrec.data import kion, yambda
from gptrec.paths import RAW_DATA_DIR

DATASETS = ("yambda", "kion")


def download_raw_data(dataset: str, directory: Path = RAW_DATA_DIR) -> Path:
    """Path of the dataset's interaction file, downloaded on first use."""
    if dataset == "yambda":
        return yambda.download_listens(directory=directory)
    if dataset == "kion":
        return kion.download_interactions(directory)
    raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")


def load_raw_events(dataset: str, raw_data_path: str | Path) -> pl.DataFrame:
    """The dataset's interactions in the raw-event format of `gptrec.data.preprocess`."""
    if dataset == "yambda":
        return yambda.load_raw_events(raw_data_path)
    if dataset == "kion":
        return kion.load_raw_events(raw_data_path)
    raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
