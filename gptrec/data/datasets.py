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


def download_side_data(dataset: str, directory: Path = RAW_DATA_DIR) -> dict[str, Path]:
    """Paths of what the dataset records beyond its interactions, downloaded on first use: Yambda's likes and dislikes, KION's item table."""
    if dataset == "yambda":
        return yambda.download_reactions(directory=directory)
    if dataset == "kion":
        return {"items": kion.download_items(directory)}
    raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")


def load_raw_events(dataset: str, raw_data_path: str | Path) -> pl.DataFrame:
    """The dataset's interactions in the raw-event format of `gptrec.data.preprocess`."""
    if dataset == "yambda":
        return yambda.load_raw_events(raw_data_path)
    if dataset == "kion":
        return kion.load_raw_events(raw_data_path)
    raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")


def load_reactions(dataset: str, side_data_paths: dict[str, str | Path]) -> pl.DataFrame | None:
    """Likes and dislikes in the reaction format of `gptrec.data.preprocess`; None for a dataset without them."""
    return yambda.load_reactions(side_data_paths) if dataset == "yambda" else None


def load_item_metadata(dataset: str, side_data_paths: dict[str, str | Path]) -> pl.DataFrame | None:
    """Names and kinds of the items, keyed by the dataset's item_id; None for a dataset without them."""
    return kion.load_item_metadata(side_data_paths["items"]) if dataset == "kion" else None
