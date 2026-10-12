"""Persisting the tables and figures an experiment produces."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

Table = TypeVar("Table", pd.DataFrame, pd.Series)


class ReportWriter:
    """Writes every table as CSV and every figure as PNG into one directory."""

    def __init__(self, directory: str | Path, show_figures: bool = True) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._show_figures = show_figures

    def table(self, name: str, table: Table) -> Table:
        """Saves the table as <name>.csv and returns it, so a notebook cell can display what was saved."""
        table.to_csv(self.directory / f"{name}.csv")
        return table

    def figure(self, name: str, figure: Figure) -> None:
        """Saves the figure as <name>.png. It stays open for the notebook to display unless figures are hidden."""
        figure.savefig(self.directory / f"{name}.png", dpi=150, bbox_inches="tight")
        if not self._show_figures:
            plt.close(figure)
