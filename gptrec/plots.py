"""Figures for the experiment reports.

Conventions: one measure per axis (measures with different units get their own panel, never a
second y-axis), a fixed colour per series across panels, a legend whenever there is more than
one series, and every reported quantity named in words (see `gptrec.labels`).
"""

from __future__ import annotations

import functools
import textwrap
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler
from matplotlib.figure import Figure

from gptrec.data.log import InteractionLog
from gptrec.labels import readable

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
REFERENCE_COLOR = "#8a8984"  # guides that are not data, e.g. the diagonal of a calibration plot
SURFACE_COLOR = "#fcfcfb"
SECONDARY_TEXT_COLOR = "#52514e"

STYLE = {
    "axes.prop_cycle": cycler(color=SERIES_COLORS),
    "figure.facecolor": SURFACE_COLOR,
    "axes.facecolor": SURFACE_COLOR,
    "axes.edgecolor": "#c9c8c2",
    "axes.grid": True,
    "grid.color": "#e7e6e1",
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 10,
    "axes.labelcolor": SECONDARY_TEXT_COLOR,
    "xtick.color": SECONDARY_TEXT_COLOR,
    "ytick.color": SECONDARY_TEXT_COLOR,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.0,
    "legend.frameon": False,
}

PANEL_WIDTH, PANEL_HEIGHT = 4.4, 3.0
MAX_POINTS_WITH_MARKERS = 30
LABEL_WIDTH = 46  # characters of a panel title or axis label before it wraps
HISTOGRAM_SAMPLE_SIZE = 2_000_000


def _styled(plot: Callable[..., Figure]) -> Callable[..., Figure]:
    @functools.wraps(plot)
    def plot_with_style(*args, **kwargs) -> Figure:
        with plt.rc_context(STYLE):
            figure = plot(*args, **kwargs)
            figure.tight_layout()
            return figure

    return plot_with_style


def _panels(num_panels: int, panels_per_row: int = 4):
    """A figure with a grid of axes; returns the figure and the first `num_panels` axes, the rest are hidden."""
    num_columns = min(num_panels, panels_per_row)
    num_rows = -(-num_panels // num_columns)
    figure, grid = plt.subplots(num_rows, num_columns, figsize=(PANEL_WIDTH * num_columns, PANEL_HEIGHT * num_rows), squeeze=False)
    axes = list(grid.flat)
    for unused in axes[num_panels:]:
        unused.set_visible(False)
    return figure, axes[:num_panels]


def _label(metric: str) -> str:
    """The quantity in words, wrapped to the width of a panel."""
    return textwrap.fill(readable(metric), LABEL_WIDTH)


def _minutes(seconds: float) -> str:
    """A duration in minutes, with a decimal only where whole minutes would not tell short items apart."""
    minutes = seconds / 60
    return f"{minutes:.1f}" if minutes < 10 else f"{minutes:.0f}"


def _marker(num_points: int) -> str | None:
    return "o" if num_points <= MAX_POINTS_WITH_MARKERS else None


@_styled
def small_multiples(curves: pd.DataFrame, x_label: str) -> Figure:
    """One panel per column of `curves`, each drawn against the index."""
    figure, axes = _panels(len(curves.columns))
    for axis, column in zip(axes, curves.columns):
        axis.plot(curves.index, curves[column], marker=_marker(len(curves)))
        axis.set(title=_label(column), xlabel=x_label)
    return figure


@_styled
def series_per_panel(panels: dict[str, pd.DataFrame], x_label: str) -> Figure:
    """One panel per measure; inside a panel one line per column. All tables must list the same columns in the same order."""
    figure, axes = _panels(len(panels))
    for axis, (measure, table) in zip(axes, panels.items()):
        for series in table.columns:
            axis.plot(table.index, table[series], marker=_marker(len(table)), label=series)
        axis.set(title=_label(measure), xlabel=x_label)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(labels))
    return figure


@_styled
def calibration_curves(calibration_tables: dict[str, pd.DataFrame]) -> Figure:
    """Mean observed against mean predicted played ratio, one panel per reward model. Tables come from `calibration_table`."""
    figure, axes = _panels(len(calibration_tables))
    for axis, (name, table) in zip(axes, calibration_tables.items()):
        axis.plot([0, 1], [0, 1], color=REFERENCE_COLOR, linestyle="--", linewidth=1.0, label="perfect calibration")
        axis.plot(table.mean_predicted, table.mean_observed, marker="o", color=SERIES_COLORS[0], label="reward model")
        axis.set(title=f"reward model: {name}", xlabel="mean predicted played ratio", ylabel="mean observed played ratio")
        axis.legend()
    return figure


@_styled
def feedback_overview(log: InteractionLog) -> Figure:
    """The played-ratio distribution, and which items are easy to finish: mean played ratio by item duration and popularity."""
    sample_size = min(log.num_events, HISTOGRAM_SAMPLE_SIZE)
    sample = np.sort(np.random.default_rng(0).choice(log.num_events, size=sample_size, replace=False))
    items = pd.DataFrame({
        "duration_seconds": log.items.duration_seconds[1:],
        "popularity": log.items.train_popularity[1:],
        "mean_played_ratio": log.items.mean_played_ratio[1:],
    })
    by_duration = items.groupby(pd.qcut(items.duration_seconds, 10, duplicates="drop"), observed=True).mean_played_ratio.mean()
    by_popularity = items.groupby(pd.qcut(items.popularity.rank(method="first"), 10, labels=False)).mean_played_ratio.mean()

    figure, (histogram, duration, popularity) = _panels(3)
    histogram.hist(np.asarray(log.played_ratio)[sample], bins=50, color=SERIES_COLORS[0])
    histogram.set(title="played ratio of events", xlabel="played ratio", ylabel="events")
    bins = range(len(by_duration))
    duration.bar(bins, by_duration.to_numpy(), width=0.7, color=SERIES_COLORS[0])
    duration.set_xticks(bins, [_minutes(interval.right) for interval in by_duration.index])
    duration.set(title="mean played ratio by item duration", xlabel="item duration up to, minutes (tenths of the catalogue)")
    popularity.bar(by_popularity.index, by_popularity.to_numpy(), width=0.7, color=SERIES_COLORS[0])
    popularity.set(title="mean played ratio by item popularity", xlabel="popularity decile (9 = most popular)")
    return figure


@_styled
def reward_relevance_frontier(metrics: pd.DataFrame, relevance_rows: list[str], reward_rows: dict[str, str], anchor: str = "pretrained") -> Figure:
    """Relevance against predicted reward, one point per run.

    metrics: one column per run (as from `metrics_table`); `anchor` is the run every other run started from
    and is drawn last, where an ever stronger pull towards it ends. reward_rows: judge name -> metric row
    holding the reward that judge predicts. One panel per relevance row, one line per judge.
    """
    runs = [run for run in metrics.columns if run != anchor] + [anchor]
    last_judge = list(reward_rows)[-1]
    figure, axes = _panels(len(relevance_rows))
    for axis, relevance_row in zip(axes, relevance_rows):
        relevance = metrics.loc[relevance_row, runs]
        for judge, reward_row in reward_rows.items():
            axis.plot(metrics.loc[reward_row, runs], relevance, marker="o", label=f"reward judged by: {judge}")
        for run in runs:
            point = (metrics.loc[reward_rows[last_judge], run], relevance[run])
            axis.annotate(run, point, textcoords="offset points", xytext=(4, 4), fontsize=8, color=SECONDARY_TEXT_COLOR)
        axis.set(title=_label(relevance_row), xlabel="predicted played ratio of the generated list")
        axis.legend()
    return figure


@_styled
def trade_off(curves: dict[str, pd.DataFrame], reward_column: str, measure_columns: list[str]) -> Figure:
    """What several methods pay for predicted reward; one panel per measure, e.g. relevance or catalogue coverage.

    curves: method -> table with one row per run and the metrics as columns. A method with several runs is a
    line through them in table order; a method with a single run is one marker.
    """
    figure, axes = _panels(len(measure_columns))
    for axis, measure_column in zip(axes, measure_columns):
        for method, runs in curves.items():
            line_style = "-" if len(runs) > 1 else "none"
            axis.plot(runs[reward_column], runs[measure_column], marker="o", linestyle=line_style, label=method)
        axis.set(title=_label(measure_column), xlabel=_label(reward_column))
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=min(len(labels), 3))
    return figure
