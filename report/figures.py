"""The figures of the paper that put several reports of a session on common axes, drawn as SVG from the CSV reports.

    python3 report/figures.py

Writes report/figures/<name>.svg for every figure of FIGURES. A point of a figure is one run of one report;
runs a session does not have are left out, so a figure fills up as sessions are added. A few figures read a table
derived from a report instead (DERIVED_REPORTS). Needs nothing beyond the standard library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import escape

from tables import DATASET_NAMES, FRONTIER_SESSIONS, REVIEW_SESSIONS, SESSIONS, Source, number, report_cells

OUTPUT = Path(__file__).resolve().parent / "figures"
FONT = "New Computer Modern"
PANEL_WIDTH, PANEL_HEIGHT = 250, 172
MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, MARGIN_BOTTOM = 48, 12, 26, 40
LEGEND_ROW_HEIGHT, LEGEND_COLUMN_WIDTH = 19, 240
HORIZONTAL_LABEL_HEIGHT = 16  # the label of the horizontal axis is written once, under the last row of panels
RIGHT_PADDING = 16  # room for the last tick label of the rightmost panel
AXIS, GRID = "#444444", "#e4e4e4"
# The colour-blind-safe palette of Okabe and Ito.
BLUE, ORANGE, GREEN, VERMILION, PURPLE, SKY, BLACK = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#222222"
SMALLEST_DISTANCE = 0.008  # nats per item; runs that moved less are drawn at this edge of the logarithmic axis
MOST_INTEGER_TICKS = 12  # a horizontal axis of run names gets a tick per integer up to this span, round ticks beyond it

KL = "training/kl_to_reference"
NDCG = "all/ndcg@10"
SHARE = "predicted/heldout/played_ratio"
LOGGED_SHARE = "logged/played_ratio_per_item"
SURPLUS = "logged/relative_surplus_per_item"
COVERAGE = "profile/catalog_coverage"


@dataclass(frozen=True)
class Series:
    """Runs drawn with one colour and one marker; `runs` are (report, run) in the order they are joined."""

    label: str
    runs: tuple[tuple[str, str], ...]
    color: str
    marker: str = "circle"
    joined: bool = True
    prefix: str = ""  # put before the quantity of each axis, for reports whose columns name a policy and a quantity
    ordered: bool = False  # join the points from left to right rather than in the order of `runs`


@dataclass(frozen=True)
class Axis:
    quantity: str | None  # None: the run's name is the value, for reports whose rows are positions or steps
    label: str
    logarithmic: bool = False
    from_zero: bool = False  # a vertical axis that starts at zero whatever its points, for a quantity that starts there


@dataclass(frozen=True)
class Figure:
    """One row of panels per session; all panels of a figure share the horizontal axis and the series."""

    name: str
    horizontal: Axis
    vertical: tuple[Axis, ...]
    series: tuple[Series, ...]
    sessions: tuple[str, ...] = tuple(SESSIONS)
    legend_columns: int = 3


def anchor_sweep(weights: tuple[str, ...], run_name: str = "kl={}", report: str = "kl_sweep") -> tuple[tuple[str, str], ...]:
    return tuple((report, run_name.format(weight)) for weight in weights)


ANCHOR_WEIGHTS = ("1.0", "0.5", "0.3", "0.2", "0.1", "0.05", "0.03", "0.0")
KL_BUDGETS = ("0.1", "0.2", "0.5")
RERANKING_WEIGHTS = ("1.0", "0.3", "0.1", "0.03", "0.01", "0.0")
BASE_RECOMMENDER = Series("base recommender, generated lists", (("kl_sweep", "pretrained"),), BLACK, marker="star", joined=False)
POSITIONS = tuple(("position_hits", str(position)) for position in range(1, 11))
MULTI_TOKEN_POSITIONS = tuple(("multi_token_position_hits", str(position)) for position in range(1, 11))
STEPS = tuple(("feedback_assumptions_on_real_sequences", str(step)) for step in range(1, 6))
# The main run is evaluated on validation requests before its first iteration and every 250 iterations after it, and
# records the KL of its sampled lists from its first policy update on.
MAIN_RUN_ITERATIONS = tuple(("rl_training", str(iteration)) for iteration in range(0, 2001, 250))
TENTHS = tuple(("impressions_by_tenth", str(tenth)) for tenth in range(1, 11))

FIGURES = (
    Figure(
        "frontier_by_distance",
        horizontal=Axis(KL, "KL from the base recommender, nats per item", logarithmic=True),
        vertical=(Axis(NDCG, "NDCG@10"), Axis(SHARE, "predicted share, held-out model"), Axis(COVERAGE, "catalogue coverage")),
        series=(
            Series("value model, fixed anchor weight", anchor_sweep(ANCHOR_WEIGHTS), BLUE),
            Series("value model, KL budget", anchor_sweep(KL_BUDGETS, "kl_budget={}", "kl_budget_sweep"), BLUE, marker="square", joined=False),
            Series("leave-one-out, fixed anchor weight", anchor_sweep(("0.3", "0.1", "0.03"), "leave-one-out, kl={}", "advantage_estimators"), ORANGE),
            Series("leave-one-out, KL budget", anchor_sweep(KL_BUDGETS, "leave-one-out, kl_budget={}", "advantage_estimators"), ORANGE, marker="square", joined=False),
            Series(
                "own reward only (γ = 0)",
                anchor_sweep(KL_BUDGETS, "discount=0, kl_budget={}", "sequence_credit_ablation") + anchor_sweep(("0.3", "0.1"), "discount=0, kl={}", "sequence_credit_ablation"),
                GREEN,
                marker="triangle",
                joined=False,
            ),
            Series(
                "one item per episode",
                anchor_sweep(KL_BUDGETS, "horizon=1, kl_budget={}", "sequence_credit_ablation") + anchor_sweep(("0.3", "0.1"), "horizon=1, kl={}", "sequence_credit_ablation"),
                PURPLE,
                marker="diamond",
                joined=False,
            ),
            Series("learning rate ×10", (("advantage_estimators", "learning rate x10.0, kl=0.3"),), VERMILION, marker="cross", joined=False),
            BASE_RECOMMENDER,
        ),
    ),
    Figure(
        "methods",
        horizontal=Axis(SHARE, "predicted share, held-out model"),
        vertical=(Axis(NDCG, "NDCG@10"), Axis(LOGGED_SHARE, "logged share per recommended item"), Axis(COVERAGE, "catalogue coverage")),
        series=(
            Series("RL on the share, anchor sweep", anchor_sweep(ANCHOR_WEIGHTS[:-3]), BLUE),
            Series("re-ranking by the reward model", anchor_sweep(RERANKING_WEIGHTS, "re-ranking, weight={}", "alternatives_to_rl"), ORANGE, marker="square"),
            Series(
                "RL on share + 10 × logged relevance",
                anchor_sweep(("0.3", "0.1", "0.03", "0.0"), "ratio + 10.0 x logged relevance, kl={}", "relevance_reward_sweep"),
                GREEN,
                marker="triangle",
            ),
            Series("RL on logged relevance alone", (("relevance_reward_sweep", "logged relevance, kl=0.0"),), GREEN, marker="cross", joined=False),
            Series("off-policy REINFORCE", anchor_sweep(("0.5", "2.0"), "off-policy REINFORCE, cap={}", "alternatives_to_rl"), PURPLE, marker="diamond"),
            Series("liked-only fine-tuning", (("alternatives_to_rl", "liked-only fine-tuning"),), VERMILION, marker="diamond", joined=False),
            Series("best of 8 sampled lists", (("alternatives_to_rl", "best of 8 sampled lists"),), SKY, marker="circle", joined=False),
            BASE_RECOMMENDER,
            Series("base recommender, ranked lists", (("baselines", "GPT top-k"),), BLACK, marker="square", joined=False),
        ),
    ),
    Figure(
        "position_hits",
        horizontal=Axis(None, "position in the list"),
        vertical=(Axis("hit_rate", "all lists"), Axis("hit_rate_after_miss", "after a miss"), Axis("hit_rate_after_hit", "after a hit")),
        series=(
            Series("base recommender, ranked list", POSITIONS, BLACK, marker="square", prefix="pretrained, ranked/"),
            Series("base recommender, generated list", POSITIONS, BLUE, prefix="pretrained, generated/"),
            Series("RL main run, generated list", POSITIONS, ORANGE, marker="triangle", prefix="rl, generated/"),
        ),
        sessions=tuple(REVIEW_SESSIONS.values()),
    ),
    Figure(
        "position_hits_with_heads",
        horizontal=Axis(None, "position in the list"),
        vertical=(Axis("hit_rate", "all lists"), Axis("hit_rate_after_miss", "after a miss"), Axis("hit_rate_after_hit", "after a hit")),
        series=(
            Series("ranked list", MULTI_TOKEN_POSITIONS, BLACK, marker="square", prefix="ranked/"),
            Series("generated list", MULTI_TOKEN_POSITIONS, BLUE, prefix="generated/"),
            Series("one head per position", MULTI_TOKEN_POSITIONS, GREEN, marker="triangle", prefix="one head per position/"),
        ),
        sessions=tuple(FRONTIER_SESSIONS.values()),
    ),
    Figure(
        "frontier_methods",
        horizontal=Axis(SHARE, "predicted share, held-out model"),
        vertical=(Axis(NDCG, "NDCG@10"), Axis(SURPLUS, "logged surplus per item")),
        series=(
            Series(
                "RL, anchor weights and KL budgets",
                (*anchor_sweep(("1.0", "0.3", "0.1")), ("rl_vs_pretrained", "rl"), *anchor_sweep(("0.5",), "kl_budget={}", "kl_budget_sweep")),
                BLUE,
                ordered=True,
            ),
            Series("re-ranking (closed form)", anchor_sweep(RERANKING_WEIGHTS, "re-ranking, weight={}", "inference_time_alignment"), ORANGE, marker="square"),
            Series("guided decoding (closed form)", anchor_sweep(("1.0", "0.3", "0.1"), "played ratio, guided, discount=0.0, strength={}", "inference_time_alignment"), GREEN, marker="triangle"),
            Series("weights between and beyond RL", anchor_sweep(("0.25", "0.5", "0.75", "1.5", "2.0"), "weights x{}", "weight_interpolation"), SKY, marker="diamond"),
            Series("RL, one offset per item", anchor_sweep(("0.1", "0.5"), "item offsets, kl_budget={}", "capacity_sweep"), VERMILION, marker="square", joined=False),
            Series("reward-weighted, relative share", anchor_sweep(("1.0", "0.3"), "reward-weighted, relative_ratio, temperature={}", "logged_preference_methods"), PURPLE, marker="diamond", joined=False),
            Series("one head per position", (("multi_token_lists", "one head per position"),), GREEN, marker="star", joined=False),
            Series("base recommender, generated", (("multi_token_lists", "generated"),), BLACK, marker="star", joined=False),
            Series("base recommender, ranked", (("multi_token_lists", "ranked"),), BLACK, marker="square", joined=False),
        ),
        sessions=tuple(FRONTIER_SESSIONS.values()),
        legend_columns=2,
    ),
    Figure(
        "feedback_assumptions",
        horizontal=Axis(None, "step: real next item of the user"),
        vertical=(Axis("bias", "bias of the prediction"), Axis("auc", "AUC")),
        series=(
            Series("earlier responses as logged", STEPS, BLACK, marker="star", prefix="train/logged/"),
            Series("expected", STEPS, BLUE, prefix="train/expected/"),
            Series("sampled", STEPS, GREEN, marker="triangle", prefix="train/sampled/"),
            Series("assumed finished", STEPS, VERMILION, marker="square", prefix="train/assume_completed/"),
            Series("unknown (masked reward model)", STEPS, PURPLE, marker="diamond", prefix="masked train/unknown/"),
        ),
        sessions=tuple(REVIEW_SESSIONS.values()),
        legend_columns=2,
    ),
    Figure(
        "main_run_training",
        horizontal=Axis(None, "RL iteration of the main run"),
        vertical=(Axis("played_ratio", "predicted share"), Axis("all/ndcg@10", "NDCG@10"), Axis("kl_to_reference", "KL from the base, nats", from_zero=True)),
        series=(
            Series("training reward model, generated lists", MAIN_RUN_ITERATIONS, BLUE, prefix="validation/predicted/train/"),
            Series("held-out reward model, generated lists", MAIN_RUN_ITERATIONS, ORANGE, marker="square", prefix="validation/predicted/heldout/"),
            Series("generated lists", MAIN_RUN_ITERATIONS, GREEN, marker="triangle", prefix="validation/next_k/"),
            Series("ranked lists", MAIN_RUN_ITERATIONS, BLACK, marker="square", prefix="validation/top_k/"),
            Series("lists sampled in training", MAIN_RUN_ITERATIONS, VERMILION, marker="diamond"),
        ),
    ),
    Figure(
        "impressions_by_tenth",
        horizontal=Axis(None, "tenth of the catalogue, from the lowest usual share or the shortest item to the highest or longest"),
        vertical=(Axis("usual_share", "% of impressions by usual share"), Axis("length", "% of impressions by length")),
        series=(
            Series("base recommender, generated lists", TENTHS, BLUE, prefix="before/"),
            Series("RL main run, generated lists", TENTHS, ORANGE, marker="triangle", prefix="after/"),
        ),
        legend_columns=2,
    ),
)


@lru_cache(maxsize=None)
def impressions_by_tenth(session: str) -> dict[str, dict[str, float]] | None:
    """The share of the test users' lists, in percent, that went to each tenth of the catalogue before and after the main run.

    The catalogue is cut twice into tenths with as many items each: by each item's mean consumed share in the training
    period ("usual_share") and by its length ("length"), lowest first. Rows are the tenths "1" to "10"; columns are
    "<before|after>/<usual_share|length>". None if the session has no exposure report.
    """
    report = report_cells(session, "rl_exposure_change")
    if report is None:
        return None
    columns, rows = report
    items = [{column: float(value) for column, value in zip(columns, values)} for values in rows.values()]
    totals = {moment: sum(item[f"exposure_{moment}"] for item in items) for moment in ("before", "after")}
    shares: dict[str, dict[str, float]] = {str(tenth): {} for tenth in range(1, 11)}
    for ordering, attribute in (("usual_share", "mean_played_ratio"), ("length", "duration_seconds")):
        ordered = sorted(items, key=lambda item: item[attribute])
        for rank, item in enumerate(ordered):
            tenth = shares[str(rank * 10 // len(ordered) + 1)]
            for moment in ("before", "after"):
                key = f"{moment}/{ordering}"
                tenth[key] = tenth.get(key, 0.0) + 100 * item[f"exposure_{moment}"] / totals[moment]
    return shares


# Tables computed from a report rather than read from one, by the name a series gives as its report.
DERIVED_REPORTS = {"impressions_by_tenth": impressions_by_tenth}


def value(session: str, report: str, run: str, quantity: str) -> float | None:
    """The number a series draws for one run: read from the session's report, or from a table derived from one."""
    if report in DERIVED_REPORTS:
        table = DERIVED_REPORTS[report](session)
        return None if table is None else table.get(run, {}).get(quantity)
    return number(Source(session, report, run, quantity))


def dataset_name(session: str) -> str:
    return DATASET_NAMES.get(session, DATASET_NAMES.get(session.split("_")[0], session))


def points_of(series: Series, session: str, horizontal: Axis, vertical: Axis) -> list[tuple[float, float]]:
    points = []
    for report, run in series.runs:
        x, y = (float(run) if axis.quantity is None else value(session, report, run, series.prefix + axis.quantity) for axis in (horizontal, vertical))
        if x is not None and y is not None:
            points.append((max(x, SMALLEST_DISTANCE) if horizontal.logarithmic else x, y))
    return sorted(points) if series.ordered else points


def linear_ticks(low: float, high: float, wanted: int = 4) -> list[float]:
    """Round numbers that cover [low, high] in about `wanted` steps."""
    span = (high - low) or abs(high) or 1.0
    magnitude = 10 ** math.floor(math.log10(span / wanted))
    step = min((factor * magnitude for factor in (1, 2, 2.5, 5, 10)), key=lambda candidate: abs(span / candidate - wanted))
    first = math.floor(low / step) * step
    count = math.ceil((high - first) / step - 1e-9) + 1
    return [round(first + index * step, 10) for index in range(count)]


def tick_text(value: float, step: float) -> str:
    decimals = max(0, -math.floor(math.log10(step) + 1e-9))
    if abs(step * 10**decimals - round(step * 10**decimals)) > 1e-6:
        decimals += 1
    return f"{value:.{decimals}f}"


def marker(shape: str, x: float, y: float, color: str, size: float = 4.2) -> str:
    if shape == "circle":
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{size:.1f}" fill="{color}"/>'
    if shape == "cross":
        arm = size * 0.95
        strokes = f"M{x - arm:.1f},{y - arm:.1f} L{x + arm:.1f},{y + arm:.1f} M{x - arm:.1f},{y + arm:.1f} L{x + arm:.1f},{y - arm:.1f}"
        return f'<path d="{strokes}" stroke="{color}" stroke-width="2.4" fill="none"/>'
    corners = {
        "square": [(-1, -1), (1, -1), (1, 1), (-1, 1)],
        "diamond": [(0, -1.35), (1.35, 0), (0, 1.35), (-1.35, 0)],
        "triangle": [(0, -1.3), (1.25, 0.95), (-1.25, 0.95)],
        "star": [(1.6 * math.sin(k * math.pi / 5) * (1 if k % 2 == 0 else 0.45), -1.6 * math.cos(k * math.pi / 5) * (1 if k % 2 == 0 else 0.45)) for k in range(10)],
    }[shape]
    outline = " ".join(f"{x + dx * size * 0.92:.1f},{y + dy * size * 0.92:.1f}" for dx, dy in corners)
    return f'<polygon points="{outline}" fill="{color}"/>'


def text(x: float, y: float, content: str, size: int = 12, anchor: str = "middle", rotate: bool = False, weight: str = "normal") -> str:
    transform = f' transform="rotate(-90 {x:.1f} {y:.1f})"' if rotate else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{BLACK}"{transform}>{escape(content)}</text>'


def panel(figure: Figure, session: str, vertical: Axis, left: float, top: float) -> list[str]:
    """One set of axes with every series that has points for the session."""
    drawn = {series: points_of(series, session, figure.horizontal, vertical) for series in figure.series}
    xs, ys = [x for points in drawn.values() for x, _ in points], [y for points in drawn.values() for _, y in points]
    if not xs:
        return []
    plot_left, plot_top = left + MARGIN_LEFT, top + MARGIN_TOP
    plot_width, plot_height = PANEL_WIDTH - MARGIN_LEFT - MARGIN_RIGHT, PANEL_HEIGHT - MARGIN_TOP - MARGIN_BOTTOM

    if figure.horizontal.logarithmic:
        low, high = math.floor(math.log10(min(xs)) + 1e-9), math.ceil(math.log10(max(xs)) - 1e-9)
        x_ticks = [(10.0**exponent, f"{10.0**exponent:g}") for exponent in range(low, high + 1)]
        x_low, x_high = math.log10(SMALLEST_DISTANCE) - 0.1, high + 0.1

        def across(x: float) -> float:
            return plot_left + (math.log10(x) - x_low) / (x_high - x_low) * plot_width
    else:
        positions = [float(run) for series in figure.series for _, run in series.runs] if figure.horizontal.quantity is None else []
        if positions and max(positions) - min(positions) <= MOST_INTEGER_TICKS:
            ticks = [float(position) for position in range(int(min(positions)), int(max(positions)) + 1)]
        else:
            ticks = linear_ticks(min(positions or xs), max(positions or xs))
        step = ticks[1] - ticks[0]
        x_ticks = [(tick, tick_text(tick, step)) for tick in ticks]
        x_low, x_high = ticks[0], ticks[-1]

        def across(x: float) -> float:
            return plot_left + (x - x_low) / (x_high - x_low) * plot_width

    y_ticks = linear_ticks(min(ys + [0.0] if vertical.from_zero else ys), max(ys))
    y_step = y_ticks[1] - y_ticks[0]

    def up(y: float) -> float:
        return plot_top + plot_height - (y - y_ticks[0]) / (y_ticks[-1] - y_ticks[0]) * plot_height

    parts = []
    for tick, label in x_ticks:
        if x_low <= (math.log10(tick) if figure.horizontal.logarithmic else tick) <= x_high:
            parts.append(f'<line x1="{across(tick):.1f}" y1="{plot_top}" x2="{across(tick):.1f}" y2="{plot_top + plot_height}" stroke="{GRID}"/>')
            parts.append(text(across(tick), plot_top + plot_height + 14, label, size=11))
    for tick in y_ticks:
        parts.append(f'<line x1="{plot_left}" y1="{up(tick):.1f}" x2="{plot_left + plot_width}" y2="{up(tick):.1f}" stroke="{GRID}"/>')
        parts.append(text(plot_left - 5, up(tick) + 4, tick_text(tick, y_step), size=11, anchor="end"))
    parts.append(f'<path d="M{plot_left},{plot_top} V{plot_top + plot_height} H{plot_left + plot_width}" stroke="{AXIS}" fill="none"/>')
    parts.append(text(plot_left + plot_width / 2, top + 15, f"{dataset_name(session)}: {vertical.label}", size=12, weight="bold"))

    for series, points in drawn.items():
        if series.joined and len(points) > 1:
            path = " L".join(f"{across(x):.1f},{up(y):.1f}" for x, y in points)
            parts.append(f'<path d="M{path}" stroke="{series.color}" stroke-width="1.6" fill="none" opacity="0.85"/>')
        parts += [marker(series.marker, across(x), up(y), series.color) for x, y in points]
    return parts


def legend(figure: Figure, width: float) -> tuple[list[str], float]:
    rows = math.ceil(len(figure.series) / figure.legend_columns)
    left = (width - figure.legend_columns * LEGEND_COLUMN_WIDTH) / 2 + 24
    parts = []
    for index, series in enumerate(figure.series):
        x, y = left + (index // rows) * LEGEND_COLUMN_WIDTH, 13 + (index % rows) * LEGEND_ROW_HEIGHT
        if series.joined:
            parts.append(f'<line x1="{x - 11}" y1="{y}" x2="{x + 11}" y2="{y}" stroke="{series.color}" stroke-width="1.6"/>')
        parts += [marker(series.marker, x, y, series.color), text(x + 18, y + 4, series.label, anchor="start")]
    return parts, rows * LEGEND_ROW_HEIGHT + 10


def svg(figure: Figure) -> str:
    width = PANEL_WIDTH * len(figure.vertical) + RIGHT_PADDING
    parts, legend_height = legend(figure, width)
    top = legend_height
    for session in figure.sessions:
        row = [part for column, vertical in enumerate(figure.vertical) for part in panel(figure, session, vertical, column * PANEL_WIDTH, top)]
        if row:
            parts += row
            top += PANEL_HEIGHT - HORIZONTAL_LABEL_HEIGHT
    top += HORIZONTAL_LABEL_HEIGHT
    parts.append(text(width / 2, top - 6, figure.horizontal.label, size=12))
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{top:.0f}" viewBox="0 0 {width} {top:.0f}" font-family="{FONT}">',
        *parts,
        "</svg>",
    ]) + "\n"


def write_figures() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for figure in FIGURES:
        (OUTPUT / f"{figure.name}.svg").write_text(svg(figure), encoding="utf-8")
        print(f"wrote {figure.name}.svg")


if __name__ == "__main__":
    write_figures()
