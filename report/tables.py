"""The tables of the paper, typeset from the CSV reports of the runs, so that no number in a table is typed by hand.

    python3 report/tables.py

Reads results/<session>/<report>.csv and writes one Typst snippet per table into report/tables/, to be pulled
into the paper with `#include`. A cell of a table is one number of one report: its row and its column together
say which session, report, run and quantity. Rows whose run a session does not have are left out and named, so
a table fills up as sessions are added. Needs nothing beyond the standard library.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUTPUT = Path(__file__).resolve().parent / "tables"
# The session of each dataset that the paper reports. A source may also name a session's folder directly.
SESSIONS = {"kion": "kion_final", "yambda": "yambda_final"}
DATASET_NAMES = {"kion": "KION", "yambda": "Yambda"}
# Reports whose rows are named by several columns; the last of them names the run.
LABEL_COLUMNS = {"list_reward_methods": 2}
GENERATED_LISTS = "next_k/"  # reports of fine-tuning runs put this before the quantities of generated lists
MINUS = "−"
MISSING = "–"
WORDS = {"True": "yes", "False": "no"}
TYPST_SPECIAL_CHARACTERS = "\\@#$*_<>[]`"


@dataclass(frozen=True)
class Source:
    """Where a number is. A row, a column, a panel and the table each know a part of it."""

    session: str | None = None
    report: str | None = None
    run: str | None = None
    quantity: str | None = None

    def completed_by(self, other: Source) -> Source:
        return Source(*(getattr(self, part.name) or getattr(other, part.name) for part in fields(Source)))


@dataclass(frozen=True)
class Row:
    label: str
    source: Source


@dataclass(frozen=True)
class Column:
    header: str
    source: Source
    decimals: int = 3
    scale: float = 1.0
    unit: str = ""


@dataclass(frozen=True)
class Panel:
    """A block of rows under a title of its own; None for a table that is one block."""

    title: str | None
    rows: tuple[Row, ...]
    session: str | None = None


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]
    panels: tuple[Panel, ...]
    report: str | None = None
    column_groups: tuple[tuple[str, int], ...] = ()  # headers that span several columns: (title, number of columns)


@dataclass(frozen=True)
class Change:
    """A row of a before-and-after table: a quantity of the base and of the fine-tuned recommender.

    The quantities of the paired comparison come with an interval for their change; the others, read from the
    full report of the main run, do not.
    """

    label: str
    quantity: str
    decimals: int = 3
    scale: float = 1.0
    has_interval: bool = True


@dataclass(frozen=True)
class ChangeTable:
    name: str
    sessions: tuple[str, ...]
    panels: tuple[tuple[str, tuple[Change, ...]], ...]


def run(label: str, name: str, report: str | None = None) -> Row:
    return Row(label, Source(run=name, report=report))


def quantity(header: str, name: str, decimals: int = 3, session: str | None = None, report: str | None = None, **formatting) -> Column:
    return Column(header, Source(session=session, report=report, quantity=name), decimals, **formatting)


def dataset_panels(rows_by_session: dict[str, tuple[Row, ...]]) -> tuple[Panel, ...]:
    return tuple(Panel(DATASET_NAMES[session], rows, session) for session, rows in rows_by_session.items())


NDCG = quantity("NDCG@10", "all/ndcg@10", decimals=4)
ON_LIKED = quantity("on liked", "liked/ndcg@10", decimals=4)
ABOVE_USUAL = quantity("above usual", "above_usual/ndcg@10", decimals=4)
SHARE = quantity("Pred. share", "predicted/heldout/played_ratio")
SHARE_TRAINING = quantity("Training model", "predicted/train/played_ratio")
RELATIVE_SHARE = quantity("Rel. share", "predicted/heldout/relative_ratio")
PERSONAL_PART = quantity("Personal part", "predicted/heldout/personal_played_ratio")
LOGGED = quantity("Logged", "logged/played_ratio_per_item", decimals=4)
SURPLUS = quantity("Surplus", "logged/relative_surplus_per_item", decimals=4)
COVERAGE = quantity("Coverage", "profile/catalog_coverage")
OVERLAP = quantity("Overlap", "profile/mean_user_pair_overlap")
KL = quantity("KL", "training/kl_to_reference")
WEIGHT = quantity("β at end", "training/anchor_weight")

MAIN_RUN = run("β = 0.3, 2,000 iterations", "rl", report="rl_vs_pretrained")


def anchor(weight: str) -> Row:
    return run(f"β = {weight.removesuffix('.0')}", f"kl={float(weight)}")


def compared_estimators(setting: str, value_model: Row, leave_one_out_run: str) -> tuple[Row, Row]:
    return (
        Row(f"{setting}: value model", value_model.source),
        run(f"{setting}: leave-one-out", leave_one_out_run, report="advantage_estimators"),
    )


def at_weight(weight: str) -> tuple[Row, Row]:
    return compared_estimators(f"β = {weight}", run("", f"kl={weight}", report="kl_sweep"), f"leave-one-out, kl={weight}")


def at_budget(budget: str) -> tuple[Row, Row]:
    return compared_estimators(f"KL budget {budget}", run("", f"kl_budget={budget}", report="kl_budget_sweep"), f"leave-one-out, kl_budget={budget}")


def credit_panel(dataset: str, setting: str, constraint: str, whole_list_report: str) -> Panel:
    title = f"{DATASET_NAMES[dataset]}, {setting}"
    rows = (
        run("whole rest of the list", constraint, report=whole_list_report),
        run("own reward only (γ = 0)", f"discount=0, {constraint}"),
        run("one item per episode", f"horizon=1, {constraint}"),
    )
    return Panel(title, rows, dataset)


def reranking(label: str, method: str, weights: tuple[str, ...]) -> tuple[Row, ...]:
    return tuple(run(f"{label}, w = {weight.removesuffix('.0')}", f"{method}, weight={float(weight)}") for weight in weights)


LIST_REWARD = "ratio + 1.0 x distance"
STAGES = {
    "pretrain_policy": "Base recommender",
    "train_reward_models": "Two reward models",
    "compare_baselines": "Baselines",
    "finetune": "Main fine-tuning run",
    "sweep_kl_weight": "Anchor sweep",
    "sweep_kl_budget": "KL budgets",
    "compare_advantage_estimators": "Advantage estimators",
    "compare_simulated_feedback": "Sampled simulated feedback",
    "explain_reward_gain": "Split of the gain, re-ranking",
    "compare_alternatives_to_rl": "Alternatives to RL",
    "sweep_rewards": "Retention rewards",
    "sweep_relevance_in_reward": "Logged relevance in the reward",
    "sweep_personal_rewards": "Personal rewards",
    "sweep_relative_reward": "Relative-share reward",
    "compare_list_reward": "List reward",
    "ablate_sequence_credit": "Credit ablations",
}

TABLES = (
    Table(
        "baselines",
        report="baselines",
        column_groups=(("KION", 3), ("Yambda", 4)),
        columns=(
            quantity("NDCG@10", "all/ndcg@10", decimals=4, session="kion"),
            quantity("Recall@10", "all/recall@10", decimals=4, session="kion"),
            quantity("Coverage", "profile/catalog_coverage", session="kion"),
            quantity("NDCG@10", "all/ndcg@10", decimals=4, session="yambda"),
            quantity("on repeats", "repeat/ndcg@10", decimals=4, session="yambda"),
            quantity("on new", "new/ndcg@10", decimals=4, session="yambda"),
            quantity("Coverage", "profile/catalog_coverage", session="yambda"),
        ),
        panels=(
            Panel(None, (
                run("Most popular", "most popular"),
                run("Replay of the history", "replay history"),
                run("Markov chain", "Markov chain"),
                run("GRU4Rec", "GRU4Rec"),
                run("SASRec", "SASRec"),
                run("BERT4Rec", "BERT4Rec"),
                run("GPT-2, ranked", "GPT top-k"),
                run("GPT-2, generated", "GPT next-k"),
            )),
        ),
    ),
    Table(
        "anchor_sweep",
        report="kl_sweep",
        columns=(KL, NDCG, ON_LIKED, SHARE_TRAINING, SHARE, LOGGED, COVERAGE, OVERLAP),
        panels=dataset_panels({
            "kion": (run("Base recommender", "pretrained"), *(anchor(weight) for weight in ("1.0", "0.5", "0.3", "0.2", "0.1", "0.0")), MAIN_RUN),
            "yambda": (run("Base recommender", "pretrained"), *(anchor(weight) for weight in ("0.3", "0.1", "0.05", "0.03", "0.0")), MAIN_RUN),
        }),
    ),
    Table(
        "estimators",
        columns=(WEIGHT, KL, NDCG, SHARE, COVERAGE, quantity("Clipped", "training/clip_fraction", decimals=1, scale=100, unit="%")),
        panels=dataset_panels({
            "kion": (
                *at_weight("0.3"),
                *at_weight("0.1"),
                *at_budget("0.1"),
                *at_budget("0.5"),
                run("β = 0.3: value model, learning rate ×10", "learning rate x10.0, kl=0.3", report="advantage_estimators"),
            ),
            "yambda": (*at_weight("0.1"), *at_weight("0.03"), *at_budget("0.2"), *at_budget("0.5")),
        }),
    ),
    Table(
        "credit",
        report="sequence_credit_ablation",
        columns=(WEIGHT, KL, NDCG, ON_LIKED, SHARE, COVERAGE),
        panels=(
            credit_panel("kion", "KL budget 0.1", "kl_budget=0.1", "kl_budget_sweep"),
            credit_panel("kion", "KL budget 0.5", "kl_budget=0.5", "kl_budget_sweep"),
            credit_panel("kion", "fixed β = 0.3", "kl=0.3", "kl_sweep"),
            credit_panel("yambda", "KL budget 0.2", "kl_budget=0.2", "kl_budget_sweep"),
            credit_panel("yambda", "KL budget 0.5", "kl_budget=0.5", "kl_budget_sweep"),
        ),
    ),
    Table(
        "feedback",
        report="simulated_feedback_sweep",
        columns=(
            KL,
            NDCG,
            COVERAGE,
            quantity("expected", "expected", report="simulated_feedback_cross_judged"),
            quantity("sampled", "sampled", report="simulated_feedback_cross_judged"),
            quantity("finished", "assume_completed", report="simulated_feedback_cross_judged"),
        ),
        column_groups=(("", 3), ("Held-out share when earlier items are", 3)),
        panels=dataset_panels({
            "kion": (
                run("Expected feedback, β = 0.3", "kl=0.3"),
                run("Sampled feedback, β = 0.3", "sampled feedback, kl=0.3"),
                run("Expected feedback, β = 0.1", "kl=0.1"),
                run("Sampled feedback, β = 0.1", "sampled feedback, kl=0.1"),
            ),
            "yambda": (run("Expected feedback, β = 0.1", "kl=0.1"), run("Sampled feedback, β = 0.1", "sampled feedback, kl=0.1")),
        }),
    ),
    Table(
        "whose_gain",
        report="reward_gain_decomposition",
        columns=(
            NDCG,
            quantity("Own user", "predicted played ratio"),
            quantity("Another user", "predicted played ratio for other users"),
            quantity("Gain", "gain"),
            quantity("Gain, another", "gain for other users"),
            quantity("Personal", "personal share of the gain", decimals=0, scale=100, unit="%"),
        ),
        panels=dataset_panels({
            session: (
                run("Base recommender", "pretrained"),
                run("RL, β = 0.3 (main run)", "RL"),
                run("RL, β = 0.1", "kl=0.1"),
                *reranking("Re-ranking, reward model", "re-ranking", ("0.3",)),
                *reranking("Re-ranking, item mean", "re-ranking by item mean", ("0.3",)),
            )
            for session in SESSIONS
        }),
    ),
    Table(
        "alternatives",
        report="alternatives_to_rl",
        columns=(NDCG, ON_LIKED, ABOVE_USUAL, SHARE, LOGGED, SURPLUS, COVERAGE),
        panels=dataset_panels({
            session: (
                run("Base recommender", "pretrained"),
                run("Base recommender, ranked", "GPT top-k", report="baselines"),
                run("RL, β = 0.3 (main run)", "RL"),
                run("Liked-only fine-tuning", "liked-only fine-tuning"),
                run("Off-policy REINFORCE, κ = 0.5", "off-policy REINFORCE, cap=0.5"),
                *reranking("Re-ranking", "re-ranking", ("1.0", "0.3", "0.1")),
                run("Best of 8 sampled lists", "best of 8 sampled lists"),
            )
            for session in SESSIONS
        }),
    ),
    Table(
        "list_reward",
        report="list_reward_methods",
        columns=(
            NDCG,
            SHARE,
            quantity("Distance", "profile/intra_list_distance"),
            quantity("List reward", "list reward (held-out)"),
            COVERAGE,
        ),
        panels=dataset_panels({
            session: (
                run("Base recommender", "pretrained"),
                *(run(f"RL on the list reward, β = {weight}", f"{LIST_REWARD}, kl={weight}") for weight in ("0.3", "0.1")),
                *reranking("Greedy re-ranking", f"greedy re-ranking, {LIST_REWARD}", ("1.0", "0.3", "0.1")),
                run("Best of 8 sampled lists", f"best of 8 sampled lists, {LIST_REWARD}"),
                *(run(f"RL on the share alone, β = {weight}", f"kl={weight}") for weight in ("0.3", "0.1")),
            )
            for session in SESSIONS
        }),
    ),
    Table(
        "rewards",
        columns=(NDCG, ON_LIKED, SHARE, RELATIVE_SHARE, LOGGED, SURPLUS, COVERAGE),
        panels=dataset_panels({
            "kion": (
                run("Base recommender", "pretrained", report="kl_sweep"),
                run("Share (the main reward), β = 0.3", "kl=0.3", report="kl_sweep"),
                run("Base recommender, ranked", "GPT top-k", report="baselines"),
                run("Relative share, β = 0.3", "relative ratio, kl=0.3", report="relative_reward_sweep"),
                run("Relative share, β = 0.1", "relative ratio, kl=0.1", report="relative_reward_sweep"),
                run("Re-ranking by relative share, w = 0.1", "re-ranking by relative ratio, weight=0.1", report="relative_reward_reranking"),
                run("Share + 10 × logged relevance, β = 0.3", "ratio + 10.0 x logged relevance, kl=0.3", report="relevance_reward_sweep"),
                run("Share + 10 × logged relevance, β = 0.1", "ratio + 10.0 x logged relevance, kl=0.1", report="relevance_reward_sweep"),
                run("Logged relevance alone, β = 0", "logged relevance, kl=0.0", report="relevance_reward_sweep"),
            ),
            "yambda": (
                run("Base recommender", "pretrained", report="kl_sweep"),
                run("Share (the main reward), β = 0.1", "kl=0.1", report="kl_sweep"),
                run("Base recommender, ranked", "GPT top-k", report="baselines"),
                run("Relative share, β = 0.1", "relative ratio, kl=0.1", report="relative_reward_sweep"),
                run("Re-ranking by relative share, w = 0.1", "re-ranking by relative ratio, weight=0.1", report="relative_reward_reranking"),
                run("Share + 10 × logged relevance, β = 0.1", "ratio + 10.0 x logged relevance, kl=0.1", report="relevance_reward_sweep"),
                run("Share + 10 × logged relevance, β = 0", "ratio + 10.0 x logged relevance, kl=0.0", report="relevance_reward_sweep"),
                run("Logged relevance alone, β = 0", "logged relevance, kl=0.0", report="relevance_reward_sweep"),
            ),
        }),
    ),
    Table(
        "repeats",
        report="alternatives_to_rl",
        columns=(
            NDCG,
            quantity("on repeats", "repeat/ndcg@10", decimals=4),
            quantity("on new", "new/ndcg@10", decimals=4),
            quantity("From history", "profile/history_repeat_rate", decimals=0, scale=100, unit="%"),
            LOGGED,
            COVERAGE,
        ),
        panels=(
            Panel(None, session="yambda", rows=(
                run("Replay of the history", "replay history", report="baselines"),
                run("Base recommender", "pretrained"),
                run("RL on the share, β = 0.3 (main run)", "RL"),
                run("Liked-only fine-tuning", "liked-only fine-tuning"),
                run("Off-policy REINFORCE, κ = 0.5", "off-policy REINFORCE, cap=0.5"),
                run("Re-ranking, w = 0.3", "re-ranking, weight=0.3"),
                run("RL on logged relevance alone, β = 0", "logged relevance, kl=0.0", report="relevance_reward_sweep"),
                run("RL on share + 10 × logged relevance, β = 0.1", "ratio + 10.0 x logged relevance, kl=0.1", report="relevance_reward_sweep"),
            )),
        ),
    ),
    Table(
        "retention_rewards",
        report="reward_sweep",
        columns=(
            NDCG,
            SHARE,
            quantity("Continues", "predicted/heldout/session_continues"),
            quantity("Returns", "predicted/heldout/returns"),
            COVERAGE,
            OVERLAP,
        ),
        panels=dataset_panels({
            session: (
                run("Base recommender", "pretrained"),
                *(run(f"{label}, β = {weight}", f"{reward}, kl={weight}") for reward, label in (("session", "Session"), ("return", "Return"), ("ratio + return", "Share + return")) for weight in ("0.3", "0.1")),
            )
            for session in SESSIONS
        }),
    ),
    Table(
        "relevance_rewards",
        report="relevance_reward_sweep",
        columns=(KL, NDCG, ON_LIKED, SHARE, LOGGED, COVERAGE, OVERLAP),
        panels=dataset_panels({
            session: (
                run("Base recommender", "pretrained"),
                run("Logged relevance alone, β = 0", "logged relevance, kl=0.0"),
                *(
                    run(f"Share + {weight} × logged relevance, β = {anchor_weight}", f"ratio + {float(weight)} x logged relevance, kl={float(anchor_weight)}")
                    for weight in ("10", "3")
                    for anchor_weight in ("0.3", "0.1", "0.03", "0")
                ),
            )
            for session in SESSIONS
        }),
    ),
    Table(
        "personal_rewards",
        report="personal_reward_sweep",
        columns=(KL, NDCG, SHARE, PERSONAL_PART, COVERAGE, OVERLAP),
        panels=dataset_panels({
            session: (
                run("Base recommender", "pretrained"),
                *(run(f"{label}, β = {weight}", f"{reward}, kl={float(weight)}") for reward, label in (("personal ratio", "Personal share"), ("ratio above item mean", "Share above the item mean")) for weight in ("0.3", "0.1", "0")),
            )
            for session in SESSIONS
        }),
    ),
    Table(
        "selection",
        report="kl_sweep_selection",
        columns=(
            quantity("NDCG@10, validation", "validation/next_k/all/ndcg@10", decimals=4),
            quantity("Pred. share, validation", "validation/predicted/heldout/played_ratio"),
            quantity("Within tolerance", "keeps_relevance"),
            quantity("Selected", "selected"),
        ),
        panels=dataset_panels({
            session: (run("Base recommender", "pretrained"), *(anchor(weight) for weight in ("1.0", "0.5", "0.3", "0.2", "0.1", "0.05", "0.03", "0.0")))
            for session in SESSIONS
        }),
    ),
    Table(
        "stage_minutes",
        report="stage_minutes",
        columns=(
            quantity("KION", "minutes", decimals=0, session="kion"),
            quantity("Yambda, first session", "minutes", decimals=0, session="yambda_diagnostics"),
            quantity("Yambda, continuation", "minutes", decimals=0, session="yambda_final"),
        ),
        panels=(Panel(None, tuple(run(label, stage) for stage, label in STAGES.items())),),
    ),
)

CHANGE_TABLES = (
    ChangeTable(
        "main_run",
        sessions=("kion", "yambda"),
        panels=(
            ("Relevance against the user's logged future", (
                Change("NDCG@10, all next items", "all/ndcg@10", decimals=4),
                Change("NDCG@10, liked items", "liked/ndcg@10", decimals=4),
                Change("NDCG@10, items consumed above usual", "above_usual/ndcg@10", decimals=4),
                Change("Recall@10, skipped items", "skipped/recall@10", decimals=4),
            )),
            ("Predicted consumed share of the list", (
                Change("Training reward model", "predicted/train/played_ratio"),
                Change("Held-out reward model", "predicted/heldout/played_ratio"),
                Change("Held-out, list shown to another user", "predicted/heldout/played_ratio_for_other_user"),
                Change("Held-out, relative to the item's usual", "predicted/heldout/relative_ratio"),
            )),
            ("Logged response to the recommended items the user went on to consume", (
                Change("Such items per list", "logged/hits_per_list", has_interval=False),
                Change("Their logged share", "logged/played_ratio_of_hits", has_interval=False),
                Change("Logged share per recommended item", "logged/played_ratio_per_item", decimals=4),
                Change("Logged share above the item's usual, per item", "logged/relative_surplus_per_item", decimals=4),
            )),
            ("Lists", (
                Change("Catalogue coverage", "profile/catalog_coverage", has_interval=False),
                Change("Overlap between two users' lists", "profile/mean_user_pair_overlap", decimals=4, has_interval=False),
                Change("Mean duration of an item, minutes", "profile/mean_duration_seconds", decimals=1, scale=1 / 60, has_interval=False),
            )),
        ),
    ),
)


@lru_cache(maxsize=None)
def report_cells(session: str, report: str) -> tuple[list[str], dict[str, list[str]]] | None:
    """A report of a session as its column names and its rows by label, every cell as text. None if the session has no such report."""
    path = RESULTS / SESSIONS.get(session, session) / f"{report}.csv"
    if not path.exists():
        return None
    num_label_columns = LABEL_COLUMNS.get(report, 1)
    with path.open(newline="", encoding="utf-8") as file:
        header, *rows = list(csv.reader(file))
    return header[num_label_columns:], {row[num_label_columns - 1]: row[num_label_columns:] for row in rows}


def cell(source: Source) -> str | None:
    """The cell of a report that `source` names, whichever of the run and the quantity the report has in its rows. None if it is not there."""
    report = report_cells(source.session, source.report)
    if report is None:
        return None
    columns, rows = report
    for quantity_name in (source.quantity, GENERATED_LISTS + source.quantity):
        if source.run in columns and quantity_name in rows:
            return rows[quantity_name][columns.index(source.run)]
        if source.run in rows and quantity_name in columns:
            return rows[source.run][columns.index(quantity_name)]
    return None


def number(source: Source) -> float | None:
    text = cell(source)
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return None if value != value else value


def formatted(value: float, decimals: int, scale: float = 1.0, unit: str = "", signed: bool = False) -> str:
    text = f"{value * scale:{'+' if signed else ''}.{decimals}f}".replace("-", MINUS)
    return text + unit


def shown(source: Source, column: Column) -> str | None:
    """A cell as it is printed: a number rounded, a word as it is, None if the report does not have it."""
    text = cell(source)
    if text is None:
        return None
    value = number(source)
    if value is not None:
        return formatted(value, column.decimals, column.scale, column.unit)
    return WORDS.get(text, MISSING if text.lower() in ("", "nan") else text)


def content(text: str) -> str:
    """`text` as a Typst content block in which nothing is read as markup."""
    return "[" + "".join(f"\\{character}" if character in TYPST_SPECIAL_CHARACTERS else character for character in text) + "]"


def typst_table(headers: list[str], column_groups: tuple[tuple[str, int], ...], body: list[list[str] | str]) -> str:
    """A Typst table as wide as the text, ruled above, below and under its header. A row of `body` that is one string is the title of a panel.

    The column of row labels takes the width the numbers leave, so that a long label does not squeeze them.
    """
    num_columns = len(headers) + 1
    header_lines = []
    if column_groups:
        group_cells, underlines, start = ["[]"], [], 1
        for title, width in column_groups:
            group_cells.append(f"table.cell(colspan: {width}, align: center, strong{content(title)})")
            if title:
                underlines.append(f"table.hline(start: {start}, end: {start + width}, stroke: 0.4pt)")
            start += width
        header_lines = [f"    {', '.join(group_cells)},", *(f"    {line}," for line in underlines)]
    header_lines.append(f"    {', '.join(['[]'] + ['strong' + content(header) for header in headers])},")

    body_lines = []
    for index, row in enumerate(body):
        if isinstance(row, str):
            if index > 0:
                body_lines.append("  table.hline(stroke: 0.4pt),")
            body_lines.append(f"  table.cell(colspan: {num_columns}, align: left, emph{content(row)}),")
        else:
            body_lines.append(f"  {', '.join(content(text) for text in row)},")
    lines = [
        "#table(",
        f"  columns: ({', '.join(['1fr'] + ['auto'] * len(headers))}),",
        f"  align: ({', '.join(['left'] + ['right'] * len(headers))}),",
        "  stroke: none,",
        "  inset: (x: 4pt, y: 2.8pt),",
        "  table.hline(stroke: 0.8pt),",
        "  table.header(",
        *header_lines,
        "  ),",
        "  table.hline(stroke: 0.4pt),",
        *body_lines,
        "  table.hline(stroke: 0.8pt),",
        ")",
    ]
    return "\n".join(lines) + "\n"


def rows_of(table: Table) -> list[list[str] | str]:
    """The body of a table. A row none of whose cells exist is left out and named, and so is a panel left without rows."""
    body: list[list[str] | str] = []
    for panel in table.panels:
        defaults = Source(session=panel.session, report=table.report)
        panel_rows, absent = [], []
        for row in panel.rows:
            cells = [shown(column.source.completed_by(row.source).completed_by(defaults), column) for column in table.columns]
            if any(text is not None for text in cells):
                panel_rows.append([row.label, *(MISSING if text is None else text for text in cells)])
            else:
                absent.append(row.label)
        if absent:
            print(f"  {table.name}{', ' + panel.title if panel.title else ''}: the reports have no {'; '.join(absent)}")
        if panel_rows:
            body += ([panel.title] if panel.title else []) + panel_rows
    return body


def rows_of_changes(table: ChangeTable) -> list[list[str] | str]:
    body: list[list[str] | str] = []
    for title, changes in table.panels:
        body.append(title)
        for change in changes:
            body.append([change.label, *(text for session in table.sessions for text in _before_after(change, session))])
    return body


def _before_after(change: Change, session: str) -> list[str]:
    """Before, after, and the relative change in percent with its interval where the paired comparison gives one."""
    if change.has_interval:
        report, before_run, after_run = "rl_vs_pretrained_significance", "before", "after"
    else:
        report, before_run, after_run = "rl_vs_pretrained", "pretrained", "rl"

    def value(run_name: str) -> float | None:
        return number(Source(session, report, run_name, change.quantity))

    before, after = value(before_run), value(after_run)
    if before is None or after is None:
        return [MISSING] * 3
    relative_change = formatted(after / before - 1, decimals=1, scale=100, signed=True)
    if change.has_interval:
        low, high = (formatted(value(bound) / before, decimals=1, scale=100, signed=True) for bound in ("change_ci_low", "change_ci_high"))
        relative_change += f" [{low}, {high}]"
    return [formatted(before, change.decimals, change.scale), formatted(after, change.decimals, change.scale), relative_change]


def write_tables() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for table in TABLES:
        body = rows_of(table)
        (OUTPUT / f"{table.name}.typ").write_text(typst_table([column.header for column in table.columns], table.column_groups, body), encoding="utf-8")
    for table in CHANGE_TABLES:
        headers = ["before", "after", "change, %"] * len(table.sessions)
        groups = tuple((DATASET_NAMES[session], 3) for session in table.sessions)
        (OUTPUT / f"{table.name}.typ").write_text(typst_table(headers, groups, rows_of_changes(table)), encoding="utf-8")
    print(f"wrote {len(TABLES) + len(CHANGE_TABLES)} tables to {OUTPUT}")


if __name__ == "__main__":
    write_tables()
