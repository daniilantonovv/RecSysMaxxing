"""The tables of the paper, typeset from the CSV reports of the runs, so that no number in a table is typed by hand.

    python3 report/tables.py

Reads results/<session>/<report>.csv and writes one Typst snippet per table into report/tables/, to be pulled
into the paper with `#include`. A cell of a table is one number of one report: its row and its column together
say which session, report, run and quantity. Rows whose run a session does not have are left out and named, so
a table fills up as sessions are added. Needs nothing beyond the standard library.
"""

from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass, fields
from functools import lru_cache, partial
from pathlib import Path
from typing import Callable

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUTPUT = Path(__file__).resolve().parent / "tables"
# The session of each dataset that the paper reports. A source may also name a session's folder directly.
SESSIONS = {"kion": "kion_final", "yambda": "yambda_final"}
DATASET_NAMES = {"kion": "KION", "yambda": "Yambda"}
# Reports whose rows are named by several columns; the last of them names the run.
LABEL_COLUMNS = {"list_reward_methods": 2, "reward_model_reactions": 2}
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
    relative_to: Source | None = None  # where set, the cell is the change in percent against the number this names instead
    only_in: tuple[str, ...] = ()  # where set, the sessions whose panels show this column; the others leave it empty


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

REVIEW_SESSIONS = {"kion": "kion_review", "yambda": "yambda_review"}


def review_panels(rows_by_dataset: dict[str, tuple[Row, ...]]) -> tuple[Panel, ...]:
    return tuple(Panel(DATASET_NAMES[dataset], rows, REVIEW_SESSIONS[dataset]) for dataset, rows in rows_by_dataset.items())


def positions(*numbers: int) -> tuple[Row, ...]:
    return tuple(run(f"position {number}", str(number)) for number in numbers)


def steps(*numbers: int) -> tuple[Row, ...]:
    return tuple(run(f"step {number}", str(number)) for number in numbers)


def history_quartiles(*groups: str) -> tuple[Row, ...]:
    """Rows for the reported groups "<lower> to <upper> events": every group but the last excludes its upper edge."""
    rows = []
    for index, group in enumerate(groups):
        lower, upper = (int(edge) for edge in group.removesuffix(" events").split(" to "))
        rows.append(run(f"{lower}–{upper if index == len(groups) - 1 else upper - 1} events", group))
    return tuple(rows)


def in_percent(header: str, name: str, decimals: int = 1, **sources) -> Column:
    return quantity(header, name, decimals=decimals, scale=100, **sources)


def guided(reward: str, strengths: tuple[str, ...] = ("0.3", "0.1")) -> tuple[Row, ...]:
    """The base recommender and its lists guided by `reward` at each strength, judging each item by its own reward and with look-ahead."""
    report = "inference_time_alignment"
    rows = [run("Base recommender", "pretrained", report=report)]
    for discount, description in (("0.0", "own reward"), ("0.9", "look-ahead")):
        rows += [run(f"Guided, {description}, β = {strength}", f"{reward}, guided, discount={discount}, strength={strength}", report=report) for strength in strengths]
    return tuple(rows)


LONG_TERM_COLUMNS = (
    KL,
    NDCG,
    Column("Δ, %", Source(quantity="all/ndcg@10"), decimals=1, relative_to=Source(run="pretrained")),
    quantity("Return", "predicted/heldout/returns_after_list"),
    quantity("P(like)", "predicted/heldout/liked", decimals=4, only_in=("yambda_review",)),
    quantity("P(adopt)", "predicted/heldout/adopts", only_in=("yambda_review", "yambda_frontier")),
    LOGGED,
    quantity("Cover.", "profile/catalog_coverage"),
    quantity("Hours", "profile/mean_duration_seconds", decimals=1, scale=1 / 3600, only_in=("kion_review", "kion_frontier")),
)
LONG_TERM_PANELS = review_panels({
    "kion": (
        run("Base recommender", "pretrained"),
        run("Return after list, β = 0.3", "return after the list, kl=0.3"),
        run("Return after list, β = 0.1", "return after the list, kl=0.1"),
        run("same, discount 0, β = 0.3", "return after the list, discount=0, kl=0.3"),
    ),
    "yambda": (
        run("Base recommender", "pretrained"),
        run("Like − dislike, β = 0.3", "likes, kl=0.3"),
        run("Like − dislike, β = 0.1", "likes, kl=0.1"),
        run("Adoption, β = 0.3", "adoption, kl=0.3"),
        run("Adoption, β = 0.1", "adoption, kl=0.1"),
        run("Return after list, β = 0.3", "return after the list, kl=0.3"),
    ),
})

REVIEW_TABLES = (
    Table(
        "concentration",
        report="exposure_concentration",
        columns=(quantity("Coverage", "catalog_coverage"), quantity("Gini of impressions", "item_gini"), in_percent("Top 1% of items, % of impressions", "top_exposure_share", decimals=0)),
        panels=review_panels({
            "kion": (
                run("Base recommender, ranked", "pretrained, ranked"),
                run("Base recommender, generated", "pretrained, generated"),
                run("RL, β = 0.3 (main run)", "RL, generated"),
                anchor("0.1"),
                anchor("0.0"),
            ),
            "yambda": (
                run("Base recommender, ranked", "pretrained, ranked"),
                run("Base recommender, generated", "pretrained, generated"),
                run("RL, β = 0.3 (main run)", "RL, generated"),
                anchor("0.05"),
            ),
        }),
    ),
    Table(
        "content_types",
        report="impressions_by_content_type",
        columns=(
            in_percent("Ranked", "pretrained, ranked", decimals=0),
            in_percent("Generated", "pretrained, generated", decimals=0),
            in_percent("RL, β = 0.3", "RL, generated", decimals=0),
            in_percent("β = 1", "kl=1.0", decimals=0),
            in_percent("β = 0.1", "kl=0.1", decimals=0),
            in_percent("β = 0", "kl=0.0", decimals=0),
        ),
        column_groups=(("Base recommender", 2), ("Fine-tuned", 4)),
        panels=(Panel(None, (run("Films, % of impressions", "film"), run("Series, % of impressions", "series")), "kion_review"),),
    ),
    Table(
        "history_length",
        report="relevance_by_history_length",
        columns=(
            quantity("Users", "num_users", decimals=0),
            quantity("Base, ranked", "pretrained, ranked/ndcg@10", decimals=4),
            quantity("Base, generated", "pretrained, generated/ndcg@10", decimals=4),
            quantity("RL", "RL, generated/ndcg@10", decimals=4),
            Column("RL change, %", Source(quantity="RL, generated/ndcg@10"), decimals=1, relative_to=Source(quantity="pretrained, generated/ndcg@10")),
            Column("Logged change, %", Source(quantity="RL, generated/logged_played_ratio_per_item"), decimals=1, relative_to=Source(quantity="pretrained, generated/logged_played_ratio_per_item")),
        ),
        column_groups=(("", 1), ("NDCG@10", 4), ("", 1)),
        panels=(
            Panel("KION", history_quartiles("5 to 9 events", "9 to 14 events", "14 to 25 events", "25 to 1143 events"), "kion_review"),
            Panel("Yambda", history_quartiles("50 to 1077 events", "1077 to 2961 events", "2961 to 6421 events", "6421 to 25401 events"), "yambda_review"),
        ),
    ),
    Table(
        "logit_shift",
        report="logit_shift_by_item",
        columns=(
            in_percent("One shift per item, % of variance", "r2_per_item", decimals=0),
            in_percent("unweighted", "r2_per_item_unweighted", decimals=0),
            quantity("Usual share", "correlation_with_mean_played_ratio", decimals=2),
            quantity("Log popularity", "correlation_with_log_popularity", decimals=2),
            quantity("Duration", "correlation_with_duration_seconds", decimals=2),
        ),
        column_groups=(("", 2), ("Correlation of the item's shift with", 3)),
        panels=review_panels({
            "kion": (run("RL, β = 0.3 (main run)", "RL"), anchor("1.0"), anchor("0.3"), anchor("0.1"), anchor("0.0")),
            "yambda": (run("RL, β = 0.3 (main run)", "RL"), anchor("0.3"), anchor("0.1"), anchor("0.05")),
        }),
    ),
    Table("long_term_rewards", report="reward_sweep", columns=LONG_TERM_COLUMNS, panels=LONG_TERM_PANELS),
    Table(
        "long_term_rewards_and_guidance",
        report="reward_sweep",
        columns=LONG_TERM_COLUMNS,
        panels=(
            *LONG_TERM_PANELS,
            Panel("KION, guided decoding by the return after the list (frontier session)", guided("return after the list"), "kion_frontier"),
            Panel("Yambda, guided decoding by adoption (frontier session)", guided("adoption"), "yambda_frontier"),
        ),
    ),
    Table(
        "position_hits",
        report="position_hits",
        columns=(
            quantity("All lists", "pretrained, ranked/hit_rate", decimals=4),
            quantity("After a miss", "pretrained, ranked/hit_rate_after_miss", decimals=4),
            quantity("After a hit", "pretrained, ranked/hit_rate_after_hit", decimals=3),
            quantity("All lists", "pretrained, generated/hit_rate", decimals=4),
            quantity("After a miss", "pretrained, generated/hit_rate_after_miss", decimals=4),
            quantity("After a hit", "pretrained, generated/hit_rate_after_hit", decimals=3),
        ),
        column_groups=(("Ranked list", 3), ("Generated list", 3)),
        panels=review_panels({dataset: positions(1, 2, 3, 5, 10) for dataset in REVIEW_SESSIONS}),
    ),
    Table(
        "feedback_assumptions",
        report="feedback_assumptions_on_real_sequences",
        columns=(
            quantity("Logged share", "train/logged/mean_played_ratio"),
            quantity("AUC", "train/logged/auc"),
            quantity("Expected", "train/expected/bias"),
            quantity("Sampled", "train/sampled/bias"),
            quantity("Finished", "train/assume_completed/bias"),
            quantity("Unknown", "masked train/unknown/bias"),
            quantity("AUC", "train/expected/auc"),
            quantity("AUC", "masked train/unknown/auc"),
        ),
        column_groups=(("Real responses", 2), ("Bias of the prediction when earlier responses are", 4), ("Expected", 1), ("Unknown", 1)),
        panels=review_panels({dataset: steps(1, 2, 3, 4, 5) for dataset in REVIEW_SESSIONS}),
    ),
    Table(
        "masked_feedback",
        columns=(KL, NDCG, ON_LIKED, SHARE, LOGGED, COVERAGE),
        panels=review_panels({
            dataset: (
                run("Base recommender", "pretrained", report="kl_sweep"),
                run("Expected feedback, β = 0.3", "kl=0.3", report="kl_sweep"),
                run("Unknown feedback, β = 0.3", "unknown feedback, kl=0.3", report="masked_feedback_sweep"),
                run("Expected feedback, β = 0.1", "kl=0.1", report="kl_sweep"),
                run("Unknown feedback, β = 0.1", "unknown feedback, kl=0.1", report="masked_feedback_sweep"),
            )
            for dataset in REVIEW_SESSIONS
        }),
    ),
    Table(
        "reaction_heads",
        report="reward_model_reactions",
        columns=(quantity("AUC", "auc"), in_percent("Base rate, %", "base_rate", decimals=2), quantity("Events", "num_events", decimals=0)),
        panels=(Panel("Yambda, held-out reward model", (run("Like", "heldout/liked"), run("Dislike", "heldout/disliked")), "yambda_review"),),
    ),
    Table(
        "signal_spread",
        report="reward_signal_spread",
        columns=(quantity("Between candidates", "spread_between_candidates"), quantity("Between users", "spread_between_users"), quantity("Relevance AUC", "relevance_auc")),
        panels=(
            Panel("KION", (run("Consumed share", "played_ratio"), run("Return", "returns")), "kion_review"),
            Panel("Yambda", (run("Consumed share", "played_ratio"), run("Return", "returns"), run("Like", "liked"), run("Adoption", "adopts")), "yambda_review"),
        ),
    ),
)
TABLES = (*TABLES, *REVIEW_TABLES)

CHANGE_TABLES = (
    ChangeTable(
        "ranking_metrics",
        sessions=("kion", "yambda"),
        panels=tuple(
            (metric, tuple(Change(f"{metric}@{k}", f"all/{name}@{k}", decimals=4, has_interval=False) for k in (1, 5, 10)))
            for metric, name in (("NDCG", "ndcg"), ("Recall", "recall"), ("MAP", "map"))
        ),
    ),
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
    rows_by_label = {row[num_label_columns - 1]: row[num_label_columns:] for row in rows}
    rows_by_label |= {"/".join(row[:num_label_columns]): row[num_label_columns:] for row in rows}
    return header[num_label_columns:], rows_by_label


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


def displayed(source: Source, column: Column) -> str | None:
    """A cell of a table: the number `source` names, or its change in percent against the number `column.relative_to` names."""
    if column.relative_to is None:
        return shown(source, column)
    value, reference = number(source), number(column.relative_to.completed_by(source))
    if value is None or not reference:
        return None
    return formatted(value / reference - 1, column.decimals, scale=100, signed=True)


def rows_of(table: Table) -> list[list[str] | str]:
    """The body of a table. A row none of whose cells exist is left out and named, and so is a panel left without rows."""
    body: list[list[str] | str] = []
    for panel in table.panels:
        defaults = Source(session=panel.session, report=table.report)
        panel_rows, absent = [], []
        for row in panel.rows:
            cells = [
                None if column.only_in and panel.session not in column.only_in else displayed(column.source.completed_by(row.source).completed_by(defaults), column)
                for column in table.columns
            ]
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


@dataclass(frozen=True)
class SeedRow:
    """A number of the study over its three seeds, as (report, row, column) of each dataset's over_three_seeds.csv."""

    label: str
    kion: tuple[str, str, str] | None
    yambda: tuple[str, str, str] | None
    decimals: int = 3
    scale: float = 1.0
    signed: bool = False


def both(report: str, row: str, column: str) -> dict[str, tuple[str, str, str]]:
    return {"kion": (report, row, column), "yambda": (report, row, column)}


def relative_change_row(label: str, quantity_name: str) -> SeedRow:
    return SeedRow(label, **both("rl_vs_pretrained", quantity_name, "relative_change"), decimals=1, scale=100, signed=True)


SEED_PANELS = (
    ("Main run (β = 0.3) against the base recommender", (
        relative_change_row("NDCG@10, change in %", "next_k/all/ndcg@10"),
        SeedRow("Predicted share, held-out model, change in %", **both("rl_vs_pretrained", "predicted/heldout/played_ratio", "relative_change"), decimals=2, scale=100, signed=True),
        relative_change_row("Logged share per recommended item, change in %", "next_k/logged/played_ratio_per_item"),
        SeedRow("Logged surplus per recommended item, change", **both("rl_vs_pretrained_significance", "logged/relative_surplus_per_item", "change"), decimals=4, signed=True),
        SeedRow("Personal share of the predicted gain, %", **both("reward_gain_decomposition", "RL", "personal share of the gain"), decimals=1, scale=100),
        SeedRow("KL from the base recommender", **both("rl_vs_pretrained", "training/kl_to_reference", "rl")),
    )),
    ("What the anchor weight means", (
        SeedRow("KL at β = 0.3, value model", ("kl_sweep", "training/kl_to_reference", "kl=0.3"), None),
        SeedRow("KL at β = 0.3, leave-one-out baseline", ("advantage_estimators", "training/kl_to_reference", "leave-one-out, kl=0.3"), None),
        SeedRow("Weight holding the KL budget, value model", ("kl_budget_sweep", "training/anchor_weight", "kl_budget=0.1"), ("kl_budget_sweep", "training/anchor_weight", "kl_budget=0.5")),
        SeedRow("Weight holding the KL budget, leave-one-out", ("advantage_estimators", "training/anchor_weight", "leave-one-out, kl_budget=0.1"), None),
        SeedRow("NDCG@10 at the budget, whole rest of the list", ("kl_budget_sweep", "next_k/all/ndcg@10", "kl_budget=0.1"), ("kl_budget_sweep", "next_k/all/ndcg@10", "kl_budget=0.5"), decimals=4),
        SeedRow("NDCG@10 at the budget, own reward only", ("sequence_credit_ablation", "next_k/all/ndcg@10", "discount=0, kl_budget=0.1"), ("sequence_credit_ablation", "next_k/all/ndcg@10", "discount=0, kl_budget=0.5"), decimals=4),
        SeedRow("NDCG@10 at the budget, one item per episode", ("sequence_credit_ablation", "next_k/all/ndcg@10", "horizon=1, kl_budget=0.1"), ("sequence_credit_ablation", "next_k/all/ndcg@10", "horizon=1, kl_budget=0.5"), decimals=4),
        SeedRow("NDCG@10 at β = 0.03 (Yambda's cliff)", None, ("kl_sweep", "next_k/all/ndcg@10", "kl=0.03"), decimals=4),
    )),
    ("Alternatives and rewards", (
        SeedRow("NDCG@10, RL main run", **both("alternatives_to_rl", "all/ndcg@10", "RL"), decimals=4),
        SeedRow("NDCG@10, re-ranking w = 0.3", **both("alternatives_to_rl", "all/ndcg@10", "re-ranking, weight=0.3"), decimals=4),
        SeedRow("Logged share per item, RL main run", **both("alternatives_to_rl", "logged/played_ratio_per_item", "RL"), decimals=4),
        SeedRow("Logged share per item, re-ranking w = 0.3", **both("alternatives_to_rl", "logged/played_ratio_per_item", "re-ranking, weight=0.3"), decimals=4),
        SeedRow("Personal share of the gain, re-ranking w = 0.3, %", **both("reward_gain_decomposition", "re-ranking, weight=0.3", "personal share of the gain"), decimals=1, scale=100),
        SeedRow("NDCG@10, off-policy REINFORCE", **both("alternatives_to_rl", "all/ndcg@10", "off-policy REINFORCE, cap=0.5"), decimals=4),
        SeedRow("NDCG@10, share + 10 × logged relevance, β = 0.3", ("relevance_reward_sweep", "next_k/all/ndcg@10", "ratio + 10.0 x logged relevance, kl=0.3"), None, decimals=4),
        SeedRow("NDCG@10, logged relevance alone, β = 0", **both("relevance_reward_sweep", "next_k/all/ndcg@10", "logged relevance, kl=0.0"), decimals=4),
        SeedRow("NDCG@10 on new items, logged relevance alone", None, ("relevance_reward_sweep", "next_k/new/ndcg@10", "logged relevance, kl=0.0"), decimals=4),
    )),
    ("Validation-selected weight on KION (β = 1)", (
        SeedRow("NDCG@10 at β = 1", ("kl_sweep", "next_k/all/ndcg@10", "kl=1.0"), None, decimals=4),
        SeedRow("Predicted share at β = 1", ("kl_sweep", "predicted/heldout/played_ratio", "kl=1.0"), None),
        SeedRow("Logged share per item at β = 1", ("kl_sweep", "next_k/logged/played_ratio_per_item", "kl=1.0"), None, decimals=4),
        SeedRow("Logged surplus per item at β = 1", ("kl_sweep", "next_k/logged/relative_surplus_per_item", "kl=1.0"), None, decimals=4),
        SeedRow("KL at β = 1", ("kl_sweep", "training/kl_to_reference", "kl=1.0"), None),
    )),
    ("Re-ranking by the relative share on KION, against the ranked lists it starts from", (
        SeedRow("NDCG@10, ranked lists", ("relative_reward_sweep", "top_k/all/ndcg@10", "pretrained"), None, decimals=4),
        SeedRow("NDCG@10, re-ranking w = 0.3", ("relative_reward_reranking", "all/ndcg@10", "re-ranking by relative ratio, weight=0.3"), None, decimals=4),
        SeedRow("NDCG@10, re-ranking w = 0.1", ("relative_reward_reranking", "all/ndcg@10", "re-ranking by relative ratio, weight=0.1"), None, decimals=4),
        SeedRow("Logged surplus per item, ranked lists", ("relative_reward_sweep", "top_k/logged/relative_surplus_per_item", "pretrained"), None, decimals=5),
        SeedRow("Logged surplus per item, re-ranking w = 0.3", ("relative_reward_reranking", "logged/relative_surplus_per_item", "re-ranking by relative ratio, weight=0.3"), None, decimals=5),
        SeedRow("Logged surplus per item, re-ranking w = 0.1", ("relative_reward_reranking", "logged/relative_surplus_per_item", "re-ranking by relative ratio, weight=0.1"), None, decimals=5),
    )),
    ("Reference", (
        SeedRow("NDCG@10, base recommender", **both("kl_sweep", "next_k/all/ndcg@10", "pretrained"), decimals=4),
        SeedRow("Predicted share, base recommender", **both("kl_sweep", "predicted/heldout/played_ratio", "pretrained")),
        SeedRow("Logged share per item, base recommender", **both("kl_sweep", "next_k/logged/played_ratio_per_item", "pretrained"), decimals=4),
        SeedRow("Logged surplus per item, base recommender", ("kl_sweep", "next_k/logged/relative_surplus_per_item", "pretrained"), None, decimals=4),
        SeedRow("NDCG@10 on new items, base recommender", None, ("kl_sweep", "next_k/new/ndcg@10", "pretrained"), decimals=4),
        SeedRow("Reward model AUC, held-out copy", **both("reward_model_played_ratio", "heldout/reward model", "auc")),
    )),
)


@lru_cache(maxsize=None)
def three_seed_statistics(dataset: str) -> dict[tuple[str, str, str], tuple[float, float]]:
    """(report, row, column) -> (mean, standard deviation) over the three seeds of a dataset."""
    path = RESULTS / f"{dataset}_seeds" / "over_three_seeds.csv"
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        return {(row["report"], row["row"], row["column"]): (float(row["mean"]), float(row["std"])) for row in csv.DictReader(file)}


def seeds_table() -> str:
    body: list[list[str] | str] = []
    for title, rows in SEED_PANELS:
        body.append(title)
        for row in rows:
            cells = []
            for dataset in DATASET_NAMES:
                key = getattr(row, dataset)
                statistic = three_seed_statistics(dataset).get(key) if key else None
                cells.append(MISSING if statistic is None else f"{formatted(statistic[0], row.decimals, row.scale, signed=row.signed)} ± {formatted(statistic[1], row.decimals, row.scale)}")
            body.append([row.label, *cells])
    return typst_table(list(DATASET_NAMES.values()), (), body)


@dataclass(frozen=True)
class Method:
    """One way of trading relevance for reward: its runs from the weakest move to the strongest, as (report, run)."""

    label: str
    runs: tuple[tuple[str, str], ...]


BASE = ("kl_sweep", "pretrained")
RELEVANCE_THRESHOLDS = (0.01, 0.05)  # the relevance a method may lose, as a share of the base recommender's NDCG@10
EQUAL_RELEVANCE_METHODS = (
    Method("Ranked list of the base recommender", (BASE, ("baselines", "GPT top-k"))),
    Method("RL on the share, anchor sweep", (BASE, *(("kl_sweep", f"kl={weight}") for weight in (1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.03, 0.0)))),
    Method("RL on share + 10 × logged relevance", (BASE, *(("relevance_reward_sweep", f"ratio + 10.0 x logged relevance, kl={weight}") for weight in (0.3, 0.1, 0.03, 0.0)))),
    Method("Re-ranking by the reward model", (("baselines", "GPT top-k"), *(("alternatives_to_rl", f"re-ranking, weight={weight}") for weight in (1.0, 0.3, 0.1, 0.03, 0.01, 0.0)))),
    Method("Off-policy REINFORCE", (BASE, ("alternatives_to_rl", "off-policy REINFORCE, cap=0.5"), ("alternatives_to_rl", "off-policy REINFORCE, cap=2.0"))),
    Method("Liked-only fine-tuning", (BASE, ("alternatives_to_rl", "liked-only fine-tuning"))),
)


def best_within(points: list[tuple[float, float, float]], lowest_ndcg: float) -> tuple[float, float] | None:
    """(predicted share, logged share) of the highest predicted share whose NDCG@10 is at least `lowest_ndcg`.

    points: (NDCG@10, predicted share, logged share) from the weakest move to the strongest; between two
    consecutive points the curve is taken to be a straight line.
    """
    candidates = [(share, logged) for ndcg, share, logged in points if ndcg >= lowest_ndcg]
    for (ndcg_0, share_0, logged_0), (ndcg_1, share_1, logged_1) in zip(points, points[1:]):
        if ndcg_0 >= lowest_ndcg > ndcg_1:
            fraction = (ndcg_0 - lowest_ndcg) / (ndcg_0 - ndcg_1)
            candidates.append((share_0 + fraction * (share_1 - share_0), logged_0 + fraction * (logged_1 - logged_0)))
    return max(candidates) if candidates else None


def equal_relevance_table() -> str:
    """Per method, the largest predicted gain it reaches without losing more than 1% or 5% of the base recommender's NDCG@10, and the logged share there."""
    body: list[list[str] | str] = []
    for dataset in DATASET_NAMES:
        session = SESSIONS[dataset]

        def point(report: str, run_name: str) -> tuple[float, float, float] | None:
            values = [number(Source(session, report, run_name, name)) for name in ("all/ndcg@10", "predicted/heldout/played_ratio", "logged/played_ratio_per_item")]
            return None if None in values else tuple(values)

        base = point(*BASE)
        body.append(DATASET_NAMES[dataset])
        for method in EQUAL_RELEVANCE_METHODS:
            points = [found for found in (point(report, run_name) for report, run_name in method.runs) if found is not None]
            if len(points) < 2:
                continue
            row = [method.label]
            for threshold in RELEVANCE_THRESHOLDS:
                best = best_within(points, (1 - threshold) * base[0])
                row += [MISSING, MISSING] if best is None else [formatted(best[0] / base[1] - 1, 1, 100, signed=True), formatted(best[1] / base[2] - 1, 1, 100, signed=True)]
            body.append(row)
    headers = ["Predicted share, %", "Logged share, %"] * len(RELEVANCE_THRESHOLDS)
    groups = tuple((f"NDCG@10 at most {threshold:.0%} below the base", 2) for threshold in RELEVANCE_THRESHOLDS)
    return typst_table(headers, groups, body)


TITLE_TRANSLITERATIONS = {"ДУХLESS": "Dukhless"}  # titles the English release of KION left in Cyrillic


def titles_table(num_items: int = 5) -> str:
    """The KION titles whose impressions grew and shrank most under the main RL run."""
    path = RESULTS / "kion_review" / "largest_impression_changes.csv"
    if not path.exists():
        return typst_table([], (), [])
    with path.open(newline="", encoding="utf-8") as file:
        items = list(csv.DictReader(file))
    body: list[list[str] | str] = []
    for direction, title in (("gained", "Gained most"), ("lost", "Lost most")):
        body.append(title)
        for item in [item for item in items if item["direction"] == direction][:num_items]:
            hours = float(item["duration_seconds"]) / 3600
            body.append([
                f"{TITLE_TRANSLITERATIONS.get(item['title'], item['title'])} ({int(float(item['release_year']))})",
                item["content_type"],
                f"{hours:.1f}",
                f"{int(float(item['train_popularity'])):,}",
                formatted(float(item["mean_played_ratio"]), 2),
                f"{int(float(item['impressions_before'])):,}",
                f"{int(float(item['impressions_after'])):,}",
            ])
    return typst_table(["Type", "Hours", "Viewers in training", "Usual share", "Lists before", "after"], (), body)


# The frontier sessions: three seeds on KION (seed 0 is the session itself, seeds 1 and 2 are pooled with it in
# over_three_seeds.csv, one number per seed), one seed on Yambda.
FRONTIER_SESSIONS = {"kion": "kion_frontier", "yambda": "yambda_frontier"}
FRONTIER_SEEDS = RESULTS / "kion_frontier_seeds" / "over_three_seeds.csv"
FRONTIER_NUM_SEEDS = 3
PREDICTED_SHARE = "predicted/heldout/played_ratio"
PERSONAL = "predicted/heldout/personal_played_ratio"
ITEM_EFFECT_SHARE = "within-history variation explained by one effect per item"

Lookup = Callable[[str, str, str], "float | None"]  # (report, run, quantity) -> the number in one seed


@lru_cache(maxsize=None)
def frontier_seed_cells() -> dict[tuple[str, str, str], tuple[str, ...]]:
    """(report, row, column) of a KION frontier report -> its text in seeds 0, 1 and 2."""
    if not FRONTIER_SEEDS.exists():
        return {}
    with FRONTIER_SEEDS.open(newline="", encoding="utf-8") as file:
        return {(row["report"], row["row"], row["column"]): tuple(row[f"seed_{seed}"] for seed in range(FRONTIER_NUM_SEEDS)) for row in csv.DictReader(file)}


def seed_number(seed: int, report: str, run_name: str, quantity_name: str) -> float | None:
    """A number of one seed of the KION frontier sessions, whichever of the run and the quantity the report has in its rows."""
    cells = frontier_seed_cells()
    for name in (quantity_name, GENERATED_LISTS + quantity_name):
        for key in ((report, name, run_name), (report, run_name, name)):
            if cells.get(key, ("",) * FRONTIER_NUM_SEEDS)[seed]:
                return float(cells[key][seed])
    return None


def frontier_lookups(dataset: str) -> list[Lookup]:
    """One lookup per seed of the dataset's frontier sessions."""
    if dataset == "kion" and frontier_seed_cells():
        return [partial(seed_number, seed) for seed in range(FRONTIER_NUM_SEEDS)]
    session = FRONTIER_SESSIONS[dataset]
    return [lambda report, run_name, quantity_name: number(Source(session, report, run_name, quantity_name))]


def gain_fraction(report: str, run_name: str, reference: str) -> Callable[[Lookup], float | None]:
    """The fraction of `reference`'s gain in predicted share over the base recommender that `run_name` reaches."""

    def value(lookup: Lookup) -> float | None:
        numbers = [lookup(report, name, PREDICTED_SHARE) for name in (run_name, reference, "pretrained")]
        if None in numbers:
            return None
        reached, full, start = numbers
        return (reached - start) / (full - start)

    return value


def relative_change(report: str, run_name: str, quantity_name: str, reference: str = "pretrained") -> Callable[[Lookup], float | None]:
    def value(lookup: Lookup) -> float | None:
        after, before = lookup(report, run_name, quantity_name), lookup(report, reference, quantity_name)
        return None if after is None or not before else after / before - 1

    return value


def plain(report: str, run_name: str, quantity_name: str) -> Callable[[Lookup], float | None]:
    return lambda lookup: lookup(report, run_name, quantity_name)


@dataclass(frozen=True)
class Derived:
    """A number computed within each seed from several cells of its reports, shown as mean ± standard deviation over the seeds."""

    label: str
    value: Callable[[Lookup], float | None]
    decimals: int = 1
    scale: float = 100.0
    signed: bool = True


def over_seeds(row: Derived, dataset: str) -> str:
    return over_lookups(row, frontier_lookups(dataset))


def over_lookups(row: Derived, lookups: list[Lookup]) -> str:
    """The row's number in every seed that `lookups` reach, as mean ± standard deviation."""
    values = [found for found in (row.value(lookup) for lookup in lookups) if found is not None]
    if not values:
        return MISSING
    text = formatted(statistics.mean(values), row.decimals, row.scale, signed=row.signed)
    if len(values) > 1:
        text += f" ± {formatted(statistics.stdev(values), row.decimals, row.scale)}"
    return text


def seeds_header(dataset: str) -> str:
    num_seeds = len(frontier_lookups(dataset))
    return f"{DATASET_NAMES[dataset]}, {num_seeds} seed{'s' if num_seeds > 1 else ''}"


def capacity_fraction(label: str, restricted: str, budget: str) -> Derived:
    return Derived(label, gain_fraction("capacity_sweep", f"{restricted}, kl_budget={budget}", f"kl_budget={budget}"), decimals=0, signed=False)


def part_of_main_run(label: str, part: str, quantity_name: str | None = None) -> Derived:
    """A part of the main run's change: its fraction of RL's gain, or the relative change of a quantity."""
    if quantity_name is None:
        return Derived(label, gain_fraction("policy_change_parts", part, "RL"), decimals=0, signed=False)
    return Derived(label, relative_change("policy_change_parts", part, quantity_name))


def item_effect_share(label: str, run_name: str) -> Derived:
    return Derived(label, plain("item_effect_share", run_name, ITEM_EFFECT_SHARE), decimals=2, scale=1.0, signed=False)


BASE_PLUS_OFFSETS, RL_MINUS_OFFSETS = "base + RL's item offsets", "RL - its item offsets"
MECHANISM_PANELS = (
    ("Fine-tuning held to a KL budget: % of full RL's gain in predicted share", (
        capacity_fraction("One score offset per item, budget 0.1", "item offsets", "0.1"),
        capacity_fraction("One score offset per item, budget 0.5", "item offsets", "0.5"),
        capacity_fraction("Rank-1 LoRA, budget 0.1", "rank-1 updates", "0.1"),
        capacity_fraction("Rank-1 LoRA, budget 0.5", "rank-1 updates", "0.5"),
    )),
    ("The main run's change split into one offset per item and the rest", (
        part_of_main_run("Base + RL's item offsets: % of RL's gain", BASE_PLUS_OFFSETS),
        part_of_main_run("Base + RL's item offsets: NDCG@10, change in %", BASE_PLUS_OFFSETS, NDCG.source.quantity),
        part_of_main_run("Base + RL's item offsets: personal part, change in %", BASE_PLUS_OFFSETS, PERSONAL),
        part_of_main_run("RL − its item offsets: % of RL's gain", RL_MINUS_OFFSETS),
        part_of_main_run("RL − its item offsets: NDCG@10, change in %", RL_MINUS_OFFSETS, NDCG.source.quantity),
        part_of_main_run("RL − its item offsets: personal part, change in %", RL_MINUS_OFFSETS, PERSONAL),
        part_of_main_run("RL: NDCG@10, change in %", "RL", NDCG.source.quantity),
        part_of_main_run("RL: personal part, change in %", "RL", PERSONAL),
    )),
    ("Share of within-history variation that one effect per item explains", (
        item_effect_share("Reward model's predictions (the optimum's change)", "reward model's predictions (the optimum's change)"),
        item_effect_share("RL, main run", "RL"),
        item_effect_share("RL, KL budget 0.5", "kl_budget=0.5"),
        item_effect_share("Rank-1 LoRA, budget 0.5", "rank-1 updates, kl_budget=0.5"),
        item_effect_share("Item offsets, budget 0.5 (their ceiling)", "item offsets, kl_budget=0.5"),
    )),
)


def mechanism_table() -> str:
    body: list[list[str] | str] = []
    for title, rows in MECHANISM_PANELS:
        body.append(title)
        body += [[row.label, *(over_seeds(row, dataset) for dataset in DATASET_NAMES)] for row in rows]
    return typst_table([seeds_header(dataset) for dataset in DATASET_NAMES], (), body)


LOGGED_METHOD_QUANTITIES = (
    ("NDCG@10", NDCG.source.quantity),
    ("Logged surplus", SURPLUS.source.quantity),
    ("Personal part", PERSONAL),
    ("Coverage", COVERAGE.source.quantity),
    ("Pred. share", PREDICTED_SHARE),
)
LOGGED_METHODS = (
    ("RL, main run (β = 0.3)", "RL"),
    ("Liked-only fine-tuning", "liked-only fine-tuning"),
    ("Reward-weighted, relative share, τ = 0.3", "reward-weighted, relative_ratio, temperature=0.3"),
    ("Reward-weighted, relative share, τ = 1", "reward-weighted, relative_ratio, temperature=1.0"),
    ("Reward-weighted, consumed share, τ = 0.3", "reward-weighted, played_ratio, temperature=0.3"),
    ("Within-user DPO, β = 0.5", "within-user DPO, beta=0.5"),
    ("Within-user DPO, β = 0.1", "within-user DPO, beta=0.1"),
)


def logged_methods_table() -> str:
    """Learning from logged responses alone against the base recommender: relative changes in percent."""
    body: list[list[str] | str] = []
    for dataset in DATASET_NAMES:
        body.append(seeds_header(dataset))
        for label, method in LOGGED_METHODS:
            cells = [over_seeds(Derived(label, relative_change("logged_preference_methods", method, name)), dataset) for _, name in LOGGED_METHOD_QUANTITIES]
            if any(text != MISSING for text in cells):
                body.append([label, *cells])
    return typst_table([header for header, _ in LOGGED_METHOD_QUANTITIES], (), body)


# The control for learning from the log: seeds 0, 1 and 2 in one session, each with its own base recommender, so every
# change is computed within a seed. Continued training is the same epoch of fine-tuning with every logged item weighing one.
CONTROL_SESSION = "kion_control"
CONTROL_SEEDS = (0, 1, 2)
CONTROL_REPORT = "continued_training_control"
CONTINUED_TRAINING = "continued training"
CONTROL_LIST_KINDS = ("generated", "ranked")
CONTROL_QUANTITIES = (
    ("NDCG@10", NDCG.source.quantity),
    ("Surplus", SURPLUS.source.quantity),
    ("Personal", PERSONAL),
    ("Coverage", COVERAGE.source.quantity),
)
CONTROL_METHODS = (
    ("Continued training", CONTINUED_TRAINING),
    ("Reward-weighted, τ = 1", "reward-weighted, relative_ratio, temperature=1.0"),
    ("Reward-weighted, τ = 0.3", "reward-weighted, relative_ratio, temperature=0.3"),
    ("Within-user DPO, β = 0.5", "within-user DPO, beta=0.5"),
)


def control_number(seed: int, report: str, run_name: str, quantity_name: str) -> float | None:
    return number(Source(f"{CONTROL_SESSION}/seed_{seed}", report, run_name, quantity_name))


def control_table() -> str:
    """Learning from the log against the base recommender and against continued training, on generated and on ranked
    lists: changes in percent, each computed within a seed of the control session."""
    lookups = [partial(control_number, seed) for seed in CONTROL_SEEDS]
    panels = (
        ("Against the base recommender's lists of the same kind", "pretrained", CONTROL_METHODS),
        ("Against continued training, lists of the same kind", CONTINUED_TRAINING, CONTROL_METHODS[1:]),
    )
    body: list[list[str] | str] = []
    for title, reference, methods in panels:
        body.append(title)
        for label, method in methods:
            changes = [
                Derived(label, relative_change(CONTROL_REPORT, f"{method}, {kind}", quantity_name, f"{reference}, {kind}"))
                for kind in CONTROL_LIST_KINDS
                for _, quantity_name in CONTROL_QUANTITIES
            ]
            body.append([label, *(over_lookups(change, lookups) for change in changes)])
    headers = [header for _ in CONTROL_LIST_KINDS for header, _ in CONTROL_QUANTITIES]
    groups = tuple((f"{kind.capitalize()} lists", len(CONTROL_QUANTITIES)) for kind in CONTROL_LIST_KINDS)
    return typst_table(headers, groups, body)


def write_tables() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "mechanism.typ").write_text(mechanism_table(), encoding="utf-8")
    (OUTPUT / "logged_methods.typ").write_text(logged_methods_table(), encoding="utf-8")
    (OUTPUT / "control.typ").write_text(control_table(), encoding="utf-8")
    (OUTPUT / "seeds.typ").write_text(seeds_table(), encoding="utf-8")
    (OUTPUT / "equal_relevance.typ").write_text(equal_relevance_table(), encoding="utf-8")
    (OUTPUT / "titles.typ").write_text(titles_table(), encoding="utf-8")
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
