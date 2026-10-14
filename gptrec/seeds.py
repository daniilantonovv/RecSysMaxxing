"""The same study under several random seeds: how much of a result is the draw of one run.

A seed fixes every source of randomness of one experiment: the initial weights of the recommender, of the
reward models and of the baselines, the order of their training batches, and the lists sampled while
fine-tuning. The interaction log and the requests the recommenders are judged on do not depend on it.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import pandas as pd
import polars as pl
import torch

from gptrec.experiment import Experiment, ExperimentConfig, smoke_test_config

# The stages repeated per seed: the base models, the main fine-tuning run and what it is compared with.
# A stage does nothing for a configuration whose grid for it is empty.
SEED_STAGES = (
    "pretrain_policy",
    "train_reward_models",
    "finetune",
    "sweep_kl_weight",
    "sweep_kl_budget",
    "compare_advantage_estimators",
    "explain_reward_gain",
    "compare_alternatives_to_rl",
    "sweep_relevance_in_reward",
    "sweep_personal_rewards",
    "sweep_relative_reward",
    "ablate_sequence_credit",
)
# Reports averaged over seeds, with the columns of each that hold its row labels.
SEED_REPORTS = {
    "policy_test_relevance": [0],
    "reward_model_played_ratio": [0, 1],
    "reward_model_relative_ratio": [0, 1],
    "reward_model_retention": [0, 1, 2],
    "rl_vs_pretrained": [0],
    "rl_vs_pretrained_significance": [0],
    "kl_sweep": [0],
    "kl_sweep_validation": [0],
    "kl_budget_sweep": [0],
    "advantage_estimators": [0],
    "sequence_credit_ablation": [0],
    "reward_gain_decomposition": [0],
    "alternatives_to_rl": [0],
    "relevance_reward_sweep": [0],
    "personal_reward_sweep": [0],
    "relative_reward_sweep": [0],
    "relative_reward_reranking": [0],
    "stage_minutes": [0],
}
TIME_SAFETY_FACTOR = 1.2  # a seed is assumed to take this much longer than the one before it


def with_seed(config: ExperimentConfig, seed: int) -> ExperimentConfig:
    """`config` with every source of randomness drawn from `seed`."""

    def reseeded(optimization):
        return replace(optimization, seed=seed)

    return replace(
        config,
        policy_optimization=reseeded(config.policy_optimization),
        reward_model_optimization=reseeded(config.reward_model_optimization),
        baselines=replace(config.baselines, optimization=reseeded(config.baselines.optimization)),
        liked_finetuning=reseeded(config.liked_finetuning),
        logged_returns=replace(config.logged_returns, optimization=reseeded(config.logged_returns.optimization)),
        rl=replace(config.rl, seed=seed),
    )


def run_seeds(
    raw_events: pl.DataFrame,
    config: ExperimentConfig,
    seeds: Sequence[int],
    home: str | Path,
    device: torch.device,
    stages: Sequence[str] = SEED_STAGES,
    time_budget_hours: float | None = None,
) -> dict[int, Path]:
    """Runs `stages` of the experiment once per seed, each in its own directory under `home`. Returns the report directory of every finished seed.

    The interaction log is built for the first seed and copied for the others. `time_budget_hours` counts
    from now: a seed that would not finish inside it, judging by the one before, is skipped and reported.
    """
    home = Path(home)
    deadline = None if time_budget_hours is None else time.monotonic() + time_budget_hours * 3600
    report_directories: dict[int, Path] = {}
    log_of_finished_seed: Path | None = None
    seconds_of_last_seed = 0.0
    for seed in seeds:
        started = time.monotonic()
        if deadline is not None and started + TIME_SAFETY_FACTOR * seconds_of_last_seed > deadline:
            print(f"skipped seed {seed}: it would not finish inside the time budget")
            continue
        hours_left = None if deadline is None else (deadline - started) / 3600
        experiment = Experiment(with_seed(config, seed), home / f"seed_{seed}", device, show_figures=False, time_budget_hours=hours_left)
        if log_of_finished_seed is not None and not experiment.log_directory.exists():
            shutil.copytree(log_of_finished_seed, experiment.log_directory)
        experiment.prepare_data(raw_events)
        for stage in stages:
            getattr(experiment, stage)()
        log_of_finished_seed = experiment.log_directory
        report_directories[seed] = experiment.reports.directory
        seconds_of_last_seed = time.monotonic() - started
        print(f"seed {seed} finished in {seconds_of_last_seed / 60:.0f} minutes")
        del experiment
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report_directories


def summarise_seeds(report_directories: dict[int, Path], output_directory: str | Path, reports: dict[str, list[int]] = SEED_REPORTS) -> dict[str, pd.DataFrame]:
    """Mean and standard deviation over seeds of every number of the named reports.

    For each report, writes "<report>_mean.csv", "<report>_std.csv", "<report>_num_seeds.csv" and
    "<report>_mean_std.csv" into `output_directory`; the last holds "mean ± std" as text and is what is returned.
    A report missing for a seed, or a run missing from a seed's report, is left out of that number's average.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for report, label_columns in reports.items():
        per_seed = {
            seed: pd.read_csv(directory / f"{report}.csv", index_col=label_columns)
            for seed, directory in report_directories.items()
            if (directory / f"{report}.csv").exists()
        }
        if not per_seed:
            continue
        stacked = pd.concat(per_seed, names=["seed"])
        over_seeds = stacked.groupby(level=list(range(1, stacked.index.nlevels)), sort=False)
        mean, deviation, num_seeds = over_seeds.mean(), over_seeds.std(), over_seeds.count()
        mean.to_csv(output_directory / f"{report}_mean.csv")
        deviation.to_csv(output_directory / f"{report}_std.csv")
        num_seeds.to_csv(output_directory / f"{report}_num_seeds.csv")
        summaries[report] = _as_text(mean) + " ± " + _as_text(deviation)
        summaries[report].to_csv(output_directory / f"{report}_mean_std.csv")
    return summaries


def run_seeds_smoke_test(raw_events: pl.DataFrame, config: ExperimentConfig, home: str | Path, device: torch.device, max_users: int = 200) -> None:
    """Two seeds of `config`, shrunk as in `gptrec.experiment.run_smoke_test`, through every stage and the summary over seeds."""
    report_directories = run_seeds(raw_events, smoke_test_config(config, max_users), seeds=(0, 1), home=home, device=device)
    summaries = summarise_seeds(report_directories, Path(home) / "over_seeds")
    if "kl_sweep" not in summaries:
        raise RuntimeError("the summary over seeds is missing the KL sweep")


def _as_text(table: pd.DataFrame) -> pd.DataFrame:
    return table.apply(lambda column: column.map("{:.4f}".format))
