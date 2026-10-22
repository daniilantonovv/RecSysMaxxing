"""The whole study as a sequence of resumable stages.

    1. prepare_data                 raw events -> interaction log
    2. pretrain_policy              the base recommender
    3. train_reward_models          "train" (drives RL) and "heldout" (only judges), on disjoint users
    4. compare_baselines            popularity, replaying history, Markov chain, GRU4Rec, SASRec, BERT4Rec against the base recommender
    5. finetune                     RL fine-tuning, compared with the base recommender
    6. sweep_kl_weight              the trade-off between relevance and reward as the anchor weakens
       sweep_kl_budget              the same trade-off with the distance from the base recommender fixed instead of the anchor weight
       compare_advantage_estimators the leave-one-out baseline against the value model, at equal weights and at equal distances
       compare_simulated_feedback   fine-tuning with a sampled response to the generated items instead of the expected one
       explain_reward_gain          how much of the gain depends on who receives the list, and how far item averages alone go
       compare_alternatives_to_rl   liked-only fine-tuning, re-ranking, off-policy REINFORCE, best of several sampled lists
       sweep_rewards                rewards for continuing the session and for returning, each at several anchor weights
       sweep_relevance_in_reward    relevance asked for by the reward itself, alone and together with the anchor
       sweep_personal_rewards       rewards that pay only for what depends on the user
       sweep_relative_reward        a reward for the played ratio relative to what is usual for the item
       compare_list_reward          a reward that depends on the list as a whole, against greedy ways of maximising it
       ablate_sequence_credit       fine-tuning that credits each item with its own reward only, or sees one item per episode
       analyse_multistep
    7. diagnose_return_label        whether a return label follows the item after the event rather than the event
       restrict_finetuning_capacity RL that may change only one score offset per item, or only low-rank updates of the transformer
       compare_group_relative_advantages  GRPO's group-normalised advantages, at equal KL budgets
       decompose_policy_change      the main run split into one offset per item and the rest; the reward's own split, before any RL
       align_at_inference           the base recommender steered by the closed form of the anchored objective, with look-ahead
       learn_from_logged_preferences  reward-weighted fine-tuning and within-user DPO on logged responses: no reward model
       predict_lists_without_feedback lists from one head per position, none of them reading an unconfirmed item
       interpolate_weights          operating points between the base and the fine-tuned weights, and beyond
       control_for_continued_training learning from the log against the same epoch of training without its weights

A stage whose grid of settings is empty in the configuration does nothing.

Every stage reads what it needs from the experiment's home directory and writes its checkpoints and
reports back there. A stage whose checkpoint already exists does not train again, so an interrupted
run continues where it stopped. A home directory belongs to one configuration; `reuse_artifacts` starts
one from the models and the finished runs of an earlier session that differed only in the grids of its sweeps.
"""

from __future__ import annotations

import functools
import json
import re
import shutil
import stat
import time
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn.functional as F

from gptrec import plots
from gptrec.baselines.suite import BaselineConfig, fit_baselines
from gptrec.data.batches import EvalExamples, EventWindows, build_eval_examples
from gptrec.data.log import InteractionLog, Period
from gptrec.data.preprocess import PreprocessingConfig, build_interaction_log
from gptrec.decoding import ContinuationTrainingConfig, GuidanceConfig, fit_continuation_values, guided_lists
from gptrec.evaluation.analyses import (
    candidate_log_probs,
    exposure_lorenz,
    feedback_assumptions_on_real_sequences,
    impressions_by_attribute,
    item_effect_share,
    kl_along_lists,
    largest_impression_changes,
    logit_shift_by_item,
    mean_logit_shift_per_item,
    position_hits,
    relevance_by_history_length,
    return_rate_by_duration,
)
from gptrec.evaluation.behavior import intra_list_distance, list_overlap, recommendation_profile
from gptrec.evaluation.policy_report import (
    EvaluationConfig,
    ListEvaluation,
    PolicyEvaluation,
    collect_lists,
    evaluate_lists,
    evaluate_policy,
    list_metrics,
    metrics_table,
    paired_comparison,
    per_step_table,
    popularity_baseline_metrics,
)
from gptrec.evaluation.relevance import ranking_metrics
from gptrec.evaluation.reward_model_metrics import (
    HeldoutFeedback,
    calibration_table,
    choice_quality,
    error_by_group,
    predict_heldout_feedback,
    quality_by_origin,
    reaction_quality,
    relative_ratio_report,
    retention_quality,
    reward_model_report,
    signal_spread_over_candidates,
)
from gptrec.models.adapters import Adaptation, interpolated, with_item_offsets
from gptrec.models.backbone import TransformerConfig
from gptrec.models.multi_token import MultiTokenHeads, load_multi_token_heads, multi_token_lists, save_multi_token_heads, train_multi_token_heads
from gptrec.models.policy import NextItemPolicy
from gptrec.models.reward_model import Feedback, RewardModel
from gptrec.models.sequence_recommender import NextItemScorer
from gptrec.models.value_model import ContinuationValueModel
from gptrec.reports import ReportWriter
from gptrec.rollout import (
    SIMULATION_ASSUMPTIONS,
    RatioBaseline,
    RewardFunction,
    RewardSpec,
    ScoredCandidates,
    SimulatedFeedback,
    best_of_sampled_lists,
    recommend_top_k,
    rerank,
    score_candidates,
)
from gptrec.training.checkpoint import load_continuation_value_model, load_policy, load_reward_model, save_continuation_value_model, save_model
from gptrec.training.off_policy import LoggedReturnConfig, finetune_on_logged_returns
from gptrec.training.precision import autocast
from gptrec.training.preference import LoggedPreferenceConfig, RewardWeightedConfig, finetune_on_logged_preferences, finetune_with_reward_weights
from gptrec.training.rl import AdvantageEstimator, RLConfig, finetune_with_rl
from gptrec.training.supervised import (
    FeedbackLossWeights,
    OptimizationConfig,
    PolicyTrainingConfig,
    RewardModelTrainingConfig,
    finetune_on_liked_items,
    train_policy,
    train_reward_model,
)

DEFAULT_REWARDS = (
    ("played ratio", RewardSpec()),
    ("played time", RewardSpec(weigh_ratio_by_duration=True)),
    ("session", RewardSpec(played_ratio_weight=0.0, session_continues_weight=1.0)),
    ("return", RewardSpec(played_ratio_weight=0.0, returns_weight=1.0)),
    ("ratio + return", RewardSpec(returns_weight=1.0)),
    ("ratio + session + return", RewardSpec(session_continues_weight=1.0, returns_weight=1.0)),
)
CHOSEN_RATIO = RewardSpec(weigh_by_choice=True)
LOGGED_RELEVANCE = RewardSpec(played_ratio_weight=0.0, logged_relevance_weight=1.0)
PERSONAL_RATIO = RewardSpec(ratio_baseline=RatioBaseline.OTHER_USER)
RATIO_ABOVE_ITEM_MEAN = RewardSpec(ratio_baseline=RatioBaseline.ITEM_MEAN)
RELATIVE_RATIO = RewardSpec(played_ratio_weight=0.0, relative_ratio_weight=1.0)
LIKES = RewardSpec(played_ratio_weight=0.0, liked_weight=1.0, disliked_weight=1.0)  # P(like) - P(dislike)
ADOPTION = RewardSpec(played_ratio_weight=0.0, adoption_weight=1.0)  # P(the user keeps coming back to a new item)
RETURN_AFTER_THE_LIST = RewardSpec(played_ratio_weight=0.0, final_returns_weight=1.0)  # P(return) after the last item only
# The signals of the held-out reward model that a reward can be aimed at: what the trade-off figures put against relevance.
JUDGED_SIGNALS = ("played_ratio", "personal_played_ratio", "session_continues", "returns", "relative_ratio")
REALISED_KL = "training/kl_to_reference"  # nats per generated item between a fine-tuned policy and the base recommender
# What a fine-tuning run looked like when it ended: metric name -> the recorded quantity it averages over the last iterations.
TRAINING_STATISTICS = {
    REALISED_KL: "kl_to_reference",
    "training/anchor_weight": "reference_kl_weight",
    "training/clip_fraction": "clip_fraction",
    "training/value_explained_variance": "value_explained_variance",
    "training/advantage_scale": "advantage_scale",  # standard deviation of the advantages before they are standardised
}
END_OF_TRAINING_STEPS = 50  # last iterations of a run over which those are averaged
INTRA_LIST_DISTANCE = "profile/intra_list_distance"  # mean cosine distance between two items of a list, in the base recommender's embeddings
FEEDBACK_ASSUMPTION_STEPS = 5  # real next items of a user on which the assumptions of a simulation are checked
TRAINING_CURVES = [
    "reward",
    "kl_to_reference",
    "reference_kl_weight",
    "entropy",
    "distinct_action_share",
    "logged_hit_rate",
    "history_repeat_rate",
    "mean_action_popularity",
    "mean_action_duration_seconds",
    "gradient_norm",
    "clip_fraction",
    "value_explained_variance",
]
TRAINING_CURVE_SMOOTHING_STEPS = 20
CONFIG_RECORD = "experiment_config.json"
RUN_TIME_SAFETY_FACTOR = 1.25  # a sweep run is assumed to take this much longer than the last run suggests
RESERVE_FOR_LAST_STAGES_SECONDS = 30 * 60  # kept free for the cheap stages that follow the sweeps
# The settings that choose which runs are made and how they are compared, as opposed to what a model or a run is.
RUN_GRIDS = (
    "kl_weights",
    "rewards",
    "reward_kl_weights",
    "choice_reward_kl_weights",
    "logged_relevance_weights",
    "logged_relevance_kl_weights",
    "personal_reward_kl_weights",
    "credit_ablation_kl_weights",
    "rerank_relevance_weights",
    "importance_weight_caps",
    "sampled_list_counts",
    "evaluate_sweeps_on_validation",
    "relevance_tolerance",
    "kl_budgets",
    "estimator_kl_weights",
    "high_learning_rate_factor",
    "sampled_feedback_kl_weights",
    "relative_reward_kl_weights",
    "diversity_weights",
    "diversity_kl_weights",
    "masked_feedback_kl_weights",
    "return_label_check",
    "capacity_kl_budgets",
    "low_ranks",
    "group_relative_kl_budgets",
    "analyse_policy_change",
    "guidance_strengths",
    "guidance_discounts",
    "guided_rewards",
    "reward_weighted_temperatures",
    "reward_weighted_responses",
    "preference_betas",
    "multi_token_lists",
    "interpolation_factors",
    "continued_training_control",
)
KL_BATCH_SIZE = 256  # histories per pass when the KL along a recommender's lists is measured: each brings K distributions over the catalogue
CONTINUED_TRAINING = "continued training"  # the base recommender trained one more epoch with every logged item weighing one
LIST_KINDS = ("ranked", "generated")  # the catalogue scored once, or the list generated item by item


@dataclass(frozen=True)
class ExperimentConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    max_users: int | None = None  # keep only this many users of the raw log; for quick end-to-end checks
    transformer: TransformerConfig = field(default_factory=TransformerConfig)  # shared by the recommender and the reward models
    window_stride: int = 64  # distance between the starts of consecutive training windows of one user
    policy_optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    reward_model_optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(epochs=5))
    reward_model_loss: FeedbackLossWeights = field(default_factory=FeedbackLossWeights)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    horizon: int = 10  # items generated per request, both in RL and in evaluation
    max_future: int = 20  # next events of a user that count as relevant
    ks: tuple[int, ...] = (1, 5, 10)
    evaluation_batch_size: int = 256
    rl: RLConfig = field(default_factory=RLConfig)  # the main run; its history length and horizon are overridden to fit the above
    sweep_steps: int = 1000  # RL steps of every sweep run
    kl_weights: tuple[float, ...] = (0.0, 0.03, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 3.0)
    rewards: tuple[tuple[str, RewardSpec], ...] = DEFAULT_REWARDS[1:]  # named alternatives to the main run's reward
    reward_kl_weights: tuple[float, ...] = (0.1, 0.3, 1.0)  # anchor weights at which each of them is tried
    # Relevance asked for by the reward instead of the anchor:
    choice_reward_kl_weights: tuple[float, ...] = (0.0, 0.1, 0.3)  # anchor weights at which the choice-weighted reward is tried
    logged_relevance_weights: tuple[float, ...] = (10.0, 3.0)  # weight of logged relevance against the played ratio
    logged_relevance_kl_weights: tuple[float, ...] = (0.0, 0.03, 0.1, 0.3)  # anchor weights at which each of those is tried
    personal_reward_kl_weights: tuple[float, ...] = (0.0, 0.1, 0.3)  # anchor weights at which the rewards with a ratio baseline are tried
    # Anchor weights at which fine-tuning without sequence credit is tried; they should be among `kl_weights`, whose runs it is compared with.
    credit_ablation_kl_weights: tuple[float, ...] = (0.1, 0.3)
    # The ways of raising the reward without on-policy RL that RL is compared with:
    liked_finetuning: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(epochs=1, learning_rate=1e-4))
    rerank_candidates: int = 100  # most likely next items that re-ranking by predicted reward chooses from
    rerank_relevance_weights: tuple[float, ...] = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0)  # weight of log-probability against reward
    logged_returns: LoggedReturnConfig = field(default_factory=LoggedReturnConfig)  # off-policy REINFORCE on logged sessions
    importance_weight_caps: tuple[float, ...] = (0.5, 2.0, 10.0)  # its knob: how far it may move from the base recommender
    sampled_list_counts: tuple[int, ...] = (2, 8, 32)  # lists sampled per user when the best of several is served
    sample_from_top: int = 4  # those lists are sampled among this many most likely items per position
    # Sweep runs are also evaluated on validation requests, so that an operating point can be chosen without the test period:
    evaluate_sweeps_on_validation: bool = True
    relevance_tolerance: float = 0.05  # share of the base recommender's validation NDCG a chosen run may lose
    # What the anchor weight means. The stages below do nothing while their grid is empty.
    kl_budgets: tuple[float, ...] = ()  # distances from the base recommender, in nats per item, that an adaptive anchor weight holds runs at
    estimator_kl_weights: tuple[float, ...] = ()  # anchor weights at which the leave-one-out baseline is compared with the value model
    leave_one_out_rollouts: int = 4  # lists per history under that baseline; the number of lists per iteration stays the same
    high_learning_rate_factor: float | None = None  # one more run with the policy's learning rate multiplied by this: does PPO's clipping ever engage
    sampled_feedback_kl_weights: tuple[float, ...] = ()  # anchor weights at which fine-tuning is repeated with sampled simulated feedback
    # Rewards that an item cannot earn on its own account:
    relative_reward_kl_weights: tuple[float, ...] = ()  # anchor weights at which the played ratio relative to the item's usual one is the reward
    diversity_weights: tuple[float, ...] = ()  # weight of intra-list distance against the main run's reward
    diversity_kl_weights: tuple[float, ...] = (0.3,)  # anchor weights at which each of those is tried
    # What a simulation is told about generated items: a reward model that can be told "unknown" needs no assumption.
    masked_feedback_kl_weights: tuple[float, ...] = ()  # anchor weights at which the policy is fine-tuned against such a model
    hidden_response_share: float = 0.5  # its training windows in which the responses to the last events are hidden
    # What RL learns, how else its optimum can be reached, and what the log alone can teach. Each stage below does
    # nothing while its grid is empty or its switch is off.
    return_label_check: bool = False  # does the return label follow the item after the event rather than the event
    capacity_kl_budgets: tuple[float, ...] = ()  # KL budgets at which fine-tuning may change only item offsets, or only low-rank updates
    low_ranks: tuple[int, ...] = (1, 8)
    item_offset_learning_rate: float = 3e-3  # offsets are scores, not weights: Adam moves each by about this much per step
    low_rank_learning_rate: float = 1e-4  # low-rank updates need about ten times the learning rate of full fine-tuning
    group_relative_kl_budgets: tuple[float, ...] = ()  # KL budgets at which GRPO's group-normalised advantages are tried
    analyse_policy_change: bool = False  # split the main run into one offset per item and the rest; compare with the reward's own split
    guidance_strengths: tuple[float, ...] = ()  # weights beta of the closed form at which the base recommender is steered at inference
    guidance_discounts: tuple[float, ...] = (0.0, 0.9)  # 0: each item judged by its own reward; above 0: plus the learned worth of what follows it
    guidance_candidates: int = 50  # the base recommender's most likely next items that each position chooses from
    guidance_batch_size: int = 256  # test users per pass; each brings `guidance_candidates` sequences to score
    guided_rewards: tuple[tuple[str, RewardSpec], ...] = (("played ratio", RewardSpec()),)
    continuation_training: ContinuationTrainingConfig = field(default_factory=ContinuationTrainingConfig)
    reward_weighted_temperatures: tuple[float, ...] = ()  # temperatures of fine-tuning in which each logged item weighs exp(response / temperature)
    reward_weighted_responses: tuple[str, ...] = ("relative_ratio",)  # the logged responses it is tried with
    preference_betas: tuple[float, ...] = ()  # DPO strengths at which within-user logged preferences are learned
    preference_margin: float = 0.1  # how far above and below the usual response the two items of a pair lie
    preference_lookahead: int = 20  # the user's next events a pair is drawn from
    preference_finetuning: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(epochs=1, learning_rate=1e-4))
    multi_token_lists: bool = False  # lists from one head per position, none of them reading an unconfirmed item
    multi_token_optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(epochs=3, learning_rate=1e-3))
    multi_token_targets_per_head: int | None = None  # positions per batch each head learns from; bounds memory on a large catalogue
    interpolation_factors: tuple[float, ...] = ()  # points on the line from the base recommender's weights (0) to the main run's (1), and beyond
    # Learning from the log against the same epoch of training with every item weighing one, on ranked and generated lists.
    continued_training_control: bool = False

    def __post_init__(self) -> None:
        if self.horizon >= self.transformer.max_positions:
            raise ValueError("the horizon must leave room for a history inside max_positions")
        if max(self.ks) > self.horizon:
            raise ValueError(f"metrics @{max(self.ks)} need generated lists of at least that length, but horizon={self.horizon}")
        if self.rerank_candidates < self.horizon:
            raise ValueError("re-ranking needs at least as many candidates as the list is long")
        if self.rl.histories_per_step % self.leave_one_out_rollouts != 0:
            raise ValueError("leave_one_out_rollouts must divide histories_per_step, so that both estimators see as many lists per iteration")

    @property
    def history_len(self) -> int:
        """Events of history shown to the models: what is left of their position budget after the generated items."""
        return self.transformer.max_positions - self.horizon

    def rl_config(self, **overrides) -> RLConfig:
        generation = replace(self.rl.generation, horizon=self.horizon)
        return replace(self.rl, **({"history_len": self.history_len, "max_future": self.max_future, "generation": generation} | overrides))

    def sweep_run(self, reward: RewardSpec, kl_weight: float, **overrides) -> RLConfig:
        """A fine-tuning run of sweep length that differs from the main run in its reward, its anchor weight and `overrides`."""
        return self.rl_config(num_steps=self.sweep_steps, reward=reward, reference_kl_weight=kl_weight, **overrides)

    def evaluation_config(self, **overrides) -> EvaluationConfig:
        """Greedy lists under the same item constraints and simulated feedback the policy is fine-tuned with."""
        generation = replace(self.rl.generation, horizon=self.horizon, greedy=True, temperature=1.0)
        config = EvaluationConfig(self.ks, generation, self.rl.simulated_feedback, batch_size=self.evaluation_batch_size)
        return replace(config, **overrides)


def smoke_test_config(config: ExperimentConfig, max_users: int) -> ExperimentConfig:
    """`config` shrunk until every stage runs in minutes, even on a CPU: few users, tiny models, a handful of steps.

    What is specific to the dataset - split lengths, session gap, generation constraints - is kept, and so is
    every grid that `config` leaves empty, so the same code paths run and a stage the configuration switches off
    stays off. The numbers a smoke test produces mean nothing.
    """
    optimization = OptimizationConfig(epochs=1, batch_size=16, warmup_steps=5)
    horizon = 4
    preprocessing = replace(config.preprocessing, max_items=1_000, min_item_events=5, min_user_events=min(config.preprocessing.min_user_events, 20))
    rl = replace(config.rl, num_steps=6, histories_per_step=16, ppo_epochs=2, value_warmup_steps=2, evaluate_every=3, mixed_precision=True)
    switched_off = {name: getattr(config, name) for name in RUN_GRIDS if not getattr(config, name)}
    shrunk = replace(
        config,
        preprocessing=preprocessing,
        max_users=max_users,
        transformer=TransformerConfig(hidden_size=32, num_layers=1, num_heads=2, max_positions=24),
        window_stride=24,
        policy_optimization=optimization,
        reward_model_optimization=optimization,
        baselines=BaselineConfig(optimization=optimization, markov_chain_successors=20),
        horizon=horizon,
        max_future=10,
        ks=(1, horizon),
        evaluation_batch_size=64,
        rl=rl,
        sweep_steps=3,
        kl_weights=(0.0, 1.0),
        reward_kl_weights=(0.3,),
        choice_reward_kl_weights=(0.0,),
        logged_relevance_weights=(1.0,),
        logged_relevance_kl_weights=(0.0, 1.0),
        personal_reward_kl_weights=(1.0,),
        credit_ablation_kl_weights=(1.0,),
        liked_finetuning=optimization,
        rerank_candidates=20,
        rerank_relevance_weights=(0.0, 1.0),
        logged_returns=replace(config.logged_returns, optimization=optimization),
        importance_weight_caps=(0.5,),
        sampled_list_counts=(2,),
        kl_budgets=(0.5,),
        estimator_kl_weights=(1.0,),
        leave_one_out_rollouts=2,
        high_learning_rate_factor=10.0,
        sampled_feedback_kl_weights=(1.0,),
        relative_reward_kl_weights=(1.0,),
        diversity_weights=(1.0,),
        diversity_kl_weights=(1.0,),
        masked_feedback_kl_weights=(1.0,),
        return_label_check=True,
        capacity_kl_budgets=(0.5,),
        low_ranks=(1,),
        group_relative_kl_budgets=(0.5,),
        analyse_policy_change=True,
        guidance_strengths=(1.0,),
        guidance_discounts=(0.0, 0.9),
        guidance_candidates=8,
        guidance_batch_size=32,
        continuation_training=replace(config.continuation_training, num_steps=3, histories_per_step=16),
        reward_weighted_temperatures=(1.0,),
        preference_betas=(0.1,),
        preference_finetuning=optimization,
        multi_token_lists=True,
        multi_token_optimization=optimization,
        multi_token_targets_per_head=None if config.multi_token_targets_per_head is None else 16,
        interpolation_factors=(0.5,),
        continued_training_control=True,
    )
    return replace(shrunk, **switched_off)


def run_smoke_test(
    raw_events: pl.DataFrame,
    config: ExperimentConfig,
    home: str | Path,
    device: torch.device,
    max_users: int = 200,
    reactions: pl.DataFrame | None = None,
    item_metadata: pl.DataFrame | None = None,
) -> None:
    """Runs all stages of `config`, shrunk, on the first `max_users` users: finds out in minutes whether anything is broken.

    A second session then starts from the artifacts of the first with one more anchor weight in its sweep, the
    way a session tops up an earlier one. `max_users` must leave enough data after filtering; datasets with
    short histories need more users.
    """
    home = Path(home)
    first_session = smoke_test_config(config, max_users)
    Experiment(first_session, home / "first_session", device, show_figures=False).run(raw_events, reactions, item_metadata)

    second_session = replace(first_session, kl_weights=(*first_session.kl_weights, 0.5))
    reuse_artifacts(home / "first_session", home / "second_session", second_session)
    continued = Experiment(second_session, home / "second_session", device, show_figures=False)
    continued.prepare_data(raw_events, reactions)
    continued.pretrain_policy()
    continued.train_reward_models()
    continued.finetune()
    continued.sweep_kl_weight()


def reuse_artifacts(previous_home: str | Path, home: str | Path, config: ExperimentConfig) -> None:
    """Starts `home` from the interaction log, the models and the finished runs of an earlier session, so that only what is missing is trained.

    A checkpoint is found by the name of its run, and the name does not say which base recommender, reward
    models or optimiser settings produced it. The earlier session must therefore have had the configuration
    `config` in everything but the grids of its sweeps (`RUN_GRIDS`).
    """
    previous_home, home = Path(previous_home), Path(home)
    absent = [part for part in (CONFIG_RECORD, "data", "artifacts") if not (previous_home / part).exists()]
    if absent:
        raise FileNotFoundError(f"{previous_home} is not the home directory of a session: it has no {', '.join(absent)}")
    previous = json.loads((previous_home / CONFIG_RECORD).read_text())
    current = json.loads(json.dumps(asdict(config)))
    differing = sorted(name for name in current if name not in RUN_GRIDS and previous.get(name) != current[name])
    if differing:
        raise ValueError(f"{previous_home} was produced with a different {', '.join(differing)}: its checkpoints do not fit this configuration")
    for part in ("data", "artifacts"):
        copied = home / part
        shutil.copytree(previous_home / part, copied, dirs_exist_ok=True, copy_function=shutil.copyfile)
        # An attached output is read-only, and a copied directory inherits that; this session has to write into it.
        for directory in (copied, *(path for path in copied.rglob("*") if path.is_dir())):
            directory.chmod(directory.stat().st_mode | stat.S_IWUSR)


def _timed(stage):
    """Makes a stage of `Experiment` record how long it took, in the "stage_minutes" report."""

    @functools.wraps(stage)
    def timed_stage(experiment: Experiment, *args, **kwargs):
        started = time.monotonic()
        try:
            return stage(experiment, *args, **kwargs)
        finally:
            experiment._record_stage_time(stage.__name__, time.monotonic() - started)

    return timed_stage


class Experiment:
    def __init__(
        self,
        config: ExperimentConfig,
        home: str | Path,
        device: torch.device,
        show_figures: bool = True,
        time_budget_hours: float | None = None,
    ) -> None:
        """`time_budget_hours` counts from now. Sweep runs that would not finish inside it are skipped and
        reported, so a session with a hard time limit ends with whatever was completed instead of with nothing."""
        self.config = config
        self.device = device
        self._deadline = None if time_budget_hours is None else time.monotonic() + time_budget_hours * 3600
        self._seconds_per_rl_step: float | None = None
        self.home = Path(home)
        self.log_directory = self.home / "data" / "processed"
        self.checkpoints = self.home / "artifacts"
        _bind_directory_to_config(self.home, config)
        self.reports = ReportWriter(self.checkpoints / "reports", show_figures)
        # The reports are archived on their own; with the configuration next to them they say what produced them.
        (self.reports.directory / CONFIG_RECORD).write_text((self.home / CONFIG_RECORD).read_text())
        self._examples: dict[str, EvalExamples] = {}
        self._test_evaluations: dict[str, PolicyEvaluation] = {}
        self._validation_metrics: dict[str, dict[str, float]] = {}
        self._reranked_evaluations: dict[str, ListEvaluation] = {}
        self._stage_seconds: dict[str, float] = {}

    def run(self, raw_events: pl.DataFrame, reactions: pl.DataFrame | None = None, item_metadata: pl.DataFrame | None = None) -> None:
        """All stages in order. `reactions` and `item_metadata` are what a dataset records beyond its interactions, if anything."""
        self.prepare_data(raw_events, reactions)
        self.pretrain_policy()
        self.train_reward_models()
        self.compare_baselines()
        self.finetune()
        self.sweep_kl_weight()
        self.sweep_kl_budget()
        self.compare_advantage_estimators()
        self.compare_simulated_feedback()
        self.explain_reward_gain()
        self.compare_alternatives_to_rl()
        self.sweep_rewards()
        self.sweep_relevance_in_reward()
        self.sweep_personal_rewards()
        self.sweep_relative_reward()
        self.compare_list_reward()
        self.ablate_sequence_credit()
        self.compare_feedback_masking()
        self.analyse_multistep()
        self.analyse_users_and_items(item_metadata)
        self.diagnose_return_label()
        self.restrict_finetuning_capacity()
        self.compare_group_relative_advantages()
        self.decompose_policy_change()
        self.align_at_inference()
        self.learn_from_logged_preferences()
        self.predict_lists_without_feedback()
        self.interpolate_weights()
        self.control_for_continued_training()

    # ---- stage 1: data -------------------------------------------------------------------------------

    @_timed
    def prepare_data(self, raw_events: pl.DataFrame, reactions: pl.DataFrame | None = None) -> pd.Series:
        """Builds the interaction log from raw events and, where the dataset has them, likes and dislikes (see `gptrec.data.preprocess`), and summarises it."""
        log_is_complete = (self.log_directory / "split.json").exists()
        if not log_is_complete:
            if self.config.max_users is not None:
                raw_events = _events_of_first_users(raw_events, self.config.max_users)
            build_interaction_log(raw_events, self.config.preprocessing, reactions).save(self.log_directory)
        self.reports.figure("feedback_overview", plots.feedback_overview(self.log))
        return self.reports.table("data_summary", _log_summary(self.log))

    # ---- stage 2: base recommender -------------------------------------------------------------------

    @_timed
    def pretrain_policy(self) -> pd.DataFrame:
        """Trains the base recommender. Returns its test relevance, ranked and generated, next to the popularity floor."""
        if not self._pretrained_policy_path.exists():
            training = PolicyTrainingConfig(self.config.transformer, self.config.policy_optimization, self.config.window_stride)
            policy, history = train_policy(self.log, training, self.device, extra_validation=self._validation_relevance)
            save_model(policy, self._pretrained_policy_path)
            history = self.reports.table("policy_training", pd.DataFrame(history).set_index("epoch"))
            self.reports.figure("policy_training", plots.small_multiples(history, x_label="epoch"))

        evaluation = self._evaluate(self.pretrained_policy, "test", reward_models={})
        popularity_floor = popularity_baseline_metrics(self.examples("test"), self.log.items, self.config.evaluation_config())
        relevance = pd.DataFrame({
            "most popular": list_metrics(popularity_floor, "top_k"),
            "GPT top-k": list_metrics(evaluation.metrics, "top_k"),
            "GPT next-k": list_metrics(evaluation.metrics, "next_k"),
        })
        return self.reports.table("policy_test_relevance", relevance)

    # ---- stage 3: reward models ----------------------------------------------------------------------

    @_timed
    def train_reward_models(self) -> pd.DataFrame:
        """Trains two reward models on disjoint halves of the users. Returns their played-ratio quality on the test period.

        "train" will produce the rewards during RL. "heldout" never influences the policy and only judges it:
        a reward gain that "heldout" does not confirm is exploitation of the errors of "train".
        """
        self._train_reward_model_pair("", self._reward_model_training())

        test_windows = EventWindows.for_heldout(self.log, "test", self.config.transformer.max_positions)
        predictions = {
            name: predict_heldout_feedback(reward_model, test_windows, self.config.evaluation_batch_size, self.device)
            for name, reward_model in self.reward_models.items()
        }
        items = self.log.items
        self.reports.table("reward_model_retention", self._retention_quality(test_predictions=predictions))
        self.reports.table("reward_model_reactions", pd.concat({name: reaction_quality(predicted) for name, predicted in predictions.items()}, names=["reward model", "signal"]))
        self.reports.table("reward_model_choice", pd.DataFrame({name: choice_quality(predicted) for name, predicted in predictions.items()}).T)
        # Which signals differ between the items a user could be shown, and which of them know about relevance.
        self.reports.table("reward_signal_spread", signal_spread_over_candidates(self._scored_test_candidates, self.examples("test").future_items))
        calibration = {name: calibration_table(predicted.played_ratio, predicted.predicted_played_ratio) for name, predicted in predictions.items()}
        self.reports.figure("reward_model_calibration", plots.calibration_curves(calibration))
        # Bias per item group shows which items the reward model flatters - the items RL will be drawn to.
        self.reports.table("reward_model_error_by_popularity_decile", error_by_group(predictions["train"], _decile(items.train_popularity)))
        self.reports.table("reward_model_error_by_duration_decile", error_by_group(predictions["train"], _decile(items.duration_seconds)))
        self.reports.table("reward_model_quality_by_origin", quality_by_origin(predictions["train"]))
        agreement = np.corrcoef(predictions["train"].predicted_played_ratio, predictions["heldout"].predicted_played_ratio)[0, 1]
        self.reports.table("reward_model_agreement", pd.Series({"correlation of predicted played ratio": agreement}))

        relative_ratio_quality = pd.concat({name: relative_ratio_report(predicted) for name, predicted in predictions.items()}, names=["reward model", "predictor"])
        self.reports.table("reward_model_relative_ratio", relative_ratio_quality)
        played_ratio_quality = pd.concat(
            {name: reward_model_report(predicted, items) for name, predicted in predictions.items()},
            names=["reward model", "predictor"],
        )
        return self.reports.table("reward_model_played_ratio", played_ratio_quality)

    # ---- stage 4: baselines --------------------------------------------------------------------------

    @_timed
    def compare_baselines(self) -> pd.DataFrame:
        """Fits the baselines and returns their test results next to the GPT recommender's, one column per model.

        Every model ranks the whole catalogue from the same histories under the same item constraints. The last
        column is the GPT recommender generating its list item by item instead of ranking.
        """
        baselines, histories = fit_baselines(
            self.log,
            self.config.transformer,
            self.config.baselines,
            self.config.window_stride,
            self.device,
            self.checkpoints / "baselines",
            extra_validation=self._validation_ranking_quality,
        )
        for name, history in histories.items():
            self.reports.table(f"baseline_training_{_file_name(name)}", pd.DataFrame(history).set_index("epoch"))
        rankers: dict[str, NextItemScorer] = baselines | {"GPT top-k": self.pretrained_policy}
        results = {name: self._evaluate_ranking(ranker).metrics for name, ranker in rankers.items()}
        results["GPT next-k"] = self._test_evaluation("pretrained", self.pretrained_policy).generated_lists().metrics
        return self.reports.table("baselines", pd.DataFrame(results))

    # ---- stage 5: RL fine-tuning ---------------------------------------------------------------------

    @_timed
    def finetune(self) -> pd.DataFrame:
        """Fine-tunes the recommender with RL. Returns every test metric of the base and the fine-tuned recommender."""
        if not self._rl_policy_path.exists():
            policy, records = self._run_rl(self.config.rl_config(), evaluate=self._monitor)
            save_model(policy, self._rl_policy_path)
            records = self.reports.table("rl_training", pd.DataFrame(records).set_index("step"))
            smoothed = records.reindex(columns=TRAINING_CURVES).rolling(TRAINING_CURVE_SMOOTHING_STEPS, min_periods=1).mean()
            self.reports.figure("rl_training_sampling", plots.small_multiples(smoothed, x_label="RL iteration"))
            self.reports.figure("rl_training_validation", plots.small_multiples(records.filter(like="validation/").dropna(), x_label="RL iteration"))

        pretrained, rl = self._test_evaluation("pretrained", self.pretrained_policy), self._test_evaluation("rl", self.rl_policy)
        pretrained.metrics.setdefault(REALISED_KL, 0.0)
        if REALISED_KL not in rl.metrics:
            rl.metrics |= self._end_of_training("rl_training")
        exposure = _exposure_change(self.log, before=pretrained, after=rl)
        self.reports.table("rl_exposure_change", exposure)
        attributes = ["popularity", "duration_seconds", "mean_played_ratio"]
        rank_correlation = exposure[attributes].rank().corrwith(exposure.exposure_change.rank())
        self.reports.table("rl_exposure_change_drivers", rank_correlation)
        self.reports.table("rl_list_overlap_with_pretrained", pd.Series({
            "next_k": list_overlap(rl.generated.actions, pretrained.generated.actions),
            "top_k": list_overlap(rl.top_k, pretrained.top_k),
        }))

        significance = paired_comparison(pretrained.generated_lists(), rl.generated_lists(), self.examples("test"), self.log.items, self.config.evaluation_config())
        self.reports.table("rl_vs_pretrained_significance", significance)

        comparison = metrics_table({"pretrained": pretrained, "rl": rl})
        comparison["relative_change"] = comparison["rl"] / comparison["pretrained"] - 1
        return self.reports.table("rl_vs_pretrained", comparison)

    # ---- stage 6: trade-off, reward design, alternatives to RL, multi-step generation ----------------

    @_timed
    def sweep_kl_weight(self) -> pd.DataFrame:
        """One fine-tuning run per KL weight: the frontier between relevance and predicted reward."""
        metrics = self._sweep("kl_sweep", self._kl_sweep_runs)
        k = max(self.config.ks)
        reward_rows = {judge: f"predicted/{judge}/played_ratio" for judge in self.reward_models}
        frontier = plots.reward_relevance_frontier(metrics, [f"top_k/all/ndcg@{k}", f"next_k/liked/ndcg@{k}"], reward_rows)
        self.reports.figure("kl_sweep_frontier", frontier)
        # The anchor weight means different things under different estimators and datasets; the distance actually travelled does not.
        runs = {"RL, KL sweep": _runs_table({"pretrained": self._pretrained_lists} | self._generated_lists_of(self._kl_sweep_runs)).sort_values(REALISED_KL)}
        measures = [f"all/ndcg@{k}", "predicted/heldout/played_ratio", "profile/catalog_coverage"]
        self.reports.figure("kl_sweep_by_realised_kl", plots.trade_off(runs, reward_column=REALISED_KL, measure_columns=measures))
        if self.config.evaluate_sweeps_on_validation:
            self.reports.table("kl_sweep_selection", self._selection_on_validation(["pretrained", *self._kl_sweep_runs]))
        return metrics.reindex(self._summary_rows)

    @_timed
    def sweep_kl_budget(self) -> pd.DataFrame:
        """One fine-tuning run per KL budget: the anchor weight adapts until the policy sits that far from the base recommender.

        An anchor weight is measured against the standardised advantages, so the same weight is a different
        constraint under another estimator, reward or dataset. A budget is the constraint itself. If the runs
        land on the frontier of the weight sweep, the budget is a way of walking that frontier in comparable steps.
        """
        if not self.config.kl_budgets:
            return pd.DataFrame()
        metrics = self._sweep("kl_budget_sweep", self._kl_budget_runs)
        curves = {
            "fixed anchor weight": self._generated_lists_of(self._kl_sweep_runs),
            "KL budget": self._generated_lists_of(self._kl_budget_runs),
            "pretrained": {"pretrained": self._pretrained_lists},
        }
        self._trade_off_figures("kl_budget_sweep", curves, signals=JUDGED_SIGNALS[:1])
        return metrics.reindex(self._summary_rows)

    @_timed
    def compare_advantage_estimators(self) -> pd.DataFrame:
        """Does the anchor weight mean the same under another advantage estimator, and do the estimators differ at equal distance?

        "leave-one-out" replaces the value model by the mean return of the other lists sampled for the same
        history. The number of lists per iteration, the learning rate and the passes over each batch are those
        of the KL sweep, so only the baseline differs from its run at the same weight. Under a KL budget both
        estimators end equally far from the base recommender and are compared on what they found there. One
        more run multiplies the policy's learning rate, to see whether PPO's clipping ever takes part.
        """
        rollouts = self.config.leave_one_out_rollouts
        leave_one_out = {
            "advantage_estimator": AdvantageEstimator.LEAVE_ONE_OUT,
            "rollouts_per_history": rollouts,
            "histories_per_step": self.config.rl.histories_per_step // rollouts,
        }
        reward, main_weight = self.config.rl.reward, self.config.rl.reference_kl_weight
        at_equal_weight = {
            f"leave-one-out, kl={weight}": self.config.sweep_run(reward, weight, **leave_one_out)
            for weight in sorted(self.config.estimator_kl_weights, reverse=True)
        }
        at_equal_distance = {
            f"leave-one-out, kl_budget={budget}": self.config.sweep_run(reward, main_weight, target_kl=budget, **leave_one_out)
            for budget in (sorted(self.config.kl_budgets) if at_equal_weight else [])
        }
        faster = {}
        if self.config.high_learning_rate_factor is not None:
            factor = self.config.high_learning_rate_factor
            faster[f"learning rate x{factor}, kl={main_weight}"] = self.config.sweep_run(reward, main_weight, learning_rate=factor * self.config.rl.learning_rate)
        if not (at_equal_weight or faster):
            return pd.DataFrame()

        metrics = self._sweep("advantage_estimators", at_equal_weight | at_equal_distance | faster)
        curves = {
            "value model, fixed anchor weight": self._generated_lists_of(self._kl_sweep_runs),
            "value model, KL budget": self._generated_lists_of(self._kl_budget_runs),
            "leave-one-out, fixed anchor weight": self._generated_lists_of(at_equal_weight),
            "leave-one-out, KL budget": self._generated_lists_of(at_equal_distance),
            "higher learning rate": self._generated_lists_of(faster),
            "pretrained": {"pretrained": self._pretrained_lists},
        }
        k = max(self.config.ks)
        by_distance = {method: _runs_table(runs).sort_values(REALISED_KL) for method, runs in curves.items() if runs}
        measures = [f"all/ndcg@{k}", "predicted/heldout/played_ratio", "profile/catalog_coverage"]
        self.reports.figure("advantage_estimators_by_realised_kl", plots.trade_off(by_distance, reward_column=REALISED_KL, measure_columns=measures))
        return metrics.reindex(self._summary_rows)

    @_timed
    def compare_simulated_feedback(self) -> pd.DataFrame:
        """Fine-tuning with a sampled response to the generated items instead of the expected one.

        The expected played ratio of an item is typically near one half, a response the reward model has hardly
        seen: real ones are close to 0 or to 1. Sampling "finished" or "skipped" with the predicted probability
        keeps the simulated history inside what the model was trained on. Every policy is then judged under each
        assumption, which separates what a policy learned from how its lists are scored.
        """
        anchor_weights = sorted(self.config.sampled_feedback_kl_weights, reverse=True)
        if not anchor_weights:
            return pd.DataFrame()
        reward = self.config.rl.reward
        with_expected = {f"kl={weight}": self.config.sweep_run(reward, weight) for weight in anchor_weights}
        with_sampled = {
            f"sampled feedback, kl={weight}": self.config.sweep_run(reward, weight, simulated_feedback=SimulatedFeedback.SAMPLED) for weight in anchor_weights
        }
        metrics = self._sweep("simulated_feedback_sweep", with_expected | with_sampled)

        predicted_reward = {}
        for run_name in [*with_expected, *with_sampled]:
            if self._sweep_checkpoint(run_name).exists():
                policy = load_policy(self._sweep_checkpoint(run_name), self.device)
                predicted_reward[run_name] = {
                    mode.value: self._evaluate(policy, "test", simulated_feedback=mode).metrics["predicted/heldout/played_ratio"] for mode in SIMULATION_ASSUMPTIONS
                }
        by_assumption = pd.DataFrame(predicted_reward).T.rename_axis("fine-tuning run").rename_axis("feedback assumed when judging", axis=1)
        self.reports.table("simulated_feedback_cross_judged", by_assumption)
        return metrics.reindex(self._summary_rows)

    @_timed
    def explain_reward_gain(self) -> pd.DataFrame:
        """Whose reward is it? Returns, per method, how much of its predicted gain other users would give the same lists.

        A list earns `played_ratio` with the user it was made for and `played_ratio_for_other_user` with somebody
        else. If a method's gain over the base recommender shows up in full for other users, the method learned
        which items everybody finishes, not what this user will finish. The last rows re-rank the base
        recommender's candidates by each item's mean played ratio in the train period - no reward model and no
        RL - to see how far that knowledge alone goes.
        """
        k = max(self.config.ks)
        by_reward_model = self._reranked("re-ranking", self._scored_test_candidates, self.config.rl.reward)
        by_item_mean = self._reranked("re-ranking by item mean", self._candidates_scored_by_item_mean, RewardSpec())
        kl_sweep = self._generated_lists_of(self._kl_sweep_runs)
        methods = {"pretrained": self._pretrained_lists, "RL": self._rl_lists} | kl_sweep | by_reward_model | by_item_mean
        metrics = _runs_table(methods)
        with_own_user, with_other_user = metrics["predicted/heldout/played_ratio"], metrics["predicted/heldout/played_ratio_for_other_user"]
        gain = with_own_user - with_own_user["pretrained"]
        gain_with_other_user = with_other_user - with_other_user["pretrained"]
        decomposition = pd.DataFrame({
            f"all/ndcg@{k}": metrics[f"all/ndcg@{k}"],
            f"liked/ndcg@{k}": metrics[f"liked/ndcg@{k}"],
            "train-period mean played ratio of the listed items": metrics["profile/mean_item_played_ratio"],
            "predicted played ratio": with_own_user,
            "predicted played ratio for other users": with_other_user,
            "gain": gain,
            "gain for other users": gain_with_other_user,
            "personal share of the gain": (1 - gain_with_other_user / gain).where(gain.abs() > 1e-9),
        })
        curves = {
            "RL, KL sweep": kl_sweep,
            "re-ranking by the reward model": by_reward_model,
            "re-ranking by item mean": by_item_mean,
            "pretrained": {"pretrained": self._pretrained_lists},
            "RL, main run": {"RL": self._rl_lists},
        }
        self._trade_off_figures("reward_gain", curves, signals=JUDGED_SIGNALS[:2])
        return self.reports.table("reward_gain_decomposition", decomposition)

    @_timed
    def compare_alternatives_to_rl(self) -> pd.DataFrame:
        """Ways of raising the reward without on-policy RL, next to RL. Returns one column per method and setting.

        "liked-only fine-tuning" continues next-item training on liked items only: logged feedback, no reward model.
        "off-policy REINFORCE" (Matveev et al.) weighs every logged item by the logged return that followed it.
        "re-ranking" reorders the base recommender's candidates by predicted reward.
        "best of N sampled lists" (the test-time scaling of Matveev et al.) samples several lists and serves
        the one with the highest predicted return.
        The figures draw relevance against what the reward model that no method used predicts: RL earns its
        complexity only where its curve lies above the others.
        """
        pretrained, rl = self._pretrained_lists, self._test_evaluation("rl", self.rl_policy).generated_lists()
        liked_only = self._test_evaluation("liked-only", self._liked_only_policy()).generated_lists()
        off_policy = {
            f"off-policy REINFORCE, cap={cap}": self._test_evaluation(f"off-policy, cap={cap}", self._logged_return_policy(cap)).generated_lists()
            for cap in self.config.importance_weight_caps
        }
        reranking = self._reranked("re-ranking", self._scored_test_candidates, self.config.rl.reward)
        best_of_sampled = {f"best of {count} sampled lists": self._evaluate_lists(self._best_of_sampled_lists(count)) for count in self.config.sampled_list_counts}

        significance = paired_comparison(pretrained, liked_only, self.examples("test"), self.log.items, self.config.evaluation_config())
        self.reports.table("liked_only_vs_pretrained_significance", significance)
        curves = {
            "RL, KL sweep": self._generated_lists_of(self._kl_sweep_runs),
            "re-ranking, weight sweep": reranking,
            "best of N sampled lists": best_of_sampled,
            "off-policy REINFORCE, cap sweep": off_policy,
            "pretrained": {"pretrained": pretrained},
            "RL, main run": {"RL": rl},
            "liked-only fine-tuning": {"liked-only fine-tuning": liked_only},
        }
        self._trade_off_figures("alternatives_to_rl", curves, signals=JUDGED_SIGNALS[:2])

        methods = {"pretrained": pretrained, "RL": rl, "liked-only fine-tuning": liked_only} | off_policy | reranking | best_of_sampled
        return self.reports.table("alternatives_to_rl", _runs_table(methods).T)

    @_timed
    def sweep_rewards(self) -> pd.DataFrame:
        """One fine-tuning run per alternative reward and anchor weight.

        The retention rewards pay for what the user is predicted to do after the item: continue the session, or
        come back for another one. Each figure puts relevance against one signal as the held-out reward model
        predicts it, one line per reward running from the strongest anchor to the weakest.
        """
        anchor_weights = sorted(self.config.reward_kl_weights, reverse=True)
        runs_by_reward = {
            reward_name: {f"{reward_name}, kl={weight}": self.config.sweep_run(reward, weight) for weight in anchor_weights}
            for reward_name, reward in self.config.rewards
        }
        # A reward paid only after the last item reaches the earlier items through the return alone; without
        # sequence credit they learn nothing, so here the credit ablation is expected to fail.
        runs_by_reward |= {
            f"{reward_name}, discount=0": {f"{reward_name}, discount=0, kl={weight}": self.config.sweep_run(reward, weight, discount=0.0) for weight in anchor_weights}
            for reward_name, reward in self.config.rewards
            if reward.rewards_the_end_of_the_list
        }
        metrics = self._sweep("reward_sweep", {name: run for runs in runs_by_reward.values() for name, run in runs.items()})
        curves = {"played ratio": self._generated_lists_of(self._kl_sweep_runs)}
        curves |= {reward_name: self._generated_lists_of(runs) for reward_name, runs in runs_by_reward.items()}
        curves["pretrained"] = {"pretrained": self._pretrained_lists}
        self._trade_off_figures("reward_sweep", curves, signals=(*JUDGED_SIGNALS, *self._signals_rewarded_by(reward for _, reward in self.config.rewards)))
        return metrics.reindex(self._summary_rows)

    @staticmethod
    def _signals_rewarded_by(rewards) -> tuple[str, ...]:
        """The predicted signals, beyond JUDGED_SIGNALS, that some of `rewards` pay for."""
        signals = []
        for reward in rewards:
            signals += ["liked"] if reward.liked_weight or reward.disliked_weight else []
            signals += ["adopts"] if reward.adoption_weight else []
            signals += ["returns_after_list"] if reward.rewards_the_end_of_the_list else []
        return tuple(dict.fromkeys(signals))

    @_timed
    def sweep_relevance_in_reward(self) -> pd.DataFrame:
        """Can the reward itself ask for relevance, and what does the anchor add to it?

        "chosen ratio" weighs the predicted played ratio by the reward model's own estimate that the user
        would take the item up at all. "logged relevance" is the reward of Petrov & Macdonald: the item is among
        the user's real next items. It runs alone with no anchor (their NDCG tuning), and added to the played
        ratio at several weights and several anchor weights: the reward term protects hits on the user's real
        continuation, the anchor protects everything else the base recommender does, such as its variety.
        """
        chosen_ratio = {f"chosen ratio, kl={weight}": self.config.sweep_run(CHOSEN_RATIO, weight) for weight in sorted(self.config.choice_reward_kl_weights, reverse=True)}
        logged_relevance_only = {"logged relevance, kl=0.0": self.config.sweep_run(LOGGED_RELEVANCE, kl_weight=0.0)}
        anchor_weights = sorted(self.config.logged_relevance_kl_weights, reverse=True)
        ratio_and_logged_relevance = {
            f"ratio + {weight} x logged relevance": {
                f"ratio + {weight} x logged relevance, kl={anchor}": self.config.sweep_run(RewardSpec(logged_relevance_weight=weight), anchor)
                for anchor in anchor_weights
            }
            for weight in sorted(self.config.logged_relevance_weights, reverse=True)
        }
        anchored_runs = {name: run for anchor_sweep in ratio_and_logged_relevance.values() for name, run in anchor_sweep.items()}
        metrics = self._sweep("relevance_reward_sweep", chosen_ratio | logged_relevance_only | anchored_runs)
        curves = {"played ratio, anchor sweep": self._generated_lists_of(self._kl_sweep_runs), "chosen ratio, anchor sweep": self._generated_lists_of(chosen_ratio)}
        curves |= {f"{reward_name}, anchor sweep": self._generated_lists_of(anchor_sweep) for reward_name, anchor_sweep in ratio_and_logged_relevance.items()}
        curves |= {"logged relevance only, no anchor": self._generated_lists_of(logged_relevance_only), "pretrained": {"pretrained": self._pretrained_lists}}
        self._trade_off_figures("relevance_reward_sweep", curves, signals=JUDGED_SIGNALS[:2])
        return metrics.reindex(self._summary_rows)

    @_timed
    def sweep_personal_rewards(self) -> pd.DataFrame:
        """Fine-tuning for the part of the played ratio that depends on who receives the list.

        "personal ratio" pays the predicted played ratio minus what the same reward model predicts for the same
        list shown to another user; a policy that ignores its user earns zero, so collapsing onto the items
        everybody finishes no longer pays. "ratio above item mean" subtracts the item's mean played ratio in the
        train period instead, which needs no second prediction but leaves the reward model's errors about
        items in the reward. Re-ranking by the personal ratio shows what the same reward gives without RL.
        """
        anchor_weights = sorted(self.config.personal_reward_kl_weights, reverse=True)
        against_other_user = {f"personal ratio, kl={weight}": self.config.sweep_run(PERSONAL_RATIO, weight) for weight in anchor_weights}
        against_item_mean = {f"ratio above item mean, kl={weight}": self.config.sweep_run(RATIO_ABOVE_ITEM_MEAN, weight) for weight in anchor_weights}
        metrics = self._sweep("personal_reward_sweep", against_other_user | against_item_mean)
        curves = {
            "played ratio, anchor sweep": self._generated_lists_of(self._kl_sweep_runs),
            "personal ratio, anchor sweep": self._generated_lists_of(against_other_user),
            "ratio above item mean, anchor sweep": self._generated_lists_of(against_item_mean),
            "re-ranking by personal ratio": self._reranked("re-ranking by personal ratio", self._scored_test_candidates, PERSONAL_RATIO),
            "pretrained": {"pretrained": self._pretrained_lists},
        }
        self._trade_off_figures("personal_reward_sweep", curves, signals=("personal_played_ratio", "played_ratio"))
        return metrics.reindex(self._summary_rows)

    @_timed
    def sweep_relative_reward(self) -> pd.DataFrame:
        """Fine-tuning for the played ratio relative to what is usual for the item.

        The plain played ratio pays for items that are easy to finish, whoever receives them: a film over a
        series, a short track over a long one. The relative ratio is the user's played ratio as a quantile among
        everyone who consumed the same item. It is 0.5 for every item on average, so a policy earns it only by
        matching items to the users who take to them more than others do. Re-ranking the base recommender's
        candidates by the same signal is the reference without RL.
        """
        anchor_weights = sorted(self.config.relative_reward_kl_weights, reverse=True)
        if not anchor_weights:
            return pd.DataFrame()
        runs = {f"relative ratio, kl={weight}": self.config.sweep_run(RELATIVE_RATIO, weight) for weight in anchor_weights}
        metrics = self._sweep("relative_reward_sweep", runs)
        reranked = self._reranked("re-ranking by relative ratio", self._scored_test_candidates, RELATIVE_RATIO)
        self.reports.table("relative_reward_reranking", _runs_table(reranked).T)
        curves = {
            "played ratio, anchor sweep": self._generated_lists_of(self._kl_sweep_runs),
            "relative ratio, anchor sweep": self._generated_lists_of(runs),
            "re-ranking by relative ratio": reranked,
            "pretrained": {"pretrained": self._pretrained_lists},
        }
        self._trade_off_figures("relative_reward_sweep", curves, signals=("relative_ratio", "played_ratio"))
        return metrics.reindex(self._summary_rows)

    @_timed
    def compare_list_reward(self) -> pd.DataFrame:
        """A reward that depends on the list as a whole: the main run's reward plus each item's distance to the items before it.

        Summed over a list, the second term is the list's mean pairwise distance in the base recommender's
        embedding space. An item's worth then depends on its neighbours, and scoring items one at a time no
        longer finds the best list. RL is compared with the greedy ways of maximising the same reward: maximal
        marginal relevance over the base recommender's candidates, which fills the list one position at a time,
        and the best of several sampled lists. The last column of the saved table is the reward itself as the
        held-out reward model sees it, per item of the list.
        """
        if not self.config.diversity_weights:
            return pd.DataFrame()
        k = max(self.config.ks)
        anchor_weights = sorted(self.config.diversity_kl_weights, reverse=True)
        rewards = {f"ratio + {weight} x distance": replace(self.config.rl.reward, diversity_weight=weight) for weight in sorted(self.config.diversity_weights)}
        rl_runs = {label: {f"{label}, kl={anchor}": self.config.sweep_run(reward, anchor) for anchor in anchor_weights} for label, reward in rewards.items()}
        metrics = self._sweep("list_reward_sweep", {name: run for runs in rl_runs.values() for name, run in runs.items()})

        methods = {}
        for label, reward in rewards.items():
            curves = {
                "RL, anchor sweep": self._generated_lists_of(rl_runs[label]),
                "greedy re-ranking, weight sweep": self._reranked(f"greedy re-ranking, {label}", self._scored_test_candidates, reward),
                "best of N sampled lists": {
                    f"best of {count} sampled lists, {label}": self._evaluate_lists(self._best_of_sampled_lists(count, reward)) for count in self.config.sampled_list_counts
                },
                "played ratio only, anchor sweep": self._generated_lists_of(self._kl_sweep_runs),
                "pretrained": {"pretrained": self._pretrained_lists},
            }
            measures = [f"all/ndcg@{k}", INTRA_LIST_DISTANCE, "profile/catalog_coverage"]
            self._trade_off_figures(f"list_reward_{_file_name(label)}", curves, signals=JUDGED_SIGNALS[:1], measures=measures)
            table = _runs_table({name: evaluation for runs in curves.values() for name, evaluation in runs.items()})
            # Over a list the distance terms add up to half its mean pairwise distance per item.
            table["list reward (held-out)"] = table["predicted/heldout/played_ratio"] + reward.diversity_weight * table[INTRA_LIST_DISTANCE] / 2
            methods[label] = table
        self.reports.table("list_reward_methods", pd.concat(methods, names=["reward", "method"]))
        return metrics.reindex(self._summary_rows)

    @_timed
    def ablate_sequence_credit(self) -> pd.DataFrame:
        """Does fine-tuning use the fact that a list is a sequence?

        "discount=0" credits every item with its own reward only, never with the rewards of the items after it.
        "horizon=1" samples one item per episode while fine-tuning, so nothing is learned from a generated
        context; its lists are still generated item by item when evaluated. Both are compared with runs that
        credit every item with the whole rest of the list: at the same anchor weights, and under the same KL
        budgets. Changing the credit changes what a weight means, so only the budget runs compare policies
        that moved equally far; for one-item episodes the distance is that of the first position only.
        """
        anchor_weights = sorted(self.config.credit_ablation_kl_weights, reverse=True)
        budgets = sorted(self.config.kl_budgets)
        reward, one_item, main_weight = self.config.rl.reward, replace(self.config.rl.generation, horizon=1), self.config.rl.reference_kl_weight
        own_reward_only = {f"discount=0, kl={weight}": self.config.sweep_run(reward, weight, discount=0.0) for weight in anchor_weights}
        own_reward_only |= {f"discount=0, kl_budget={budget}": self.config.sweep_run(reward, main_weight, discount=0.0, target_kl=budget) for budget in budgets}
        one_item_episodes = {f"horizon=1, kl={weight}": self.config.sweep_run(reward, weight, generation=one_item) for weight in anchor_weights}
        one_item_episodes |= {f"horizon=1, kl_budget={budget}": self.config.sweep_run(reward, main_weight, generation=one_item, target_kl=budget) for budget in budgets}
        metrics = self._sweep("sequence_credit_ablation", own_reward_only | one_item_episodes)
        whole_rest_of_the_list = [f"kl={weight}" for weight in anchor_weights] + [f"kl_budget={budget}" for budget in budgets]
        curves = {
            "whole rest of the list": self._generated_lists_of(whole_rest_of_the_list),
            "own reward only (discount 0)": self._generated_lists_of(own_reward_only),
            "one item per episode (horizon 1)": self._generated_lists_of(one_item_episodes),
            "pretrained": {"pretrained": self._pretrained_lists},
        }
        self._trade_off_figures("sequence_credit_ablation", curves, signals=JUDGED_SIGNALS[:1])
        return metrics.reindex(self._summary_rows)

    @_timed
    def analyse_multistep(self) -> pd.DataFrame:
        """How the generated list changes along its positions. Returns NDCG of ranked against generated lists.

        From the second position on, the recommender conditions on items the user never consumed and the
        reward model on feedback that was never observed; drift shows as a trend over positions.
        """
        examples, items = self.examples("test"), self.log.items
        evaluations = {"pretrained": self._test_evaluation("pretrained", self.pretrained_policy), "rl": self._test_evaluation("rl", self.rl_policy)}
        step_tables = {name: self.reports.table(f"per_step_{name}", per_step_table(evaluation, examples, items)) for name, evaluation in evaluations.items()}
        measures = [
            "in_true_continuation",
            "policy_entropy",
            "predicted_played_ratio/train",
            "predicted_is_chosen/train",
            "predicted_returns/train",
            "played_ratio_disagreement/train_vs_heldout",
        ]
        by_measure = {measure: pd.DataFrame({name: table[measure] for name, table in step_tables.items()}) for measure in measures}
        self.reports.figure("per_step", plots.series_per_panel(by_measure, x_label="position in the generated list"))

        # If the predicted reward of later positions depends on what the reward model is told about the
        # generated items, the multi-step estimate is an artefact of the simulation.
        predicted_by_mode = {}
        for mode in SIMULATION_ASSUMPTIONS:
            evaluation = self._evaluate(self.rl_policy, "test", simulated_feedback=mode)
            predicted_by_mode[mode.value] = per_step_table(evaluation, examples, items)["predicted_played_ratio/train"]
        predicted_by_mode = self.reports.table("per_step_by_simulated_feedback", pd.DataFrame(predicted_by_mode))
        by_mode_figure = plots.series_per_panel({"predicted played ratio (RL policy)": predicted_by_mode}, x_label="position in the generated list")
        self.reports.figure("per_step_by_simulated_feedback", by_mode_figure)

        # A generated item the user never consumed is part of the context of the items after it; a ranked list has no such context.
        lists = {f"{name}, {kind}": lists for name, evaluation in evaluations.items() for kind, lists in (("ranked", evaluation.top_k), ("generated", evaluation.generated.actions))}
        self.reports.table("position_hits", position_hits(lists, examples.future_items))

        # Which assumption about the response to an item nobody consumed agrees with what users did, on their real next items.
        reward_models = {"train": self.reward_models["train"]}
        if self._reward_model_path("masked_train").exists():
            reward_models["masked train"] = self.masked_reward_models["masked_train"]
        # A history leaves room for a list of `horizon` items, so no more real items than that fit after it.
        steps = min(FEEDBACK_ASSUMPTION_STEPS, self.config.horizon)
        assumptions = {}
        for name, reward_model in reward_models.items():
            modes = [SimulatedFeedback.LOGGED, *SIMULATION_ASSUMPTIONS] + ([SimulatedFeedback.UNKNOWN] if reward_model.config.unknown_feedback else [])
            assumptions[name] = feedback_assumptions_on_real_sequences(
                reward_model, examples, modes, steps, self.config.evaluation_batch_size, self.device, mixed_precision=True
            )
        self.reports.table("feedback_assumptions_on_real_sequences", pd.concat(assumptions, axis=1, names=["reward model", "quantity"]))

        generation_gap = pd.DataFrame({
            (name, list_name): {k: evaluation.metrics[f"{list_name}/all/ndcg@{k}"] for k in self.config.ks}
            for name, evaluation in evaluations.items()
            for list_name in ("top_k", "next_k")
        }).rename_axis("k")
        return self.reports.table("ndcg_top_k_vs_next_k", generation_gap)

    @_timed
    def compare_feedback_masking(self) -> pd.DataFrame:
        """A reward model that can be told "the response to this item is unknown", against models that must be told an assumed one.

        The masked reward models are trained like the others, on the same halves of the users, except that in a
        share of the training windows the responses to the last events are hidden (see `hide_last_responses`).
        A generated item can then be appended as what it is, shown with an unknown response, and the simulation
        needs no assumption. The policy is fine-tuned against the masked training model with unknown feedback and
        judged like every other run; the runs of the anchor sweep are also judged by the masked held-out model.
        """
        anchor_weights = sorted(self.config.masked_feedback_kl_weights, reverse=True)
        if not anchor_weights:
            return pd.DataFrame()
        masked = self.masked_reward_models
        test_windows = EventWindows.for_heldout(self.log, "test", self.config.transformer.max_positions)
        quality = {
            name: reward_model_report(predict_heldout_feedback(reward_model, test_windows, self.config.evaluation_batch_size, self.device), self.log.items).loc["reward model"]
            for name, reward_model in (self.reward_models | masked).items()
        }
        self.reports.table("reward_model_masked_quality", pd.DataFrame(quality).T)

        runs = {
            f"unknown feedback, kl={weight}": self.config.sweep_run(self.config.rl.reward, weight, simulated_feedback=SimulatedFeedback.UNKNOWN)
            for weight in anchor_weights
        }
        metrics = self._sweep("masked_feedback_sweep", runs, reward_model=masked["masked_train"])

        judged_by_masked_model = {}
        for run_name in ["pretrained", *self._kl_sweep_runs, *runs]:
            if run_name == "pretrained" or self._sweep_checkpoint(run_name).exists():
                policy = self.pretrained_policy if run_name == "pretrained" else load_policy(self._sweep_checkpoint(run_name), self.device)
                evaluation = self._evaluate(policy, "test", reward_models={"masked_heldout": masked["masked_heldout"]}, simulated_feedback=SimulatedFeedback.UNKNOWN)
                judged_by_masked_model[run_name] = {
                    "predicted played ratio, unknown feedback": evaluation.metrics["predicted/masked_heldout/played_ratio"],
                    "predicted played ratio, expected feedback": self._sweep_or_pretrained_metric(run_name, "predicted/heldout/played_ratio"),
                }
        self.reports.table("masked_feedback_judged", pd.DataFrame(judged_by_masked_model).T)
        return metrics.reindex(self._summary_rows)

    def _sweep_or_pretrained_metric(self, run_name: str, metric: str) -> float:
        evaluation = self._test_evaluation("pretrained", self.pretrained_policy) if run_name == "pretrained" else self._sweep_evaluation(run_name)
        return evaluation.metrics[metric]

    @_timed
    def analyse_users_and_items(self, item_metadata: pl.DataFrame | None = None) -> pd.DataFrame:
        """Who loses relevance, what the policy learned about items, and how impressions spread over the catalogue.

        Users are grouped by how many events they had before the request. The logit shift asks whether the
        fine-tuned policy changed its preferences per user or one item at a time. With `item_metadata` (item_id
        plus attributes such as the title and whether it is a film or a series), the items that gained and lost
        the most impressions are named. Returns the logit-shift table: one row per fine-tuned policy.
        """
        examples, items, k = self.examples("test"), self.log.items, max(self.config.ks)
        pretrained, rl = self._test_evaluation("pretrained", self.pretrained_policy), self._test_evaluation("rl", self.rl_policy)
        user_starts = np.asarray(self.log.user_offsets[:-1])
        test_starts, _ = self.log.period_rows("test")
        users = examples.user.numpy()
        prior_events = test_starts[users] - user_starts[users]
        main_lists = {"pretrained, ranked": pretrained.top_k, "pretrained, generated": pretrained.generated.actions, "RL, generated": rl.generated.actions}
        by_length = relevance_by_history_length(main_lists, examples, prior_events, k, self.config.evaluation_config().liked_threshold)
        self.reports.table("relevance_by_history_length", by_length)

        policies = {"RL": self.rl_policy} | {name: load_policy(self._sweep_checkpoint(name), self.device) for name in self._kl_sweep_runs if self._sweep_checkpoint(name).exists()}
        shift = pd.DataFrame({
            name: logit_shift_by_item(policy, self.pretrained_policy, examples, items, self.config.evaluation_batch_size, self.device) for name, policy in policies.items()
        }).T
        self.reports.table("logit_shift_by_item", shift)

        swept_lists = {name: self._sweep_evaluation(name).generated.actions for name in self._kl_sweep_runs if self._sweep_checkpoint(name).exists()}
        all_lists = main_lists | swept_lists
        self.reports.table("exposure_lorenz", exposure_lorenz(all_lists, items.num_items))
        concentration = {
            name: {"catalog_coverage": profile["catalog_coverage"], "item_gini": profile["item_gini"], "top_exposure_share": profile["top_exposure_share"]}
            for name, lists in all_lists.items()
            for profile in [recommendation_profile(lists, examples.history.items, items)]
        }
        self.reports.table("exposure_concentration", pd.DataFrame(concentration).T)

        if item_metadata is not None:
            item_table = self._item_table(item_metadata)
            if "content_type" in item_table:
                self.reports.table("impressions_by_content_type", impressions_by_attribute(all_lists, item_table["content_type"].to_numpy()))
            self.reports.table("largest_impression_changes", largest_impression_changes(pretrained.generated.actions, rl.generated.actions, item_table))
        return shift

    def _item_table(self, item_metadata: pl.DataFrame) -> pd.DataFrame:
        """One row per item token (row 0 is padding): the attributes of `item_metadata` next to the log's statistics of the item."""
        items = self.log.items
        tokens = pd.DataFrame({
            "item_id": items.raw_id,
            "train_popularity": items.train_popularity,
            "mean_played_ratio": items.mean_played_ratio,
            "duration_seconds": items.duration_seconds,
        })
        metadata = pd.DataFrame({column: item_metadata[column].to_numpy() for column in item_metadata.columns}).drop_duplicates("item_id")
        return tokens.merge(metadata, on="item_id", how="left").set_axis(pd.RangeIndex(len(tokens), name="item_token"))

    # ---- what the stages share -----------------------------------------------------------------------

    # ---- stage 7: what RL learns, inference-time alignment, learning from the log alone ---------------

    @_timed
    def diagnose_return_label(self) -> pd.DataFrame:
        """Whether a user's return after an event is recorded by the event or by the item that came after it.

        Data only. On a log with one row per user and item, dated by the last view, the item after an event decides
        when the next row appears; a reward for returns then pays for the bookkeeping, see `return_rate_by_duration`.
        """
        if not self.config.return_label_check:
            return pd.DataFrame()
        return self.reports.table("return_rate_by_duration", return_rate_by_duration(self.log))

    @_timed
    def restrict_finetuning_capacity(self) -> pd.DataFrame:
        """How much of what RL reaches needs more than one offset per item: fine-tuning held to item offsets or to low-rank updates.

        Every run holds the same KL budgets, so they are compared at an equal distance from the base recommender.
        Item offsets can only re-weight the catalogue, for everybody alike. Low-rank updates of the transformer change
        how a history is read, but no item's embedding. If the offsets reach the full run's predicted reward at its
        relevance, what RL learned is a re-weighting of the catalogue. If rank 1 matches full fine-tuning, a policy
        gradient carried as little information per episode as is reported for language models.
        """
        if not self.config.capacity_kl_budgets:
            return pd.DataFrame()
        reward, weight = self.config.rl.reward, self.config.rl.reference_kl_weight
        # Named and configured like the runs of `sweep_kl_budget`, so a budget both stages share is trained once.
        full = {f"kl_budget={budget}": self.config.sweep_run(reward, weight, target_kl=budget) for budget in sorted(self.config.capacity_kl_budgets)}
        capacity_runs = self._capacity_runs
        metrics = self._sweep("capacity_sweep", full | capacity_runs)
        curves = {"every weight": self._generated_lists_of(full)}
        for label in dict.fromkeys(name.split(", kl_budget=")[0] for name in capacity_runs):
            curves[f"{label} only"] = self._generated_lists_of({name: run for name, run in capacity_runs.items() if name.startswith(f"{label},")})
        curves["pretrained"] = {"pretrained": self._pretrained_lists}
        self._trade_off_figures("capacity_sweep", curves, signals=JUDGED_SIGNALS[:1])
        return metrics.reindex(self._summary_rows)

    @_timed
    def compare_group_relative_advantages(self) -> pd.DataFrame:
        """GRPO (Shao et al., 2024), the estimator of current generative-recommender RL, at the KL budgets of the other estimators.

        Group-relative advantages compare the lists sampled for one history with each other and divide by their
        spread, so every history weighs the same; nothing else differs from the leave-one-out runs.
        """
        runs = self._group_relative_runs
        if not runs:
            return pd.DataFrame()
        metrics = self._sweep("group_relative_advantages", runs)
        leave_one_out = {f"leave-one-out, kl_budget={budget}": None for budget in sorted(self.config.group_relative_kl_budgets)}
        curves = {
            "value model": self._generated_lists_of(self._kl_budget_runs),
            "leave-one-out": self._generated_lists_of(leave_one_out),
            "group-relative": self._generated_lists_of(runs),
            "pretrained": {"pretrained": self._pretrained_lists},
        }
        self._trade_off_figures("group_relative_advantages", curves, signals=JUDGED_SIGNALS[:1])
        return metrics.reindex(self._summary_rows)

    @_timed
    def decompose_policy_change(self) -> pd.DataFrame:
        """What the main run changed, split into one offset per item and the rest, and whether the reward predicted the split.

        The run's change of log-probabilities is averaged per item over validation histories. "base + RL's item
        offsets" adds those averages to the base recommender; "RL - its item offsets" removes them from the
        fine-tuned one. If the first reproduces RL and the second does little, RL learned a re-weighting of the
        catalogue. The returned table then compares, on the base recommender's candidates for every test user, the
        share of within-history variation that one effect per item explains: in the reward model's predictions,
        which the closed form pi* = pi_0 exp(r / beta) / Z turns into the optimal change of log-probabilities at any
        beta, and in what each fine-tuned recommender changed. RL above the reward's share amplified the item part.
        """
        if not self.config.analyse_policy_change:
            return pd.DataFrame()
        offsets = mean_logit_shift_per_item(self.rl_policy, self.pretrained_policy, self.examples("validation"), self.config.evaluation_batch_size, self.device)
        parts = {
            "pretrained": self._pretrained_lists,
            "RL": self._rl_lists,
            "base + RL's item offsets": self._test_evaluation("base + RL's item offsets", with_item_offsets(self.pretrained_policy, offsets)).generated_lists(),
            "RL - its item offsets": self._test_evaluation("RL - its item offsets", with_item_offsets(self.rl_policy, -offsets)).generated_lists(),
        }
        self.reports.table("policy_change_parts", _runs_table(parts).T)

        candidates, examples, batch_size = self._scored_test_candidates, self.examples("test"), self.config.evaluation_batch_size
        weights = candidates.log_probs.exp()
        reward = RewardFunction(self.config.rl.reward, self.log.items, "cpu").item_reward(
            candidates.feedback, candidates.items, feedback_for_other_user=candidates.feedback_for_other_user
        )
        shares = {"reward model's predictions (the optimum's change)": item_effect_share(reward, candidates.items, weights)}
        base_log_probs = candidate_log_probs(self.pretrained_policy, examples, candidates.items, batch_size, self.device)
        for name, policy in self._finetuned_policies().items():
            shift = candidate_log_probs(policy, examples, candidates.items, batch_size, self.device) - base_log_probs
            shares[name] = item_effect_share(shift, candidates.items, weights)
        return self.reports.table("item_effect_share", pd.Series(shares, name="within-history variation explained by one effect per item"))

    @_timed
    def align_at_inference(self) -> pd.DataFrame:
        """The base recommender steered at inference by the closed form of the anchored objective: no fine-tuning.

        Each position takes the candidate with the highest log pi_0 + Q / beta among the base recommender's most likely
        next items, after the items already chosen (see `guided_lists`). With discount 0, Q is the item's own reward;
        above 0 it adds the learned worth of what follows. Re-ranking once after the real history is the same closed
        form without either. For a reward paid only after the last item, nothing but look-ahead can see it coming.
        Returns one column per method: RL, re-ranking, and each reward, discount and strength of guidance.
        """
        strengths = sorted(self.config.guidance_strengths, reverse=True)
        if not strengths:
            return pd.DataFrame()
        guided_by_reward = {}
        for reward_name, reward in self.config.guided_rewards:
            reward_function = RewardFunction(reward, self.log.items, self.device)
            guided = {}
            for discount in self.config.guidance_discounts:
                value_model = self._continuation_value_model(reward_name, reward, discount) if discount > 0 else None
                for strength in strengths:
                    name = f"{reward_name}, guided, discount={discount}, strength={strength}"
                    guidance = GuidanceConfig(strength=strength, discount=discount, num_candidates=self.config.guidance_candidates)
                    guided[name] = self._evaluate_lists(self._guided_lists(reward_function, value_model, guidance))
            guided_by_reward[reward_name] = guided

        reranked = self._reranked("re-ranking", self._scored_test_candidates, self.config.rl.reward)
        rl_on_rewards = self._generated_lists_of({f"{name}, kl={weight}": None for name, _ in self.config.rewards for weight in self.config.reward_kl_weights})
        for (reward_name, reward), guided in zip(self.config.guided_rewards, guided_by_reward.values()):
            curves = {"RL, anchor sweep": self._generated_lists_of(self._kl_sweep_runs), "re-ranking once, real history": reranked}
            for discount in self.config.guidance_discounts:
                curves[f"guided, discount={discount}"] = {name: lists for name, lists in guided.items() if f", discount={discount}," in name}
            curves["RL on this reward"] = {name: lists for name, lists in rl_on_rewards.items() if name.startswith(f"{reward_name}, kl=")}
            curves["pretrained"] = {"pretrained": self._pretrained_lists}
            self._trade_off_figures(f"inference_time_alignment_{_file_name(reward_name)}", curves, signals=(*JUDGED_SIGNALS[:1], *self._signals_rewarded_by([reward])))

        methods = {"pretrained": self._pretrained_lists, "RL": self._rl_lists} | rl_on_rewards | reranked
        for guided in guided_by_reward.values():
            methods |= guided
        return self.reports.table("inference_time_alignment", _runs_table(methods).T)

    @_timed
    def learn_from_logged_preferences(self) -> pd.DataFrame:
        """Fine-tuning on logged responses alone, with current preference-learning methods: no reward model, no generated lists.

        Reward-weighted fine-tuning weighs each logged item by exp(response / temperature); within-user DPO learns
        from pairs of items one user consumed after the same history, one above and one below the usual response for
        it. Each method is compared with the base recommender user by user, and with RL and liked-only fine-tuning.
        Returns one column per method.
        """
        temperatures, betas = self.config.reward_weighted_temperatures, self.config.preference_betas
        if not (temperatures or betas):
            return pd.DataFrame()
        methods = {}
        for response in self.config.reward_weighted_responses:
            for temperature in temperatures:
                name = f"reward-weighted, {response}, temperature={temperature}"
                methods[name] = self._test_evaluation(name, self._reward_weighted_policy(response, temperature)).generated_lists()
        for beta in betas:
            name = f"within-user DPO, beta={beta}"
            methods[name] = self._test_evaluation(name, self._logged_preference_policy(beta)).generated_lists()

        pretrained, evaluation_config = self._pretrained_lists, self.config.evaluation_config()
        for name, lists in methods.items():
            significance = paired_comparison(pretrained, lists, self.examples("test"), self.log.items, evaluation_config)
            self.reports.table(f"logged_preferences_significance_{_file_name(name)}", significance)
        liked_only = self._test_evaluation("liked-only", self._liked_only_policy()).generated_lists()
        k = max(self.config.ks)
        curves = {
            "RL, anchor sweep": self._generated_lists_of(self._kl_sweep_runs),
            **{name: {name: lists} for name, lists in methods.items()},
            "liked-only fine-tuning": {"liked-only fine-tuning": liked_only},
            "pretrained": {"pretrained": pretrained},
        }
        measures = [f"all/ndcg@{k}", "logged/relative_surplus_per_item", "profile/catalog_coverage"]
        self._trade_off_figures("logged_preferences", curves, signals=JUDGED_SIGNALS[:1], measures=measures)
        everything = {"pretrained": pretrained, "RL": self._rl_lists, "liked-only fine-tuning": liked_only} | methods
        return self.reports.table("logged_preference_methods", _runs_table(everything).T)

    @_timed
    def predict_lists_without_feedback(self) -> pd.DataFrame:
        """Lists from one head per position, read off the real history, against ranked and generated lists of the base recommender.

        Generation feeds each chosen item back as if it had been consumed; ranking scores every position from the real
        history but with one distribution. One head per position does neither. The position hits show what each costs
        after a miss and gains after a hit. Returns one column per kind of list.
        """
        if not self.config.multi_token_lists:
            return pd.DataFrame()
        heads, generation = self._multi_token_heads(), self.config.evaluation_config().generation

        def one_head_per_position(history):
            return multi_token_lists(self.pretrained_policy, heads, history, generation)

        examples = self.examples("test")
        predicted = collect_lists(one_head_per_position, examples, self.config.evaluation_batch_size, self.device)
        pretrained = self._test_evaluation("pretrained", self.pretrained_policy)
        lists = {"ranked": pretrained.top_k, "generated": pretrained.generated.actions, "one head per position": predicted}
        self.reports.table("multi_token_position_hits", position_hits(lists, examples.future_items))
        evaluations = {"ranked": self._evaluate_lists(pretrained.top_k), "generated": pretrained.generated_lists(), "one head per position": self._evaluate_lists(predicted)}
        return self.reports.table("multi_token_lists", _runs_table(evaluations).T)

    @_timed
    def interpolate_weights(self) -> pd.DataFrame:
        """Operating points from weight space: theta_0 + a (theta_RL - theta_0) for each factor a, against the anchor sweep.

        No point needs training. Below 1 the factor interpolates towards the base recommender (WiSE-FT, WARP);
        above 1 it extrapolates past the main run (ExPO). Every recommender's distance from the base one is measured
        the same way, as the KL along its own lists, since an interpolated one has no training record.
        Returns one column per recommender.
        """
        factors = sorted(self.config.interpolation_factors)
        if not factors:
            return pd.DataFrame()
        generation, examples = self.config.evaluation_config().generation, self.examples("test")

        def with_distance(name: str, policy: NextItemPolicy) -> ListEvaluation:
            evaluation = self._test_evaluation(name, policy)
            lists = evaluation.generated_lists()
            lists.metrics["kl_along_own_lists"] = kl_along_lists(policy, self.pretrained_policy, examples, evaluation.generated.actions, generation, KL_BATCH_SIZE, self.device)
            return lists

        interpolations = {f"weights x{factor}": with_distance(f"weights x{factor}", interpolated(self.pretrained_policy, self.rl_policy, factor)) for factor in factors}
        anchor_sweep = {name: with_distance(name, load_policy(self._sweep_checkpoint(name), self.device)) for name in self._kl_sweep_runs if self._sweep_checkpoint(name).exists()}
        pretrained = with_distance("pretrained", self.pretrained_policy)
        main_run = with_distance("rl", self.rl_policy)
        curves = {
            "RL, anchor sweep": anchor_sweep,
            "weights between and beyond": interpolations | {"weights x1.0": main_run},
            "pretrained": {"pretrained": pretrained},
        }
        self._trade_off_figures("weight_interpolation", curves, signals=JUDGED_SIGNALS[:1])
        k = max(self.config.ks)
        by_distance = {method: _runs_table(runs).sort_values("kl_along_own_lists") for method, runs in curves.items() if runs}
        measures = [f"all/ndcg@{k}", "predicted/heldout/played_ratio", "profile/catalog_coverage"]
        self.reports.figure("weight_interpolation_by_kl", plots.trade_off(by_distance, reward_column="kl_along_own_lists", measure_columns=measures))
        everything = {"pretrained": pretrained, "RL": main_run} | anchor_sweep | interpolations
        return self.reports.table("weight_interpolation", _runs_table(everything).T)

    @_timed
    def control_for_continued_training(self) -> pd.DataFrame:
        """Learning from the log against the same epoch of training without its weights, on ranked as well as generated lists.

        Reward-weighted fine-tuning and within-user DPO train the base recommender for one more epoch at a lower
        learning rate. The base recommender stopped when its validation loss stopped falling, before its learning rate
        had decayed, so that epoch may raise relevance by itself. Continued training, in which every logged item weighs
        one, is the epoch alone: what a method gains over it is what its weights or its preferences add. Every
        recommender is judged on its ranked lists too, since the base recommender's ranked lists are its stronger ones
        on some datasets. Returns one column per recommender and kind of list.
        """
        if not self.config.continued_training_control:
            return pd.DataFrame()
        recommenders = {"pretrained": self.pretrained_policy, CONTINUED_TRAINING: self._continued_training_policy()}
        for response in self.config.reward_weighted_responses:
            for temperature in self.config.reward_weighted_temperatures:
                recommenders[f"reward-weighted, {response}, temperature={temperature}"] = self._reward_weighted_policy(response, temperature)
        for beta in self.config.preference_betas:
            recommenders[f"within-user DPO, beta={beta}"] = self._logged_preference_policy(beta)

        lists: dict[tuple[str, str], ListEvaluation] = {}
        for name, policy in recommenders.items():
            evaluation = self._test_evaluation(name, policy)
            lists[(name, "ranked")] = self._evaluate_lists(evaluation.top_k)
            lists[(name, "generated")] = evaluation.generated_lists()

        learned_from_the_log = [name for name in recommenders if name not in ("pretrained", CONTINUED_TRAINING)]
        compared = [(CONTINUED_TRAINING, "pretrained")] + [(name, reference) for reference in ("pretrained", CONTINUED_TRAINING) for name in learned_from_the_log]
        examples, evaluation_config = self.examples("test"), self.config.evaluation_config()
        significance = {
            (f"{name} vs {reference}", kind): paired_comparison(lists[(reference, kind)], lists[(name, kind)], examples, self.log.items, evaluation_config)
            for name, reference in compared
            for kind in LIST_KINDS
        }
        self.reports.table("continued_training_significance", pd.concat(significance, names=["comparison", "lists", "quantity"]))
        table = _runs_table({f"{name}, {kind}": evaluation for (name, kind), evaluation in lists.items()}).T
        return self.reports.table("continued_training_control", table)

    @property
    def _capacity_runs(self) -> dict[str, RLConfig]:
        """Runs of the main reward held at each capacity KL budget, with fine-tuning restricted to item offsets or to low-rank updates."""
        reward, weight = self.config.rl.reward, self.config.rl.reference_kl_weight
        budgets = sorted(self.config.capacity_kl_budgets)
        runs = {
            f"item offsets, kl_budget={budget}": self.config.sweep_run(
                reward, weight, target_kl=budget, adaptation=Adaptation.ITEM_OFFSETS, learning_rate=self.config.item_offset_learning_rate
            )
            for budget in budgets
        }
        for rank in self.config.low_ranks:
            runs |= {
                f"rank-{rank} updates, kl_budget={budget}": self.config.sweep_run(
                    reward, weight, target_kl=budget, adaptation=Adaptation.LOW_RANK, low_rank=rank, learning_rate=self.config.low_rank_learning_rate
                )
                for budget in budgets
            }
        return runs

    @property
    def _group_relative_runs(self) -> dict[str, RLConfig]:
        """Runs of the main reward with GRPO's advantages: as many lists per iteration as the other runs, in groups per history."""
        rollouts = self.config.leave_one_out_rollouts
        group_relative = {
            "advantage_estimator": AdvantageEstimator.GROUP_RELATIVE,
            "rollouts_per_history": rollouts,
            "histories_per_step": self.config.rl.histories_per_step // rollouts,
            "normalize_advantages": False,
        }
        reward, weight = self.config.rl.reward, self.config.rl.reference_kl_weight
        return {f"group-relative, kl_budget={budget}": self.config.sweep_run(reward, weight, target_kl=budget, **group_relative) for budget in sorted(self.config.group_relative_kl_budgets)}

    def _finetuned_policies(self) -> dict[str, NextItemPolicy]:
        """The main run and every trained run of the main reward's sweeps: anchor weights, KL budgets, restricted capacity, GRPO."""
        full_capacity = [f"kl_budget={budget}" for budget in sorted(self.config.capacity_kl_budgets)]
        names = dict.fromkeys([*self._kl_sweep_runs, *self._kl_budget_runs, *full_capacity, *self._capacity_runs, *self._group_relative_runs])
        return {"RL": self.rl_policy} | {name: load_policy(self._sweep_checkpoint(name), self.device) for name in names if self._sweep_checkpoint(name).exists()}

    def _continuation_value_model(self, reward_name: str, reward: RewardSpec, discount: float) -> ContinuationValueModel:
        """U for `reward` at `discount`, learned from the base recommender's own lists on first use."""
        path = self.checkpoints / "continuation_values" / f"{_file_name(reward_name)}_discount_{discount}.pt"
        if not path.exists():
            training = replace(
                self.config.continuation_training,
                history_len=self.config.history_len,
                max_future=self.config.max_future,
                min_history=self.config.rl.min_history,
                generation=replace(self.config.rl.generation, horizon=self.config.horizon),
                discount=discount,
                mixed_precision=self.config.rl.mixed_precision,
            )
            value_model, records = fit_continuation_values(self.pretrained_policy, self.reward_models["train"], self.log, reward, training, self.device)
            save_continuation_value_model(value_model, path)
            self.reports.table(f"continuation_training_{_file_name(reward_name)}_discount_{discount}", pd.DataFrame(records).set_index("step"))
        return load_continuation_value_model(path, self.device)

    def _guided_lists(self, reward_function: RewardFunction, value_model: ContinuationValueModel | None, guidance: GuidanceConfig) -> torch.Tensor:
        """One guided list per test user, steered by the "train" reward model."""
        generation = self.config.evaluation_config().generation

        def guided(history):
            with autocast(self.device, self.config.rl.mixed_precision):
                return guided_lists(self.pretrained_policy, self.reward_models["train"], reward_function, value_model, history, guidance, generation)

        return collect_lists(guided, self.examples("test"), self.config.guidance_batch_size, self.device)

    def _reward_weighted_policy(self, response: str, temperature: float) -> NextItemPolicy:
        path = self.checkpoints / f"policy_reward_weighted_{response}_{temperature}.pt"
        if not path.exists():
            config = RewardWeightedConfig(self.config.preference_finetuning, response, temperature)
            policy, history = finetune_with_reward_weights(self.pretrained_policy, self.log, config, self.config.window_stride, self.device)
            save_model(policy, path)
            self.reports.table(f"reward_weighted_training_{response}_{temperature}", pd.DataFrame(history).set_index("epoch"))
        return load_policy(path, self.device)

    def _continued_training_policy(self) -> NextItemPolicy:
        """The base recommender trained for one more epoch exactly as reward-weighted fine-tuning trains it, with every logged item weighing one."""
        path = self.checkpoints / "policy_continued_training.pt"
        if not path.exists():
            config = RewardWeightedConfig(self.config.preference_finetuning, response=None)
            policy, history = finetune_with_reward_weights(self.pretrained_policy, self.log, config, self.config.window_stride, self.device)
            save_model(policy, path)
            self.reports.table("continued_next_item_training", pd.DataFrame(history).set_index("epoch"))
        return load_policy(path, self.device)

    def _logged_preference_policy(self, beta: float) -> NextItemPolicy:
        path = self.checkpoints / f"policy_within_user_dpo_{beta}.pt"
        if not path.exists():
            config = LoggedPreferenceConfig(self.config.preference_finetuning, beta, self.config.preference_margin, self.config.preference_lookahead)
            policy, history = finetune_on_logged_preferences(self.pretrained_policy, self.log, config, self.config.window_stride, self.device)
            save_model(policy, path)
            self.reports.table(f"within_user_dpo_training_{beta}", pd.DataFrame(history).set_index("epoch"))
        return load_policy(path, self.device)

    def _multi_token_heads(self) -> MultiTokenHeads:
        path = self.checkpoints / "multi_token_heads.pt"
        if not path.exists():
            heads, history = train_multi_token_heads(
                self.pretrained_policy,
                self.log,
                self.config.horizon,
                self.config.multi_token_optimization,
                self.config.window_stride,
                self.device,
                max_targets_per_head=self.config.multi_token_targets_per_head,
            )
            save_multi_token_heads(heads, path)
            self.reports.table("multi_token_training", pd.DataFrame(history).set_index("epoch"))
        return load_multi_token_heads(path, self.device)

    @cached_property
    def log(self) -> InteractionLog:
        return InteractionLog.load(self.log_directory)

    @cached_property
    def pretrained_policy(self) -> NextItemPolicy:
        return load_policy(self._pretrained_policy_path, self.device)

    @cached_property
    def rl_policy(self) -> NextItemPolicy:
        return load_policy(self._rl_policy_path, self.device)

    @cached_property
    def reward_models(self) -> dict[str, RewardModel]:
        return {name: load_reward_model(self._reward_model_path(name), self.device) for name in ("train", "heldout")}

    @cached_property
    def masked_reward_models(self) -> dict[str, RewardModel]:
        """The two reward models that can read "unknown response" (see `compare_feedback_masking`), trained on first use."""
        training = replace(self._reward_model_training(), hidden_response_share=self.config.hidden_response_share, max_hidden_responses=self.config.horizon)
        self._train_reward_model_pair("masked_", training)
        return {name: load_reward_model(self._reward_model_path(name), self.device) for name in ("masked_train", "masked_heldout")}

    def _reward_model_training(self) -> RewardModelTrainingConfig:
        return RewardModelTrainingConfig(self.config.transformer, self.config.reward_model_optimization, self.config.window_stride, loss_weights=self.config.reward_model_loss)

    def _train_reward_model_pair(self, prefix: str, training: RewardModelTrainingConfig) -> None:
        """Trains the reward models named <prefix>train and <prefix>heldout on disjoint halves of the users, unless they exist."""
        all_users = np.arange(self.log.num_users)
        for name, users in {f"{prefix}train": all_users[all_users % 2 == 0], f"{prefix}heldout": all_users[all_users % 2 == 1]}.items():
            if not self._reward_model_path(name).exists():
                reward_model, history = train_reward_model(self.log, training, self.device, users=users)
                save_model(reward_model, self._reward_model_path(name))
                self.reports.table(f"reward_model_{name}_training", pd.DataFrame(history).set_index("epoch"))

    def examples(self, period: Period) -> EvalExamples:
        """One recommendation request per user at the start of `period`."""
        if period not in self._examples:
            self._examples[period] = build_eval_examples(self.log, period, self.config.history_len, self.config.max_future)
        return self._examples[period]

    @property
    def _pretrained_policy_path(self) -> Path:
        return self.checkpoints / "policy_pretrained.pt"

    @property
    def _rl_policy_path(self) -> Path:
        return self.checkpoints / "policy_rl.pt"

    def _reward_model_path(self, name: str) -> Path:
        return self.checkpoints / f"reward_model_{name}.pt"

    def _evaluate(self, policy: NextItemPolicy, period: Period, reward_models: dict[str, RewardModel] | None = None, **evaluation_overrides) -> PolicyEvaluation:
        reward_models = self.reward_models if reward_models is None else reward_models
        config = self.config.evaluation_config(**evaluation_overrides)
        evaluation = evaluate_policy(policy, self.examples(period), self.log.items, reward_models, config, self.device)
        # Distances are measured in the base recommender's embeddings, which do not exist while it is being trained.
        if self._pretrained_policy_path.exists():
            evaluation.metrics[f"top_k/{INTRA_LIST_DISTANCE}"] = self._mean_intra_list_distance(evaluation.top_k)
            evaluation.metrics[f"next_k/{INTRA_LIST_DISTANCE}"] = self._mean_intra_list_distance(evaluation.generated.actions)
        return evaluation

    def _evaluate_lists(self, lists: torch.Tensor) -> ListEvaluation:
        """Test-period evaluation of one given list per test user."""
        evaluation = evaluate_lists(lists, self.examples("test"), self.log.items, self.reward_models, self.config.evaluation_config(), self.device)
        evaluation.metrics[INTRA_LIST_DISTANCE] = self._mean_intra_list_distance(evaluation.lists)
        return evaluation

    @cached_property
    def _unit_item_embeddings(self) -> torch.Tensor:
        """The base recommender's item embeddings scaled to unit length, on the CPU: the space in which lists are called varied. [V+1, d]"""
        return F.normalize(self.pretrained_policy.item_embedding.weight.detach().float().cpu(), dim=-1)

    def _mean_intra_list_distance(self, lists: torch.Tensor) -> float:
        return intra_list_distance(lists.cpu(), self._unit_item_embeddings).mean().item()

    def _evaluate_ranking(self, ranker: NextItemScorer) -> ListEvaluation:
        """Test-period evaluation of the lists a model produces by ranking the catalogue once, as long as the generated lists."""
        config = self.config.evaluation_config()

        def top_of_ranking(history):
            return recommend_top_k(ranker, history, self.config.horizon, config.generation)

        return self._evaluate_lists(collect_lists(top_of_ranking, self.examples("test"), config.batch_size, self.device))

    def _liked_only_policy(self) -> NextItemPolicy:
        path = self.checkpoints / "policy_liked_only.pt"
        if not path.exists():
            liked_threshold = self.config.evaluation_config().liked_threshold
            policy, history = finetune_on_liked_items(
                self.pretrained_policy, self.log, self.config.liked_finetuning, self.config.window_stride, liked_threshold, self.device
            )
            save_model(policy, path)
            self.reports.table("liked_only_training", pd.DataFrame(history).set_index("epoch"))
        return load_policy(path, self.device)

    def _logged_return_policy(self, importance_weight_cap: float) -> NextItemPolicy:
        """The base recommender fine-tuned with off-policy REINFORCE on logged sessions, for the main run's reward."""
        path = self.checkpoints / f"policy_off_policy_cap_{importance_weight_cap}.pt"
        if not path.exists():
            config = replace(self.config.logged_returns, reward=self.config.rl.reward, importance_weight_cap=importance_weight_cap)
            policy, history = finetune_on_logged_returns(self.pretrained_policy, self.log, config, self.config.window_stride, self.device)
            save_model(policy, path)
            self.reports.table(f"off_policy_training_cap_{importance_weight_cap}", pd.DataFrame(history).set_index("epoch"))
        return load_policy(path, self.device)

    def _best_of_sampled_lists(self, num_lists: int, reward: RewardSpec | None = None) -> torch.Tensor:
        """For every test user, the best of `num_lists` lists sampled from the base recommender, as the "train" reward model sees it.

        Lists are compared by `reward`, the main run's unless another is given.
        """
        config = self.config.evaluation_config()
        sampling = replace(config.generation, sample_from_top=self.config.sample_from_top)
        reward_function = RewardFunction(reward or self.config.rl.reward, self.log.items, self.device, item_embeddings=self.pretrained_policy.item_embedding.weight)

        def best_list(history):
            return best_of_sampled_lists(
                self.pretrained_policy, self.reward_models["train"], reward_function, history, num_lists, self.config.rl.discount, sampling, config.simulated_feedback
            )

        return collect_lists(best_list, self.examples("test"), config.batch_size, self.device)

    @cached_property
    def _scored_test_candidates(self) -> ScoredCandidates:
        """The base recommender's most likely next items for every test user, each judged by the "train" reward model. On the CPU."""
        config = self.config.evaluation_config()
        scored = [
            score_candidates(self.pretrained_policy, self.reward_models["train"], batch.history.to(self.device), self.config.rerank_candidates, config.generation).cpu()
            for batch in self.examples("test").batches(config.batch_size)
        ]
        return ScoredCandidates.concatenate(scored)

    def _retention_quality(self, test_predictions: dict[str, HeldoutFeedback]) -> pd.DataFrame:
        """How well both reward models predict the retention signals, on held-out events whose outcome the log knows.

        A return window as long as the test period leaves almost no test event with a known outcome. The
        validation period then carries the evidence, although it also picked the best training epoch.
        """
        validation_windows = EventWindows.for_heldout(self.log, "validation", self.config.transformer.max_positions)
        quality = {}
        for name, reward_model in self.reward_models.items():
            validation_predictions = predict_heldout_feedback(reward_model, validation_windows, self.config.evaluation_batch_size, self.device)
            quality[(name, "validation")] = retention_quality(validation_predictions)
            quality[(name, "test")] = retention_quality(test_predictions[name])
        return pd.concat(quality, names=["reward model", "period", "signal"])

    @property
    def _pretrained_lists(self) -> ListEvaluation:
        return self._test_evaluation("pretrained", self.pretrained_policy).generated_lists()

    @property
    def _rl_lists(self) -> ListEvaluation:
        return self._test_evaluation("rl", self.rl_policy).generated_lists()

    def _sweep_evaluation(self, run_name: str) -> PolicyEvaluation:
        """Test evaluation of a trained sweep run, together with how far the run ended from the base recommender."""
        if run_name not in self._test_evaluations:
            self._test_evaluations[run_name] = self._evaluate(load_policy(self._sweep_checkpoint(run_name), self.device), "test")
        evaluation = self._test_evaluations[run_name]
        if REALISED_KL not in evaluation.metrics:
            evaluation.metrics |= self._end_of_training(f"sweep_training_{_file_name(run_name)}")
        return evaluation

    def _end_of_training(self, training_report: str) -> dict[str, float]:
        """How a recorded fine-tuning run ended: each of `TRAINING_STATISTICS` averaged over its last iterations.

        The first is the KL from the fine-tuned policy to the base recommender, in nats per generated item.
        """
        records = pd.read_csv(self.reports.directory / f"{training_report}.csv")
        return {
            statistic: float(records[column].dropna().tail(END_OF_TRAINING_STEPS).mean()) if column in records else float("nan")
            for statistic, column in TRAINING_STATISTICS.items()
        }

    def _metrics_on_validation(self, run_name: str) -> dict[str, float]:
        """All metrics of the base recommender ("pretrained") or of a trained sweep run on validation requests, computed once."""
        if run_name not in self._validation_metrics:
            policy = self.pretrained_policy if run_name == "pretrained" else load_policy(self._sweep_checkpoint(run_name), self.device)
            self._validation_metrics[run_name] = self._evaluate(policy, "validation").metrics
        return self._validation_metrics[run_name]

    def _selection_on_validation(self, run_names: list[str]) -> pd.DataFrame:
        """Which of the runs would be chosen without looking at the test period. One row per run evaluated on validation requests.

        The chosen run has the highest reward on validation requests, as the held-out reward model predicts it,
        among the runs that keep the validation NDCG within `relevance_tolerance` of the base recommender's.
        """
        k = max(self.config.ks)
        relevance, reward = f"next_k/all/ndcg@{k}", "predicted/heldout/played_ratio"
        evaluated = [name for name in run_names if name in self._validation_metrics]
        selection = pd.DataFrame({
            f"validation/{relevance}": {name: self._validation_metrics[name][relevance] for name in evaluated},
            f"validation/{reward}": {name: self._validation_metrics[name][reward] for name in evaluated},
        })
        lowest_relevance = (1.0 - self.config.relevance_tolerance) * selection.loc["pretrained", f"validation/{relevance}"]
        selection["keeps_relevance"] = selection[f"validation/{relevance}"] >= lowest_relevance
        best_admissible = selection[f"validation/{reward}"].where(selection["keeps_relevance"]).idxmax()
        selection["selected"] = selection.index == best_admissible
        return selection

    def _generated_lists_of(self, sweep_runs) -> dict[str, ListEvaluation]:
        """Test evaluation of the lists generated by each named sweep run that was trained, in the given order."""
        return {name: self._sweep_evaluation(name).generated_lists() for name in sweep_runs if self._sweep_checkpoint(name).exists()}

    def _reranked(self, label: str, candidates: ScoredCandidates, reward: RewardSpec) -> dict[str, ListEvaluation]:
        """Test evaluation of the candidates re-ranked by `reward` at every relevance weight, computed once per label."""
        reward_function = RewardFunction(reward, self.log.items, "cpu", item_embeddings=self.pretrained_policy.item_embedding.weight)
        evaluations = {}
        for weight in self.config.rerank_relevance_weights:
            name = f"{label}, weight={weight}"
            if name not in self._reranked_evaluations:
                self._reranked_evaluations[name] = self._evaluate_lists(rerank(candidates, reward_function, self.config.horizon, weight))
            evaluations[name] = self._reranked_evaluations[name]
        return evaluations

    @cached_property
    def _candidates_scored_by_item_mean(self) -> ScoredCandidates:
        """The same candidates with the reward model's prediction replaced by each item's mean played ratio in the train period."""
        candidates = self._scored_test_candidates
        item_mean = torch.as_tensor(self.log.items.mean_played_ratio)[candidates.items]
        feedback = Feedback(
            played_ratio=item_mean,
            session_continues=torch.zeros_like(item_mean),
            returns=torch.zeros_like(item_mean),
            is_chosen=torch.ones_like(item_mean),
            relative_ratio=torch.full_like(item_mean, 0.5),  # every item's mean relative ratio, by construction
            liked=torch.zeros_like(item_mean),
            disliked=torch.zeros_like(item_mean),
            adopts=torch.zeros_like(item_mean),
        )
        return ScoredCandidates(candidates.items, candidates.log_probs, feedback, feedback)

    def _trade_off_figures(
        self, report_name: str, curves: dict[str, dict[str, ListEvaluation]], signals: tuple[str, ...], measures: list[str] | None = None
    ) -> None:
        """One figure per signal: `measures` against that signal as the held-out reward model predicts it.

        curves: method -> its runs in the order its line is drawn. Methods without runs are left out.
        measures: metrics of the lists, by default relevance on all and on liked items and catalogue coverage.
        """
        k = max(self.config.ks)
        measures = measures or [f"all/ndcg@{k}", f"liked/ndcg@{k}", "profile/catalog_coverage"]
        tables = {method: _runs_table(runs) for method, runs in curves.items() if runs}
        for signal in signals:
            figure = plots.trade_off(tables, reward_column=f"predicted/heldout/{signal}", measure_columns=measures)
            self.reports.figure(f"{report_name}_by_{signal}", figure)

    @property
    def _kl_sweep_runs(self) -> dict[str, RLConfig]:
        return {f"kl={weight}": self.config.rl_config(num_steps=self.config.sweep_steps, reference_kl_weight=weight) for weight in self.config.kl_weights}

    @property
    def _kl_budget_runs(self) -> dict[str, RLConfig]:
        """Runs of the main reward whose anchor weight starts at the main run's and adapts to hold each KL budget."""
        reward, starting_weight = self.config.rl.reward, self.config.rl.reference_kl_weight
        return {f"kl_budget={budget}": self.config.sweep_run(reward, starting_weight, target_kl=budget) for budget in sorted(self.config.kl_budgets)}

    def _record_stage_time(self, stage: str, seconds: float) -> None:
        self._stage_seconds[stage] = self._stage_seconds.get(stage, 0.0) + seconds
        self.reports.table("stage_minutes", pd.Series({name: total / 60 for name, total in self._stage_seconds.items()}, name="minutes"))

    def _test_evaluation(self, name: str, policy: NextItemPolicy) -> PolicyEvaluation:
        """Test-period evaluation of a named recommender, computed once per experiment."""
        if name not in self._test_evaluations:
            self._test_evaluations[name] = self._evaluate(policy, "test")
        return self._test_evaluations[name]

    def _validation_ranking_quality(self, ranker: NextItemScorer) -> dict[str, float]:
        """NDCG of ranking the catalogue for validation users, under the name the GPT recommender reports it with.

        Losses of different models are not comparable; this number, printed after every epoch, is.
        """
        config, k, examples = self.config.evaluation_config(), max(self.config.ks), self.examples("validation")

        def top_of_ranking(history):
            return recommend_top_k(ranker, history, k, config.generation)

        lists = collect_lists(top_of_ranking, examples, config.batch_size, self.device)
        return {f"top_k/all/ndcg@{k}": ranking_metrics(lists, examples.future_items, [k])[f"ndcg@{k}"]}

    def _validation_relevance(self, policy: NextItemPolicy) -> dict[str, float]:
        k = max(self.config.ks)
        metrics = self._evaluate(policy, "validation", reward_models={}).metrics
        return {name: metrics[name] for name in (f"top_k/all/ndcg@{k}", f"next_k/all/ndcg@{k}")}

    def _monitor(self, policy: NextItemPolicy) -> dict[str, float]:
        """What is tracked on validation users while RL runs: relevance, both predicted rewards, catalogue coverage."""
        k = max(self.config.ks)
        monitored = [
            f"top_k/all/ndcg@{k}",
            f"next_k/all/ndcg@{k}",
            f"next_k/liked/ndcg@{k}",
            "predicted/train/played_ratio",
            "predicted/heldout/played_ratio",
            "predicted/heldout/personal_played_ratio",
            "predicted/heldout/relative_ratio",
            "predicted/heldout/returns",
            "next_k/logged/played_ratio_per_item",
            "next_k/profile/catalog_coverage",
        ]
        metrics = self._evaluate(policy, "validation").metrics
        return {name: metrics[name] for name in monitored}

    @property
    def _summary_rows(self) -> list[str]:
        k = max(self.config.ks)
        return [
            f"top_k/all/ndcg@{k}",
            f"next_k/all/ndcg@{k}",
            f"next_k/liked/ndcg@{k}",
            f"next_k/skipped/recall@{k}",
            f"next_k/above_usual/ndcg@{k}",
            f"next_k/below_usual/recall@{k}",
            f"next_k/new/ndcg@{k}",
            f"next_k/repeat/ndcg@{k}",
            *TRAINING_STATISTICS,
            "next_k/logged/hits_per_list",
            "next_k/logged/played_ratio_of_hits",
            "next_k/logged/relative_ratio_of_hits",
            "next_k/logged/played_ratio_per_item",
            "next_k/logged/relative_surplus_per_item",
            "predicted/train/played_ratio",
            "predicted/heldout/played_ratio",
            "predicted/heldout/played_ratio_for_other_user",
            "predicted/heldout/personal_played_ratio",
            "predicted/heldout/relative_ratio",
            "predicted/heldout/chosen_played_ratio",
            "predicted/heldout/is_chosen",
            "predicted/heldout/played_seconds",
            "predicted/heldout/session_continues",
            "predicted/heldout/returns",
            "predicted/heldout/returns_after_list",
            "predicted/heldout/liked",
            "predicted/heldout/disliked",
            "predicted/heldout/adopts",
            "next_k/logged/liked_per_item",
            "next_k/logged/adopted_per_item",
            "next_k/profile/catalog_coverage",
            "next_k/profile/top_exposure_share",
            "next_k/profile/mean_user_pair_overlap",
            f"next_k/{INTRA_LIST_DISTANCE}",
            "next_k/profile/head_item_share",
            "next_k/profile/mean_duration_seconds",
            "next_k/profile/history_repeat_rate",
        ]

    def _sweep(self, report_name: str, runs: dict[str, RLConfig], reward_model: RewardModel | None = None) -> pd.DataFrame:
        """Fine-tunes one policy per run and tabulates all test metrics, one column per run plus the base recommender.

        The policies learn from `reward_model`, by default the training reward model. Runs that no longer fit the
        time budget are left out of the table.
        """
        evaluations = {"pretrained": self._test_evaluation("pretrained", self.pretrained_policy)}
        evaluations["pretrained"].metrics.setdefault(REALISED_KL, 0.0)
        for run_name, rl_config in runs.items():
            if self._is_finetuned(run_name, rl_config, reward_model):
                evaluations[run_name] = self._sweep_evaluation(run_name)
        if self.config.evaluate_sweeps_on_validation:
            on_validation = pd.DataFrame({run_name: self._metrics_on_validation(run_name) for run_name in evaluations})
            self.reports.table(f"{report_name}_validation", on_validation)
        return self.reports.table(report_name, metrics_table(evaluations))

    def _is_finetuned(self, run_name: str, rl_config: RLConfig, reward_model: RewardModel | None = None) -> bool:
        """Trains the policy of a sweep run unless its checkpoint exists. False if there is no time left to train it."""
        path = self._sweep_checkpoint(run_name)
        if not path.exists():
            if not self._fits_time_budget(rl_config):
                print(f"skipped the run '{run_name}': it would not finish inside the time budget")
                return False
            policy, records = self._run_rl(rl_config, reward_model=reward_model)
            save_model(policy, path)
            self.reports.table(f"sweep_training_{_file_name(run_name)}", pd.DataFrame(records).set_index("step"))
        return True

    def _sweep_checkpoint(self, run_name: str) -> Path:
        return self.checkpoints / "sweep" / f"{_file_name(run_name)}.pt"

    def _run_rl(self, rl_config: RLConfig, evaluate=None, reward_model: RewardModel | None = None) -> tuple[NextItemPolicy, list[dict[str, float]]]:
        """One RL fine-tuning of the base recommender, timed so that later runs can be checked against the time budget."""
        started = time.monotonic()
        reward_model = self.reward_models["train"] if reward_model is None else reward_model
        policy, records = finetune_with_rl(self.pretrained_policy, reward_model, self.log, rl_config, self.device, evaluate=evaluate)
        self._seconds_per_rl_step = (time.monotonic() - started) / rl_config.num_steps
        return policy, records

    def _fits_time_budget(self, rl_config: RLConfig) -> bool:
        """Whether an RL run of this length would end before the deadline, judging by the speed of the last run."""
        if self._deadline is None or self._seconds_per_rl_step is None:
            return True
        expected_seconds = rl_config.num_steps * self._seconds_per_rl_step * RUN_TIME_SAFETY_FACTOR
        return time.monotonic() + expected_seconds + RESERVE_FOR_LAST_STAGES_SECONDS < self._deadline


def _file_name(name: str) -> str:
    """`name` with everything that is awkward in a file name replaced by underscores."""
    return re.sub(r"[^0-9a-zA-Z.]+", "_", name)


def _runs_table(runs: dict[str, ListEvaluation]) -> pd.DataFrame:
    """One row per run, one column per metric."""
    return pd.DataFrame({name: evaluation.metrics for name, evaluation in runs.items()}).T


def _bind_directory_to_config(home: Path, config: ExperimentConfig) -> None:
    """Records the configuration in `home`, and refuses a directory that holds the results of a different one."""
    home.mkdir(parents=True, exist_ok=True)
    record = home / CONFIG_RECORD
    serialized = json.dumps(asdict(config), indent=2, sort_keys=True)
    if record.exists() and json.loads(record.read_text()) != json.loads(serialized):
        raise ValueError(f"{home} holds the results of a different configuration: use another home directory, or delete this one")
    record.write_text(serialized)


def _events_of_first_users(raw_events: pl.DataFrame, num_users: int) -> pl.DataFrame:
    first_users = raw_events.select(pl.col("user_id").unique().sort().head(num_users))
    return raw_events.join(first_users, on="user_id", how="semi")


def _log_summary(log: InteractionLog) -> pd.Series:
    def num_events_in(period: Period) -> int:
        starts, ends = log.period_rows(period)
        return int((ends - starts).sum())

    return pd.Series({
        "users": log.num_users,
        "items": log.num_items,
        "events": log.num_events,
        "train events": num_events_in("train"),
        "validation events": num_events_in("validation"),
        "test events": num_events_in("test"),
        "median events per user": float(np.median(np.diff(log.user_offsets))),
        "mean played ratio": float(np.mean(log.played_ratio)),
        "share of events played at least half": float(np.mean(np.asarray(log.played_ratio) >= 0.5)),
        "share of events that continue the session": _rate_where_known(log.session_continues, log.session_continues_known),
        "share of events after which the user returns": _rate_where_known(log.returns, log.returns_known),
        "share of events whose return outcome is known": float(np.mean(log.returns_known)),
        "share of organic events": float(np.mean(log.is_organic)),
        "share of events the user liked": _rate_where_known(log.liked, log.reactions_known),
        "share of events the user disliked": _rate_where_known(log.disliked, log.reactions_known),
        "share of first events with an item that the user came back to": _rate_where_known(log.adopts, log.adopts_known),
    })


def _rate_where_known(signal: np.ndarray, is_known: np.ndarray) -> float:
    """How often a signal is true among the events whose outcome the log knows; NaN where it knows none."""
    is_known = np.asarray(is_known)
    return float(np.asarray(signal)[is_known].mean()) if is_known.any() else float("nan")


def _decile(values_per_item: np.ndarray) -> np.ndarray:
    """Decile (0 = lowest) of every item token by the given attribute."""
    return pd.qcut(pd.Series(values_per_item).rank(method="first"), 10, labels=False).to_numpy()


def _exposure_change(log: InteractionLog, before: PolicyEvaluation, after: PolicyEvaluation) -> pd.DataFrame:
    """Per item, the number of test users whose generated list contains it under each recommender, biggest winners first.

    Next to the item's popularity, duration and average played ratio, this shows what the fine-tuned recommender learned to prefer.
    """
    def exposure(evaluation: PolicyEvaluation) -> np.ndarray:
        return np.bincount(evaluation.generated.actions.flatten().numpy(), minlength=log.num_items + 1)

    table = pd.DataFrame({
        "popularity": log.items.train_popularity,
        "duration_seconds": log.items.duration_seconds,
        "mean_played_ratio": log.items.mean_played_ratio,
        "exposure_before": exposure(before),
        "exposure_after": exposure(after),
    }).iloc[1:].rename_axis("item_token")
    table["exposure_change"] = table.exposure_after - table.exposure_before
    return table.sort_values("exposure_change", ascending=False)
