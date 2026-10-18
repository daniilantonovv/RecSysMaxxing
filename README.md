# RecSysMaxxing

GPTRec + RL: aligning generation to high-expected-reward sequences.

A baseline for studying what happens when a generative next-item recommender is fine-tuned with
reinforcement learning against a learned reward model:

1. **Base recommender** – a GPT-2 decoder over item-ID tokens, trained for next-item prediction.
2. **Reward model** – a second GPT-2 decoder that predicts the user's response to an item given
   their history: the played ratio, whether the session continues, whether the user returns for
   another session, whether the user would take the item up at all, and, where the log records them,
   likes, dislikes and whether a new item is adopted (played again in a later session).
3. **RL fine-tuning** – the recommender generates lists item by item, the reward model scores
   them, and PPO with a KL anchor to the base recommender moves probability towards lists with a
   higher predicted return.

Around that core, the pipeline answers the two questions a reader will ask first:

- **Is the base recommender any good?** It is compared with popularity, replaying the user's own
  history, a Markov chain, GRU4Rec, SASRec and BERT4Rec, trained on the same windows and evaluated on
  the same requests.
- **Does it take RL?** RL is compared with four ways of raising the reward without on-policy RL:
  supervised fine-tuning on liked items only, off-policy REINFORCE on logged sessions (Matveev et al.),
  re-ranking the base recommender's candidates by predicted reward, and serving the best of several
  sampled lists (the test-time scaling of Matveev et al.).

And it studies the reward itself: rewards for continuing the session and for returning; two ways of
letting the reward ask for relevance, so that the anchor does not have to (weighing it by the reward
model's own estimate that the user would choose the item, and adding the logged-relevance reward of
Petrov & Macdonald); rewards that pay only for what depends on the user; a reward for the played ratio
relative to what is usual for the item, which no item earns by being easy to finish; and a reward that
depends on the list as a whole. Long-term rewards pay for likes, for adoption and for the return after the
whole list, and a reward model trained with hidden responses checks what simulating the responses to generated
items does to the reward.

Datasets: [Yambda](https://huggingface.co/datasets/yandex/yambda) (Yandex Music, the 50M variant) and
[KION](https://github.com/irsafilo/KION_DATASET) (MTS online cinema). They differ in what makes the task
hard: Yambda has long histories with repeats and second-level timestamps; KION has short histories,
one row per user-item pair and dates only.

> **Status.** `report/paper_v4.pdf` is the current write-up (15 pages), organised around the task's five questions;
> `report/paper_v4_cuts.md` lists what its 33-page version of commit `839539d` had and it leaves out. It reports
> the final sessions (`results/kion_final`, and `results/yambda_final`, which continued `results/yambda_diagnostics`),
> the review sessions (`results/kion_review`, `results/yambda_review`: long-term rewards, reward models with hidden
> responses, analyses by position, user and item) and seeds 1 and 2 of the core stages (`results/kion_seeds`,
> `results/yambda_seeds`, with `over_three_seeds.csv` for all three seeds). Its tables are typeset from those files by
> `report/tables.py` and its SVG figures drawn by `report/figures.py`. Earlier write-ups, kept for the record:
> `report/paper_v3.pdf` (one seed), `report/paper_v2.pdf` (seven earlier runs), `report/report_v2_ru.pdf` and
> `report/first_iteration.pdf` (Russian), `report/theory_ru.pdf` (background, Russian). The code is written on a machine without a GPU or
> PyTorch, so changes are statically checked, not run, before they reach Kaggle. Every entry point therefore
> starts with a few-minute end-to-end smoke test (`gptrec.experiment.run_smoke_test`,
> `gptrec.seeds.run_seeds_smoke_test`).

## Findings in one page

Offline; numbers are for generated lists of ten on the test requests, and "±" is the standard deviation over three
seeds. The paper (`report/paper_v4.pdf`) is organised around the same five questions.

| Question | Answer |
|---|---|
| 1. How does the model's behaviour change after RL fine-tuning? | The reward is maximised as far as the reward models can tell: on KION the predicted consumed share rises by 23.9% ± 2.1 and a reward model trained on other users agrees. But the policy learns which items everybody finishes, not what each user wants: 78% of its change is one shift per item (correlated 0.90 with the item's usual consumed share), films go from 66% to 80% of impressions, and measured against what is usual for each title users took to the recommendations less (logged surplus −0.0007 in every seed). Two causes: the reward scores only items a user has started (and KION's share of a series covers all its episodes), and RL extracts the per-item part of a reward model that knows more (re-ranking by the same model is 8.9% personal on KION and 55% on Yambda; RL 0.2% and 16%). |
| 2. Is relevance preserved? Does the model recommend the same items to everyone? | Relevance is traded along a frontier. At the weight selected on validation requests (β = 1) KION loses 1.8% of NDCG@10 for +8% predicted share; at the main run (β = 0.3) 11.5% ± 1.3 for +24%, at every cut-off of NDCG, Recall and MAP and most for users with 5 to 8 earlier events (−13.5%), while the most shown 1% of titles take 76% of impressions instead of 72%. Yambda keeps both. The anchor keeps relevance; its weight is not comparable between estimators (effective weight β·σ_A), the KL distance describes much more, and without the anchor every user gets the same list. |
| 3. Can we get closer to long-term goals? | Rewards for return, likes and adoption raise their predictions by a few percent at a moderate distance; adoption on Yambda is the most promising (+19% at no loss of relevance, a fifth less coverage). The return after the list on KION rises 18% by drifting to short titles at −69% NDCG@10, following an artifact of KION's one-row-per-title log that both reward models learned. Offline we can test predictability, spread between candidates and the cost in relevance; whether users return more needs the online test designed in the paper. |
| 4. What happens when generating several steps ahead? | A generated list gains after a hit and loses after a miss: on Yambda the item at position 5 is a hit in 54% of lists after a hit at position 4, against 25% in a ranked list, and in 4% against 6% after a miss. Simulated responses bias the predicted reward of later steps (+0.06 to +0.18 when earlier items are assumed finished); a reward model trained with hidden responses is unbiased and as accurate. At an equal KL distance, crediting each item with its own reward only is at least as good as crediting it with the rest of the list. |
| 5. What is the trade-off between relevance and reward? | RL settings choose a point on the frontier; only the reward function moves the frontier. Re-ranking by the reward model computes the KL-anchored optimum in closed form: within 1% of the base NDCG@10 on KION it gains 15% of predicted share, against 4% for RL on the share and 17% for RL with logged relevance in the reward. Re-ranking by the share relative to the item's usual raises NDCG@10 (+1.1%) and the logged surplus (+15%) at once, in every seed. |

| KION, test requests (seed 0) | NDCG@10 | Predicted share (held-out model) | Logged share per recommended item | Coverage |
|---|---|---|---|---|
| Base recommender, generated lists | 0.1173 | 0.455 | 0.0173 | 0.495 |
| RL on the share, anchor weight 0.3 | 0.1047 | 0.553 | 0.0174 | 0.449 |
| Base recommender, ranked lists | 0.1220 | 0.466 | 0.0193 | 0.574 |
| Re-ranking by the reward model, w = 0.3 | 0.1127 | 0.545 | 0.0201 | 0.531 |
| RL on share + 10 x logged relevance, anchor weight 0.3 | 0.1194 | 0.495 | 0.0185 | 0.426 |

![Every fine-tuning run against the distance it ended at from the base recommender](report/figures/frontier_by_distance.svg)

Every fine-tuning run of the default reward, against the KL distance it ended at from the base recommender: runs
that differ in the advantage estimator, in how the constraint is set and in how credit is assigned lie on one curve.

To rebuild the paper from the reports:

```bash
python3 report/tables.py && python3 report/figures.py && typst compile report/paper_v4.typ
```

## Setup

```bash
pip install -e ".[notebooks]"
```

Python ≥ 3.10 and a CUDA GPU. With the default sizes (30k items, 128-token windows, batch 128)
next-item training needs roughly 8–10 GB of GPU memory; lower `batch_size` or `max_items` if needed.

## Running on Kaggle

`kaggle/` holds seven self-contained notebooks. Each carries the `gptrec` source in `%%writefile` cells, so
nothing has to be cloned or uploaded besides the notebook itself (*File -> Import Notebook*).

| Notebook | Kaggle settings | Does |
|---|---|---|
| `1_full_pipeline_t4` | GPU T4 x2, internet on | downloads the chosen dataset and runs the whole study on one T4 |
| `2_prepare_offline_bundle` | no accelerator, internet on | packs dependency wheels and both datasets into `/kaggle/working/gptrec_bundle`: the interactions, Yambda's likes and dislikes, KION's item table |
| `3_train_offline` | large GPU, internet off, notebook 2's output attached as input | full-size run with the wheels and the data of the bundle |
| `4_seeds_kion_1_2`, `4_seeds_yambda_1_2` | as notebook 3 | the core of the study once per random seed, with mean and standard deviation over the seeds of the session |
| `5_review_kion`, `5_review_yambda` | as notebook 3 | long-term rewards (likes, coming back to a new track, the return after the list), a reward model that can be told a response is unknown, and analyses along the list and across users and items |

Notebooks 1 and 3 handle one dataset per run: set `DATASET = "yambda"` or `"kion"` in the configuration cell.
Notebook 3 also takes `SESSION`: `"diagnostics"` runs the stages that test what the anchor weight means and the
two new rewards on reduced grids, `"final"` runs every stage on the full grids. The seeds notebooks use the
`"seeds"` configuration of their dataset and come with the dataset and the seeds set: seeds 1 and 2, which with
seed 0 of notebook 3's sessions make three seeds per dataset (one Yambda seed takes about 4.7 hours, one KION seed
about 2). A seed that would not finish inside the time budget is skipped. Each
session writes `<dataset>_seeds_<seeds>_reports.zip`.

A session of notebook 3 can continue an earlier one of the same dataset: attach the earlier session's output
as an input and name its home directory in `EARLIER_SESSIONS` (the "Earlier sessions" cell). The interaction log,
the models and the finished fine-tuning runs are copied, and only the missing runs are trained. This is how a
study that does not fit one 12-hour session is finished in two: on Yambda, `"final"` on top of `"diagnostics"`.
The two sessions may differ only in the grids of their sweeps; anything else is refused before training starts.

For the offline route: run notebook 2 with *Save Version -> Save & Run All* (only a saved version has
an attachable output), then in notebook 3, 4 or 5 use *Add Input -> Notebooks* to attach it and run. The
training notebooks run their own copy of the code and smoke-test it on their GPU first, so the bundle has to
be rebuilt only when a dependency or a dataset changes.

Outputs go to `/kaggle/working/<dataset>_<session>/artifacts`: checkpoints, and `reports/` with every table
(CSV), every figure (PNG) and the configuration that produced them; `<dataset>_<session>_reports.zip` is the same
folder in one file. Stages whose checkpoint exists are skipped, so a session that is still alive can be resumed
by re-running the cells.

After changing anything under `gptrec/`, regenerate the notebooks:

```bash
python3 kaggle/build_notebooks.py
```

## Running elsewhere

`gptrec.experiment.Experiment` runs the same stages anywhere:

```python
from gptrec.data.datasets import download_raw_data, load_raw_events
from gptrec.experiment import Experiment, ExperimentConfig
from gptrec.training.precision import default_device

raw_events = load_raw_events("yambda", download_raw_data("yambda"))
Experiment(ExperimentConfig(), home="runs/yambda", device=default_device()).run(raw_events)
```

The notebooks in `notebooks/` do the same steps one cell at a time, for interactive work.
Run them in order.

| Notebook | Does | Writes |
|---|---|---|
| `01_prepare_data` | downloads Yambda listens, builds the interaction log, sanity plots | `data/processed/yambda-50m/` |
| `02_pretrain_policy` | trains the base recommender, top-k vs next-k relevance, popularity floor | `artifacts/policy_pretrained.pt` |
| `03_reward_model` | trains two reward models on disjoint user halves, held-out quality, calibration, bias by item group | `artifacts/reward_model_{a,b}.pt` |
| `04_rl_finetune` | RL fine-tuning, training dynamics, base vs RL on the test period | `artifacts/policy_rl.pt` |
| `05_tradeoff_and_multistep` | KL sweep (relevance/reward frontier), reward variants, multi-step analysis | `artifacts/sweep/*.pt` |

## Where each question is answered

Stage names are methods of `gptrec.experiment.Experiment`; report names are files in `artifacts/reports/`.

| Question | Stage | What to look at |
|---|---|---|
| Is the base recommender competitive? | `compare_baselines` | `baselines.csv`: popularity, replay of history, Markov chain, GRU4Rec, SASRec, BERT4Rec, GPT ranked and generated |
| 1. How does behaviour change, and why? | `finetune`, `analyse_users_and_items` | `predicted/train/*` vs `predicted/heldout/*`; `*/profile/*`; `rl_exposure_change.csv`; `logit_shift_by_item.csv`: how much of the change is one shift per item; `impressions_by_content_type.csv` and `largest_impression_changes.csv` (KION titles) |
| 2. Is relevance kept? Does everyone get the same items? | `finetune`, `analyse_users_and_items` | `*/all/*`, `*/liked/*`, `*/skipped/*` Recall/NDCG/MAP/HitRate at 1, 5, 10; `rl_vs_pretrained_significance.csv`; `relevance_by_history_length.csv`; `exposure_concentration.csv`: coverage, Gini, share of the most shown 1% |
| 3. Long-term goals | `train_reward_models`, `sweep_rewards` | `reward_model_retention.csv` and `reward_model_reactions.csv`: are return, likes and adoption predictable; `reward_signal_spread.csv`: can choosing items move them; `reward_sweep*`: rewards that pay for them (`LIKES`, `ADOPTION`, `RETURN_AFTER_THE_LIST`), each at several anchor weights |
| 4. Several steps ahead | `analyse_multistep`, `compare_feedback_masking` | `position_hits.csv`: hit rate by position after a miss and after a hit; `feedback_assumptions_on_real_sequences.csv`: the reward model on real next items under each `SimulatedFeedback` assumption; `masked_feedback_sweep*`: RL against a reward model trained with hidden responses |
| 5. Relevance vs reward trade-off | `sweep_kl_weight`, `sweep_rewards`, `sweep_relevance_in_reward` | sweeps over `reference_kl_weight` and `RewardSpec`; `*_by_<signal>.png`: relevance against each signal the held-out reward model predicts |
| Whose reward is it? | `explain_reward_gain` | `reward_gain_decomposition.csv`: how much of each method's gain other users would give the same lists; re-ranking by item averages, with no reward model, as the reference |
| Can the reward ask for relevance itself? | `sweep_relevance_in_reward` | `relevance_reward_sweep*`: the logged-relevance reward of Petrov & Macdonald alone and together with the anchor; the choice-weighted reward |
| Can the reward be made personal? | `sweep_personal_rewards` | `personal_reward_sweep*`: the played ratio measured against another user or against the item's mean, and re-ranking by the same reward |
| Does fine-tuning use the sequence? | `ablate_sequence_credit` | `sequence_credit_ablation*`: discount 0 and one-item episodes against the full return |
| Does it take RL at all? | `compare_alternatives_to_rl` | `alternatives_to_rl*`: RL against liked-only fine-tuning, off-policy REINFORCE, re-ranking and the best of several sampled lists, judged by the held-out reward model |
| What does the anchor weight mean? | `sweep_kl_budget`, `compare_advantage_estimators` | `kl_budget_sweep*`: runs held at a fixed distance from the base recommender; `advantage_estimators*`: the leave-one-out baseline against the value model at equal weights and at equal distances, and `training/clip_fraction` |
| Does the simulated feedback decide the result? | `compare_simulated_feedback` | `simulated_feedback_cross_judged.csv`: policies fine-tuned with the expected and with a sampled response, each judged under every assumption |
| Is the gain more than "items that are easy to finish"? | `sweep_relative_reward` | `relative_reward_sweep*`: the reward is the played ratio as a quantile inside the item; `predicted/heldout/relative_ratio` |
| Does RL win when the reward depends on the whole list? | `compare_list_reward` | `list_reward_methods.csv`: RL against maximal marginal relevance and the best of several sampled lists, on a reward with an intra-list distance term |
| Is any of it confirmed without a model? | every evaluation | `*/logged/*`: the logged response to the recommended items the user went on to consume |
| Was anything chosen on the test period? | every sweep | `<sweep>_validation.csv`: the same metrics on validation requests; `kl_sweep_selection.csv`: the run chosen there |
| How much is the seed? | `gptrec.seeds.run_seeds` | `over_seeds/<report>_mean_std.csv` |

In the interactive notebooks, questions 1-2 are in notebook 4 and questions 3-5 in notebook 5.

## Design decisions worth knowing

- **One time axis, one split.** Train / validation / test are consecutive time ranges. Item
  vocabulary, popularity and item statistics come from the train range only.
- **Tokens are popularity ranks.** Token 1 is the most popular train item; token 0 is padding.
- **Right padding, no attention masks.** Attention is causal, so a real event never sees the
  padding behind it.
- **A fixed position budget.** Models are trained on windows of `max_positions` events. At
  recommendation time the history and the generated items share that budget
  (`history_len + horizon <= max_positions`), so no position is used that was never trained.
- **The reward model reads (item, previous feedback) pairs.** Position *t* sees item *t* and the
  feedback on item *t − 1*; its output predicts the feedback on item *t*. One forward pass scores
  every event of a real history, and a generated item can be scored before its outcome exists.
- **Two reward models.** `train` generates rewards during RL, `heldout` is fitted on other users and
  only evaluates. A reward gain that `heldout` does not confirm is exploitation of `train`'s errors.
- **The reward model knows what users chose, not only how they responded.** Logged events are all items
  somebody picked, so a model of the played ratio alone says nothing about an item this user would never
  pick. A fourth head is trained to tell the logged item from another item put in its place, one swap
  per window, the replacement drawn as often as items are consumed. `RewardSpec(weigh_by_choice=True)`
  multiplies the reward by that probability, and `predicted/<judge>/chosen_played_ratio` is the one
  predicted signal that falls when lists stop being relevant.
- **Retention is two signals, and absence needs evidence.** `session_continues`: the next event follows
  within the session gap. `returns`: another session starts within the return window. "Nothing followed"
  is a fact only once the waiting time has passed inside the observed data; until then the label is
  unknown and is left out of training and scoring. Train-period events are observed until the train
  period ends, so their labels never look into the held-out periods.
- **A reward can be measured against a baseline.** Without one, a policy earns the most by showing everybody the
  items everybody finishes. `RatioBaseline.OTHER_USER` subtracts what the same reward model predicts for the same
  list shown to another user: a policy that ignores its user then earns zero, and errors of the reward model that
  depend on the item alone cancel. `RatioBaseline.ITEM_MEAN` subtracts the item's train-period mean instead. Every
  evaluated list is likewise scored for another user, and `predicted/<judge>/personal_played_ratio` is the difference.
- **Trade-offs have three axes.** Every trade-off figure shows relevance on all and on liked items and catalogue
  coverage: a reward that keeps hits can still shrink the lists onto popular items. Runs are also placed by the KL
  divergence they ended at (`training/kl_to_reference`), because the anchor weight means different things under
  different estimators and datasets.
- **Rewards can come from three places.** From the reward model (any item, but only as good as the
  model), from the log on the policy's own samples (`logged_relevance_weight`: the item is among the
  user's real next items, the NDCG reward of Petrov & Macdonald), and from the log on logged actions
  (`gptrec.training.off_policy`: the REINFORCE of Matveev et al., which generates nothing).
- **A model-free check.** Relevance is reported separately against future items the user played
  (`liked`) and skipped (`skipped`). Avoiding skips should lower the second without hurting the first.
  That split favours items that are easy to finish, so `above_usual` and `below_usual` split at the item's
  own mean played ratio, and `repeat` and `new` split by whether the item is in the visible history.
- **A reward no model predicts.** A recommended item that is among the user's next events comes with the
  response the log recorded. `logged/played_ratio_of_hits` and `logged/relative_ratio_of_hits` average it over
  those items; `logged/played_ratio_per_item` counts the others as nothing and is compared user by user.
  Hits are a selected sample of the recommendations: this says how the reached items were received.
- **The played ratio relative to the item.** `relative_ratio` is the played ratio as a quantile among the
  item's train-period events, so every item's mean is 0.5 and a series that few finish stands on one scale
  with a film that most finish. The reward model predicts it with a fifth head, and
  `RewardSpec(relative_ratio_weight=...)` pays for it.
- **A KL budget.** The anchor weight is measured against standardised advantages, so the same weight is a
  different constraint under another estimator, reward or dataset. `RLConfig(target_kl=...)` adapts the weight
  until the policy sits that many nats per item from the base recommender; runs under one budget moved equally far.
- **A reward for the list.** `RewardSpec(diversity_weight=...)` adds each item's cosine distance, in the base
  recommender's embeddings, to the items before it. The greedy maximiser of the same reward
  (`gptrec.rollout.pick_greedily`, maximal marginal relevance) is the baseline RL has to beat on it.
- **Nothing is chosen on the test period.** Early stopping uses the validation period. Every sweep run is also
  evaluated on validation requests, and `kl_sweep_selection.csv` names the run chosen there.
- **RL is PPO with an anchor.** A value model started from the recommender's transformer estimates
  what each state promises; generalised advantage estimation turns rewards into advantages; every
  batch of sampled lists is used for `ppo_epochs` clipped gradient steps. Dropout is off so that
  recorded and recomputed log-probabilities agree. On top of PPO sits
  `reference_kl_weight * KL(policy || base policy)`: the reward never asks for relevance, so this
  penalty is the only thing that does. `AdvantageEstimator.LEAVE_ONE_OUT` with `ppo_epochs=1`
  gives the REINFORCE variant the first runs used.
- **The reward model is only trustworthy where it was trained.** It predicts logged feedback well
  (AUC 0.87, calibrated) and is still easy to exploit: with a weak anchor the policy moved to a
  handful of obscure tracks with extreme logged completion. Read `predicted/train/*` only together
  with `predicted/heldout/*` and the relevance on liked items.
- **Runs respect a time budget.** `Experiment(time_budget_hours=...)` skips sweep runs that would
  not finish, judging by the measured speed of the previous run, so a session with a hard limit
  ends with what was completed.
- **Generation constraints are part of the policy.** A list never repeats an item
  (`block_generated_items`). Re-recommending history items is allowed on Yambda, where repeats are
  normal and one of the behaviours under study, and blocked on KION, where each user-item pair
  occurs once (`block_history_items`). Baselines and re-ranking obey the same constraints.
- **Baselines differ in objective and architecture, not in size.** The neural baselines take the GPT
  recommender's width, depth, dropout and sequence length, and the same training windows. SASRec keeps
  its original loss (binary cross-entropy against sampled negatives); GRU4Rec uses the full softmax;
  BERT4Rec is trained on the cloze task and predicts by appending a mask. None of them is tuned per dataset.
- **RL has to beat two cheaper methods.** Liked-only fine-tuning uses logged feedback and no reward
  model; re-ranking uses the reward model but no training. Both are placed on the same
  relevance / reward plane as the RL runs, with the reward read from the held-out reward model.
- **Differences come with intervals.** `paired_comparison` bootstraps users for the change in
  relevance and predicted reward between two recommenders. That is the noise of the user sample, not of
  training: `gptrec.seeds.run_seeds` repeats the core stages once per seed, each in its own home directory, and
  `summarise_seeds` gives the mean and standard deviation of every reported number.
- **What the retention signals mean depends on the clock.** On Yambda the session continues with
  another listen within 30 minutes, and the user returns by starting a new session within a day. KION
  only has dates: a session is one calendar day, and the user returns by watching on another day within
  a week. The order of a KION user's titles within a day is the order of the file, so on KION
  `session_continues` says how many titles the day had more than which title kept the user watching.

## Layout

```
gptrec/
  data/          log.py (on-disk log), preprocess.py (raw events -> log), yambda.py and kion.py (adapters),
                 datasets.py (datasets by name), batches.py (tensors)
  models/        backbone.py (GPT-2 stack), sequence_recommender.py (shared scoring), policy.py, reward_model.py
  baselines/     counting.py (popularity, Markov chain), gru4rec.py, sasrec.py, bert4rec.py, suite.py (fits them all)
  rollout.py     item-by-item generation, simulated feedback, reward function, re-ranking, best of several sampled lists
  training/      supervised.py (all supervised training), rl.py (PPO fine-tuning), off_policy.py (REINFORCE on logged
                 sessions), checkpoint.py, precision.py
  evaluation/    relevance.py, significance.py, behavior.py, logged_feedback.py, reward_model_metrics.py, policy_report.py,
                 analyses.py (by position, by user, by item)
  experiment.py  the whole study as resumable stages; reports.py, plots.py write its tables and figures
  seeds.py       the core stages once per random seed, and the summary over seeds
  labels.py      names of the reported quantities in words, for figures and tables
notebooks/       the five interactive notebooks
kaggle/          build_notebooks.py and the four Kaggle notebooks it generates
results/         the CSV reports and the configuration of every session the paper reports, one folder per session
report/          the write-ups; tables.py typesets the paper's tables from results/, figures.py draws its figures
```

To add a dataset, write an adapter that returns a polars frame with the columns in
`gptrec.data.preprocess.RAW_EVENT_COLUMNS` and register it in `gptrec/data/datasets.py`.

## First run

1. Let the smoke test pass. It runs every stage on a small slice of the data with tiny models and
   exists to surface crashes in minutes; its numbers mean nothing.
2. On the first real run, check the invariants that guard against silent errors: the recommender's
   validation loss is below `ln(num_items)` and it beats the popularity floor; the reward model beats
   the `item mean` predictor; `kl_to_reference` is ~0 at RL step 1.
3. Then tune what could not be tuned blind: the RL learning rate and number of steps (watch
   `kl_to_reference` and `reward` in `rl_training.csv`), and batch sizes for the GPU at hand.

## References

- A. Petrov, C. Macdonald. *Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning.* arXiv:2403.04875.
- A. Matveev et al. *Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction.* arXiv:2607.02818.
- A. Ploshkin et al. *Yambda-5B: A Large-Scale Multi-modal Dataset for Ranking and Retrieval.* arXiv:2505.22238.
