"""Generates the Kaggle notebooks from the `gptrec` sources: python3 kaggle/build_notebooks.py"""

from __future__ import annotations

import ast
import io
import json
import tokenize
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
PACKAGE = REPOSITORY / "gptrec"
OUTPUT_DIRECTORY = Path(__file__).resolve().parent

WORKING = "/kaggle/working"
BUNDLE = f"{WORKING}/gptrec_bundle"
REQUIRED_PACKAGES = ["torch", "transformers", "polars", "pandas", "numpy", "huggingface_hub", "tqdm", "matplotlib"]
HIDDEN_INPUT = {"_kg_hide-input": True, "jupyter": {"source_hidden": True}}


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _source_lines(text)}


def code(text: str, metadata: dict | None = None) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": metadata or {}, "outputs": [], "source": _source_lines(text)}


def _source_lines(text: str) -> list[str]:
    lines = text.strip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def write_notebook(name: str, cells: list[dict]) -> None:
    for cell in cells:
        if cell["cell_type"] == "code":
            _require_no_comments("".join(cell["source"]))
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    (OUTPUT_DIRECTORY / name).write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")


def _python_source(cell_source: str) -> str:
    """The cell without its IPython magics."""
    return "\n".join(line for line in cell_source.split("\n") if not line.startswith("%"))


def _comments(source: str) -> list[tokenize.TokenInfo]:
    return [token for token in tokenize.generate_tokens(io.StringIO(source).readline) if token.type == tokenize.COMMENT]


def _require_no_comments(cell_source: str) -> None:
    comments = _comments(_python_source(cell_source))
    if comments:
        raise ValueError(f"a notebook cell contains a comment: {comments[0].string}")


def without_comments(source: str) -> str:
    """`source` with every comment removed; lines that held nothing else are dropped. The syntax tree is unchanged."""
    lines = source.split("\n")
    comment_only_lines = set()
    for comment in _comments(source):
        row, column = comment.start[0] - 1, comment.start[1]
        lines[row] = lines[row][:column].rstrip()
        if not lines[row]:
            comment_only_lines.add(row)
    stripped = "\n".join(line for row, line in enumerate(lines) if row not in comment_only_lines)
    if ast.dump(ast.parse(stripped)) != ast.dump(ast.parse(source)):
        raise ValueError("removing the comments changed the code")
    return stripped


def package_source_cells(source_directory: str) -> list[dict]:
    """Cells that write the `gptrec` package to `source_directory`."""
    modules = sorted(path for path in PACKAGE.rglob("*.py") if path.read_text().strip())
    packages = sorted({str(path.parent.relative_to(REPOSITORY)) for path in PACKAGE.rglob("*.py")})
    create_packages = code(
        f"""
from pathlib import Path

SOURCE_DIR = Path("{source_directory}")
for package in {packages}:
    (SOURCE_DIR / package).mkdir(parents=True, exist_ok=True)
    (SOURCE_DIR / package / "__init__.py").touch()
""",
        HIDDEN_INPUT,
    )
    write_modules = [
        code(f"%%writefile {source_directory}/{path.relative_to(REPOSITORY)}\n{without_comments(path.read_text())}", HIDDEN_INPUT) for path in modules
    ]
    return [markdown("## Package source"), create_packages, *write_modules]


ABSTRACT = markdown(r"""
# RL fine-tuning of a generative recommender against a learned reward model

A GPT-2 decoder over item identifiers is trained to predict a user's next item and is then fine-tuned with
reinforcement learning to generate recommendation lists with a higher predicted user response. The response is
predicted by a second decoder, the reward model. The study measures what the fine-tuning changes: the predicted
response, the relevance of the lists, and how much the lists of different users differ.

**Notation.** A user's log is a sequence of events $(i_t, \rho_t, c_t, u_t)$: item $i_t$, played ratio
$\rho_t \in [0, 1]$, session continuation $c_t \in \{0, 1\}$ and return $u_t \in \{0, 1\}$. The recommender is
$\pi_\theta$, and $\pi_0$ is its state before fine-tuning. A recommendation list $a_{1:K}$ is generated item by
item from a history $h$, and $F$ is the set of the next $M$ items the user consumed after $h$.
""")

CHECK_HARDWARE = code("""
import os

import polars
import torch
import transformers

print(f"torch {torch.__version__} | transformers {transformers.__version__} | polars {polars.__version__} | {os.cpu_count()} CPU cores")
assert torch.cuda.is_available(), "no CUDA device"
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(f"cuda:{index}: {properties.name}, {properties.total_memory / 2**30:.0f} GB, compute capability {torch.cuda.get_device_capability(index)}")

device = torch.device("cuda:0")
assert torch.isfinite((torch.randn(64, 64, device=device) @ torch.randn(64, 64, device=device)).sum()).item()
torch.set_float32_matmul_precision("high")
""")

IMPORTS = code("""
import shutil
from dataclasses import replace

import pandas as pd

from gptrec.baselines.suite import BaselineConfig
from gptrec.data.datasets import download_raw_data, load_raw_events
from gptrec.data.preprocess import PreprocessingConfig
from gptrec.experiment import DEFAULT_REWARDS, Experiment, ExperimentConfig, reuse_artifacts, run_smoke_test
from gptrec.models.backbone import TransformerConfig
from gptrec.rollout import GenerationConfig
from gptrec.seeds import run_seeds, run_seeds_smoke_test, summarise_seeds
from gptrec.training.rl import RLConfig
from gptrec.training.supervised import OptimizationConfig

SECONDS_PER_DAY = 86_400


def show(table):
    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.float_format", "{:.4f}".format):
        display(table)
""")

CHOOSE_DATASET = code("""
config = CONFIGS[(DATASET, SESSION)]
HOME = WORKING / f"{DATASET}_{SESSION}"
TIME_BUDGET_HOURS = 11.0
""")

SMOKE_TEST = [
    markdown("""
## Smoke test

All stages on a subsample of users, with small models and a few optimisation steps.
"""),
    code("""
run_smoke_test(raw_events, config, WORKING / "smoke_test" / DATASET, device, max_users=SMOKE_TEST_USERS[DATASET])
shutil.rmtree(WORKING / "smoke_test")
"""),
]

STAGES = [
    markdown(r"""
## 1. Data

Events are split on one time axis: the last $T_{\text{test}}$ days form the test period, the $T_{\text{val}}$ days
before them the validation period, everything earlier the training period. The catalogue consists of the most
frequent items of the training period, and item statistics are computed on the training period only.

Two retention signals are defined for an event at time $t$ from the times $t' \ge t$ of the same user's later events:

$$c = \mathbb{1}\big[\min_{t'}\,(t' - t) \le g\big], \qquad u = \mathbb{1}\big[\exists\, t' :\; g < t' - t \le w\big],$$

with session gap $g$ and return window $w$. An outcome counts as observed only if its waiting time ends inside the
observed data: $t + g \le T$ for $c$ and $t + w \le T$ for $u$, where $T$ is the end of the training period for
training events and the end of the log for held-out events. Unobserved outcomes enter neither training nor
evaluation.
"""),
    code("""
experiment = Experiment(config, HOME, device, time_budget_hours=TIME_BUDGET_HOURS)
summary = experiment.prepare_data(raw_events)
del raw_events
show(summary)
"""),
    markdown(r"""
## 2. Base recommender

A causal transformer (GPT-2) with tied input and output item embeddings $E$ models the item sequence of a user:

$$\pi_\theta(i_{t+1} = j \mid i_{1:t}) = \operatorname{softmax}_j\,(E\, s_t), \qquad
\mathcal{L}_{\text{next}}(\theta) = -\sum_t \log \pi_\theta(i_{t+1} \mid i_{1:t}),$$

where $s_t$ is the state of the transformer after $t$ items. A list of $K$ items is produced in two ways:

- **top-K**: the $K$ most probable next items under $\pi_\theta(\cdot \mid h)$;
- **next-K**: item by item, $a_k = \arg\max_j \pi_\theta(j \mid h, a_{1:k-1})$, without repeating an item.

Relevance is measured on one request per test user, issued at the start of the test period, against the user's
next $M$ events: all of them, those with $\rho \ge 0.5$ (*liked*) and those with $\rho < 0.5$ (*skipped*). A second
split compares the played ratio with the mean $\bar\rho(i)$ of the same item in the training period: $\rho \ge \bar\rho(i)$
(*above usual*) and $\rho < \bar\rho(i)$ (*below usual*); unlike the first, it does not favour items that are easy to
finish. A third split separates next items that are in the visible history (*repeat*) from those that are not
(*new*): replaying the history finds the first without any model. The metrics are Recall@K, NDCG@K, MAP@K and
HitRate@K.

The *logged* statistics need no model at all. A recommended item that is among the user's next events comes with
the response the log recorded: its played ratio $\rho$, and its relative played ratio $r$ defined in section 3.
They are reported over those items, and per recommended item with an item the user did not go on to consume
counting as nothing.
"""),
    code("""
show(experiment.pretrain_policy())
"""),
    markdown(r"""
## 3. Reward model

A second causal transformer reads, at position $t$, the item $i_t$ together with the quantised played ratio
$\rho_{t-1}$ of the previous item. Its state $z_t$ depends on $(i_{1:t}, \rho_{1:t-1})$: it has seen the candidate
item and the history, but not the response to the candidate. Five heads predict that response,

$$(\hat\rho_t,\ \hat c_t,\ \hat u_t,\ \hat q_t,\ \hat r_t) = \sigma(W z_t),$$

the played ratio, the session continuation, the return, the probability that the user chooses the item at all, and
the *relative played ratio*. The last is the played ratio as a quantile among the training-period events of the
same item,

$$r = \frac{\#\{\rho' < \rho\} + \tfrac{1}{2}\,\#\{\rho' = \rho\}}{n(i)},$$

so its mean is $0.5$ for every item, whatever the item's length. The loss is

$$\mathcal{L} = \operatorname{BCE}(\hat\rho, \rho) + \lambda_c \operatorname{BCE}(\hat c, c) + \lambda_u \operatorname{BCE}(\hat u, u) + \lambda_q\, \mathcal{L}_q + \lambda_r \operatorname{BCE}(\hat r, r),$$

with $\rho$ and $r$ as soft labels and the retention terms taken over observed outcomes only. Logged events contain only
chosen items, so $\hat q$ is learned by contrast. One event per sequence is scored a second time with its item
replaced by the item $j$ of another event of the batch, so that replacements follow the empirical item
distribution $p$:

$$\mathcal{L}_q = -\tfrac{1}{2}\, \mathbb{E}\big[\log \hat q(i_t \mid h_t)\big] - \tfrac{1}{2}\, \mathbb{E}_{j \sim p}\big[\log\big(1 - \hat q(j \mid h_t)\big)\big],
\qquad \hat q^{*}(j \mid h) = \frac{P(j \mid h)}{P(j \mid h) + p(j)}.$$

Two copies are trained on disjoint halves of the users. $R_{\text{train}}$ provides the rewards during fine-tuning;
$R_{\text{held-out}}$ only evaluates the resulting lists.

The first table compares the predicted played ratio on test-period events, each predicted from the events before
it, with three reference predictors: the training mean, the mean of the item, and the mean over the user's preceding
events. The second does the same for the relative played ratio, which no statistic of the item can predict.
"""),
    code("""
show(experiment.train_reward_models())
show(pd.read_csv(experiment.reports.directory / "reward_model_relative_ratio.csv", index_col=[0, 1]))
"""),
    markdown(r"""
**Error structure.** Quality of $\hat\rho$ separately for events the user initiated and events a recommender served;
mean signed error $\hat\rho - \rho$ by decile of item popularity and of item duration; correlation between the
predictions of the two copies on the same events.
"""),
    code("""
for report in ["reward_model_quality_by_origin", "reward_model_error_by_popularity_decile", "reward_model_error_by_duration_decile", "reward_model_agreement"]:
    print(report)
    show(pd.read_csv(experiment.reports.directory / f"{report}.csv", index_col=0))
"""),
    markdown(r"""
**What each signal offers as a reward.**

- *Retention*: prediction quality of $\hat c$ and $\hat u$ on events with observed outcomes.
- *Choice*: AUC of $\hat q$ for the logged item against a replaced one.
- *Spread over candidates*: for the $N$ most probable next items of each test user under $\pi_0$, the standard
  deviation of each predicted signal across the candidates of one user, averaged over users; the standard deviation
  of its per-user mean across users; and the AUC of the signal for the event that the candidate is in $F$. The last
  row gives the same for $\log \pi_0$.
"""),
    code("""
for report, index_columns in [("reward_model_retention", [0, 1, 2]), ("reward_model_choice", 0), ("reward_signal_spread", 0)]:
    print(report)
    show(pd.read_csv(experiment.reports.directory / f"{report}.csv", index_col=index_columns))
"""),
    markdown(r"""
## 4. Baselines

| Model | Definition |
|---|---|
| Most popular | the most frequent items of the training period |
| Replay history | the items of the user's history, most played first; the rest by popularity |
| Markov chain | first order, $P(j \mid i) \propto n(i \to j)$ |
| GRU4Rec [5] | GRU over the item sequence, full-softmax cross-entropy |
| SASRec [6] | causal transformer, binary cross-entropy against one sampled negative per position |
| BERT4Rec [7] | bidirectional transformer trained on masked-item prediction; a mask is appended at inference |

The neural baselines share the width, depth, dropout, sequence length and training windows of the recommender.
Every model ranks the catalogue for the same test requests under the same item constraints.
"""),
    code("""
show(experiment.compare_baselines())
"""),
    markdown(r"""
## 5. Reinforcement-learning fine-tuning

Generating a list is an episode of $K$ steps with state $s_k = (h, a_{1:k-1})$ and action
$a_k \sim \pi_\theta(\cdot \mid s_k)$. The reward is the response $R_{\text{train}}$ predicts to $a_k$; for the items
generated before $a_k$ the reward model is given its own expected played ratio as their feedback:

$$r_k = \hat\rho\,(a_k \mid h, a_{1:k-1}), \qquad G_k = \sum_{l \ge k} \gamma^{\,l-k}\, r_l .$$

A value model $V_\phi$, initialised from the recommender, regresses on $G_k$ and yields the advantages by
generalised advantage estimation [4],

$$\delta_k = r_k + \gamma V_\phi(s_{k+1}) - V_\phi(s_k), \qquad A_k = \sum_{l \ge 0} (\gamma\lambda)^l\, \delta_{k+l},$$

which are standardised within each batch. The policy minimises the clipped surrogate of PPO [3] with a KL penalty
towards the base recommender:

$$\mathcal{L}(\theta) = -\,\mathbb{E}\Big[\min\big(w_k A_k,\ \operatorname{clip}(w_k, 1 - \epsilon, 1 + \epsilon)\, A_k\big)\Big]
+ \beta\, \mathbb{E}\Big[\operatorname{KL}\big(\pi_\theta(\cdot \mid s_k)\,\|\,\pi_0(\cdot \mid s_k)\big)\Big],
\qquad w_k = \frac{\pi_\theta(a_k \mid s_k)}{\pi_{\theta_{\text{old}}}(a_k \mid s_k)} .$$

The reward contains no relevance term, so $\beta$ is the only constraint that ties the policy to observed
behaviour.

Fine-tuned lists are compared with those of $\pi_0$ on the test requests by relevance, by the response both reward
models predict, and by concentration: catalogue coverage, the Gini coefficient of item exposure, the share of users
who receive the most frequent item, and the expected overlap between the lists of two users.
"""),
    code("""
show(experiment.finetune())
"""),
    markdown(r"""
Per-user change from $\pi_0$ to $\pi_\theta$ with a 95% paired-bootstrap interval over users.
"""),
    code("""
show(pd.read_csv(experiment.reports.directory / "rl_vs_pretrained_significance.csv", index_col=0))
"""),
    markdown(r"""
The exposure of an item is the number of test users whose list contains it. The items with the largest gain and
the largest loss from $\pi_0$ to $\pi_\theta$:
"""),
    code("""
exposure = pd.read_csv(experiment.reports.directory / "rl_exposure_change.csv", index_col=0)
show(exposure.head(15))
show(exposure.tail(15))
"""),
    markdown(r"""
## 6. Relevance against reward

The fine-tuning is repeated for several values of $\beta$. Each run is a point: the predicted played ratio of its
lists against their NDCG@K. The ratio is predicted by $R_{\text{train}}$ and by $R_{\text{held-out}}$; a gap between
the two is a gain that only the optimised model predicts.

The second figure places the same runs by the divergence they ended at,
$\mathbb{E}\big[\operatorname{KL}(\pi_\theta(\cdot \mid s_k)\,\|\,\pi_0(\cdot \mid s_k))\big]$ in nats per generated item. The
weight $\beta$ acts relative to the standardised advantages, so its effect depends on the estimator and the data;
the divergence reached does not.

Every run is also evaluated on validation requests. The last table marks the run that would be chosen without the
test period: the highest predicted played ratio on validation requests among the runs whose validation NDCG@K stays
within a tolerance of the base recommender's.
"""),
    code("""
show(experiment.sweep_kl_weight())
show(pd.read_csv(experiment.reports.directory / "kl_sweep_selection.csv", index_col=0))
"""),
    markdown(r"""
## 7. A KL budget instead of an anchor weight

With a budget $\delta$ the anchor weight is no longer fixed. After every iteration it is multiplied by
$\exp\big(\eta\,(\operatorname{KL} / \delta - 1)\big)$, so it tightens while the policy is further than $\delta$ nats per
item from $\pi_0$ and loosens while it is nearer. Runs under the same budget have moved equally far, whatever the
advantage estimator, the reward or the dataset.
"""),
    code("""
show(experiment.sweep_kl_budget())
"""),
    markdown(r"""
## 8. Advantage estimators

The leave-one-out baseline replaces the value model: $R$ lists are sampled for each history, and
$A_k^{(m)} = G_k^{(m)} - \frac{1}{R - 1} \sum_{m' \ne m} G_k^{(m')}$. The number of lists per iteration, the learning
rate and the number of passes over each batch are those of section 6, so only the baseline differs. The runs are
compared at equal $\beta$ and under equal KL budgets. A last run multiplies the learning rate of the policy and
reports the share of actions whose probability ratio leaves $[1 - \epsilon, 1 + \epsilon]$.
"""),
    code("""
show(experiment.compare_advantage_estimators())
"""),
    markdown(r"""
## 9. Simulated feedback while fine-tuning

The reward model needs a response to every generated item before it can score the next one. Sections 5 and 6 feed
it its own expectation $\hat\rho_k$, a value near one half that logged responses rarely take. Here the response is
sampled, $\rho_k \sim \operatorname{Bernoulli}(\hat\rho_k)$. The second table judges each policy under every assumption:
the expectation, a sample, and "everything was played".
"""),
    code("""
show(experiment.compare_simulated_feedback())
if (experiment.reports.directory / "simulated_feedback_cross_judged.csv").exists():
    show(pd.read_csv(experiment.reports.directory / "simulated_feedback_cross_judged.csv", index_col=0))
"""),
    markdown(r"""
## 10. Whose reward is it

A list $a$ generated for a user with history $h$ can also be shown to another user with history $h'$. The reward
model then gives two values,

$$R_{\text{own}} = \mathbb{E}\big[\hat\rho(a \mid h)\big], \qquad R_{\text{other}} = \mathbb{E}\big[\hat\rho(a \mid h')\big],$$

and $R_{\text{own}} - R_{\text{other}}$ is the part of the predicted response that depends on who receives the list; it
is zero for any recommender that ignores the user. The gain of a method over $\pi_0$ splits accordingly into the gain
that other users would give the same lists, and the rest, its personal share:

$$\Delta R_{\text{own}} = \Delta R_{\text{other}} + \Delta\,(R_{\text{own}} - R_{\text{other}}).$$

As a reference that uses no reward model, the $N$ candidates of $\pi_0$ are re-ranked by
$\bar\rho(j) + w \log \pi_0(j \mid h)$, where $\bar\rho(j)$ is the mean played ratio of item $j$ in the training period.
"""),
    code("""
show(experiment.explain_reward_gain())
"""),
    markdown(r"""
## 11. Alternatives to on-policy RL

**Liked-only fine-tuning.** Next-item training restricted to targets the user liked:
$\mathcal{L}(\theta) = -\sum_{t:\ \rho_{t+1} \ge 0.5} \log \pi_\theta(i_{t+1} \mid i_{1:t})$.

**Off-policy REINFORCE on logged sessions** [2]. Logged items are the actions and $\pi_0$ is the behaviour policy:

$$\mathcal{L}(\theta) = -\frac{\sum_t \gamma^{\tau_t}\, \bar\omega_t\, G_t\, \log \pi_\theta(i_t \mid i_{1:t-1})}{\sum_t \gamma^{\tau_t}\, \bar\omega_t},
\qquad \bar\omega_t = \min\Big(\kappa,\ \prod_{l=t}^{t+L-1} \frac{\pi_\theta(i_l \mid i_{1:l-1})}{\pi_0(i_l \mid i_{1:l-1})}\Big),$$

where $G_t$ is the discounted sum of logged rewards from $t$ to the end of its session, $\tau_t$ is the position of
$t$ in the session, and the product runs over at most $L$ steps of the same session. The cap $\kappa$ bounds the
distance from $\pi_0$.

**Re-ranking.** The $N$ most probable next items under $\pi_0$ are scored one by one,
$\operatorname{score}(j) = r(j \mid h) + w \log \pi_0(j \mid h)$, and the $K$ best form the list.

**Best of $n$ sampled lists**, the test-time scaling of [2]. $n$ lists are sampled from $\pi_0$, restricted at each
step to its $m$ most probable items, and the list with the largest $\sum_k \gamma^{k-1} r_k$ is served.

Every method is placed by the response $R_{\text{held-out}}$ predicts to its lists, $R_{\text{own}}$ in the first figure
and $R_{\text{own}} - R_{\text{other}}$ in the second, against NDCG@K on all and on liked items and against catalogue
coverage.
"""),
    code("""
show(experiment.compare_alternatives_to_rl())
"""),
    markdown(r"""
## 12. Rewards for continuation and return

The reward generalises to

$$r_k = w_\rho\,\big(\hat\rho_k - b_k\big) + w_c\, \hat c_k + w_u\, \hat u_k + w_F\, \frac{\mathbb{1}[a_k \in F]}{\log_2(k + 1)},$$

with a baseline $b_k$ that is zero unless stated otherwise.

| Reward | $w_\rho$ | $w_c$ | $w_u$ |
|---|---|---|---|
| played ratio | 1 | 0 | 0 |
| session | 0 | 1 | 0 |
| return | 0 | 0 | 1 |
| ratio + return | 1 | 0 | 1 |
| ratio + session + return | 1 | 1 | 1 |

*Played time* replaces $\hat\rho_k$ by $\hat\rho_k\, d(a_k) / \operatorname{median}(d)$, where $d$ is the duration of
an item. The rewards selected in the configuration are run for several $\beta$. The figures show NDCG@K and catalogue
coverage against one signal predicted by $R_{\text{held-out}}$, one line per reward.
"""),
    code("""
show(experiment.sweep_rewards())
"""),
    markdown(r"""
## 13. Relevance inside the reward

**Logged relevance**, $r_k = \mathbb{1}[a_k \in F] / \log_2(k + 1)$, is the term of DCG@K, the NDCG reward of [1]. It is
run alone with $\beta = 0$, and added to the played ratio,

$$r_k = \hat\rho_k + w_F\, \frac{\mathbb{1}[a_k \in F]}{\log_2(k + 1)},$$

for several $w_F$ and several $\beta$. The two constraints protect different things: the reward term protects the hits
on the user's real continuation, the KL penalty protects the whole distribution of $\pi_0$, including its variety.
"""),
    code("""
show(experiment.sweep_relevance_in_reward())
"""),
    markdown(r"""
## 14. Personal rewards

A reward without a baseline pays the most for items that everybody finishes, whoever receives them. Two rewards pay
only for what depends on the user:

$$r_k = \hat\rho(a_k \mid h, a_{1:k-1}) - \hat\rho(a_k \mid h', a_{1:k-1}) \quad \text{(personal ratio)},
\qquad r_k = \hat\rho(a_k \mid h, a_{1:k-1}) - \bar\rho(a_k) \quad \text{(ratio above item mean)},$$

where $h'$ is the history of another user of the batch. For a policy that ignores the user the first has zero
expectation, $\mathbb{E}_{h, h'}\big[\hat\rho(a \mid h) - \hat\rho(a \mid h')\big] = 0$, so a list that is the same for
everyone earns nothing; it also cancels every error of the reward model that depends on the item alone. Both rewards
are run for several $\beta$. Re-ranking the candidates of $\pi_0$ by the personal ratio gives the same reward without RL.
"""),
    code("""
show(experiment.sweep_personal_rewards())
"""),
    markdown(r"""
## 15. A length-neutral reward

The played ratio pays for items that are easy to finish, whoever receives them. The reward here is the predicted
relative played ratio, $r_k = \hat r(a_k \mid h, a_{1:k-1})$, whose mean is $0.5$ for every item: a policy earns it
only by matching items to the users who take to them more than others do. Re-ranking the candidates of $\pi_0$ by
$\hat r$ is the reference without RL.
"""),
    code("""
show(experiment.sweep_relative_reward())
"""),
    markdown(r"""
## 16. A reward for the list as a whole

$$r_k = \hat\rho_k + w_D\, \frac{1}{K - 1} \sum_{l < k} \big(1 - \cos(e_{a_k}, e_{a_l})\big),$$

where $e$ are the item embeddings of $\pi_0$. Summed over a list, the second term is $K / 2$ times the mean pairwise
distance of its items, so what an item earns depends on its neighbours. Scoring items one at a time no longer finds
the best list. RL is compared with the greedy maximiser of the same reward, which fills the list one position at a
time from the candidates of $\pi_0$ (maximal marginal relevance), and with the best of several sampled lists.
"""),
    code("""
show(experiment.compare_list_reward())
if (experiment.reports.directory / "list_reward_methods.csv").exists():
    show(pd.read_csv(experiment.reports.directory / "list_reward_methods.csv", index_col=[0, 1]))
"""),
    markdown(r"""
## 17. Credit along the list

Two ablations test whether fine-tuning uses the sequential structure of a list. With $\gamma = 0$ the return reduces to
the reward of the item itself, $G_k = r_k$, so no item is credited with what follows it. With a horizon of one, a
single item is sampled per episode during fine-tuning, so the policy never learns from a context that contains
generated items; its lists are still generated item by item at evaluation. Both are compared with runs that credit
every item with the rest of the list, at the same $\beta$ and under the same KL budgets. Changing the credit changes
what $\beta$ means, so only the budget runs compare policies that moved equally far.
"""),
    code("""
show(experiment.ablate_sequence_credit())
"""),
    markdown(r"""
## 18. Multi-step generation

For each list position $k$: the probability that $a_k \in F$, the entropy of $\pi_\theta(\cdot \mid s_k)$, the
predicted signals, and the disagreement $|\hat\rho^{\,\text{train}}_k - \hat\rho^{\,\text{held-out}}_k|$ between the two
reward models.

A generated item has no observed feedback, but the reward model needs one to score the next item. Three
substitutes are compared: the expected ratio $\hat\rho_k$, a sample from $\operatorname{Bernoulli}(\hat\rho_k)$, and $1$.

The table gives NDCG@k of top-K and next-K lists before and after fine-tuning.
"""),
    code("""
show(experiment.analyse_multistep())
"""),
    markdown("""
## Reports
"""),
    code("""
show(pd.read_csv(experiment.reports.directory / "stage_minutes.csv", index_col=0))
shutil.make_archive(str(WORKING / f"{DATASET}_{SESSION}_reports"), "zip", experiment.reports.directory)
sorted(path.name for path in experiment.reports.directory.iterdir())
"""),
    markdown("""
## References

1. A. V. Petrov, C. Macdonald. Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning. arXiv:2403.04875.
2. A. Matveev, S. Makeev, A. Krasilnikov, V. Baikalov, S. Liamaev, K. Khrylchenko. Session-Level Optimization for
   Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction. arXiv:2607.02818.
3. J. Schulman, F. Wolski, P. Dhariwal, A. Radford, O. Klimov. Proximal Policy Optimization Algorithms. arXiv:1707.06347.
4. J. Schulman, P. Moritz, S. Levine, M. Jordan, P. Abbeel. High-Dimensional Continuous Control Using Generalized
   Advantage Estimation. arXiv:1506.02438.
5. B. Hidasi, A. Karatzoglou, L. Baltrunas, D. Tikk. Session-based Recommendations with Recurrent Neural Networks.
   arXiv:1511.06939.
6. W.-C. Kang, J. McAuley. Self-Attentive Sequential Recommendation. arXiv:1808.09781.
7. F. Sun et al. BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer.
   arXiv:1904.06690.
8. A. Ploshkin et al. Yambda-5B: A Large-Scale Multi-modal Dataset for Ranking and Retrieval. arXiv:2505.22238.
9. A. Petrov, I. Safilo, D. Tikhonovich, D. Ignatov. MTS Kion Implicit Contextualised Sequential Dataset for Movie
   Recommendation. arXiv:2209.00325.
"""),
]


T4_CONFIGS = code("""
DATASET = "kion"
SESSION = "t4"

CONFIGS = {
    ("yambda", "t4"): ExperimentConfig(
        preprocessing=PreprocessingConfig(max_items=30_000, min_item_events=20, min_user_events=50, validation_days=7, test_days=7),
        transformer=TransformerConfig(hidden_size=256, num_layers=4, num_heads=4, dropout=0.1, max_positions=128),
        window_stride=64,
        policy_optimization=OptimizationConfig(epochs=6, batch_size=64, learning_rate=1e-3),
        reward_model_optimization=OptimizationConfig(epochs=4, batch_size=128, learning_rate=1e-3),
        baselines=BaselineConfig(optimization=OptimizationConfig(epochs=3, batch_size=64, learning_rate=1e-3)),
        horizon=10,
        max_future=20,
        ks=(1, 5, 10),
        evaluation_batch_size=256,
        rl=RLConfig(num_steps=400, histories_per_step=256, value_warmup_steps=25, reference_kl_weight=0.3, learning_rate=3e-5, evaluate_every=100, mixed_precision=True),
        sweep_steps=250,
        kl_weights=(0.0, 0.1, 0.3, 1.0),
        rewards=DEFAULT_REWARDS[3:5],
        reward_kl_weights=(0.3,),
        choice_reward_kl_weights=(),
        logged_relevance_weights=(10.0,),
        logged_relevance_kl_weights=(0.0, 0.1),
        personal_reward_kl_weights=(0.0, 0.3),
        credit_ablation_kl_weights=(0.3,),
        importance_weight_caps=(0.5,),
        sampled_list_counts=(4,),
    ),
    ("kion", "t4"): ExperimentConfig(
        preprocessing=PreprocessingConfig(max_items=20_000, min_item_events=10, min_user_events=5, validation_days=7, test_days=7,
                                          session_gap_seconds=0, return_window_seconds=7 * SECONDS_PER_DAY),
        transformer=TransformerConfig(hidden_size=256, num_layers=4, num_heads=4, dropout=0.2, max_positions=48),
        window_stride=24,
        policy_optimization=OptimizationConfig(epochs=20, batch_size=256, learning_rate=1e-3),
        reward_model_optimization=OptimizationConfig(epochs=10, batch_size=256, learning_rate=1e-3),
        baselines=BaselineConfig(optimization=OptimizationConfig(epochs=20, batch_size=256, learning_rate=1e-3)),
        horizon=10,
        max_future=20,
        ks=(1, 5, 10),
        evaluation_batch_size=512,
        rl=RLConfig(num_steps=400, histories_per_step=256, value_warmup_steps=25, min_history=3, generation=GenerationConfig(block_history_items=True),
                    reference_kl_weight=0.3, learning_rate=3e-5, evaluate_every=100, mixed_precision=True),
        sweep_steps=250,
        kl_weights=(0.0, 0.1, 0.3, 1.0),
        rewards=DEFAULT_REWARDS[3:5],
        reward_kl_weights=(0.3,),
        choice_reward_kl_weights=(),
        logged_relevance_weights=(10.0,),
        logged_relevance_kl_weights=(0.0, 0.1),
        personal_reward_kl_weights=(0.0, 0.3),
        credit_ablation_kl_weights=(0.3,),
        importance_weight_caps=(0.5,),
        sampled_list_counts=(4,),
    ),
}
SMOKE_TEST_USERS = {"yambda": 200, "kion": 30_000}
""")

OFFLINE_CONFIGS = code("""
DATASET = "yambda"
SESSION = "final"

YAMBDA = ExperimentConfig(
    preprocessing=PreprocessingConfig(max_items=50_000, min_item_events=20, min_user_events=50, validation_days=7, test_days=7),
    transformer=TransformerConfig(hidden_size=256, num_layers=4, num_heads=4, dropout=0.1, max_positions=128),
    window_stride=32,
    policy_optimization=OptimizationConfig(epochs=14, batch_size=256, learning_rate=1e-3),
    reward_model_optimization=OptimizationConfig(epochs=8, batch_size=512, learning_rate=1e-3),
    baselines=BaselineConfig(optimization=OptimizationConfig(epochs=14, batch_size=256, learning_rate=1e-3)),
    horizon=10,
    max_future=20,
    ks=(1, 5, 10),
    evaluation_batch_size=1024,
    rl=RLConfig(num_steps=2000, histories_per_step=512, reference_kl_weight=0.3, learning_rate=1e-5, evaluate_every=250, mixed_precision=True),
    sweep_steps=1000,
    kl_weights=(0.0, 0.03, 0.05, 0.1, 0.3),
    rewards=DEFAULT_REWARDS[3:4],
    reward_kl_weights=(0.1,),
    choice_reward_kl_weights=(),
    logged_relevance_weights=(10.0,),
    logged_relevance_kl_weights=(0.0,),
    personal_reward_kl_weights=(0.1,),
    credit_ablation_kl_weights=(),
    importance_weight_caps=(0.5,),
    sampled_list_counts=(8,),
)
KION = ExperimentConfig(
    preprocessing=PreprocessingConfig(max_items=20_000, min_item_events=10, min_user_events=5, validation_days=7, test_days=7,
                                      session_gap_seconds=0, return_window_seconds=7 * SECONDS_PER_DAY),
    transformer=TransformerConfig(hidden_size=256, num_layers=4, num_heads=4, dropout=0.2, max_positions=48),
    window_stride=24,
    policy_optimization=OptimizationConfig(epochs=30, batch_size=512, learning_rate=1e-3),
    reward_model_optimization=OptimizationConfig(epochs=15, batch_size=512, learning_rate=1e-3),
    baselines=BaselineConfig(optimization=OptimizationConfig(epochs=30, batch_size=512, learning_rate=1e-3)),
    horizon=10,
    max_future=20,
    ks=(1, 5, 10),
    evaluation_batch_size=2048,
    rl=RLConfig(num_steps=2000, histories_per_step=512, min_history=3, generation=GenerationConfig(block_history_items=True),
                reference_kl_weight=0.3, learning_rate=1e-5, evaluate_every=250, mixed_precision=True),
    sweep_steps=1000,
    kl_weights=(0.0, 0.1, 0.2, 0.3, 0.5, 1.0),
    rewards=DEFAULT_REWARDS[2:5],
    reward_kl_weights=(0.1, 0.3),
    choice_reward_kl_weights=(),
    logged_relevance_weights=(10.0, 3.0),
    logged_relevance_kl_weights=(0.0, 0.03, 0.1, 0.3),
    personal_reward_kl_weights=(0.0, 0.1, 0.3),
    credit_ablation_kl_weights=(0.1, 0.3),
    importance_weight_caps=(0.5, 2.0),
    sampled_list_counts=(8,),
)

CONFIGS = {
    ("kion", "diagnostics"): replace(
        KION,
        kl_weights=(0.0, 0.1, 0.3),
        rewards=(),
        logged_relevance_weights=(),
        personal_reward_kl_weights=(),
        credit_ablation_kl_weights=(0.3,),
        importance_weight_caps=(0.5,),
        kl_budgets=(0.1, 0.5),
        estimator_kl_weights=(0.1, 0.3),
        high_learning_rate_factor=10.0,
        sampled_feedback_kl_weights=(0.1, 0.3),
        relative_reward_kl_weights=(0.0, 0.1, 0.3),
        diversity_weights=(1.0,),
        diversity_kl_weights=(0.1, 0.3),
    ),
    ("kion", "final"): replace(
        KION,
        kl_budgets=(0.1, 0.5),
        estimator_kl_weights=(0.1, 0.3),
        high_learning_rate_factor=10.0,
        sampled_feedback_kl_weights=(0.1, 0.3),
        relative_reward_kl_weights=(0.0, 0.1, 0.3),
        diversity_weights=(1.0,),
        diversity_kl_weights=(0.1, 0.3),
    ),
    ("kion", "seeds"): replace(
        KION,
        kl_weights=(0.1, 0.3, 0.5, 1.0),
        rewards=(),
        logged_relevance_weights=(10.0,),
        logged_relevance_kl_weights=(0.3,),
        personal_reward_kl_weights=(),
        credit_ablation_kl_weights=(),
        importance_weight_caps=(0.5,),
        sampled_list_counts=(),
        kl_budgets=(0.1,),
        estimator_kl_weights=(0.3,),
        relative_reward_kl_weights=(0.3,),
    ),
    ("yambda", "diagnostics"): replace(
        YAMBDA,
        kl_weights=(0.0, 0.03, 0.1),
        rewards=(),
        logged_relevance_weights=(),
        personal_reward_kl_weights=(),
        kl_budgets=(0.5,),
        estimator_kl_weights=(0.03, 0.1),
        relative_reward_kl_weights=(0.03, 0.1),
    ),
    ("yambda", "final"): replace(
        YAMBDA,
        logged_relevance_kl_weights=(0.0, 0.1),
        kl_budgets=(0.2, 0.5),
        estimator_kl_weights=(0.03, 0.1),
        sampled_feedback_kl_weights=(0.1,),
        relative_reward_kl_weights=(0.03, 0.1),
        diversity_weights=(1.0,),
        diversity_kl_weights=(0.1,),
    ),
    ("yambda", "seeds"): replace(
        YAMBDA,
        kl_weights=(0.03, 0.1),
        rewards=(),
        logged_relevance_weights=(),
        logged_relevance_kl_weights=(),
        personal_reward_kl_weights=(),
        importance_weight_caps=(0.5,),
        sampled_list_counts=(),
        kl_budgets=(0.5,),
    ),
}
SMOKE_TEST_USERS = {"yambda": 200, "kion": 30_000}
""")


def full_pipeline_on_t4() -> list[dict]:
    source_directory = f"{WORKING}/src"
    return [
        ABSTRACT,
        *package_source_cells(source_directory),
        markdown("## Setup"),
        code(f"""
import importlib.util
import os
import subprocess
import sys

WORKING = Path("{WORKING}")
os.environ["GPTREC_HOME"] = str(WORKING)
sys.path.insert(0, str(SOURCE_DIR))

missing = [package for package in {REQUIRED_PACKAGES} if importlib.util.find_spec(package) is None]
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", *missing])
"""),
        CHECK_HARDWARE,
        IMPORTS,
        markdown("## Configuration"),
        T4_CONFIGS,
        CHOOSE_DATASET,
        code("""
raw_events = load_raw_events(DATASET, download_raw_data(DATASET))
print(f"{len(raw_events):,} raw events")
"""),
        *SMOKE_TEST,
        *STAGES,
    ]


def prepare_offline_bundle() -> list[dict]:
    return [
        markdown("""
# Offline bundle

The package source, the wheels of its dependencies and both datasets in one directory, with the result of a smoke
test of the pipeline on each dataset.
"""),
        code("""
DOWNLOAD_TORCH = False
"""),
        *package_source_cells(f"{BUNDLE}/src"),
        markdown("## Dependency wheels"),
        code(f"""
import subprocess
import sys

BUNDLE_DIR = Path("{BUNDLE}")
WHEELS_DIR = BUNDLE_DIR / "wheels"
packages = {[package for package in REQUIRED_PACKAGES if package != "torch"]} + (["torch"] if DOWNLOAD_TORCH else [])
subprocess.check_call([sys.executable, "-m", "pip", "download", "--quiet", "--only-binary=:all:", "--dest", str(WHEELS_DIR), *packages])
(BUNDLE_DIR / "builder_environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
print(len(list(WHEELS_DIR.iterdir())), "wheels")
"""),
        markdown("## Datasets"),
        code("""
import os

os.environ["GPTREC_HOME"] = str(BUNDLE_DIR)
sys.path.insert(0, str(SOURCE_DIR))

from gptrec.data.datasets import DATASETS, download_raw_data

raw_data_paths = {dataset: download_raw_data(dataset) for dataset in DATASETS}
for dataset, path in raw_data_paths.items():
    print(dataset, path, f"{path.stat().st_size / 2**20:.0f} MB")
"""),
        markdown("""
## Smoke test

All stages of each dataset's pipeline on a subsample of users, with small models and a few optimisation steps.
"""),
        IMPORTS,
        OFFLINE_CONFIGS,
        code(f"""
import traceback

import torch

smoke_test_results = {{}}
for dataset in DATASETS:
    try:
        smoke_test_events = load_raw_events(dataset, raw_data_paths[dataset])
        smoke_test_home = Path("{WORKING}/smoke_test") / dataset
        run_smoke_test(smoke_test_events, CONFIGS[(dataset, "final")], smoke_test_home / "experiment", torch.device("cpu"), max_users=SMOKE_TEST_USERS[dataset])
        run_seeds_smoke_test(smoke_test_events, CONFIGS[(dataset, "seeds")], smoke_test_home / "seeds", torch.device("cpu"), max_users=SMOKE_TEST_USERS[dataset])
        smoke_test_results[dataset] = "passed"
    except Exception:
        smoke_test_results[dataset] = "FAILED\\n" + traceback.format_exc()
    print(f"smoke test on {{dataset}}:", smoke_test_results[dataset])
shutil.rmtree("{WORKING}/smoke_test", ignore_errors=True)
"""),
        markdown("## Manifest"),
        code("""
import json

manifest = {
    "raw_data": {dataset: os.path.relpath(path, BUNDLE_DIR) for dataset, path in raw_data_paths.items()},
    "wheels": sorted(path.name for path in WHEELS_DIR.iterdir()),
    "smoke_tests": smoke_test_results,
    "python": sys.version,
}
(BUNDLE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

bundle_bytes = sum(path.stat().st_size for path in BUNDLE_DIR.rglob("*") if path.is_file())
print(f"bundle: {bundle_bytes / 2**30:.2f} GB in {BUNDLE_DIR}")
for dataset, result in smoke_test_results.items():
    print(f"smoke test on {dataset}:", result.splitlines()[0])
"""),
    ]


OFFLINE_SETUP = code(f"""
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

MAX_MOUNT_DEPTH = 4
manifests = [
    manifest_path
    for depth in range(1, MAX_MOUNT_DEPTH + 1)
    for manifest_path in sorted(Path("/kaggle/input").glob("*/" * depth + "gptrec_bundle/manifest.json"))
]
assert manifests, "gptrec_bundle not found under /kaggle/input"
BUNDLE_DIR = manifests[0].parent
manifest = json.loads((BUNDLE_DIR / "manifest.json").read_text())
print("bundle:", BUNDLE_DIR)
for dataset, result in manifest["smoke_tests"].items():
    print(f"smoke test on {{dataset}}:", result.splitlines()[0])

WORKING = Path("{WORKING}")
os.environ.update(GPTREC_HOME=str(WORKING), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
sys.path.insert(0, str(SOURCE_DIR))

missing = [package for package in {REQUIRED_PACKAGES} if importlib.util.find_spec(package) is None]
if missing:
    wheels = BUNDLE_DIR / "wheels"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-index", f"--find-links={{wheels}}", *missing])
""")

REUSE_EARLIER_SESSIONS = [
    markdown("""
## Earlier sessions

A session named in `EARLIER_SESSIONS` continues an earlier one instead of repeating it: the interaction log, the
models and the finished fine-tuning runs of the earlier session are copied from its attached output, and only the
runs it does not have are trained. The earlier session must have had the same configuration in everything but the
grids of its sweeps. The models listed below are the ones that will not be trained again.
"""),
    code("""
EARLIER_SESSIONS = {
    ("yambda", "final"): "/kaggle/input/datasets/ventihi/yambda-diagnostics/yambda_diagnostics",
}

earlier_session = EARLIER_SESSIONS.get((DATASET, SESSION))
if earlier_session is None:
    print("no earlier session is continued: every model is trained in this one")
else:
    reuse_artifacts(earlier_session, HOME, config)
    reused_models = sorted(str(path.relative_to(HOME / "artifacts")) for path in (HOME / "artifacts").rglob("*.pt"))
    print(f"continuing {earlier_session}: {len(reused_models)} trained models")
    print("\\n".join(reused_models))
"""),
]

LOAD_BUNDLED_EVENTS = code("""
raw_events = load_raw_events(DATASET, BUNDLE_DIR / manifest["raw_data"][DATASET])
print(f"{len(raw_events):,} raw events")
""")


def train_offline() -> list[dict]:
    return [
        ABSTRACT,
        *package_source_cells(f"{WORKING}/src"),
        markdown("## Setup"),
        OFFLINE_SETUP,
        CHECK_HARDWARE,
        IMPORTS,
        markdown("## Configuration"),
        OFFLINE_CONFIGS,
        CHOOSE_DATASET,
        LOAD_BUNDLED_EVENTS,
        *SMOKE_TEST,
        *REUSE_EARLIER_SESSIONS,
        *STAGES,
    ]


# Seed 0 is the seed of notebook 3's sessions, so two more seeds give three per dataset. Two Yambda seeds (about
# 4.7 hours each) are what fits the 12-hour limit.
SEED_SESSIONS = {
    "4_seeds_kion_1_2.ipynb": ("kion", (1, 2)),
    "4_seeds_yambda_1_2.ipynb": ("yambda", (1, 2)),
}


def seeds_offline(dataset: str, seeds: tuple[int, ...]) -> list[dict]:
    return [
        ABSTRACT,
        markdown("""
This notebook repeats the core of the study under several random seeds. A seed fixes the initial weights of the
recommender and of the reward models, the order of their training batches, and the lists sampled while fine-tuning;
the interaction log and the test requests are the same for every seed. Each table gives the mean and the standard
deviation over the seeds of this session. Seed 0 is the seed of the single-seed sessions of notebook 3, so it is
not repeated here.
"""),
        *package_source_cells(f"{WORKING}/src"),
        markdown("## Setup"),
        OFFLINE_SETUP,
        CHECK_HARDWARE,
        IMPORTS,
        markdown("## Configuration"),
        OFFLINE_CONFIGS,
        code(f"""
DATASET = "{dataset}"
SEEDS = {seeds}
SESSION = "seeds"
config = CONFIGS[(DATASET, SESSION)]
HOME = WORKING / f"{{DATASET}}_seeds_{{'_'.join(map(str, SEEDS))}}"
TIME_BUDGET_HOURS = 11.0
"""),
        LOAD_BUNDLED_EVENTS,
        markdown("""
## Smoke test

Two seeds on a subsample of users, with small models and a few optimisation steps.
"""),
        code("""
run_seeds_smoke_test(raw_events, config, WORKING / "smoke_test" / DATASET, device, max_users=SMOKE_TEST_USERS[DATASET])
shutil.rmtree(WORKING / "smoke_test")
"""),
        markdown("""
## Runs

Per seed: the base recommender, both reward models, the main fine-tuning run, the anchor sweep, a run under a KL
budget with its leave-one-out and myopic-credit counterparts, the split of the reward gain between the user and
other users, the alternatives to on-policy RL, and the rewards with a logged-relevance term and with the relative
played ratio. A stage whose grid is empty for the dataset is skipped.
"""),
        code("""
report_directories = run_seeds(raw_events, config, SEEDS, HOME, device, time_budget_hours=TIME_BUDGET_HOURS)
del raw_events
"""),
        markdown("## Mean and standard deviation over seeds"),
        code("""
summaries = summarise_seeds(report_directories, HOME / "over_seeds")
for report, summary in summaries.items():
    print(report)
    show(summary)
"""),
        markdown("## Reports"),
        code("""
archive = WORKING / f"{HOME.name}_reports"
shutil.copytree(HOME / "over_seeds", archive / "over_seeds", dirs_exist_ok=True)
for seed, directory in report_directories.items():
    shutil.copytree(directory, archive / f"seed_{seed}", dirs_exist_ok=True)
shutil.make_archive(str(archive), "zip", archive)
sorted(path.name for path in (archive / "over_seeds").iterdir())
"""),
    ]


if __name__ == "__main__":
    write_notebook("1_full_pipeline_t4.ipynb", full_pipeline_on_t4())
    write_notebook("2_prepare_offline_bundle.ipynb", prepare_offline_bundle())
    write_notebook("3_train_offline.ipynb", train_offline())
    for name, (dataset, seeds) in SEED_SESSIONS.items():
        write_notebook(name, seeds_offline(dataset, seeds))
    print("wrote", sorted(path.name for path in OUTPUT_DIRECTORY.glob("*.ipynb")))
