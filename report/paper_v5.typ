// The paper, version 5.
// Build: python3 report/tables.py && python3 report/figures.py && typst compile report/paper_v5.typ
// Tables come from results/<session>/*.csv through report/tables.py, the SVG figures through report/figures.py; the
// numbers quoted in the prose are read from the same reports. Bibliography: report/references.bib.
// report/archive/paper_v5_changes.md lists what changed from version 4 (report/archive/paper_v4.typ).

#set document(
  title: "What Keeps a Generative Recommender Relevant? An Empirical Study of Reinforcement Fine-Tuning Against a Learned Reward Model",
  author: "Daniil Antonov",
)
#set page(paper: "a4", margin: (x: 2.1cm, y: 2.2cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt, lang: "en")
#set par(justify: true, leading: 0.6em, spacing: 0.85em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set block(above: 1.3em, below: 0.7em)
#show heading.where(level: 2): set block(above: 1.1em, below: 0.6em)
#show figure.caption: set text(size: 8.5pt)
#show figure.where(kind: table): set figure.caption(position: top)
#show table: set text(size: 8pt, hyphenate: false)
#show table: set par(justify: false)
#set list(indent: 0.6em, spacing: 0.6em)
#set enum(indent: 0.6em)
#set math.equation(numbering: "(1)")
#show math.equation.where(block: false): box
#set figure(placement: auto, gap: 0.6em)

// A table with rules above, below and under the header only.
#let results-table(columns: (), align: auto, header: (), inset-x: 4pt, ..rows) = table(
  columns: columns,
  align: align,
  stroke: none,
  inset: (x: inset-x, y: 2.8pt),
  table.hline(stroke: 0.8pt),
  table.header(..header.map(cell => strong(cell))),
  table.hline(stroke: 0.4pt),
  ..rows,
  table.hline(stroke: 0.8pt),
)

// A table typeset by report/tables.py from the reports of the runs.
#let report-table(name, caption, size: 8pt) = figure(
  kind: table,
  {
    show table: set text(size: size)
    include "tables/" + name + ".typ"
  },
  caption: caption,
)

// The answer to one of the five questions, closing its section.
#let answer(question, body) = block(
  width: 100%,
  breakable: false,
  inset: (x: 7pt, y: 5pt),
  radius: 2pt,
  stroke: (left: 1.6pt + luma(110)),
  fill: luma(246),
)[*Answer to #question.* #body]

// One box of the pipeline figure.
#let stage(title, body) = rect(width: 100%, height: 2.75cm, inset: 5pt, radius: 2pt, stroke: 0.6pt + luma(90))[
  #set text(size: 7.5pt)
  #set par(justify: false, leading: 0.45em)
  #set align(left + top)
  #strong(title) \
  #body
]

#align(center)[
  #text(size: 15pt, weight: "bold")[
    What Keeps a Generative Recommender Relevant? \
    An Empirical Study of Reinforcement Fine-Tuning Against a Learned Reward Model
  ]
  #v(0.5em)
  #text(size: 11pt)[Daniil Antonov] \
  ITMO University, Saint Petersburg, Russia \
  #text(size: 9pt)[October 2026]
]

#pad(x: 1cm)[
  #set text(size: 9.5pt)
  *Abstract.* A recommender that generates its list item by item is a policy, so reinforcement learning (RL) can
  fine-tune it towards feedback that next-item training does not see. We study the setting with the fewest
  safeguards. PPO fine-tunes a GPT-2 recommender to raise the share of each item that users consume, as predicted by a
  second GPT-2, the reward model. The only term that asks for relevance is a KL penalty towards the pretrained
  recommender, the _anchor_. Because every reward gain is itself a prediction, we check it with signals the policy
  cannot influence: a reward model trained on other users, the logged responses, and the same lists scored for another
  user. We run on KION (video) and Yambda-50M (music), with three seeds for every central comparison. On KION, RL
  raises the predicted share by 8.3% for a 1.8% loss of NDCG\@10 at the anchor weight chosen on validation, and by 24%
  for an 11.5% loss at a weaker one. What RL learns is mostly one re-weighting of the catalogue, the same for every
  user, towards the titles that viewers finish most often. This re-weighting is most of the reward's own optimum. RL
  settings therefore only move the policy along a frontier between relevance and reward, and the closed-form optimum,
  computed at inference, reaches the same frontier without training. The frontier itself moves only when training
  draws on the log or the list format changes. Reward-weighted fine-tuning on the logged share relative to what is
  typical for each item needs no reward model. Over the same extra epoch of plain training, it raises users' logged
  response beyond what is typical for each item by 21% and personalisation by 19%. Predicting each list position from
  the real history is 6% more relevant than generating the list.
]

= Introduction

Generative sequential recommenders @kang2018sasrec @petrov2023gptrec @rajput2023tiger @zhai2024actions @deng2025onerec
are trained to continue logged sequences. But the log records what users did, not what served them: 36% of music
listens in Yambda stop before half of the track, and 52% of KION titles are abandoned before half of their length. A
recommender that generates a list item by item is a policy, so RL can move it towards what will be received well
@chen2018topk @ie2019slateq @petrov2024aligning @matveev2026session. How safe this is depends on where the reward
comes from. Petrov and Macdonald @petrov2024aligning keep relevance inside the reward. Matveev et al.
@matveev2026session learn from logged feedback and use a learned feedback model to evaluate and select
recommendations at inference time. Letting a learned model of user feedback _be_ the training reward is the option with
the fewest protections. A policy optimised against a learned model drifts to where the model is wrong
@gao2022scaling @skalse2022hacking, and in recommendation this means popularity and filter bubbles
@gao2023dorl @jiang2019degenerate.

We study this option in its barest form. A GPT-2 @radford2019gpt2 recommender is fine-tuned to raise the consumed
share that a second GPT-2 predicts. A KL penalty towards the pretrained recommender, the _anchor_, is the only term
that asks for relevance. Since a model reports every reward gain in this setting, the study rests on instruments that
the optimised model cannot influence, and on three seeds for every central comparison. Beyond measuring the trade-off,
we ask why it arises and what moves it. For this we borrow probes from the post-training of language models: RL
restricted to one score offset per item or to rank-1 updates; the policy's change split into a per-item part and the
rest; the closed-form optimum computed at inference, with and without a learned look-ahead; learning from logged
responses alone; and lists predicted position by position from the real history.

The study answers five questions about RL fine-tuning of a next-item recommender. @tab-answers answers each in brief,
@sec-results gives the evidence in one subsection per question, and @tab-glossary defines the terms that recur.

#figure(
  results-table(
    columns: (3.3cm, 1fr),
    align: (left, left),
    header: ([Question], [Answer]),
    [*Q1.* How does the model's behaviour change after RL fine-tuning?],
    [The predicted reward rises ($+24%$ on KION), and a reward model trained on other users agrees. But what RL learns
      is mostly one re-weighting of the catalogue, the same for every user, towards titles that everybody finishes.
      That re-weighting is most of the reward's own optimum.],
    [*Q2.* Is relevance preserved? Does the model recommend the same items to everyone?],
    [Relevance is traded along a frontier. NDCG\@10 falls by 1.8% at the anchor weight chosen on validation and by
      11.5% at the main run, most for users with short histories. Without the anchor every user receives the same
      list. On Yambda the main run keeps both relevance and the spread of impressions.],
    [*Q3.* Can we get closer to long-term goals?],
    [Return, likes and adoption can be predicted and rewarded, but their predictions rise by only a few percent. The
      largest predicted gain lies beyond anything in the log; only an online test can show whether users return.],
    [*Q4.* What happens when generating several steps ahead?],
    [A generated list hits more often after a hit and less often after a miss, and simulated responses bias the
      reward of later steps. Predicting each position from the real history removes the loss ($+6.3%$ NDCG\@10).],
    [*Q5.* What is the trade-off between relevance and reward?],
    [RL settings choose a point on the frontier, and inference reaches it without training. The frontier moves when
      training draws on the log: reward-weighted fine-tuning on the logged relative share raises the logged surplus by
      21% and personalisation by 19% over the same epoch of plain training.],
  ),
  caption: [The five questions and their answers, on KION unless Yambda is named.],
  placement: auto,
) <tab-answers>

#figure(
  results-table(
    columns: (3.4cm, 1fr),
    align: (left, left),
    header: ([Term], [Meaning]),
    [Base recommender $pi_0$], [The pretrained GPT-2 next-item model. Every method starts from it.],
    [Generated, ranked list], [Ten items chosen one at a time, each appended to the history as if consumed; or the ten
      items $pi_0$ scores highest for the real history.],
    [Hit, miss], [A recommended item that is, or is not, among the user's next 20 logged items.],
    [Consumed share $rho$], [The share of a title watched or a track played (`played_ratio` in the code).],
    [Relative share $q$], [$rho$ as a quantile among the item's training-period responses. 0.5 is usual for the item,
      so no item earns it by being easy to finish.],
    [Reward model], [A second GPT-2 that predicts the response to an item before it is consumed. The _training_ copy
      gives the rewards; the _held-out_ copy, trained on other users, only judges.],
    [Anchor $beta$, KL distance], [The KL penalty towards $pi_0$ and its weight; and how far a policy ended from
      $pi_0$, in nats per generated item. The distance can be compared between runs; the weight cannot.],
    [Main run], [RL on the default reward at $beta = 0.3$. It shows a clear move along the frontier; it is not the
      operating point we recommend, which is chosen on validation.],
    [Frontier], [The highest predicted share reached at each level of relevance. Moving along it trades one for the
      other; a method that moves it gains both.],
    [Logged share, logged surplus], [Per recommended item, the response the log recorded if the item was a hit, else
      zero; the logged share mostly counts hits. The surplus counts $q - 1 \/ 2$ instead: it is length-neutral and
      positive only if users took to the items more than usual. Neither needs a model.],
    [Personal part], [The predicted share of a list for its own user minus that of the same list scored for another
      user. "$x%$ of a gain is personal": the personal part's share of the gain.],
    [Per-item share], [The share of a change within one history that one effect per item explains (@eq-per-item): 1 if
      every user's items move alike.],
    [Item offsets], [RL restricted to one score offset per item, the same for every user.],
    [Re-ranking, guided decoding], [The closed-form optimum $pi_0 exp(r \/ beta)$ computed at inference, over the
      candidates of $pi_0$ at once or one position at a time.],
    [Reward-weighted fine-tuning], [Next-item training on logged items, each weighted by $exp(r \/ tau)$ of its logged
      response. It needs no reward model.],
    [One head per position], [A linear head per list position that predicts the item $k$ events ahead from the real
      history.],
  ),
  caption: [Terms used throughout.],
  placement: auto,
) <tab-glossary>

= Related Work

Next-item models @hidasi2015gru4rec @kang2018sasrec @sun2019bert4rec run autoregressively as generative recommenders
over items @petrov2023gptrec or semantic identifiers @rajput2023tiger, up to industrial scale
@zhai2024actions @deng2025onerec. RL enters recommendation through off-policy policy gradients @chen2018topk, slate
values @ie2019slateq and RL heads @xin2020sqn, and learned simulators replace the user @ie2019recsim @zhao2023kuaisim.
A policy optimised against a learned model exploits its errors @yu2020mopo @kidambi2020morel @gao2022scaling
@skalse2022hacking. An ensemble of reward models does not stop this, because models trained alike share their errors
@eisenstein2023helping. Penalising divergence from a pretrained model is KL-control
@jaques2019wayoff @ziegler2019finetuning @ouyang2022instructgpt; Gao et al. @gao2022scaling compare best-of-$n$
sampling with RL along this divergence. Consumption time is confounded by item duration
@zhan2022d2q @zhao2023watchtime, retention has been optimised directly @zou2019longterm @cai2023retention, and maximal
marginal relevance @carbonell1998mmr greedily optimises a list-dependent reward.

Our probes come from the post-training of language models: low-rank updates @hu2022lora, which match full RL even at
rank 1 @schulman2025lora; the KL-anchored optimum computed at decoding time with a learned value of what follows
@mudgal2024controlled @zhou2025qsharp; learning from logged rewards without a reward model @peng2019awr
@chidambaram2026robust and from preference pairs @rafailov2023dpo; weight interpolation and extrapolation
@wortsman2022wiseft @rame2024warp @zheng2025expo; group-relative advantages @shao2024deepseekmath; and one head per
future position @gloeckle2024mtp.

Closest to this study, Chidambaram et al. @chidambaram2026robust find that a learned reward model for a generative
recommender predicts no better than item means. PPO and DPO against it collapse ranking metrics, while exponential
reward-weighted fine-tuning on logged rewards does better. They derive the KL-anchored optimum in closed form and note
that standardising rewards rescales its temperature. OneRec-V2 @zhou2025onerecv2 found its reward model open to reward
hacking and moved to user feedback, scoring watch time as a percentile within the user's own history of videos of
similar duration. Our reward model, in contrast, predicts responses far better than item means (AUC 0.755 against
0.656 on KION), yet RL extracts only its per-item part. We trace this before any collapse, at controlled KL distances
and on two domains. We show whether the reward or the optimiser sets the per-item part, and how much of RL's gain and
loss it carries. Finally, we find what moves the frontier that it defines: their reward-weighted fine-tuning with a
reward that compares a response with what is typical for the item, and lists predicted from the real history.

= Method <sec-method>

#figure(
  [
    #grid(
      columns: (1fr, 12pt, 1fr, 12pt, 1fr, 12pt, 1fr),
      align: horizon + center,
      stage[1. Pretrain][Recommender $pi_0$: GPT-2 over item tokens, next-item cross-entropy on logged sequences.],
      sym.arrow.r,
      stage[2. Generate][Policy $pi_theta$, started from $pi_0$, samples a list of $K = 10$ items for a logged
        history; each item is appended as if consumed.],
      sym.arrow.r,
      stage[3. Reward][The training reward model predicts the response to each item: consumed share, relative share,
        continuation, return; on Yambda like, dislike, adoption.],
      sym.arrow.r,
      stage[4. Update][Policy gradient on the predicted returns plus the anchor $beta dot "KL"(pi_theta || pi_0)$.
        Steps 2--4 repeat.],
    )
    #v(3pt)
    #rect(width: 100%, inset: 5pt, radius: 2pt, stroke: (paint: luma(90), thickness: 0.6pt, dash: "dashed"))[
      #set text(size: 7.5pt)
      #set par(justify: false)
      #set align(left)
      *Evaluation only:* a held-out reward model fitted on the other half of the users #sym.dot.c the logged response
      to recommended items the user went on to consume #sym.dot.c the same lists scored for another user #sym.dot.c
      relevance by outcome, user and position #sym.dot.c concentration #sym.dot.c the reward model on real sequences
      #sym.dot.c the per-item share of the policy's and of the optimum's change.
    ]
  ],
  caption: [The pipeline. The reward model of step 3 is the only source of reward; the instruments in the dashed box
    never influence the policy.],
  placement: none,
) <fig-pipeline>

*Models.* A log is a set of events with a user, an item, a time and the _consumed share_ $rho in [0, 1]$ of the track
or title. The recommender $pi_0$ is a GPT-2 decoder @radford2019gpt2 over item tokens, trained with the full-softmax
next-item cross-entropy. A list of $K = 10$ is obtained by _ranking_, which scores the catalogue once, or by
_generation_, which appends each chosen item to the history as if consumed @petrov2023gptrec. RL fine-tunes the
generation mode.

The _reward model_ is a second GPT-2. Its token at position $t$ carries the item $i_t$ and the consumed share of the
previous item, so it scores a candidate before its outcome exists. Its heads are trained with binary cross-entropy and
predict:
- the consumed share $hat(rho)$, with $rho$ as a soft label;
- the _relative share_ $hat(q)$: the share as a mid-rank quantile among the item's training-period shares. It
  averages one half for every item, so an item cannot earn it by being easy to finish;
- the probabilities that the session continues, $hat(c)$, and that the user returns, $hat(u)$;
- on Yambda, the probabilities of a like and of a dislike, and of _adoption_ $hat(alpha)$: a first play followed by
  another play in a later session within 14 days.

The log holds only items that users chose, so every head describes the response to an item _given_ that the user
starts it. Two copies are trained on disjoint halves of the users: the _training_ model gives the rewards, the
_held-out_ model only judges. A third pair is trained with _hidden responses_. In half of its training windows, the
responses to the last 1 to 10 events are replaced by an "unknown" value, which is the situation of a generated list.

*Fine-tuning.* An episode generates one list for a training history. The reward model is the environment, and it is
fed its own expected response for each earlier generated item. The policy gradient uses one of three estimators: a
learned value model with generalised advantage estimates @schulman2015gae and four steps of PPO's clipped surrogate
per batch @schulman2017ppo; a leave-one-out baseline over four lists per history @kool2019buy; or group-relative
advantages (GRPO) over the same four lists @shao2024deepseekmath. Advantages are standardised in every case. The loss
adds the exact divergence from $pi_0$ at every position:
$ cal(L)(theta) = cal(L)_"PG" (theta) + beta dot EE_k ["KL"(pi_theta (dot | s_k) || pi_0 (dot | s_k))]. $ <eq-loss>
Instead of a fixed $beta$, a run may set a _KL budget_ $delta$: $beta$ is then adapted after every iteration so that
the policy settles at $delta$ nats per item. Two ablations remove credit along the list: discount $gamma = 0$, and
episodes of a single item.

For a reward $r(j | s)$ of item $j$ in state $s$, the anchored objective at one position has the closed-form
maximiser @jaques2019wayoff @chidambaram2026robust
$ pi^*(j | s) = (pi_0 (j | s) exp(r(j | s) \/ beta)) / Z(s). $ <eq-optimum>
Standardising the advantages makes the policy-gradient term scale-free but leaves the KL term as it is. RL therefore
pursues this optimum with an effective weight $beta sigma_A$, where $sigma_A$ is the scale of the advantages before
standardisation @chidambaram2026robust. Since $sigma_A$ depends on the estimator and drifts during training, the weight
$beta$ cannot be compared between runs. The KL distance a run ends at can, and we report it throughout.

*Rewards.* All rewards are instances of
$ r_k = w_rho hat(rho)_k + w_q hat(q)_k + w_u hat(u)_k + w_L (hat(l)_k - hat(d)_k) + w_A hat(alpha)_k
  + w_R bb(1)[k = K] hat(u)_K + w_F (bb(1)[a_k in F]) / (log_2 (k + 1)). $ <eq-reward>
The default is $w_rho = 1$ alone. The _return after the list_ ($w_R$) is paid at the last item only, so it belongs to
the list as a whole. The _logged-relevance_ term of @petrov2024aligning ($w_F$) uses the user's real continuation $F$
of a training history and sums to DCG\@K. A _personal_ reward subtracts what the model predicts for the same list
shown to another user, and a _diversity_ reward adds the mean distance between the items of the list. The return,
like and adoption terms are studied in @sec-q3.

*Alternatives to on-policy RL* (@sec-q3, @sec-q5). _Liked-only fine-tuning_ continues next-item training on targets
with $rho >= 0.5$. _Off-policy REINFORCE_ @matveev2026session weights logged items by the logged reward that followed
them, with importance weights capped at $kappa = 0.5$. _Re-ranking_ keeps the $K$ items, of the 100 best candidates of
$pi_0$, with the highest $r(j | h) + w log pi_0 (j | h)$. This orders items exactly as $pi^*$ of @eq-optimum with
$beta = w$, so re-ranking computes the top of the anchored optimum for the real history, without training. _Guided
decoding_ applies @eq-optimum one position at a time. Among the 50 most likely next items of $pi_0$, it takes the one
with the highest $log pi_0 + Q \/ beta$, given the items already chosen; we call $beta$ here the _strength_ of
guidance. $Q$ is the item's predicted reward plus $gamma$ times a value $U$ of what follows it, and $U$ is regressed on
the discounted rewards of lists that $pi_0$ samples. This is controlled decoding @mudgal2024controlled, the first step
of Q$sharp$ @zhou2025qsharp. With $gamma = 0$, each item is judged by its own reward alone. _Weight interpolation_
takes $theta_0 + a (theta_"RL" - theta_0)$ for $a$ from 0.25 to 2 @wortsman2022wiseft @rame2024warp @zheng2025expo.

*What RL learns* (@sec-q1). _Item offsets_ restrict RL to one score offset per item, the same for every user.
_Rank-1 LoRA_ @hu2022lora updates only the transformer's projections. Both are held to the KL budgets of full RL, with
learning rates of $3 dot 10^(-3)$ and $10^(-4)$ against full RL's $10^(-5)$. _Policy surgery_ averages the main run's
change of log-probabilities per item over validation histories. Adding this average to $pi_0$ gives "base + RL's item
offsets"; subtracting it from the fine-tuned policy gives "RL $-$ its item offsets".

*Learning from the log* (@sec-q5). _Reward-weighted fine-tuning_ continues next-item training with each logged target
weighted by $exp(r \/ tau)$, normalised over the batch, where $r$ is the logged relative or consumed share. It fits
@eq-optimum to the log @peng2019awr @chidambaram2026robust. _Within-user DPO_ @rafailov2023dpo learns from pairs of
items that one user consumed within 20 events after the same history: one with a relative share of at least 0.6 and
one of at most 0.4. Neither item is therefore preferred for having been started or for being short. Both methods run
one epoch at a learning rate of $10^(-4)$.

*One head per position* (@sec-q4) @gloeckle2024mtp predicts a list without reading its own items. On the frozen $pi_0$, a linear map
per offset $k = 2, ..., K$, initialised as the identity, turns the state after the real history into one from which
$pi_0$'s output layer scores the item $k$ events ahead. Position $k$ takes head $k$'s best allowed item.

*Instruments.* Let $F$ be the next 20 items the user really consumed; a recommended item in $F$ is a _hit_.
- _Held-out reward model._ A gain it does not confirm is exploitation of the training model.
- _Logged share and logged surplus._ Neither needs a model. The logged share
  $1/K sum_k bb(1)[a_k in F] rho(a_k)$ counts hits as much as responses. The logged surplus
  $1/K sum_k bb(1)[a_k in F] (q(a_k) - 1/2)$ is positive only if users took to the recommended items more than is
  usual for those items, so an item's length does not raise it.
- _Personal part._ The same lists scored for another user give the part of the predicted share that depends on who
  receives the list.
- _Relevance_ is NDCG, Recall and MAP at $K = 1, 5, 10$ against $F$. NDCG\@10 or Recall\@10 is also broken down by
  outcome (liked, $rho >= 0.5$, or skipped; above or below the item's usual share), by the length of the history, and
  by position after a hit or a miss.
- _Concentration_ is catalogue coverage, the Gini coefficient of impressions, the share of impressions of the 1% most
  shown items, and the overlap of two users' lists.
- _Per-item share._ For a quantity $x_(h j)$ of candidate $j$ after history $h$, let $tilde(x)_(h j)$ be $x_(h j)$
  minus its mean within the history, with weights $w_(h j) = pi_0 (j | h)$ over the 100 best candidates of $pi_0$.
  The per-item share is the part of the within-history variation that one effect $e_j$ per item explains:
  $ S = 1 - (sum_(h, j) w_(h j) (tilde(x)_(h j) - e_j)^2) / (sum_(h, j) w_(h j) tilde(x)_(h j)^2), quad
    e_j = (sum_h w_(h j) tilde(x)_(h j)) / (sum_h w_(h j)). $ <eq-per-item>
  $S = 1$ if every history orders the items alike, and $S = 0$ if an item's value depends on the history alone. For
  $x = log pi_theta - log pi_0$, $S$ measures how much of a policy's change is one preference per item applied to every
  user. For the reward model's predictions, $S$ measures the same for the optimum: by @eq-optimum the optimum changes
  $log pi_0$ by $r \/ beta$ plus a constant per history, at any $beta$.
- _Simulation on real sequences_ runs the reward model on the user's real next five items, with the earlier responses
  as logged or as a simulation has to assume them.

= Experimental Setup <sec-setup>

#figure(
  results-table(
    columns: (2.5cm, 1fr, 1fr),
    align: (left, left, left),
    header: ([], [KION (main dataset)], [Yambda-50M (replication)]),
    [Log], [Online cinema @petrov2022kion: one row per user and title, dated by its last view, with the share watched;
      an item table names every title and its type], [Music service @ploshkin2025yambda: one row per listen, timed to the
      second, with the share played; likes and dislikes],
    [Kept], [Users with at least 5 and titles with at least 10 training events], [Users with at least 50 training
      events; the 50k most played tracks],
    [Size], [256k users, 8k titles, 3.5M training events; a median of 10 per user], [8.7k users, 50k tracks, 35M
      training events; a median of 2,638 per user],
    [Split], [The last 7 days are the test period, the 7 days before them the validation period], [The same],
    [Test requests], [59,997, one per user at the start of the test week; the history is the last 38 events, the
      targets the next 20], [7,593; the last 118 events and the next 20],
    [Session, return], [The next event on the same calendar day; a new day within a week], [The next event within 30
      minutes; a new session within a day],
    [Models], [GPT-2, 4 layers of width 256, windows of 48 events], [The same, windows of 128 events],
    [Fine-tuning], [512 lists of 10 per iteration, $gamma = 0.99$, learning rate $10^(-5)$; the main run has
      $beta = 0.3$ and 2,000 iterations, every other run 1,000], [The same],
    [History items], [Never recommended again: one row per user and title], [Allowed: repeats are normal],
    [Chosen on validation], [$beta = 1$], [$beta = 0.05$],
  ),
  caption: [Data, split, models and fine-tuning.],
  placement: auto,
) <tab-setup>

*Data.* @tab-setup lists the data, the split and the models. On Yambda 0.77% of events are liked and 0.16% disliked,
and 24% of first plays are adopted. The consumed share is bimodal, and on KION it depends on length. Films are watched
to 0.5 on average. The longest tenth of the catalogue, mostly series, is watched to 0.22, since the share is taken of
the whole series.

*Training and evaluation.* Fine-tuning cuts its histories at a random moment of the training period of a random user
with at least 8 training events. Lists of ten are generated greedily unless ranked lists are named, and every baseline
reads the recommender's window of history. The KL distance of a run is the exact divergence over the catalogue on its
own sampled training lists, averaged over the last 50 iterations. Operating points are selected on validation requests
only: the highest held-out predicted share within 5% of the base recommender's NDCG\@10. The main run's $beta = 0.3$
is not such an operating point. It was carried over from earlier iterations of the study, and we keep it because it
moves clearly along the frontier, which makes its changes easy to analyse.

*No leakage.* Every trained model reads windows of the training period only: the recommender, the reward models, the
baselines and every kind of fine-tuning. RL samples its histories, and the logged continuation of each, inside the
training period. The catalogue and every item statistic, including the quantiles of the relative share, come from the
training period. Labels that look ahead, such as continuation, return, likes and adoption, are computed for training
events from events before the validation period; an outcome whose waiting time is not observed is left out. Early
stopping and operating points use the validation period. The history of a test request includes the validation week,
on which no model trained. On KION a title watched across the boundary appears only as a test target, and test events
on items outside the training catalogue are dropped for every method alike. The repository's README lists each check
next to the code that implements it.

*Sessions and seeds.* The experiments ran as several sessions per dataset, and every table and figure names the
session it comes from (@sec-sessions). Seeds 0, 1 and 2 repeat every central comparison. "$x plus.minus s$" is the mean
and standard deviation over the three seeds, with $s$ in the unit of $x$ (percentage points when $x$ is a percentage);
each quantity is computed within its seed. Other numbers are from seed 0. Every conclusion below holds in each seed.

= Results <sec-results>

*Base models.* On KION the recommender's ranked lists are level with GRU4Rec, 13% above SASRec and 5% below BERT4Rec
(@tab-baselines). On Yambda it is the best trained model. Replaying the user's own history from the same 118 events
does better still (NDCG\@10 0.187 against 0.117), mostly through repeated tracks. On Yambda we therefore report
relevance as a change against this recommender, not against the best list possible. The reward model predicts whether
a held-out event is consumed at least half with AUC 0.755 on KION and 0.868 on Yambda; the item's mean share reaches
0.656 and 0.607. The like and dislike heads reach 0.74 and 0.73. The two copies are calibrated on their own users and
differ in level on the same lists (0.417 and 0.455 for the base recommender on KION), so we compare their relative
changes.

== Q1. How does the model's behaviour change after RL fine-tuning? <sec-q1>

*Is the reward maximised?* As far as the reward models can tell, yes. On KION the main run raises the consumed share
predicted by the training reward model by 20.3%, and by the held-out one by 21.5% (seed 0; $23.9% plus.minus 2.1$ over
three seeds; @tab-main). The two models rise together from the first evaluation. Most of the change comes early: 85%
of the gain and 90% of the loss of NDCG\@10 arrive within the first 250 of 2,000 iterations (@fig-training). After
that the policy keeps moving away from $pi_0$, from 0.06 to 0.09 nats per item, for little more reward.

#report-table("main_run", [The main run ($beta = 0.3$) against the base recommender, final session, seed 0. Changes
  are relative, in percent, with a 95% paired-bootstrap interval over users where the quantity is defined per user.],
  size: 7pt) <tab-main>

*Do users respond better?* The log says no. The logged surplus, users' response to recommended titles beyond what is
usual for each title, falls by $61% plus.minus 2$, from 0.0012 to 0.0005 per recommended title. At the
validation-selected $beta = 1$ it falls by 15%. The logged share per recommended item, which mostly counts hits, rises
by $1.1% plus.minus 0.4$. Users reached fewer of the recommended titles (hits $-8.8% plus.minus 0.9$) but finished
those more often ($+10.8% plus.minus 1.6$), because they are titles that everybody finishes.

*How the recommendations differ.* The lists shift towards films. Films take 66% of the impressions of the base
recommender's generated lists and 80% of the main run's; at $beta = 0.1$ they take 89%, and without an anchor all of
them. The titles that gained most are films of 1.4 to 2.2 hours, usually watched to 0.54--0.69. Four of the five that
lost most are series of 9.5 to 22 hours, usually watched to 0.08--0.29 (@tab-titles). Popularity is not what the
policy follows: one of the largest gains is a film that only 953 users watched in the training period.
@fig-impressions shows the shift across the catalogue. The four tenths of titles with the lowest usual share fall from
34% to 13% of the impressions, and the three highest rise from 29% to 46%. The longest tenth, mostly series, falls
from 26% to 14%. On Yambda impressions move the same way, by a few points.

#figure(
  image("figures/main_run_training.svg", width: 100%),
  caption: [The main run over its iterations (final sessions): the predicted share of the generated lists for
    validation requests as each reward model predicts it; NDCG\@10 of the generated and ranked lists for the same
    requests; and the KL from the base recommender, in nats per generated item, of the lists sampled in training,
    recorded from the first update of the policy.],
) <fig-training>

#figure(
  image("figures/impressions_by_tenth.svg", width: 72%),
  caption: [Where the impressions went (final sessions): the share of all impressions in the test users' lists that
    each tenth of the catalogue received from the base recommender and from the main run. The catalogue is cut into
    tenths with as many items each, by the item's usual consumed share in the training period and by its length.],
) <fig-impressions>

*One re-weighting of the catalogue.* Three tests show that the policy's change on KION is mostly one preference per
item, the same for every user (@tab-mechanism).
- _Restricting RL to item offsets loses little._ Held to the same KL budget as full RL, RL that may change only one
  score offset per item reaches $93% plus.minus 3$ of full RL's gain in predicted share at a budget of 0.1 nats, and
  $89% plus.minus 4$ at 0.5. Its NDCG\@10 is within 0.002 of full RL's.
- _The gain does not depend on the user._ The same lists scored for another user earn the same gain: only
  $0.2% plus.minus 0.5$ of it is personal. Re-ranking by each item's mean share in the training log, with no reward
  model, reaches the main run's point (predicted share 0.549 at NDCG\@10 0.109, against 0.553 at 0.105).
- _The per-item part carries the whole effect._ Adding the main run's average change per item to $pi_0$ reproduces
  $88% plus.minus 2$ of its gain and all of its loss of relevance (NDCG\@10 $-11.0% plus.minus 1.8$, against RL's own
  $-11.6% plus.minus 1.5$ in the same sessions). Subtracting it from the fine-tuned policy leaves
  $21% plus.minus 2$ of the gain at the base relevance ($-0.1% plus.minus 0.4$).

The two parts pull personalisation in opposite directions. The per-item part lowers the personal part of the
predicted share by $15.7% plus.minus 3.1$, the rest raises it by $20.5% plus.minus 2.5$, and in RL they nearly cancel
($+2.2% plus.minus 2.9$). What the per-item part ranks is clear from its correlations: an item's mean shift correlates
0.90 with its usual consumed share, 0.12 with its log popularity and $-0.36$ with its duration (review session). The
policy learned a ranking of titles by how often they are finished, and it applies this ranking to every user.

#report-table("mechanism", [What RL learns (frontier sessions). Restricted runs and full RL hold the same KL budget,
  except that item offsets end short of 0.5: at a KL of 0.41 on KION (full RL 0.53), and on Yambda at 0.25 with the
  anchor weight at zero. The parts of the main run are compared with the base recommender. The per-item share
  (@eq-per-item) is measured on the base recommender's 100 best candidates for every test user; that of the reward
  model's predictions is the optimum's at any $beta$ (@eq-optimum).]) <tab-mechanism>

*The reward or the optimiser?* @eq-optimum answers this before any RL. The per-item share of the reward model's
predictions is that of the optimum's change, at any $beta$. On KION it is $0.79 plus.minus 0.01$. Every run of the
anchor sweep, the KL budgets and LoRA stays below it, in every seed (main run $0.77 plus.minus 0.01$). On KION the
reward thus sets the re-weighting, and the optimiser follows it.

The reason lies in what the reward measures. $hat(rho)$ is the share a user consumes _given_ that they start the
item, so it cannot penalise an item the user would never start. And on KION the share of a series is taken over all
its episodes. An item that everybody finishes therefore earns a high reward, whoever receives it.

On Yambda the reward is mostly personal: its per-item share is 0.42. Yet every RL run is more per-item than its reward
(0.46 to 0.55). A policy gradient averages a noisy signal over histories, and the direction that is the same for all of
them accumulates first. Item offsets alone fall short on Yambda: even with the anchor weight driven to zero they stop
at a KL of 0.25. At a budget of 0.5 they reach 34% of full RL's gain, less than full RL reaches at a KL of 0.11. The
reward model also knows more about the user than RL uses: $55% plus.minus 9$ of re-ranking's gain is personal, against
$16% plus.minus 16$ of RL's. Rank-1 LoRA stays on full RL's frontier on both datasets, with 87% to 95% of its gain at a
higher NDCG\@10, as reported for language models @schulman2025lora.

#answer[Q1][The reward is maximised as far as the reward models can tell. But what RL learns is one re-weighting of
  the catalogue, the same for every user, and this re-weighting carries RL's whole loss of relevance. Measured against
  what is usual for each title, users took to the recommendations less. On KION the reward sets the re-weighting, since
  79% of its optimum's change is per item. On Yambda, where the reward is mostly personal, the optimiser amplifies the
  per-item part.]

== Q2. Is relevance preserved? Does the model recommend the same items to everyone? <sec-q2>

*A frontier, not a verdict.* How much relevance is kept depends on the operating point (@fig-frontier). On KION the
weight selected on validation requests is $beta = 1$. There NDCG\@10 falls by 1.8%, while the predicted share rises by
8.3% and the logged share by 3.3% (three seeds). At the main run, $beta = 0.3$, NDCG\@10 falls by
$11.5% plus.minus 1.3$ for a 24% higher predicted share. On Yambda the validation rule selects $beta = 0.05$ (NDCG\@10
$-4.9%$, predicted share $+5.0%$). This is one grid step from $beta = 0.03$, which collapses the policy in every seed.
The Yambda main run moves little.

*Relevance by Recall, NDCG and MAP.* At the main run on KION every metric falls at every cut-off, by 8 to 14%
(@tab-ranking). The loss already appears at $K = 1$ ($-7.9%$). The first item is chosen from the real history alone,
so this part of the loss is the policy's own preference; from the second position on, the generated context adds to
it (@sec-q4). The loss is uneven across outcomes. NDCG\@10 on liked titles is kept ($+1.3%$), while Recall\@10 on
skipped titles falls by 18.2% and NDCG\@10 on titles watched above their usual share by 9.4%. It is also uneven across
users: those with the least history lose most, $-13.5%$ of NDCG\@10 for 5 to 8 earlier events against $-7.4%$ for 25
or more (@tab-history). On Yambda every metric stays within 1.3% of the base recommender, and no group of users loses.

#report-table("ranking_metrics", [Ranking metrics of the main run against the base recommender (generated lists,
  final session). At $K = 1$ NDCG and MAP coincide.]) <tab-ranking>

*The same items for everyone?* Not at the main run, but closer to it (@tab-concentration). On KION the base
recommender's generated lists already give 72% of the impressions to the 1% most shown titles (62% for ranked lists).
The main run raises this to 76%, and the Gini coefficient of impressions from 0.963 to 0.969. A weaker anchor makes the
lists converge: 92% at $beta = 0.1$. Without an anchor nearly every user receives the same list (overlap between two
users' lists 0.98 on KION, 1.00 on Yambda), and the held-out model still scores it 51% above the base recommender. On
Yambda the share of the top 1% moves only from 18% to 20%. What makes the lists alike is the per-item part of the
policy (@sec-q1).

*What keeps relevance.* Only the anchor asks for relevance, but its weight does not say how much relevance it keeps;
the KL distance does. Two estimators at the same $beta = 0.3$ on KION end far apart: the leave-one-out estimator ends
6.7 times further from $pi_0$ than the value model (KL $0.543 plus.minus 0.010$ against $0.081 plus.minus 0.004$). Held
to the same KL budget of 0.1, the two end at the same relevance (NDCG\@10 $0.1037 plus.minus 0.0021$ and
$0.1013 plus.minus 0.0026$), with weights 2.9 times apart. The effective weight $beta sigma_A$ (@sec-method) explains
most of this. Before standardisation, the value model's advantages are 2.2 times larger at that budget ($sigma_A$ 0.95
against 0.43), and 1.7 times larger at a budget of 0.5, where the weights differ 2.7 times. On a logarithmic scale this
accounts for 55% to 75% of the gap between the weights. On Yambda $beta = 0.1$ keeps relevance, and $beta = 0.03$
collapses the policy in every seed (NDCG\@10 $0.006 plus.minus 0.004$; @tab-sweep).

#figure(
  image("figures/frontier_by_distance.svg", width: 88%),
  caption: [Every fine-tuning run of the default reward against the KL distance it ended at from the base
    recommender (logarithmic axis; the base recommender at the left edge; final session).],
) <fig-frontier>

#answer[Q2][Relevance is traded along a frontier. On KION, NDCG\@10 falls by 1.8% at the validation-selected weight
  and by 11.5% at the main run, at every cut-off and most for users with short histories, while impressions
  concentrate further. The per-item part of the policy causes both. On Yambda the main run keeps both relevance and the
  spread of impressions. The anchor is what keeps relevance, and the KL distance a run ends at tells how much it keeps
  far better than the anchor weight does. Without the anchor every user receives the same list.]

== Q3. Can we get closer to long-term goals? <sec-q3>

*Which signals.* Four logged signals describe what follows an item: the session continues, the user returns, the user
likes or dislikes the item, and the user adopts it. The last two exist on Yambda only. A reward can steer a policy only
if it differs between the items a user could be shown. The predicted return barely does: it varies between one user's
candidates four times (KION) and nine times (Yambda) less than between users. The probability of a like varies by only
0.003 between a user's candidates. Adoption is the exception. It varies between candidates half as much as between
users (0.077 against 0.165), and it is the only signal that leans towards the items users really consumed next (AUC
0.56).

*Rewards for return and continued interaction.* We rewarded each signal per item, and the return also once per list,
after its last item. Rewarding the return after each item raises it by 3 to 7% on KION, at a cost of 8 to 23% of
NDCG\@10, and does not move it on Yambda (@tab-retention).

The _return after the list_ moves the KION policy further than any per-item reward (KL 0.51 at $beta = 0.3$). The
predicted return rises from 0.73 to 0.86 ($+18%$), and the held-out model agrees. But NDCG\@10 falls by 69%, the
logged share by 64%, and the mean duration of a recommended title from 5.7 to 2.2 hours (@tab-long-term). Nothing in
the log supports a return rate of 0.86. Neither title length nor KION's format, one row per user and title dated by its
last view, produces it: in every decile of the duration of an event's own title or of the next title, the logged return
rate lies between 0.68 and 0.77. Both reward models extrapolate beyond the log in the same way, so the held-out model
cannot catch it. This is the one reward for which credit along the list matters: with $gamma = 0$ the policy hardly
moves (KL 0.006). On Yambda the return after the list does not move.

#report-table("long_term_rewards_and_guidance", [Long-term rewards (review session), and guided decoding by them, which
  leaves the recommender as it is (frontier session; "β": the strength of guidance). "Δ": change of NDCG\@10 in
  percent; "Return": the held-out model's predicted return after the last item; P(like), P(adopt): its predicted
  probabilities (Yambda only); "Cover.": catalogue coverage; "Hours": mean duration of a recommended KION title.])
<tab-long-term>

*Likes and adoption (Yambda).* The like reward at $beta = 0.3$ raises the predicted probability of a like from 1.08%
to 1.17%, for 2% of NDCG\@10. At $beta = 0.1$ likes rise by 49%, but dislikes rise by 53%: the policy finds tracks
that provoke reactions, not tracks that are liked more. Adoption is the one long-term reward that moved without a loss
of relevance. At $beta = 0.3$ its predicted probability rises from 0.233 to 0.277 ($+19%$), NDCG\@10 by 1.5% and the
logged share by 2.5%, while coverage falls from 0.52 to 0.41.

*Steering at inference.* Guided decoding steers the base recommender towards these rewards without fine-tuning it
(@tab-long-term). On Yambda, guidance by predicted adoption raises adoption by 6.5% and NDCG\@10 by 3.7% at strength
0.3. At strength 0.1 it raises adoption by 20% for 0.8% of NDCG\@10; the RL run gained 19% of adoption and 1.5% of
NDCG\@10. Judging each track also by a learned value of what follows it adds 3 and 6 points of adoption. On KION the
return after the list is paid only at the end, so guidance that judges each title by its own reward sees it only at the
last position. Look-ahead nearly doubles the gain: at strength 0.1 the predicted return rises from 0.729 to 0.742
without it and to 0.753 with it, for 2.3% of NDCG\@10. RL reached 0.86 but lost 69% of NDCG\@10. The guided lists also
keep the length of their titles.

*Offline or online.* Offline we can check whether a signal is predictable, whether it varies between the items a
policy chooses from, whether a held-out model agrees, and what it costs in relevance. Whether users return, continue
or adopt more under a new policy is counterfactual, for three reasons. The log shows these outcomes only under the
policy that produced it. Adoption needs 14 days after a first play, which a test week does not have. And a prediction
can leave the log's range, as the return after the list did. This needs the online experiment of @sec-online.

#answer[Q3][Rewards for return, likes and adoption can be learned and optimised, but at a moderate KL distance their
  predictions rise by only a few percent, and the largest gain lies outside the log. Adoption is the most promising
  signal: steering the base recommender by it at inference, with a learned look-ahead, raises adoption and relevance
  together. Only an online test can show whether users return or adopt more.]

== Q4. What happens when generating several steps ahead? <sec-q4>

A generated list appends each chosen item to the history as if the user had consumed it. It thus plans several steps
ahead under an assumption that is false for most items: at most 17% of the items at any position are hits.

#figure(
  image("figures/position_hits_with_heads.svg", width: 88%),
  caption: [Hit rate by position of the list, for all lists, after a miss and after a hit at the position before:
    the base recommender's ranked and generated lists, and the lists of one head per position (frontier sessions).],
) <fig-positions>

*Effect on the following recommendations.* After a miss, a generated list continues from a context that did not
happen, and it hits less often than a ranked list: 0.0273 against 0.0312 at position 5 on KION, 0.040 against 0.063 on
Yambda (@fig-positions). After a hit it hits more often: 0.112 against 0.093 at position 2 on KION. On Yambda, where
listening continues within albums and artists, the effect is larger: 0.54 against 0.25 at position 5. On KION misses
dominate, and generation costs 4% of NDCG\@10 against ranking. On Yambda generation is ahead at positions 2 to 7.

A list predicted position by position from the real history removes the loss. After a miss, one head per position
hits at least as often as a ranked list (0.033 at position 5 on KION, 0.073 on Yambda); after a hit, as often as a
ranked list. On KION it is $6.3% plus.minus 0.4$ more relevant than generation and $2.5% plus.minus 0.1$ more than
ranking, with a logged surplus $25% plus.minus 1$ above generation's. On Yambda, where hits after hits matter more, it
is level with generation ($+0.4%$) and 3.4% above ranking, with less coverage.

Steering shows the same pattern. Guided decoding decides each position after the items already chosen, as generation
does. On KION it falls slightly behind re-ranking (predicted share at the main run's NDCG\@10:
$0.556 plus.minus 0.024$ against $0.564 plus.minus 0.025$; RL $0.548 plus.minus 0.023$). On Yambda, where generation
beats ranking, it beats re-ranking (@fig-methods).

*Effect on the reward estimate.* The reward at later positions depends on what the reward model assumes about the
responses to earlier items. We test these assumptions on the users' real next five items (@fig-assumptions in the
appendix). With the logged responses fed back, the AUC of the predicted share rises along the list (0.757 to 0.782 on
KION, 0.640 to 0.819 on Yambda). Under any simulated response it stays near its first-step level. Simulated responses
also bias the level of the prediction on Yambda. Expected responses underestimate the next share by 0.06 to 0.09,
sampled ones overestimate it slightly (at most $+0.035$), and assuming that earlier items were finished overestimates
it by 0.06 to 0.18. The reward model trained with hidden responses and fed "unknown" is nearly unbiased ($+0.001$ to
$+0.013$). It also reads observed responses as well as the standard model (AUC 0.755 against 0.758 on KION). Policies
fine-tuned against it, or with sampled responses, land on the same frontier: the assumption sets the level of a
multi-step predicted reward, not which policy wins.

*Does planning help?* Not for the default reward. At an equal KL budget, crediting each item only with its own reward
is at least as relevant as crediting it with the rest of the list. On KION, NDCG\@10 is $0.1070 plus.minus 0.0027$
against $0.1013 plus.minus 0.0026$ (one-item episodes $0.1043 plus.minus 0.0024$), at predicted shares 1.4 to 1.6
points lower. On Yambda it is $0.1076 plus.minus 0.0014$ against $0.0960 plus.minus 0.0008$. The myopic policies lie on
the same frontier of relevance against reward. Only the return after the list, paid at the end, needed full credit
(@sec-q3). Look-ahead at inference agrees. On KION a learned value of what follows lowers the predicted share that
guided decoding reaches at the main run's NDCG\@10 (0.548 against 0.567, seed 0). It helps only rewards that depend on
what follows an item: the return after the list and adoption.

#answer[Q4][A generated list hits more often after a hit and less often after a miss. Simulated responses bias the
  reward of later steps, unless the reward model was trained to read unknown ones. One head per position predicts the
  list from the real history and removes the loss after a miss: it gives the most relevant lists on KION and is level
  with generation on Yambda. Credit along the list and look-ahead help only rewards that depend on what follows an
  item.]

== Q5. What is the trade-off between relevance and reward? <sec-q5>

*How RL settings and the reward shape it.* Every RL setting we varied moves the policy along one curve: the anchor
weight, the KL budget, the estimator, the credit assignment, the simulated responses and the capacity (@fig-frontier,
@sec-q4). *Among RL settings, only the reward moves the frontier; the others choose a point on it.* @eq-optimum says
why for per-item rewards: the settings change how far the policy moves towards $pi^*$, while only $r$ changes where
$pi^*$ lies. Rank-1 LoRA lies on the frontier. GRPO is the one setting that may lie slightly above it on KION: at equal
KL it reaches 7% to 10% more predicted share than the value model, at no lower NDCG\@10. On Yambda it trades along the
frontier (76% of the value model's gain at a higher NDCG\@10; seed 0).

*Can the reward rise without a noticeable loss of relevance?* @tab-equal-relevance in the appendix compares methods at
equal relevance: the largest gain each reaches while staying within 1% or 5% of the base recommender's NDCG\@10.
Between two runs of a method we draw a straight line, which a random mixture of the two policies achieves. On KION, RL
on the share gains 4.2% and 14.1% of predicted share at the two levels, and 1.8% and 3.7% of logged share. Adding ten
times the logged relevance to the reward moves the frontier: 16.9% and 23.1% of predicted share. Re-ranking by the
same reward model reaches 14.9% and 20.7%, with the largest logged gain (15.6%). Most of that logged gain comes from
where re-ranking starts: the base recommender's ranked lists alone already have an 11.7% higher logged share than its
generated lists. On Yambda the differences at 1% are within the noise between seeds, and the methods that learn from
the log lead in logged share through repeats.

*Does it take RL?* No. Re-ranking with weight $w$ computes the top of the anchored optimum for $beta = w$. At the main
run's predicted share, re-ranking is more relevant (NDCG\@10 $0.1119 plus.minus 0.0013$ against
$0.1041 plus.minus 0.0020$) and has a higher logged share ($0.0201 plus.minus 0.0001$ against
$0.0176 plus.minus 0.0002$). It is also more personal: $8.9% plus.minus 0.3$ of its gain is personal, against 0.2% of
RL's (@tab-alternatives). Most of the relevance gap is a head start. Ranked lists are 4% more relevant than generated
ones on KION, and from its own start each method pays about the same relevance for the same predicted share.

*Does it take training?* Not to reach the frontier (@fig-methods). Interpolating the weights between the base
recommender and the main run gives policies on the anchor sweep's curve, or slightly above it at equal predicted share
(NDCG\@10 $+0.0001 plus.minus 0.0002$ to $+0.0009 plus.minus 0.0002$). Extrapolating to twice the change stays on it
($-0.0009 plus.minus 0.0014$). On Yambda, whose main run barely moved, extrapolation falls 0.003 below the run at
$beta = 0.1$. Guided decoding beats RL at equal relevance on both datasets. On Yambda at strength 0.3 it raises
NDCG\@10 by 2.1% and the logged surplus by 9%, at about RL's predicted share.

#figure(
  image("figures/frontier_methods.svg", width: 72%),
  caption: [Methods against the predicted share (frontier sessions, seed 0). RL, its interpolated and extrapolated
    weights, re-ranking and guided decoding share one frontier; reward-weighted fine-tuning on the relative share and
    one head per position leave it towards relevance and a higher logged surplus. Re-ranking starts from the ranked
    lists, the other methods from the generated ones.],
) <fig-methods>

*Rewards that move the frontier.* On KION, the share plus ten times logged relevance at $beta = 0.3$ beats the base
recommender's generated lists on three counts at once: relevance (NDCG\@10 $0.1202 plus.minus 0.0007$ against
$0.1177 plus.minus 0.0008$), predicted share ($+10%$) and logged share ($+7%$). It does not reach the relevance of the
base recommender's ranked lists ($0.1221 plus.minus 0.0004$). RL on the relative share removes the drift towards films
and keeps relevance, but gains almost nothing (@tab-rewards). Re-ranking by the relative share does better against
the ranked lists it starts from. At $w = 0.3$ it raises NDCG\@10 from $0.1221 plus.minus 0.0004$ to
$0.1234 plus.minus 0.0006$ and the logged surplus by 15%. At $w = 0.1$ it raises the surplus by 38% for 0.5% of
NDCG\@10, in every seed (@tab-seeds). On Yambda, logged relevance alone raises NDCG\@10 by 23%, but by replaying
histories: relevance on new tracks falls from 0.0260 to 0.0198.

*Learning from the log.* Every method above reads the reward model. Reading the log instead does better on the
outcomes that the reward model cannot inflate. On KION, reward-weighted fine-tuning on the logged relative share
($tau = 0.3$) raises NDCG\@10 by $1.9% plus.minus 0.5$, the logged surplus by $28% plus.minus 3$, the personal part by
$19% plus.minus 1$ and coverage by $5% plus.minus 2$ (@tab-logged). For NDCG\@10 and the surplus, the per-user
intervals exclude zero in every seed. It lowers the predicted share by 3.8%: it moves towards what a user takes to more
than usual, not towards what the reward model rates.

On the plain consumed share, the same fine-tuning reproduces RL's drift without any reward model. The predicted share
rises by $20.8% plus.minus 0.5$ (RL: $24.0% plus.minus 2.0$), NDCG\@10 falls by $13.4% plus.minus 0.3$, the logged
surplus by 75%, and the recommended titles are half as long. The drift therefore lies in the definition of the share,
not in RL. Within-user DPO buys a larger surplus ($+42% plus.minus 5$) and personal part ($+35% plus.minus 6$), but
pays in relevance ($-3.7% plus.minus 1.6$) and coverage ($-22% plus.minus 2$). On Yambda, reward-weighted fine-tuning
raises the surplus by 13% at unchanged relevance, and liked-only fine-tuning, its hard-threshold form, is more relevant
still (NDCG\@10 $+2.2%$).

#report-table("logged_methods", [Learning from logged responses alone, without a reward model (frontier sessions):
  change against the base recommender's generated lists, in percent. "Logged surplus": the logged response to
  recommended items relative to what is typical for each item; "Personal part": the held-out reward model's
  prediction for the user minus that for another user.]) <tab-logged>

*A control for the extra epoch.* Reward-weighted fine-tuning gives $pi_0$ one more epoch of training at a tenth of its
learning rate. Two effects other than the weights could explain its gain. First, $pi_0$ had stopped while its learning
rate was still high, so any extra epoch might help. Second, on KION ranked lists are 4% more relevant than generated
ones anyway. A control session with seeds 0 to 2 separates these effects, with its hypotheses written down before the
run (@tab-control).

The same epoch with every weight equal to one, _continued training_, does help. It lowers the next-item loss on the
validation period in every seed. It raises coverage by 10%, NDCG\@10 of generated lists by 0.7% (significant in two
seeds of three) and the logged surplus by 5% (in one seed of three). The weights add more on top of it, in every seed:
$21% plus.minus 4$ of logged surplus, $19% plus.minus 2$ of personal part and $1.3% plus.minus 0.5$ of NDCG\@10 on
generated lists, and 17%, 15% and 0.7% on ranked lists. Against continued training, coverage falls by 3% to 4%. The
fine-tuned recommender's ranked lists are also $0.9% plus.minus 0.3$ more relevant than those of $pi_0$, with 23% more
logged surplus, in every seed. In sum, the extra epoch gives a third of the relevance gain and all of the coverage
gain. The weights give the other two thirds of the relevance gain and almost all of the gain in logged surplus and
personalisation.

#report-table("control", [Learning from the log against the same epoch of training without its weights (KION,
  control session, seeds 0 to 2): change in percent, mean $plus.minus$ standard deviation over the seeds, each change
  computed within its seed and against lists of the same kind. "Continued training" weighs every logged item one;
  "Surplus": the logged surplus; "Personal": the personal part of the predicted share.], size: 7pt)
<tab-control>

#answer[Q5][Among RL settings, only the reward moves the frontier; the others choose a point on it, and re-ranking,
  guided decoding and weight interpolation reach that point without RL training. The frontier moves when training
  draws on the log. Logged relevance in the reward moves it, and reward-weighted fine-tuning on the logged relative
  share moves it most. Without a reward model, this fine-tuning raises the logged surplus by 17--21% and
  personalisation by 15--19% over the same epoch of plain training, with relevance up, on generated and on ranked
  lists.]

= Discussion <sec-discussion>

*What a second reward model catches.* A held-out reward model, the usual defence against reward hacking, catches
exploitation of one particular set of weights. Without an anchor, the training model predicts a consumed share of 0.84
for its policy's lists, but the held-out model only 0.69 (@tab-sweep). It misses what any model of this kind invites,
as Eisenstein et al. @eisenstein2023helping found for ensembles. Both of our models learned, correctly, that some titles
are finished more often than others. And both extrapolate the return after the list beyond any rate in the log.

*Cheap checks that do catch it.* Three instruments expose this at little cost: scoring the same lists for another
user, splitting the policy's change into a per-item part and the rest, and counting the logged response to recommended
items. One quantity even predicts it before any RL: the per-item share of the reward's predictions says how much of a
policy's change will be one re-weighting of the catalogue.

*What moved the frontier.* For a reward that sums per-item predictions, the anchored optimum can be reached without a
policy gradient (@eq-optimum). What moved the frontier came from two other places. One was the log, through a reward
that compares a response with what is typical for the item. The other was the list format, which should follow how
strongly one consumption predicts the next.

*In practice:*
- compute the reward's per-item share before fine-tuning;
- report the KL distance a run ended at, not its anchor weight;
- re-rank or decode by the reward model before training against it;
- fit the anchored optimum of a length-neutral logged reward directly, by reward-weighted fine-tuning;
- choose the list format (generated, ranked or one head per position) on validation.

== An online experiment <sec-online>

Offline evidence settles the collapse, the role of the anchor and the impersonality of the predicted gain. Whether
users consume more of the fine-tuned lists, return more often or adopt more needs an experiment that serves them. Its
arms are set by KL distance, not by anchor weight: on Yambda one grid step separates a working weight from a collapsed
policy, so a fixed weight does not fix a distance.
- _Arms:_ (A) the base recommender; (B) RL held at the KL distance of the validation-selected run (0.015 nats per item
  on KION, 0.29 on Yambda); (C) re-ranking by the reward model at the predicted share of B; (D) reward-weighted
  fine-tuning on the relative share, $tau = 0.3$; (E) one head per position; (F) on the music service, decoding guided
  by adoption with look-ahead at strength 0.3.
- _Primary metrics, per user, over four weeks of hashed assignment_ (one week for exposure to settle, fourteen days
  of adoption window, one more week for returns): the consumed share of served recommendations and its surplus over
  what is typical for each item, the return rate, and on Yambda the share of first plays of recommended tracks played
  again within 14 days.
- _Guardrails:_ consumption per user, likes and dislikes, coverage and Gini of impressions, the share of new items,
  and every metric for the quartile of users with the shortest histories.
- _Analysis and size:_ per-user differences with a CUPED adjustment on the four weeks before, bootstrap intervals, and
  history quartiles as pre-registered segments. A 1% relative change in a binary outcome near 0.7 needs about
  $2 (1.96 + 0.84)^2 dot 0.7 dot 0.3 \/ 0.007^2 approx 67,000$ users per arm ($alpha = 0.05$, power 0.8).

= Limitations, Future Work and Conclusion <sec-limitations>

*Limitations.* All results are offline, and the logged check sees only the recommended items that users reached on
their own. Yambda's frontier session, the long-term rewards and the analyses by position, user and item have one seed,
and adoption could not be verified on a test week. The models are small (4 layers of width 256), so a larger reward
model may know more about users. Yambda's models read 118 events of a median 2,638, its grids are narrower than
KION's, and the baselines are untuned. Guided decoding calls the reward model for 50 candidates at every position, the
heads per position sit on a frozen recommender, and reward-weighted fine-tuning and DPO ran at one learning rate.
Early stopping kept the base recommender while its learning rate was still high. One more epoch of plain training
lowers its validation loss and raises NDCG\@10 of its generated lists by 0.7%, so every change measured against the
base recommender includes this slack. It is small next to RL's loss, and a third of the relevance gained by learning
from the log. Finally, the equal-relevance curves interpolate between runs, the per-item share is measured on 100
candidates, and the personal part is the reward model's own belief.

*Future work.* The online experiment of @sec-online is the decisive next step. Offline, we see six directions:
- a base recommender trained until its learning rate has decayed, as the reference for every method;
- the per-user counterpart of the relative share from OneRec-V2 @zhou2025onerecv2 as the logged reward of
  reward-weighted fine-tuning;
- heads per position trained together with the recommender and combined with reward-weighted fine-tuning;
- guided decoding with the value of the optimum rather than of $pi_0$, as Q$sharp$ iterates it @zhou2025qsharp;
- a larger reward model, whose per-item share would say in advance whether RL can learn more than a re-weighting of
  the catalogue;
- a longer Yambda context, and a test period long enough to observe adoption.

*Conclusion.* We fine-tuned a generative recommender against a learned reward model, with only a KL anchor to keep it
relevant. The predicted reward rose, and a reward model trained on other users confirmed the rise. But what RL had
learned was largely one re-weighting of the catalogue towards the items finished most often. Measured against what is
typical for each item, the logged response to the recommendations fell. Relevance was traded along a frontier that RL
settings only move along and that inference reaches without training. What moved the frontier came from the log and
from the list format. Reward-weighted fine-tuning on the consumed share relative to what is typical for the item
improved the logged surplus, personalisation and relevance beyond the same epoch of plain training, on generated and
on ranked lists. Lists predicted position by position from the real history were more relevant than generated ones. A
reward model is worth optimising with RL only if it knows something about the user that cannot be looked up per item,
and its per-item share shows whether it does before any training.

*Code and runs.* Code, configurations and every table are public at
#link("https://github.com/daniilantonovv/RecSysMaxxing")[`github.com/daniilantonovv/RecSysMaxxing`]. The runs were
executed in the Kaggle notebook
#link("https://www.kaggle.com/code/daniilantonov5/gptrec-rtx-pro-6000")[`kaggle.com/code/daniilantonov5/gptrec-rtx-pro-6000`],
whose published versions keep their outputs.

#heading(numbering: none)[References]
#{
  set text(size: 7pt)
  set par(justify: false, leading: 0.5em)
  columns(2, gutter: 14pt, bibliography("references.bib", style: "ieee", title: none))
}

#pagebreak()
#counter(heading).update(0)
#set heading(numbering: "A", supplement: [Appendix])
#set figure(placement: none)

= Sessions and seeds <sec-sessions>

Each dataset ran three sessions on one 96 GB GPU. A _final_ session ran the full grid (6.1 hours on KION, 12.3 on
Yambda). A _review_ session added the long-term rewards, the reward models with hidden responses, and the analyses by
position, user and item. A _frontier_ session ran the methods that probe what RL learns (4.8 and 6.0 hours); Yambda's
ran without likes and dislikes, which none of its rewards use. On KION two more sessions followed. The _control_
session ran seeds 0 to 2 itself (20 minutes per seed) to test learning from the log against continued training. The
_quick_ session repeated the core of the study with seed 0 in 55 minutes and reproduced the final session's numbers.

Seeds 1 and 2 repeat the core on both datasets: the base models, the main run, the anchor sweep, a KL budget, the
credit and estimator comparisons, the alternatives to RL and logged relevance. On KION they also repeat the frontier
comparisons. A seed fixes the initial weights of every model and the sampling of fine-tuning. Every session retrains
every model with the same seed, so a run present in two sessions differs only in the third decimal.

= Supporting tables

#report-table("seeds", [The central comparisons over three seeds (0, 1, 2): mean $plus.minus$ standard deviation.
  A seed fixes the initial weights of every model and the sampling of fine-tuning. "Budget" is a KL budget of 0.1
  nats on KION and 0.5 on Yambda; "–": not run with three seeds on that dataset; "\<0.0001": a spread below the four
  decimals at which seeds 1 and 2 were saved.], size: 7pt) <tab-seeds>

#report-table("equal_relevance", [Methods at equal relevance (final session): the largest change of the predicted
  share (held-out model) reached without losing more than 1% or 5% of the base recommender's NDCG\@10, and the change
  of the logged share at that point, in percent. "–": the method does not reach that relevance.], size: 7pt)
<tab-equal-relevance>

#figure(
  image("figures/feedback_assumptions.svg", width: 48%),
  caption: [The reward model on the users' real next five items: bias and AUC of the predicted share at each step,
    with the earlier responses as logged, expected, sampled, finished, or unknown to the model trained with hidden
    responses (review session).],
) <fig-assumptions>

#report-table("titles", [KION: the titles whose number of lists among the 59,997 test requests grew and shrank most
  under the main run (review session). Titles as given in the English release of the KION item table, which
  translates the Russian titles; "usual share": the title's mean consumed share in the training period.], size: 7pt) <tab-titles>

#report-table("history_length", [NDCG\@10 and the change of the logged share per recommended item by the user's
  number of events before the request, in quartiles (review session, main run).], size: 7pt) <tab-history>

#report-table("concentration", [Concentration of impressions: catalogue coverage, Gini coefficient of the impressions
  per item, and the share of impressions taken by the 1% most shown items (review session).], size: 7pt) <tab-concentration>

#report-table("anchor_sweep", [Anchor sweep with the value-model estimator, 1,000 iterations per run (final
  session). KL is in nats per generated item; "Training model" is the share predicted by the model that produced the
  rewards.], size: 7pt) <tab-sweep>

#report-table("baselines", [Baselines on generated or ranked lists of ten (final session); every baseline reads the
  recommender's window of history. On KION the history's items are blocked, so replay of the history reduces to
  popularity and the two rows coincide. The last two rows are the recommender that is fine-tuned.], size: 7pt) <tab-baselines>

#report-table("alternatives", [Ways of raising the reward without on-policy RL (final session). Re-ranking starts
  from the ranked lists of the base recommender, every other method from its generated lists.], size: 7pt) <tab-alternatives>

#report-table("rewards", [Rewards that an item cannot earn on its own account, and logged relevance (final
  session). "Rel. share" is the held-out model's predicted share relative to what is typical for the item; 0.5 is "as
  usual".], size: 7pt) <tab-rewards>

#report-table("retention_rewards", [Rewards for continuing the session and for returning, per item (final session).
  "Continues" and "returns" are the held-out model's predictions; on KION a session is a calendar day.], size: 7pt) <tab-retention>
