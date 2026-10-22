// Paper draft, version 3.
// Build: python3 report/tables.py && python3 report/figures.py && typst compile --root report report/archive/paper_v3.typ
// Tables come from results/<session>/*.csv through report/tables.py, figures 2 and 3 through report/figures.py; the
// numbers quoted in the prose are read from the same reports. Bibliography: report/references.bib.

#set document(
  title: "What Keeps a Generative Recommender Relevant? An Empirical Study of Reinforcement Fine-Tuning Against a Learned Reward Model",
  author: "Daniil Antonov",
)
#set page(paper: "a4", margin: (x: 2.3cm, y: 2.4cm), numbering: "1")
#set text(font: "New Computer Modern", size: 10pt, lang: "en")
#set par(justify: true, leading: 0.62em, spacing: 0.9em)
#set heading(numbering: "1.1")
#show heading.where(level: 1): set block(above: 1.5em, below: 0.8em)
#show heading.where(level: 2): set block(above: 1.2em, below: 0.6em)
#show figure.caption: set text(size: 9pt)
#show figure.where(kind: table): set figure.caption(position: top)
#show table: set text(size: 8.5pt, hyphenate: false)
#show table: set par(justify: false)
#set list(indent: 0.8em)
#set enum(indent: 0.8em)
#set math.equation(numbering: "(1)")
#set figure(placement: auto)

// A table with rules above, below and under the header only.
#let results-table(columns: (), align: auto, header: (), inset-x: 5pt, ..rows) = table(
  columns: columns,
  align: align,
  stroke: none,
  inset: (x: inset-x, y: 3.2pt),
  table.hline(stroke: 0.8pt),
  table.header(..header.map(cell => strong(cell))),
  table.hline(stroke: 0.4pt),
  ..rows,
  table.hline(stroke: 0.8pt),
)

// A table typeset by report/tables.py from the reports of the runs.
#let report-table(name, caption, size: 8.5pt) = figure(
  kind: table,
  {
    show table: set text(size: size)
    include "../tables/" + name + ".typ"
  },
  caption: caption,
)

// One box of the pipeline figure.
#let stage(title, body) = rect(width: 100%, height: 3.1cm, inset: 6pt, radius: 2pt, stroke: 0.6pt + luma(90))[
  #set text(size: 8pt)
  #set par(justify: false, leading: 0.5em)
  #set align(left + top)
  #strong(title) \
  #body
]

#align(center)[
  #text(size: 16pt, weight: "bold")[
    What Keeps a Generative Recommender Relevant? \
    An Empirical Study of Reinforcement Fine-Tuning \ Against a Learned Reward Model
  ]
  #v(0.7em)
  #text(size: 11.5pt)[Daniil Antonov]
  #v(0.05em)
  ITMO University, Saint Petersburg, Russia
  #v(0.2em)
  #text(size: 9pt)[Draft, version 3 #h(0.6em) 14 October 2026]
]

#v(0.6em)

#pad(x: 1.1cm)[
  #align(center, text(weight: "bold")[Abstract])
  #set text(size: 9.5pt)
  A recommender that generates a list item by item can be fine-tuned with reinforcement learning (RL) towards feedback
  that next-item training does not see. We study the least protected form of this idea, in which a learned model of
  user feedback is the only source of reward: a GPT-2 recommender is fine-tuned with a policy gradient to raise the
  consumed share of its items as a second GPT-2 predicts it, held near the pretrained recommender by a KL anchor, on a
  video log (KION) and a music log (Yambda-50M). Four instruments that the optimised model cannot influence judge the
  result: a reward model fitted on other users, the logged response to the recommended items that users went on to
  consume, the same lists scored for another user, and relevance split by the logged outcome. On KION the predicted
  share rises by 21.5% and the held-out model agrees, yet the logged share per recommended item does not change, and
  99% of the gain is what any other user would give the same lists: the policy learned which titles everybody
  finishes. A reward that is neutral to how easy an item is to finish removes this drift and leaves RL little to
  gain. The anchor weight is not a comparable quantity: at one weight a leave-one-out baseline moves the policy seven
  times further than a value model. The KL distance is comparable: at equal distance the two estimators give the same
  policy, and crediting an item with its own reward only is no worse than crediting it with the rest of the list.
  Re-ranking the pretrained recommender's candidates matches or beats RL at every moderate operating point, also for
  a reward that depends on the list as a whole. The one reward term that improves logged outcomes is itself read
  from the log. All results are offline and single-seed.

  #v(0.3em)
  *Keywords:* generative recommendation, sequential recommendation, reinforcement learning, reward model, reward
  hacking, offline evaluation.
]

#v(0.4em)

= Introduction

Sequential recommenders built as autoregressive transformers over item or semantic identifiers
@kang2018sasrec @petrov2023gptrec @rajput2023tiger @zhai2024actions @deng2025onerec are trained to continue logged
sequences. The log records what users did, not what served them: in the two logs studied here, 36% of music listens
stop before half of the track and 52% of video titles are abandoned before half of their length. A recommender that
generates a list item by item is also a policy that takes a sequence of decisions, so reinforcement learning (RL) is
the natural tool for moving it from "what comes next" towards "what will be received well"
@chen2018topk @ie2019slateq @petrov2024aligning @matveev2026session.

Where the reward comes from decides how safe that step is. Two recent studies that report gains without a loss of
relevance both keep the reward tied to the log. Petrov and Macdonald @petrov2024aligning fine-tune GPTRec with PPO on
its own generated lists, with a reward that contains NDCG against the user's true next item. Matveev et al.
@matveev2026session train a retriever with off-policy REINFORCE on logged sessions, with logged feedback as the
reward, and use a learned feedback model only to evaluate. The third option is to let a learned model of user
feedback _be_ the reward. It is attractive because such a model scores any item for any user, so the policy can be
trained on its own generations and for signals that the log holds only for the items users chose. It is also the
option with the fewest protections: a policy optimised against a learned model drifts to where the model is wrong
@yu2020mopo @kidambi2020morel @gao2022scaling @skalse2022hacking, and in recommendation the drift takes the form of
popularity and filter-bubble effects @gao2023dorl @gao2022cirs @jiang2019degenerate.

This paper studies the third option in its barest form. A GPT-2 @radford2019gpt2 next-item recommender is fine-tuned
with a policy gradient to raise the share of an item that a second GPT-2, the _reward model_, predicts the user will
consume. Nothing in this reward asks for relevance; a KL penalty towards the pretrained recommender, the _anchor_, is
the only term that does. We ask five questions.

#block(inset: (left: 0.8em))[
  #set enum(numbering: n => strong[RQ#n], indent: 0pt, body-indent: 0.7em)
  + Does fine-tuning raise the reward, and is the gain real outside the model that was optimised?
  + Is relevance preserved, do users still receive different lists, and which mechanism preserves either?
  + Does the policy use the fact that a list is a sequence, and what does the reward of a generated list depend on?
  + Can other signals serve as rewards: retention, a share that is neutral to item length, a reward for the whole
    list?
  + What is the trade-off between reward and relevance, and does reaching it take RL?
]

Every reward gain in this setting is reported by a model. The pipeline therefore carries instruments that do not
depend on the model being optimised, and the study is organised around what they show. Our contributions are the
following.

- *A protocol for judging RL against a learned reward offline.* A second reward model fitted on a disjoint half of
  the users; the logged response to those recommended items that the user went on to consume, which no model stands
  behind; the same lists scored for another user, which separates what the policy learned about the user from what
  it learned about items; and relevance split by the logged outcome. Operating points are selected on validation
  requests only (@sec-instruments, @sec-setup).
- *The gain is confirmed by a second model and by nothing else.* On KION the predicted consumed share rises by 21.5%
  and a reward model trained on other users agrees. The logged share per recommended item does not move
  ($+0.6%$, interval $[-0.6, +1.7]$), the logged share relative to what is usual for the item falls, and 99% of the
  predicted gain is what other users would give the same lists. Re-ranking candidates by an item's mean share in
  the training log, with no reward model, reaches the same point. A reward that pays for the share _relative to the
  item's usual_ removes the drift towards easily finished items and leaves the policy almost nothing to gain
  (@sec-main, @sec-whose, @sec-rewards-results).
- *What the anchor constrains.* The weight of the anchor is measured against standardised advantages and has no
  meaning across estimators: at the same weight a leave-one-out baseline moves the policy 7 to 15 times further from
  the pretrained recommender than a learned value model. A budget on the KL divergence is comparable. At an equal
  budget the two estimators arrive at the same policy, and a policy credited with each item's own reward only, or
  trained on one-item episodes, is at least as relevant as one credited with the whole rest of the list: we find
  no evidence that the policy gradient uses the sequential structure of a list. PPO's clipping hardly took part
  (@sec-anchor, @sec-credit).
- *Reaching the trade-off does not take RL.* Re-ranking the 100 best candidates of the pretrained recommender by
  the same reward model reaches the frontier of RL without any training, and raises the logged share where RL does
  not. This also holds for a reward that depends on the list as a whole, where a greedy construction is not optimal
  in principle. The reward term that improves logged outcomes under RL is logged relevance, and on a log with
  repeated consumption it does so by replaying the user's history (@sec-alternatives, @sec-rewards-results).

All results are offline, and every configuration was trained with one seed. @sec-discussion separates the claims
that offline evidence settles from those that need an online experiment, and @sec-limitations lists what remains
open.

= Related Work

*Generative recommenders and RL.* GRU4Rec @hidasi2015gru4rec, SASRec @kang2018sasrec and BERT4Rec
@sun2019bert4rec made next-item prediction the default formulation of sequential recommendation; generative
recommenders use the same models autoregressively, over items @petrov2023gptrec or semantic identifiers
@rajput2023tiger, up to industrial scale @zhai2024actions @botta2025pinrec @deng2025onerec. We keep the simplest
member of the family so that the effects of fine-tuning are not confounded with tokenisation. RL enters
recommendation through off-policy policy gradients @chen2018topk, slate values @ie2019slateq @deffayet2023slate and
RL heads on supervised models @xin2020sqn @xin2021sa2c. The two studies we build on sit at opposite ends: on-policy
PPO over the recommender's own generations with a reward read from the true next item @petrov2024aligning, and
off-policy REINFORCE over logged sessions @matveev2026session. Learned models enter rewards when the wanted signal
is sparse @christakopoulou2022reward and replace the user altogether in simulators
@ie2019recsim @shi2018virtualtaobao @zhao2023kuaisim; ours is a simulator learned from the log the recommender was
trained on.

*Learned rewards and their exploitation.* Model-based offline RL trusts a model only where data supports it
@yu2020mopo @kidambi2020morel @janner2019mbpo. For language models fine-tuned against reward models, a "gold"
reward first rises and then falls while the proxy keeps rising @gao2022scaling; ensembles mitigate this
@coste2023ensembles, and reward hacking has a formal treatment @skalse2022hacking. Our held-out reward model plays
the part of the gold model, and we find that it is not gold enough: two models trained on the same kind of data
share what they have learned about items. In recommendation, conservative offline RL amplifies the Matthew effect
@gao2023dorl and feedback loops degenerate @gao2022cirs @jiang2019degenerate.

*Constraints and rewards.* TRPO and PPO bound each update relative to the previous iterate
@schulman2015trpo @schulman2017ppo; the adaptive-penalty variant of PPO adjusts a KL coefficient towards a target
divergence, the rule that our KL budget applies to the pretrained policy. Penalising divergence from a _pretrained_
model is KL-control @jaques2016sequencetutor @jaques2019wayoff, used when language models are fine-tuned from human
preferences @ziegler2019finetuning @ouyang2022instructgpt and as behaviour regularisation in offline RL
@wu2019brac. Consumption time is confounded by item duration @zhan2022d2q @zhao2023watchtime; we observe the
confounding from the policy's side and remove it with a within-item quantile. Long-term engagement and retention
have been optimised directly @zou2019longterm @cai2023retention and through proxies
@mann2018delayed @su2023exploration. The diversity of a list is an objective that no single item can earn; maximal
marginal relevance @carbonell1998mmr is its greedy optimiser and our reference for a list-dependent reward.

*Generation and evaluation.* Feeding a model its own outputs takes it away from the inputs it was trained on
@bengio2015scheduled @ranzato2015sequence, and spending inference compute on several candidates that a learned
model judges is the recommender form of test-time scaling @matveev2026session @guo2026promise. Next-item protocols
hide deficiencies of RL recommenders @deffayet2023offline, and off-policy estimators such as the doubly robust one
@dudik2011dr need logged propensities, which neither log provides. Results selected and judged by the same feedback
model are not independent evidence @matveev2026session, and baselines need one protocol
@ferraridacrema2019progress.

= Method

@fig-pipeline shows the pipeline: a recommender is pretrained on next-item prediction, two reward models are fitted
on disjoint halves of the users, and the recommender is fine-tuned on the reward that one of them predicts for its
own generated lists.

#figure(
  [
    #grid(
      columns: (1fr, 14pt, 1fr, 14pt, 1fr, 14pt, 1fr),
      align: horizon + center,
      stage[1. Pretrain][Recommender $pi_0$: a GPT-2 decoder over item tokens, next-item cross-entropy on logged
        sequences.],
      sym.arrow.r,
      stage[2. Generate][The policy $pi_theta$, started from $pi_0$, samples a list of $K = 10$ items one by one for
        a logged history $h$.],
      sym.arrow.r,
      stage[3. Reward][The training reward model predicts the response to each generated item: consumed share
        $hat(rho)$, relative share $hat(q)$, continuation $hat(c)$, return $hat(u)$.],
      sym.arrow.r,
      stage[4. Update][Policy gradient on the predicted returns plus the anchor
        $beta dot "KL"(pi_theta || pi_0)$. Steps 2--4 repeat.],
    )
    #v(4pt)
    #rect(width: 100%, inset: 6pt, radius: 2pt, stroke: (paint: luma(90), thickness: 0.6pt, dash: "dashed"))[
      #set text(size: 8pt)
      #set par(justify: false)
      #set align(left)
      *Evaluation only.* A held-out reward model fitted on the other half of the users #sym.dot.c the logged response
      to the recommended items the user went on to consume #sym.dot.c the same lists scored for another user
      #sym.dot.c relevance against the user's real future, split by the logged outcome #sym.dot.c list concentration.
    ]
  ],
  caption: [The pipeline. The reward model of step 3 is the only source of reward during fine-tuning; the instruments
    in the dashed box never influence the policy.],
) <fig-pipeline>

== Setting and models

A log is a set of events, each with a user, an item $i_t$, a time and the _consumed share_ $rho_t in [0, 1]$: the
share of the track that was played or of the title that was watched. A recommender is a distribution
$pi_theta (j | h)$ over the next item given the history $h$; $pi_0$ is the recommender before fine-tuning, and a
recommendation is a list $a_(1:K)$ with $K = 10$.

*Base recommender.* A GPT-2 decoder @radford2019gpt2 over item tokens with tied input and output embeddings, trained
with the full-softmax next-item cross-entropy. A list is obtained by _ranking_ (Top-K: the catalogue is scored once)
or by _generation_ (Next-K: each chosen item is appended to the history as if the user had consumed it, and no item
repeats within a list) @petrov2023gptrec. Fine-tuning changes the second mode, in which the recommender is a policy
over a sequence of decisions.

*Reward model.* A second GPT-2 decoder of the same size. The token at position $t$ carries the item $i_t$ together
with the consumed share of the _previous_ item in ten buckets, so the state at position $t$ has seen the candidate
item and the whole history with its responses, but not the response it predicts. A generated item can thus be scored
before any outcome for it exists. Five heads read the state, each trained with binary cross-entropy: the consumed
share $hat(rho)$ (against $rho$ as a soft label); the _relative share_ $hat(q)$ defined below; the probability
$hat(c)$ that the session continues (the user's next event follows within a gap $g$); the probability $hat(u)$ that
the user returns (a new session starts within a window $w$); and a choice head, which an earlier iteration of the
study found useless as a relevance signal and which no reward here uses. A retention outcome counts as known only if
its whole waiting time lies inside the observed data.

*Relative share.* The consumed share pays for items that are easy to finish, whoever receives them: a film over a
series, a short track over a long one. To separate the user's response from the item's, each event also gets its
mid-rank quantile among the training-period events of the same item,
$ q_t = (\#{rho' < rho_t} + 1/2 \#{rho' = rho_t}) / n_(i_t), $ <eq-relative>
where $rho'$ runs over the $n_(i_t)$ training-period consumed shares of item $i_t$. Its mean is exactly one half for
every item, so no item is better than another on average, and a policy can raise it only by matching items to the
users who take to them more than others do.

*Two copies.* Two reward models are trained on disjoint halves of the users. The _training_ reward model produces
the rewards during fine-tuning. The _held-out_ reward model never influences the policy and only scores its lists.

== Reinforcement fine-tuning <sec-rl>

One episode is the generation of one list for a history sampled from the training period. The state is
$s_k = (h, a_(1:k-1))$, the action is the next item $a_k$, the reward $r_k$ is predicted by the training reward
model (by default $r_k = hat(rho)_k$), and the return is $G_k = sum_(l >= k) gamma^(l-k) r_l$. The reward model is
the environment. To score the $k$-th item it needs the responses to the items generated before it, which nobody
observed; by default it is fed its own expectation $hat(rho)$ for them, and @sec-credit compares this with sampling
a response.

The policy gradient $EE[sum_k A_k nabla_theta log pi_theta (a_k | s_k)]$ is estimated with one of two advantage
estimators. The default uses a learned _value model_ $V_phi$, a copy of the recommender's transformer with a scalar
head, and generalised advantage estimates @schulman2015gae; each batch of lists is used for four gradient steps on
the clipped surrogate of PPO @schulman2017ppo. The alternative is a _leave-one-out_ baseline: several lists are
sampled for each history, and the baseline of a list is the mean return of the others
@williams1992reinforce @kool2019buy. Advantages are standardised to zero mean and unit variance within a batch in
both cases. The clipping of the surrogate touched at most 0.3% of sampled items in any run that did not collapse
(@sec-anchor), so we do not attribute anything to PPO: in effect the default is an actor--critic policy gradient
with several steps per batch.

*Anchor and KL budget.* The loss adds a penalty on the divergence from the frozen pretrained recommender, computed
exactly over the catalogue at every position:
$ cal(L)(theta) = cal(L)_"PG" (theta) + beta dot EE_k ["KL"(pi_theta (dot | s_k) || pi_0 (dot | s_k))]. $ <eq-loss>
Since the default reward has no relevance term, $beta$ is the only thing that asks the policy to stay relevant. The
penalty is weighed against standardised advantages, so the same $beta$ is a different constraint under another
estimator, reward or dataset. A _KL budget_ $delta$ fixes the constraint itself: after every iteration the weight is
multiplied by $exp(eta (overline("KL") \/ delta - 1))$, with $eta = 0.03$ and the mean divergence $overline("KL")$
of the iteration, so that the policy settles at the distance $delta$ in nats per generated item.

*Credit along the list.* Two ablations test whether fine-tuning uses the sequential structure of a list. With
$gamma = 0$ the return is the reward of the item itself. With _one-item episodes_ a single item is sampled per
episode, so the policy never learns from a context that contains generated items; at evaluation its lists are still
generated item by item.

== Rewards <sec-rewards>

All rewards are instances of
$ r_k = w_rho (hat(rho)_k - b_k) + w_q hat(q)_k + w_c hat(c)_k + w_u hat(u)_k
  + w_F (bb(1)[a_k in F]) / (log_2 (k + 1)) + w_D / (K - 1) sum_(l < k) d(a_k, a_l). $ <eq-reward>
The default is $w_rho = 1$ and everything else zero. The _relative-share_ reward is $w_q = 1$ alone, and the
retention rewards use $w_c$ and $w_u$. The fifth term is the _logged-relevance_ reward of @petrov2024aligning: for a
history from the training period the user's real continuation $F$ is known, and the term sums to DCG\@K over the
list. A baseline $b_k$ makes the reward _personal_: it subtracts what the same reward model predicts for the same
list shown to another user, or the item's mean share in the training period. The last term makes the reward depend
on the _list as a whole_: $d$ is the cosine distance between the embeddings of two items in $pi_0$, and over a list
the term sums to $w_D K \/ 2$ times its mean pairwise distance. An item's worth then depends on its neighbours, and
scoring items one at a time no longer finds the best list.

== Alternatives to on-policy RL <sec-alt-methods>

Four methods raise the same rewards without generating lists in a training loop. *Liked-only fine-tuning* continues
next-item training for one epoch on targets with $rho >= 0.5$; it uses logged feedback and no reward model.
*Off-policy REINFORCE* @matveev2026session raises the log-probability of every logged item in proportion to the
discounted logged reward that followed it in the session, with importance weights against $pi_0$ capped at $kappa$.
*Re-ranking* takes the 100 best candidates of $pi_0$ and keeps the $K$ with the highest
$r(j | h) + w log pi_0 (j | h)$, where $r$ is the reward predicted for $j$ as the next item after the real history;
nothing is trained, and $w$ plays the part of the anchor. For the list reward it becomes greedy maximal marginal
relevance @carbonell1998mmr: the list is filled one position at a time, and the distance to the items already
picked is added to the score. *Best of $n$ lists* samples $n$ lists from $pi_0$ and serves the one with the highest
predicted return, the test-time scaling of @matveev2026session.

== Instruments <sec-instruments>

For evaluation, $F$ is the set of the next $M = 20$ items the user really consumed after $h$.

- *Held-out reward model.* A gain that the reward model of the other half of the users does not confirm is
  exploitation of the errors of the training model.
- *Logged response.* A recommended item that is in $F$, a _hit_, comes with the response the log recorded, and no
  model stands between that response and the recommender. We report the number of hits per list, their mean logged
  share, and two quantities per recommended item in which an item that was not hit counts as nothing: the _logged
  share_ $1/K sum_k bb(1)[a_k in F] rho(a_k)$ and the _logged surplus_
  $1/K sum_k bb(1)[a_k in F] (q(a_k) - 1/2)$, which is positive only if users took to the recommended items more than
  people usually take to those items. Hits are a selected sample of the recommendations: these numbers say how the
  items the user did reach were received, not how the rest of the list would have been.
- *Another user.* A list generated for a user with history $h$ is also scored for a user with another history $h'$.
  The difference $R_"own" - R_"other"$ is the part of the predicted response that depends on who receives the list;
  it is zero for a recommender that ignores the user. The gain of a method over $pi_0$ splits into the gain other
  users would give the same lists and the rest, whose share of the total is the _personal share of the gain_.
- *Relevance by outcome.* Recall and NDCG against $F$ and against its parts: _liked_ ($rho >= 0.5$) and _skipped_
  items; items consumed _above_ and _below usual_ ($rho$ against the item's mean share in the training period),
  which asks the same question without favouring items that are easy to finish; and, where repetition is allowed,
  items that are _repeats_ from the visible history and _new_ ones.
- *Concentration.* Catalogue coverage (the share of items that appear in at least one list) and the expected overlap
  between the lists of two users (_pair overlap_).

= Experimental Setup <sec-setup>

*Data.* Yambda-50M @ploshkin2025yambda is a listening log of Yandex Music: each event has the track, a timestamp,
the played share, the track length and a flag telling whether the user chose the track or a recommender served it.
KION @petrov2022kion is a log of an online cinema with one row per user and title: the date of the last viewing and
the watched share of the title. Both logs are split along one time axis: the last 7 days are the test period, the 7
days before them the validation period, the rest the training period; the catalogue and all item statistics come
from the training period only (@tab-data). The consumed share is bimodal in both logs, so the reward model is in
effect a skip classifier. On KION the share depends strongly on the length of the title: films are watched to 0.5 on
average, and the longest tenth of the catalogue, which is series, to 0.22, because the share is taken of the whole
series. A reward built on the share prefers films before any model is involved. On Yambda the session continues if
the user has another event within 30 minutes and the user returns by starting a new session within a day; KION only
has dates, so a session is a calendar day and a return is a viewing on another day within a week.

#figure(
  results-table(
    columns: (auto, auto, auto),
    align: (left, right, right),
    header: ([], [KION], [Yambda-50M]),
    [Users], [256,407], [8,711],
    [Items in the catalogue], [7,992], [50,000],
    [Events, millions (training / validation / test)], [3.51 / 0.19 / 0.18], [35.0 / 1.13 / 1.08],
    [Events per user, median], [10], [2,638],
    [Mean consumed share], [0.50], [0.65],
    [Events consumed at least half], [48%], [64%],
    [Events that continue the session / are followed by a return], [42% / 71%], [95% / 92%],
    [Events the user chose (not served by a recommender)], [not recorded], [51%],
    [Test requests], [59,997], [7,593],
  ),
  caption: [The two processed logs.],
) <tab-data>

*Models and training.* The recommender and the reward models have 4 layers, width 256 and 4 attention heads, with
windows of 48 events on KION and 128 on Yambda, trained with AdamW @loshchilov2017adamw and stopped by the
validation loss. Re-recommending items from the history is allowed on Yambda, where repeated listening is normal,
and blocked on KION, where every user--title pair occurs once; every method obeys the same constraint. Fine-tuning
samples one list for each of 512 histories per iteration ($gamma = 0.99$, $lambda = 0.9$, clipping threshold 0.2,
learning rates $10^(-5)$ for the policy and $10^(-4)$ for the value model, which is trained alone for the first 50
iterations). The leave-one-out estimator samples 4 lists for each of 128 histories, so both see as many lists per
iteration. The _main run_ has $beta = 0.3$ and 2,000 iterations; every other fine-tuning run has 1,000.

*Baselines.* The most popular items, a replay of the user's own history (most played first), a first-order Markov
chain, GRU4Rec @hidasi2015gru4rec, SASRec with its original loss @kang2018sasrec and BERT4Rec @sun2019bert4rec. The
neural baselines take the width, depth, dropout and window length of the recommender and are stopped by their own
validation loss; none was tuned.

*Evaluation.* Each test user with a history makes one request at the start of the test week: the last 38 events on
KION (118 on Yambda) are the history and the next at most 20 events are the ground truth. All quantities are
measured on generated lists of ten. The change from the base to the fine-tuned recommender is reported with a 95%
interval from a paired bootstrap over users, which measures the noise of the user sample and not the noise between
training runs.

*Selection.* No model, hyperparameter or operating point was chosen on the test period within this protocol. Model
sizes and learning rates were fixed before the study. Sweeps are reported in full, and where one point of a sweep
has to be named we name the run with the highest held-out predicted share on _validation_ requests among the runs
that keep the validation NDCG\@10 within 5% of the base recommender's. The weight $beta = 0.3$ of the main run is
the exception: it was carried over from earlier iterations of this study, in which test metrics of anchor sweeps had
been seen, so it is an illustration and not a selected optimum. By the validation rule the selected weights are
$beta = 1$ on KION and $beta = 0.05$ on Yambda (@tab-selection).

*Sessions and repeatability.* The KION results come from one session of 49 fine-tuning runs (6.1 hours on one 96 GB
GPU). The Yambda results come from two: a first session of 12 runs (8.1 hours, of which 4.4 go to the base models
and the baselines) and a continuation that reused its models and runs and trained 13 more (4.2 hours; @tab-time).
The continuation re-evaluated every reused run and reproduced the numbers of the first session to four decimals.
The Yambda grids are still narrower than KION's (@tab-config). Every configuration was trained with one seed. A
second KION session with the same seed reproduced the base recommender exactly and the 26 fine-tuning runs the two
sessions share with a median difference of 0.0003 in NDCG\@10 and 0.001 in predicted share (at most 0.003 and
0.007): the noise of the hardware, a lower bound on the variation between seeds.

= Results

In the tables, "Pred. share" is the consumed share of the generated lists as the held-out reward model predicts it,
and "Logged" and "Surplus" are the logged share and the logged surplus per recommended item of @sec-instruments.

== Base models <sec-pretrained>

*Recommender.* On KION the recommender is mid-table (@tab-baselines): level with GRU4Rec, 13% above SASRec and 5%
below BERT4Rec for ranked lists, and popularity is a strong floor. Generating a list item by item costs 4% of
NDCG\@10 against ranking. On Yambda the recommender is the best of the trained models, and all of them are far below
a replay of the user's own history: NDCG\@10 0.187 against 0.117. The split into repeats and new items shows where
the trained models stand. Replay scores only on repeats; on tracks the user has not played in the visible history,
the recommender is the best ranked model (0.032) and popularity reaches a third of it. Every statement below about
relevance on Yambda is relative to this recommender, and the split is reported wherever a method could gain by
replaying.

#report-table("baselines", [Baselines on generated or ranked lists of ten for the test requests. The last two rows
  are the recommender that is fine-tuned, ranked and generated. On KION items of the history are blocked, so every
  hit is a new item.]) <tab-baselines>

*Reward model.* On held-out events the reward model predicts the consumed share with AUC 0.755 on KION and 0.868 on
Yambda (@tab-reward-model), against 0.656 and 0.607 for the item's mean share and 0.673 and 0.798 for the user's
recent mean. The two copies agree (correlation of predictions 0.94 and 0.95). The right half of the table measures
what a signal offers a recommender that chooses among its own candidates: how much it varies between the 100
candidates of one user, against how much its mean varies between users. On Yambda the predicted share varies five
times less between one user's candidates than between users: it is mostly a property of the listener, which leaves a
policy little to act on. Retention is predictable on KION (AUC 0.72 for return) and barely on Yambda (0.61), and on
both logs it varies several times less between candidates than between users.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto, auto, auto),
    align: (left, right, right, right, right, right, right),
    stroke: none,
    inset: (x: 5pt, y: 3.2pt),
    table.hline(stroke: 0.8pt),
    table.header(
      [], table.cell(colspan: 3, align: center, strong[KION]), table.cell(colspan: 3, align: center, strong[Yambda]),
      table.hline(start: 1, end: 4, stroke: 0.4pt),
      table.hline(start: 4, end: 7, stroke: 0.4pt),
      [], strong[AUC], strong[Candidates], strong[Users], strong[AUC], strong[Candidates], strong[Users],
    ),
    table.hline(stroke: 0.4pt),
    [Consumed share $hat(rho)$], [0.755], [0.144], [0.153], [0.868], [0.052], [0.250],
    [Relative share $hat(q)$], [0.702], [0.042], [0.091], [0.833], [0.039], [0.145],
    [Session continues $hat(c)$], [0.657], [0.050], [0.093], [0.603], [0.006], [0.016],
    [Return $hat(u)$], [0.722], [0.033], [0.125], [0.611], [0.009], [0.094],
    table.hline(stroke: 0.8pt),
  ),
  caption: [The held-out reward model on test events (AUC; the share heads against the outcome thresholded at one
    half), and the standard deviation of each prediction between the 100 candidates of one user ("candidates") and of
    its mean over candidates between users ("users").],
) <tab-reward-model>

== What fine-tuning changes <sec-main>

@tab-main compares the main run with the base recommender.

#report-table("main_run", [The main fine-tuning run ($beta = 0.3$) against the base recommender. Changes are
  relative, in percent, with a 95% paired-bootstrap interval over users where the quantity is defined per user.],
  size: 7.9pt) <tab-main>

*KION: the models agree, the log does not.* The training reward model reports a gain of 20.3% and the held-out one
21.5%: by the standard check for exploitation the gain is real. Relevance falls by 10.7% overall and by 18.2% on
skipped titles, and it is kept on liked ones ($+1.3%$), which looks like a policy that has learned to avoid what
users abandon. The other instruments disagree. The policy recommends fewer titles that the user went on to watch
(hits $-8.3%$), and those are finished more often ($+9.8%$) because they are titles that everybody finishes more
often: the mean duration of a recommended title falls from 5.7 to 3.4 hours, away from series. The two effects
cancel, and the logged share per recommended item is unchanged ($+0.6%$, $[-0.6, +1.7]$). Measured against what is
usual for each title, users took to the recommended titles less than before: the logged surplus falls by 62%, and
relevance on titles watched above usual falls by 9.4%, almost as much as overall. The predicted relative share agrees
($-1.0%$). Finally, the same lists shown to _other_ users are predicted to earn the same gain ($+22.0%$).

*Yambda: little to gain.* The same weight moves the policy less than half as far (KL 0.037 against 0.092 nats per
item) and changes little: $+1.7%$ of predicted share, with NDCG\@10 inside its interval and a small gain on liked
tracks. The logged share per recommended item rises by 2.4% ($[+0.3, +4.6]$) and the surplus does not move; the
policy has shifted slightly towards popular tracks (mean popularity $+13%$). The value model explains 78% of the
variance of returns here, against 49% on KION: the reward is mostly determined by the listener, as
@tab-reward-model predicted.

== The anchor: a weight is not a distance <sec-anchor>

#report-table("anchor_sweep", [Anchor sweep with the value-model estimator, 1,000 iterations per run. KL is the
  divergence from the base recommender at the end of fine-tuning, in nats per generated item; "Training model" is
  the share predicted by the reward model that produced the rewards.], size: 8pt)
<tab-sweep>

*Without an anchor the policy collapses, and the held-out model partly applauds.* At $beta = 0$ nearly every user
receives the same list on both datasets (pair overlap 0.98 and 1.00, @tab-sweep). The training reward model reports
a share of 0.84 on KION and the held-out one 0.69: the gap is exploitation, but the held-out value is still 51%
above the base recommender. A second model trained on the same kind of data is fooled in the same direction, only
less. The model-free instruments are unambiguous: NDCG\@10 and the logged share are near zero.

*The trade-off is smooth on KION and a cliff on Yambda.* On KION relevance falls and the predicted share rises
steadily as the anchor weakens. The logged share is highest at $beta = 0.5$ ($+4%$) and is below the base
recommender from $beta = 0.2$ on. By the validation rule the selected run is $beta = 1$: 1.6% below the base
recommender in NDCG\@10, 7% above in predicted share, 3% above in logged share. On Yambda relevance and predicted
share move by a few percent down to $beta = 0.05$ (KL 0.29: NDCG\@10 $-4.9%$, predicted share $+5.0%$), and
$beta = 0.03$ collapses the policy (KL 4.4): the cliff lies between these two weights, and the validation rule selects
$beta = 0.05$.

#figure(
  image("../figures/frontier_by_distance.svg", width: 100%),
  caption: [Every fine-tuning run of the default reward against the distance it ended at from the base recommender
    (logarithmic axis; the base recommender is drawn at the left edge). Runs that differ in the advantage estimator
    and in how the constraint is set lie on one curve. Runs with myopic credit lie on it at the KL budgets and, at a
    fixed weight, far to the right, above it in relevance and below it in coverage. For one-item episodes the
    distance is that of the first position only.],
) <fig-frontier>

*The weight means different things under different estimators; the distance does not.* @tab-estimators changes
only the baseline of the policy gradient, with the number of lists per iteration, the learning rate and the passes
over a batch held equal. At $beta = 0.3$ on KION the leave-one-out estimator ends 6.8 times further from the base
recommender than the value model (KL 0.55 against 0.08), and at $beta = 0.1$ it collapses where the value model
does not; on Yambda the factor at $beta = 0.1$ is 15. Under a KL budget the two estimators end at nearly the same
distance and on the same frontier, while the weights that hold the budget differ by a factor of 2.3 to 3.3. Where
one of them settles slightly further away (on KION the value model, KL 0.104 against 0.099 and 0.56 against 0.50;
on Yambda at the smaller budget the leave-one-out run, 0.200 against 0.189), it is correspondingly less relevant at
a higher predicted share. The budget runs lie on the frontier of the weight sweep (@fig-frontier). On Yambda the
budget also controls what the weight cannot: budgets of 0.2 and 0.5 nats end at almost the same weight (0.066 and
0.067). In this region the distance at which a run settles depends on more than its weight, so a fixed weight
hardly selects an operating point there, while a budget does ($delta = 0.5$: NDCG\@10 $-19%$, predicted share
$+11%$).

#report-table("estimators", [The value model against the leave-one-out baseline at equal anchor weights and at
  equal KL budgets. "$beta$ at end" is the weight the run finished with. "Clipped" is the share of sampled items
  whose probability ratio left PPO's clipping range.]) <tab-estimators>

One reading of the factor is that standardising the advantages gives the policy-gradient term the same scale under
both estimators while the share of that scale that is signal differs: a leave-one-out advantage compares lists for
the same history, a value-model advantage with an imperfect critic still carries noise between histories, which
averages out over a batch. We did not test this reading; the practical point does not depend on it. An anchor
weight is not a quantity that can be compared between estimators, rewards, datasets or papers, and a KL budget is.

*Clipping hardly engaged.* The share of sampled items beyond the clipping threshold is at most 0.3% in every run
that did not collapse, the run with ten times the learning rate of the policy included; only collapsing runs reach a
few percent. The small learning rate and the anchor keep almost every update inside the threshold.

== Sequence credit and simulated feedback <sec-credit>

#report-table("credit", [Credit ablations on the default reward at equal KL budgets and, for KION, at an equal
  anchor weight.]) <tab-credit>

*At an equal distance, myopic credit is no worse.* At a fixed weight the ablated runs are not comparable with the
full one: with $gamma = 0$ the policy ends 31 times further from the base recommender (@tab-credit, last KION
panel), presumably because per-item rewards standardise to larger advantages than returns that share most of their
terms. At an equal KL budget the comparison is fair. On KION a policy credited with each item's own reward only is
6% more relevant than the policy credited with the whole rest of the list at a predicted share 1.5 points lower,
which puts it on or slightly above the same frontier (@fig-frontier); one-item episodes give the same picture, with
lower coverage at the larger budget, and the second session reproduces these numbers to the third decimal. On
Yambda the picture is the same at the budget of 0.2 (NDCG\@10 $+1.5%$ and $+3.1%$ at nearly the same predicted
share, with lower coverage), and at 0.5 both ablated policies lie close to the line between the full-credit runs.
We find no sign that crediting an item with what follows it, or learning from generated contexts, contributes to
the result.

*Sampling the simulated response changes the scale, not the policy.* The reward of the $k$-th item depends on the
responses assumed for the items before it. Fine-tuning with a response sampled from the predicted probability,
instead of its expectation, gives nearly the same policy (@tab-feedback in the appendix): NDCG\@10 0.103 against
0.104 on KION at $beta = 0.3$, 0.073 against 0.071 on KION and 0.118 against 0.116 on Yambda at $beta = 0.1$, where
the sampled-feedback runs also ended somewhat nearer to the base recommender. What the assumption changes is the
level of every number: the same policy has a predicted share of 0.55, 0.62 or 0.69 on KION, and 0.62, 0.73 or 0.87
on Yambda, depending on whether earlier items are assumed to have received the expected response, a sampled one,
or to have been finished. A multi-step predicted reward is in part a property of the simulation, and only
differences between policies under one assumption are interpretable.

== Whose reward is it <sec-whose>

#report-table("whose_gain", [The gain in predicted share (held-out model) and the part of it that other users would
  give the same lists. "Personal" is the share of the gain that depends on who receives the list.]) <tab-whose>

*The policy gradient learns items, not users.* On KION 99% of the gain of the main run is what the held-out model
predicts for the same lists shown to other users; at a weaker anchor the personal share is negative (@tab-whose). On
Yambda the small gain is almost entirely impersonal as well. Re-ranking the base recommender's candidates by the same
reward model has a personal share of 9% on KION and 44% on Yambda: the reward model does know something
about the user, and a method that asks it about each candidate uses that knowledge, while the policy gradient does
not find it.

*A table of item means reaches the same point.* Re-ranking candidates by $macron(rho)(j) + w log pi_0 (j | h)$,
where $macron(rho)(j)$ is the mean consumed share of item $j$ in the training period, uses no reward model and no
RL. At $w = 0.3$ on KION it reaches a predicted share of 0.549 at NDCG\@10 0.109; the main RL run reaches 0.553 at
0.105. What RL extracts from the reward model on this reward is, to a first approximation, a ranking of items by how
often they are finished.

== Does it take RL? <sec-alternatives>

#figure(
  image("../figures/methods.svg", width: 100%),
  caption: [Methods compared at the predicted share they reach (held-out model): relevance, the logged share per
    recommended item, and coverage. Points of a line are one method at several strengths. RL and the fine-tuning
    methods start from the generated lists of the base recommender, re-ranking from its ranked lists.],
) <fig-methods>

#report-table("alternatives", [Ways of raising the reward without on-policy RL. Re-ranking starts from the ranked
  lists of the base recommender, every other method from its generated lists.], size: 8pt) <tab-alternatives>

*Re-ranking reaches the frontier of RL on KION.* At nearly the same predicted share as the main RL run (0.545
against 0.553), re-ranking with $w = 0.3$ has NDCG\@10 0.113 against 0.105, coverage 0.53 against 0.45 and a logged
share per recommended item of 0.0201 against 0.0174 (@tab-alternatives, @fig-methods). Most of this difference is a
head start: re-ranking begins from the ranked lists of the base recommender and RL from its generated lists, which
are 4% less relevant and cover less of the catalogue. Measured from its own starting point, each method pays about
the same relevance for the same predicted share ($-7.6%$ of NDCG\@10 for $+0.08$ of share by re-ranking, $-10.7%$
for $+0.10$ by RL). The logged share is the one axis on which re-ranking moves and RL does not: $+4%$ from its
starting point against $+0.6%$. Liked-only fine-tuning and off-policy REINFORCE, which never consult the reward
model, land next to the main RL run in predicted share and relevance. No comparison favours on-policy RL, and
re-ranking needs no training.

*On Yambda the log-based methods lead.* Off-policy REINFORCE on logged sessions raises NDCG\@10 by 6% and the
logged share by 10% at the predicted share that RL reaches, through repeats; relevance on new tracks is unchanged
(@tab-repeats). Re-ranking buys more predicted share for a small loss of relevance. No method moves the predicted
share by more than a few points without leaving the region where the lists are still relevant.

*A reward for the whole list does not change the picture.* With a reward that adds the distance of each item to the
items before it, an item's worth depends on its neighbours, and the argument that re-ranking suffices because the
reward is a sum over items no longer applies. RL does optimise it: the list reward rises from 0.76 to 0.85 at
$beta = 0.3$ on KION, for 5% of NDCG\@10 (@tab-list in the appendix). Greedy re-ranking with the same reward
reaches 0.85 for 1.6% of the NDCG\@10 of its starting point, and 0.93 at the relevance RL had. On Yambda RL at
$beta = 0.1$ raises the list reward from 0.96 to 0.98 without losing relevance, and greedy re-ranking reaches 1.00
while staying above the relevance of the ranked lists it starts from. A greedy construction is not optimal for this
reward in principle; in practice it is ahead of the policy gradient.

== Other rewards <sec-rewards-results>

#report-table("rewards", [Rewards that an item cannot earn on its own account. "Rel. share" is the held-out model's
  prediction of the share relative to the item's usual (@eq-relative); 0.5 is "as usual".], size: 8pt)
<tab-rewards>

*A length-neutral reward removes the drift and most of the gain.* Under the relative-share reward the policy no
longer moves towards films: the mean duration of a recommended title rises from 5.7 to 6.7 hours where the share
reward took it to 3.4, and relevance is kept ($-0.4%$ at $beta = 0.3$, @tab-rewards). What is left to gain is
small. The predicted relative share rises from 0.529 to 0.541, and the logged surplus, which the share reward
halved, stays where it was (0.0013 against 0.0012). Re-ranking by the same signal gains more: a logged surplus of
0.0020 against 0.0014 for the ranked lists it starts from, for 1% of NDCG\@10. On Yambda the reward raises the
predicted relative share from 0.469 to 0.488 and the logged surplus from 0.0033 to 0.0040 for 5.5% of NDCG\@10.
This is the one place where RL moved a logged, length-neutral quantity more than re-ranking by the same signal did
(0.0034). It rests on one run, and the interval of this quantity over the user sample alone is $plus.minus 13%$ in
the main run, so we do not build on it. Once the reward cannot be earned by choosing easy items, what remains is
the personal part, and it is small.

*Logged relevance is the reward term that helps, and on Yambda it helps by replaying.* On KION the share plus ten
times the logged-relevance term with $beta = 0.3$ is the only RL configuration that improves on the base recommender
in relevance, predicted share and logged share at once: NDCG\@10 $+1.8%$, predicted share $+8.8%$, logged share
$+7%$. The term protects hits and the anchor protects variety: without the anchor, coverage falls from 0.50 to 0.11
(@tab-relevance). On Yambda logged relevance alone,
with no anchor, raises NDCG\@10 by 23%, which is the largest gain in relevance of the study. @tab-repeats shows
what it consists of: relevance on repeats rises by a third, relevance on new tracks falls by 23%, and the share of
recommendations taken from the history rises from 51% to 62%. The policy moves towards the replay baseline, which is
what the reward pays for on a log where repeats dominate. It is a legitimate gain on this metric and should not be
read as better recommendation of new music. With an anchor the move is milder: the share plus ten times logged
relevance at $beta = 0.1$ raises NDCG\@10 by 12% and the logged share by 13%, through repeats ($+15%$), while
relevance on new tracks falls by 4% and coverage is kept.

#report-table("repeats", [Yambda: relevance on tracks from the visible history and on new tracks.]) <tab-repeats>

*Retention and personal rewards.* Rewarding the predicted return raises it by 3 to 7% on KION for a loss of 8 to 23%
of NDCG\@10 and does not move it on Yambda (@tab-retention in the appendix); the signal varies four to ten times
less between a user's candidates than between users, so there is little for a policy to choose. The session signal
on KION is derived from calendar days and the order of rows within a day, and we do not interpret its reward. A
reward that subtracts what the same model predicts for another user cannot collapse onto one list even without an
anchor (pair overlap 0.33 against 0.98 on KION), and raises the personal part at $beta = 0.1$ from 0.014 to 0.024 on
KION for 3% of NDCG\@10, and by 5% on Yambda for 0.5% (@tab-personal).

= Discussion <sec-discussion>

*What the instruments add.* The usual defence against reward hacking is a second model, and here it fails in an
instructive way: it catches the exploitation that is specific to one set of weights (0.84 against 0.69 without an
anchor) and misses the exploitation that any model of this kind invites. Both reward models have learned, correctly,
that some titles are finished more often than others, and a policy that serves those titles to everyone raises both.
The instruments that expose this are cheap: score the same lists for another user, and count what the log says about
the recommended items the user reached.

*What RL added.* For a reward that is a sum of per-item predictions and is mostly determined by the item, the
optimum of the anchored objective is close to a re-ranking of $pi_0$ by item quality, and a policy gradient is an
expensive way to approximate it: a fine-tuning run costs minutes of GPU time per operating point, a re-ranking
weight costs nothing. The estimator comparison and the credit ablations say the same from the inside:
at an equal distance from $pi_0$, nothing that distinguishes one policy-gradient method from another mattered. The
list-dependent reward was the case built to favour sequential decisions, and a greedy heuristic still led.

*Relation to the studies we build on.* Petrov and Macdonald @petrov2024aligning keep relevance inside the reward;
our results agree that this term is what keeps hits, and add that it keeps whatever is easiest to hit, so that an
anchor or a diversity term has to keep the rest. Matveev et al. @matveev2026session learn from logged actions and
logged feedback; on the log with a recommender flag and long sessions this was the best method in logged terms,
and their warning about judging with the model that selected applies to our predicted shares in full.

#figure(
  results-table(
    columns: (4.4cm, 5.4cm, 1fr),
    align: (left, left, left),
    header: ([Claim], [Offline evidence here], [What an online experiment would add]),
    [Without an anchor the lists collapse], [Conclusive: a fact about the policy, measured without any model of the
      user], [Nothing],
    [An anchor weight is not comparable, a KL distance is], [Conclusive for this setup: a statement about the
      optimiser], [Nothing],
    [The predicted gain is impersonal], [Conclusive as a statement about the reward model's predictions],
      [Whether real responses to the fine-tuned lists depend on the user more than predicted],
    [The predicted gain on KION is not a gain in consumed share], [Suggestive: the logged share per recommended item
      is unchanged, but only for the items users reached on their own], [The consumed share of served
      recommendations, the decisive measurement],
    [Re-ranking is at least as good as RL], [Under the same reward model and the same logged checks],
      [Real responses; serving cost of scoring 100 candidates against one generation pass],
    [Retention rewards buy little], [Weak: return is a counterfactual outcome that the log shows for one policy],
      [Return rates under each policy],
    [Greedy construction suffices for a diversity reward], [Conclusive for the modelled reward], [Whether users
      value the diversity that is rewarded],
  ),
  caption: [What the offline evidence settles and what needs an online experiment.],
) <tab-offline>

*Practical implications.* Report the KL distance a fine-tuned policy ended at, not only the weight that was set.
Before fine-tuning against a feedback model, re-rank by the same model and by item means: if the frontier is the
same, the model's contribution is knowledge about items. Score lists for another user. Prefer a reward that an item
cannot earn on its own account, and expect it to be worth less than the raw signal suggested.

= Limitations <sec-limitations>

- *One seed per configuration.* The repeat of the KION session bounds hardware noise, not the variation between
  seeds; differences of a few thousandths of NDCG\@10, such as those between credit assignments, need the seed runs.
- *Everything is offline.* @tab-offline lists what this leaves open. The logged check sees only recommended items
  that users reached without the recommender's help.
- *The Yambda grids are narrower than KION's.* Sampled feedback, the list reward, and the retention and personal
  rewards were run at one anchor weight each, and the credit ablations at the two budgets only.
- *The base recommender on Yambda is below a replay heuristic,* and gains in relevance there are gains over this
  recommender.
- *The main run's weight is not a blind choice* (@sec-setup).
- *The explanation of why weights differ between estimators is a hypothesis* that no run isolates.
- *The personal part is the reward model's own belief* about how much of its prediction depends on the user.
- *Re-ranking and RL start from different lists,* ranked and generated; on KION this favours re-ranking by about 4%.
- *The distance in the list reward is measured in the base recommender's own embedding space,* which the greedy
  method and the evaluation share.
- *Baselines are untuned.*

= Conclusion

We fine-tuned a generative recommender against a learned reward model in the setting with the fewest protections.
The reward rose, and a model trained on other users confirmed it. The log did not: users consumed no more of what
was recommended, and less of it relative to what is usual for the items. Other users would have given the same
lists the same reward, and a table of item means reproduces the result. Making the reward neutral to the item
removes the artefact together with most of the gain. On the side of optimisation, the anchor weight turned out not
to be a quantity one can report or transfer, while the KL distance is; at an equal distance the choice of advantage
estimator and of credit assignment made no difference we could detect, and PPO's clipping hardly acted. At every
moderate operating point, re-ranking the pretrained recommender's candidates was at least as good as RL, including
for a reward that depends on the whole list.

What moved logged outcomes came from the log: a logged-relevance term inside the reward, and off-policy learning
from logged sessions. For a reward model to be worth optimising with RL, it has to know something about the user
that cannot be looked up per item, and the reward has to be built so that only that knowledge pays. In our data
that part was small. Whether it is small in general, and whether real users respond to the fine-tuned lists as the
log suggests, are questions for more seeds and for an online experiment.

#heading(numbering: none, level: 2)[Code and data availability]

The code, the configuration and every table of every session are in the repository
`github.com/daniilantonovv/RecSysMaxxing` (`results/`); the tables of this paper are typeset from those files by
`report/tables.py`.

#bibliography("../references.bib", style: "ieee", title: "References")

#pagebreak()
#counter(heading).update(0)
#set heading(numbering: "A.1", supplement: [Appendix])
#set figure(placement: none)

= Configurations and cost

@tab-config lists what differs between the datasets; everything else is in `results/<session>/experiment_config.json`.

#figure(
  results-table(
    columns: (auto, auto, auto),
    align: (left, left, left),
    header: ([], [KION], [Yambda-50M]),
    [Catalogue], [titles with at least 10 training events], [50,000 most frequent tracks],
    [Users], [at least 5 training events], [at least 50 training events],
    [Window, dropout], [48 events, 0.2], [128 events, 0.1],
    [Epochs: recommender / reward models], [at most 30 / 15, early stopping], [at most 14 / 8, early stopping],
    [Session gap, return window], [calendar day, 7 days], [30 minutes, 1 day],
    [Items of the history in a list], [blocked], [allowed],
    [Anchor weights], [0, 0.1, 0.2, 0.3, 0.5, 1], [0, 0.03, 0.05, 0.1, 0.3],
    [KL budgets], [0.1, 0.5], [0.2, 0.5],
    [Leave-one-out at weights], [0.1, 0.3], [0.03, 0.1],
    [Credit ablations], [weights 0.1, 0.3; both budgets], [both budgets],
    [Relative-share reward at weights], [0, 0.1, 0.3], [0.03, 0.1],
    [Logged relevance], [alone at 0; with the share at 0, 0.03, 0.1, 0.3], [alone at 0; with the share at 0, 0.1],
    [Sampled feedback, list reward at weights], [0.1, 0.3], [0.1],
    [Retention rewards], [session, return, share + return at 0.1, 0.3], [return at 0.1],
    [Personal rewards at weights], [0, 0.1, 0.3], [0.1],
  ),
  caption: [What the sessions ran; for Yambda, the two sessions together. Fine-tuning: 512 lists per iteration, 4
    gradient steps per batch, 2,000 iterations for the main run and 1,000 for every other run.],
) <tab-config>

#report-table("stage_minutes", [Wall-clock minutes by stage on one RTX PRO 6000 (96 GB). One fine-tuning run of
  1,000 iterations takes 6.6 minutes on KION and 18.5 on Yambda, with its evaluation on test and validation
  requests. The Yambda continuation reused the base models, the baselines and the runs of the first session.])
<tab-time>

#report-table("selection", [Selection of the anchor weight on validation requests: the highest predicted share
  among the runs within 5% of the base recommender's NDCG\@10.]) <tab-selection>

#report-table("feedback", [Fine-tuning with expected and with sampled simulated feedback, and the held-out
  predicted share of each policy under three assumptions about the response to earlier items of the list.])
<tab-feedback>

= Further rewards

#report-table("retention_rewards", [Retention rewards. "Continues" and "returns" are the held-out model's
  predictions for the generated lists.]) <tab-retention>

#report-table("relevance_rewards", [Logged relevance in the reward, alone and added to the share.], size: 8pt)
<tab-relevance>

#report-table("personal_rewards", [Rewards with a baseline. "Personal part" is the predicted share for the user
  minus that for another user.]) <tab-personal>

#report-table("list_reward", [A list-dependent reward: predicted share plus the mean distance between the items of
  the list in the base recommender's embedding space. "List reward" is the reward per item as the held-out model
  sees it. Greedy re-ranking is maximal marginal relevance over 100 candidates.]) <tab-list>

= Further figures

#figure(
  grid(
    columns: (1fr,),
    row-gutter: 6pt,
    image("figures/final_kion_rl_training_validation.png", width: 100%),
    image("figures/final_yambda_rl_training_validation.png", width: 100%),
  ),
  caption: [The main run on validation requests during fine-tuning: KION (top) and Yambda (bottom).],
) <fig-training>

#figure(
  grid(
    columns: (1fr, 1fr),
    column-gutter: 8pt,
    image("figures/final_kion_per_step_by_simulated_feedback.png", width: 100%),
    image("figures/final_yambda_per_step_by_simulated_feedback.png", width: 100%),
  ),
  caption: [Predicted share by position of the list for the base recommender under three assumptions about the
    response to the items before it: KION (left) and Yambda (right).],
) <fig-per-step>

#figure(
  grid(
    columns: (1fr, 1fr),
    column-gutter: 8pt,
    image("figures/final_kion_reward_model_calibration.png", width: 100%),
    image("figures/final_yambda_reward_model_calibration.png", width: 100%),
  ),
  caption: [Calibration of the two reward models on test events: KION (left) and Yambda (right).],
) <fig-calibration>

= Earlier iterations of the study <sec-history>

The pipeline was completed seven times before the sessions reported here (version 2 of this draft describes those
runs in full). They differ from the present sessions in the reward models, which had fewer heads, and in the
instruments available, and they agree with them where they overlap. The main run on KION was made three times with
the same base recommender and other reward models: NDCG\@10 changed by $-11.4%$, $-11.4%$ and $-11.1%$ and the
held-out predicted share by $+22.8%$, $+23.1%$ and $+23.1%$, against $-10.7%$ and $+21.5%$ here. On Yambda two
earlier runs gave $+1.1%$ and $+1.8%$ of predicted share, against $+1.7%$ here. The first two runs used REINFORCE
with a leave-one-out baseline on Yambda, where the policy held at $beta = 0.3$ and collapsed at $beta = 0.1$, while
the value-model runs held at $0.1$ and collapsed at $0.03$. The comparison was confounded by the learning rate and
the number of lists per history; @tab-estimators repeats it with those held equal and confirms that the estimator
alone moves the transition. Two findings of the earlier runs are not repeated here: a choice head trained to tell
logged items from popularity-distributed replacements did not work as a relevance signal inside the reward, and a
reward weighted by item duration moved the policy along the duration axis in the direction the normalisation
favoured.
