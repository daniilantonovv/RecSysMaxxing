// Paper draft, version 4.
// Build: python3 report/tables.py && python3 report/figures.py && typst compile --root report report/archive/paper_v4.typ
// Tables come from results/<session>/*.csv through report/tables.py, the SVG figures through report/figures.py; the
// numbers quoted in the prose are read from the same reports. Bibliography: report/references.bib.
// report/archive/paper_v4_cuts.md lists what the long version of 15 October (commit 839539d) had and this one leaves out.

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
    include "../tables/" + name + ".typ"
  },
  caption: caption,
)

// The answer to one of the five questions, closing its section.
#let answer(question, body) = block(
  width: 100%,
  inset: (x: 7pt, y: 5pt),
  radius: 2pt,
  stroke: (left: 1.6pt + luma(110)),
  fill: luma(246),
)[*Answer to #question.* #body]

// One box of the pipeline figure.
#let stage(title, body) = rect(width: 100%, height: 2.45cm, inset: 5pt, radius: 2pt, stroke: 0.6pt + luma(90))[
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
  #text(size: 9pt)[Draft, version 4 #h(0.6em) 18 October 2026]
]

#pad(x: 1cm)[
  #set text(size: 9.5pt)
  *Abstract.* A recommender that generates a list item by item can be fine-tuned with reinforcement learning (RL)
  towards feedback that next-item training does not see. We study the least protected form of this idea: a GPT-2
  recommender is fine-tuned with a policy gradient to raise the consumed share of its items as a second GPT-2, the
  reward model, predicts it, held near the pretrained recommender only by a KL anchor. On a video log (KION) and a
  music log (Yambda-50M), with three seeds for every central comparison, instruments that the optimised model cannot
  influence judge the result: a reward model fitted on other users, the logged response to recommended items that
  users went on to consume, and the same lists scored for another user. Relevance is traded for predicted reward
  along a frontier: at the operating point selected on validation requests KION loses 1.8% of NDCG\@10 for an 8%
  higher predicted share, further along it 11.5% for 24%. The policy learns which items everybody finishes rather
  than what each user wants. RL settings only choose a point on the frontier, the reward function moves it, and
  re-ranking by the same model, which computes the KL-anchored optimum in closed form, reaches more reward at equal
  relevance. Rewards for return and adoption buy little offline, and the largest such gain lies beyond any return
  rate in the log.
]

= Introduction

Generative sequential recommenders @kang2018sasrec @petrov2023gptrec @rajput2023tiger @zhai2024actions @deng2025onerec
are trained to continue logged sequences, and the log records what users did, not what served them: 36% of music
listens in Yambda stop before half of the track, and 52% of KION titles are abandoned before half of their length. A
recommender that generates a list item by item is a policy, so RL can move it towards what will be received well
@chen2018topk @ie2019slateq @petrov2024aligning @matveev2026session. Where the reward comes from decides how safe this
is. Petrov and Macdonald @petrov2024aligning keep relevance inside the reward. Matveev et al. @matveev2026session
learn from logged feedback, and use a learned feedback model to evaluate and to select recommendations at inference
time. Letting a learned model of user feedback _be_ the training reward is the option with the fewest protections: a
policy optimised against a learned model drifts to where the model is wrong @gao2022scaling @skalse2022hacking, in
recommendation towards popularity and filter bubbles @gao2023dorl @jiang2019degenerate.

We study this option in its barest form. A GPT-2 @radford2019gpt2 recommender is fine-tuned to raise the consumed
share that a second GPT-2 predicts, and a KL penalty towards the pretrained recommender, the _anchor_, is the only
term that asks for relevance. Since every reward gain in this setting is reported by a model, the study rests on
instruments the optimised model cannot influence, and on three seeds for every central comparison. It answers five
questions about RL fine-tuning of a next-item recommender; @tab-answers answers each in one sentence, and @sec-q1 to
@sec-q5 give the evidence.

#figure(
  results-table(
    columns: (3.3cm, 1fr),
    align: (left, left),
    header: ([Question], [Answer in one sentence]),
    [*Q1.* How does the model's behaviour change after RL fine-tuning?],
    [The predicted reward is maximised ($+24%$ on KION, confirmed by a held-out reward model), but the policy learns
      which titles everybody finishes rather than what each user wants: it moves impressions from long series to short
      films, 78% of its change is one shift per item, and users take to the recommended titles less than is usual for
      them, because the reward scores only items a user has started and RL extracts its per-item part.],
    [*Q2.* Is relevance preserved? Does the model recommend the same items to everyone?],
    [Relevance is traded along a frontier: at the weight selected on validation requests KION's NDCG\@10 falls by
      1.8% for an 8% higher predicted share, at the main run by $11.5% plus.minus 1.3$ for 24% (at every cut-off and
      most for users with short histories, while the most shown 1% of titles take 76% of impressions instead of 72%);
      Yambda keeps both, and without the anchor every user receives the same list.],
    [*Q3.* Can we get closer to long-term goals?],
    [Return, likes and adoption can be predicted and rewarded, but at a moderate distance their predictions rise by a
      few percent (adoption: $+19%$ at no loss of relevance, with a fifth less coverage); the largest gain, the return
      after the list on KION ($+18%$), lies beyond any return rate in the log, and only an online test can show
      whether users return more.],
    [*Q4.* What happens when generating several steps ahead?],
    [A generated list gains after a hit and loses after a miss (on Yambda the item at position 5 is a hit in 54% of
      lists after a hit at position 4, against 25% in a ranked list, and in 4% against 6% after a miss); the predicted
      reward of later steps depends on an assumption about responses nobody observed, which a reward model trained with
      hidden responses makes unbiased; crediting an item with the rest of the list helps no per-item reward.],
    [*Q5.* What is the trade-off between relevance and reward?],
    [RL settings choose a point on the frontier; only the reward function moves the frontier: within 1% of the base
      recommender's NDCG\@10 on KION, RL on the share gains 4% of predicted share, re-ranking by the same reward model,
      which computes the anchored optimum, 15%, RL with logged relevance in the reward 17%, and re-ranking by the
      share relative to the item's usual raises NDCG\@10 and a length-neutral logged response at once.],
  ),
  caption: [The five questions and their answers. Numbers are from KION unless Yambda is named; "$plus.minus$" is the
    standard deviation over three seeds.],
  placement: none,
) <tab-answers>

= Related Work

Next-item models @hidasi2015gru4rec @kang2018sasrec @sun2019bert4rec run autoregressively as generative recommenders
over items @petrov2023gptrec or semantic identifiers @rajput2023tiger, up to industrial scale
@zhai2024actions @deng2025onerec. RL enters recommendation through off-policy policy gradients @chen2018topk, slate
values @ie2019slateq and RL heads @xin2020sqn, and learned simulators replace the user @ie2019recsim @zhao2023kuaisim. A
policy optimised against a learned model exploits its errors @yu2020mopo @kidambi2020morel @gao2022scaling @skalse2022hacking, and an
ensemble of reward models does not stop this, because models trained alike share their errors
@eisenstein2023helping. Penalising divergence from a pretrained model is KL-control
@jaques2019wayoff @ziegler2019finetuning @ouyang2022instructgpt; Gao et al. @gao2022scaling compare best-of-$n$
sampling with RL along this divergence. Consumption time is confounded by item duration
@zhan2022d2q @zhao2023watchtime, retention has been optimised directly @zou2019longterm @cai2023retention, and maximal
marginal relevance @carbonell1998mmr greedily optimises a list-dependent reward.

Closest to this study, Chidambaram et al. @chidambaram2026robust find that a learned reward model for a generative
recommender predicts no better than item means, that PPO and DPO against it collapse ranking metrics, and that
exponential reward-weighted fine-tuning on logged rewards does better; they derive the KL-anchored optimum in closed
form and note that standardising rewards rescales its temperature. OneRec-V2 @zhou2025onerecv2 found its reward model
open to reward hacking and moved to user feedback, scoring watch time as a percentile within the user's own history
of videos of similar duration. Our setting differs in what the reward model knows: it predicts responses far better
than item means (AUC 0.755 against 0.656 on KION), yet RL extracts only its per-item part. We trace this before any
collapse, at controlled KL distances, on two domains with three seeds, with instruments that separate what the policy
learned about items from what it learned about users, and through generation several steps ahead.

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
      relevance by outcome, user and position #sym.dot.c concentration #sym.dot.c the reward model on real sequences.
    ]
  ],
  caption: [The pipeline. The reward model of step 3 is the only source of reward; the instruments in the dashed box
    never influence the policy.],
  placement: none,
) <fig-pipeline>

*Models.* A log is a set of events with a user, an item, a time and the _consumed share_ $rho in [0, 1]$ of the track
or title. The recommender $pi_0$ is a GPT-2 decoder @radford2019gpt2 over item tokens, trained with the full-softmax
next-item cross-entropy. A list of $K = 10$ is obtained by _ranking_ (the catalogue is scored once) or by _generation_
(each chosen item is appended to the history as if consumed) @petrov2023gptrec; fine-tuning changes the second mode.
The _reward model_ is a second GPT-2 whose token at position $t$ carries the item $i_t$ and the consumed share of the
previous item, so it scores a candidate before its outcome exists. Its heads, trained with binary cross-entropy,
predict the consumed share $hat(rho)$ (with $rho$ as a soft label); the _relative share_ $hat(q)$, the share as a
mid-rank quantile among the item's training-period shares, which averages one half for every item and so cannot be
earned by an item that is easy to finish; the probability that the session continues, $hat(c)$, and that the user
returns, $hat(u)$; and on Yambda the probabilities of a like and a dislike, and of _adoption_ $hat(alpha)$: a first
play followed by another play in a later session within 14 days. Since the log holds only items that users chose,
every head describes the response to an item _given_ that the user starts it. Two copies are trained on disjoint
halves of the users: the _training_ model gives the rewards, the _held-out_ model only judges. A third pair is trained
with _hidden responses_: in half of the training windows the responses to the last 1 to 10 events are replaced by an
"unknown" value, which is the situation of a generated list.

*Fine-tuning.* An episode generates one list for a training history; the reward model is the environment and is fed
its own expectation for the responses to earlier generated items. The policy gradient uses a learned value model with
generalised advantage estimates @schulman2015gae and four steps of PPO's clipped surrogate per batch
@schulman2017ppo, or a leave-one-out baseline over four lists per history @kool2019buy; advantages are standardised
in both cases. The loss adds the exact divergence from $pi_0$ at every position,
$ cal(L)(theta) = cal(L)_"PG" (theta) + beta dot EE_k ["KL"(pi_theta (dot | s_k) || pi_0 (dot | s_k))], $ <eq-loss>
and a _KL budget_ $delta$ adapts $beta$ after every iteration so that the policy settles at $delta$ nats per item.
Two ablations remove credit along the list: $gamma = 0$, and one-item episodes. For a reward $r(j | s)$ of item $j$
in state $s$, the anchored objective at one position has the closed-form maximiser
@jaques2019wayoff @chidambaram2026robust
$ pi^*(j | s) = (pi_0 (j | s) exp(r(j | s) \/ beta)) / Z(s). $ <eq-optimum>
Standardising the advantages makes the policy-gradient term scale-free but
leaves the KL term as it is, so RL pursues this optimum with an effective weight $beta sigma_A$, where $sigma_A$ is
the scale of the advantages before standardisation @chidambaram2026robust. $sigma_A$ depends on the estimator and
drifts during training: a weight cannot be compared between runs, the KL distance a run ends at can.

*Rewards.* All rewards are instances of
$ r_k = w_rho hat(rho)_k + w_q hat(q)_k + w_u hat(u)_k + w_L (hat(l)_k - hat(d)_k) + w_A hat(alpha)_k
  + w_R bb(1)[k = K] hat(u)_K + w_F (bb(1)[a_k in F]) / (log_2 (k + 1)). $ <eq-reward>
The default is $w_rho = 1$ alone. The _return after the list_ ($w_R$) is paid at the last item only, so it belongs to
the list as a whole. The _logged-relevance_ term of @petrov2024aligning ($w_F$) uses the user's real continuation $F$
of a training history and sums to DCG\@K. A _personal_ reward subtracts what the model predicts for the same list
shown to another user, and a _diversity_ reward adds the mean distance between the items of the list.

*Alternatives to on-policy RL.* _Liked-only fine-tuning_ continues next-item training on targets with $rho >= 0.5$.
_Off-policy REINFORCE_ @matveev2026session weights logged items by the logged reward that followed them, with
importance weights capped at $kappa = 0.5$. _Re-ranking_ keeps the $K$ of the 100 best candidates of $pi_0$ with the
highest $r(j | h) + w log pi_0 (j | h)$, which orders items exactly as $pi^*$ of @eq-optimum with $beta = w$: it
computes the top of the anchored optimum for the real history, without training.

*Instruments.* Let $F$ be the next 20 items the user really consumed. A gain the _held-out reward model_ does not
confirm is exploitation of the training model. The _logged response_ needs no model: the _logged share_
$1/K sum_k bb(1)[a_k in F] rho(a_k)$ counts hits as much as responses, and the _logged surplus_
$1/K sum_k bb(1)[a_k in F] (q(a_k) - 1/2)$ is positive only if users took to the recommended items more than is usual
for those items. Scoring a user's lists for _another user_ gives the share of a gain that depends on who receives the
list, its _personal share_. _Relevance_ is NDCG, Recall and MAP at $K = 1, 5, 10$ against $F$, and NDCG\@10 or
Recall\@10 by outcome (liked, $rho >= 0.5$, or skipped; above or below the item's usual share), by the length of the
history, and by position after a hit or a miss. _Concentration_ is catalogue coverage, the Gini coefficient of
impressions, the share of impressions of the 1% most shown items, and the overlap of two users' lists. The _per-item
shift_ is the $pi_0$-weighted share of the variance of $log pi_theta (j | h) - log pi_0 (j | h)$ over histories and
items that the item's mean explains: 1 for one preference per item applied to every user, and a lower bound, since the
per-history normaliser is left in. _Simulation on real sequences_ runs the reward model on the user's real next five
items, with the earlier responses as logged or as a simulation has to assume them.

= Experimental Setup <sec-setup>

*Data.* KION @petrov2022kion is a log of an online cinema with one row per user and title, dated by its last view;
we keep users with at least 5 and titles with at least 10 training events (256k users, 8k titles, 3.5M training
events, a median of 10 per user), and its English item table gives each title's name and type. Yambda-50M
@ploshkin2025yambda is a listening log of Yandex Music; we keep users with at least 50 training events and the 50k
most played tracks (8.7k users, 35M training events, a median of 2,638 per user), with likes (0.77% of events) and
dislikes (0.16%); 24% of first plays are adopted. A session continues if the next event follows within 30 minutes on
Yambda and on the same calendar day on KION; the user returns if a new session starts within a day on Yambda and
within a week on KION. The last 7 days of each log are the test period and the 7 days before them the validation
period; the catalogue and every item statistic come from the training period, so no model sees an event after its
request. The consumed share is bimodal, and on KION it depends on length: films are watched to 0.5 on average and
the longest tenth of the catalogue, series, to 0.22, since the share is taken of the whole series.

*Training and evaluation.* All transformers have 4 layers of width 256 and windows of 48 (KION) or 128 (Yambda)
events. Fine-tuning samples 512 lists per iteration ($gamma = 0.99$, learning rate $10^(-5)$) for histories cut at a
random moment of the training period of a random user with at least 8 training events; the _main run_ has
$beta = 0.3$ and 2,000 iterations, every other run 1,000. Items of the history are blocked on KION and allowed on
Yambda, where repeats are normal. Each test user makes one request at the start of the test week: the last 38 (KION)
or 118 (Yambda) events are the history, the same window that every baseline reads, and lists of ten are generated
greedily unless ranked lists are named. The KL distance of a run is the exact divergence over the catalogue on its
own sampled training lists, averaged over the last 50 iterations. Operating points are selected on validation
requests only: the highest held-out predicted share within 5% of the base recommender's NDCG\@10 selects $beta = 1$
on KION and $beta = 0.05$ on Yambda. The main run's $beta = 0.3$ was carried over from earlier iterations of the study
and shows a clearly visible move along the frontier, not an optimum.

*Sessions and seeds.* Each dataset ran a _final_ session with the full grid (6.1 hours on KION, 12.3 on Yambda, on
one 96 GB GPU), a _review_ session with the long-term rewards, the reward models with hidden responses and the
analyses by position, user and item, and two sessions with seeds 1 and 2 of the core: base models, main run, anchor
sweep, a KL budget, credit and estimator comparisons, alternatives to RL and logged relevance. "$plus.minus$" is the
standard deviation over the three seeds (@tab-seeds); other numbers are from seed 0. The review session retrains every
model with the same seed, so a run present in both sessions differs in the third decimal (the main run's coverage is
0.452 and 0.449); every table names its session. Every conclusion below holds in each seed.

= Results <sec-results>

*Base models.* On KION the recommender is level with GRU4Rec, 13% above SASRec and 5% below BERT4Rec for ranked
lists; on Yambda it is the best trained model, but below a replay of the user's own history, which reads the same 118
events (NDCG\@10 0.117 against 0.187, mostly on repeated tracks), so relevance there is measured against this
recommender (@tab-baselines). The reward model predicts whether a held-out event is consumed at least half with AUC
0.755 on KION and 0.868 on Yambda, against 0.656 and 0.607 for the item's mean share; the like and dislike heads reach
0.74 and 0.73. The two copies are calibrated on their own users and differ in level on the same lists (0.417 and
0.455 for the base recommender on KION), so we compare their relative changes.

== Q1. How does the model's behaviour change after RL fine-tuning? <sec-q1>

*Is the reward maximised?* As far as the reward models can tell, yes: on KION the main run raises the consumed share
predicted by the training reward model by 20.3% and by the held-out one by 21.5% ($23.9% plus.minus 2.1$; @tab-main).
The log does not show users responding better. Against what is usual for each title, users took to the recommended
titles less: the logged surplus falls from 0.0012 to 0.0005, by 0.0007 in every seed, and by 15% at the
validation-selected $beta = 1$. The logged share per recommended item, which mostly counts hits, rises by
$1.1% plus.minus 0.4$: users reached fewer recommended titles (hits $-8.3%$) and finished those more often ($+9.8%$),
because they are titles everybody finishes.

*How the recommendations differ.* Films take 66% of the impressions of the base recommender's generated lists and
80% of the main run's (89% at $beta = 0.1$, and all of them without an anchor). The titles that gained most are films
of 1.4 to 2.2 hours usually watched to 0.54--0.69; four of the five that lost most are series of 9.5 to 22 hours
whose usual share is 0.08 to 0.29 (@tab-titles). Popularity is not what the policy follows: one of the largest gains
is a film that 953 users watched in the training period.

#report-table("titles", [KION: the titles whose number of lists among the 59,997 test requests grew and shrank most
  under the main run (review session). Titles as given in the English release of the KION item table, which
  translates the Russian titles; "usual share": the title's mean consumed share in the training period.]) <tab-titles>

*One shift per item.* On KION the item's mean change explains 78% of the change of the policy's log-probabilities,
and an item's mean shift correlates 0.90 with its usual consumed share, 0.12 with its log popularity and $-0.36$ with
its duration (2,048 test histories, review session). The policy learned a ranking of titles by how often they are finished and applies it to
every user. On Yambda the item means explain 44%.

*Why.* There are two causes. The first is the reward: $hat(rho)$ is the share a user consumes given that they start
the item, so it cannot penalise an item the user would never start, and on KION the share of a series is taken over
all its episodes, so an item that everybody finishes earns a high reward whoever receives it. The second is the
optimiser: the reward model knows something about users ($8.9% plus.minus 0.3$ of the gain of re-ranking by it is
personal), but a policy gradient, averaging a noisy signal over histories, accumulates first the direction that is
the same for all of them. The same lists scored for another user earn the same gain (personal share
$0.2% plus.minus 0.5%$), and re-ranking by the item's mean share in the training log, with no reward model, reaches
the main run's point (predicted share 0.549 at NDCG\@10 0.109, against 0.553 at 0.105). On Yambda the reward model
knows more about the user than RL uses ($55% plus.minus 9$ of re-ranking's gain is personal, $16% plus.minus 16$ of
RL's), and the policy changes little (predicted share $+1.67% plus.minus 0.01$, NDCG\@10 $+0.8% plus.minus 0.7$).

#answer[Q1][The reward is maximised as far as the reward models can tell, but the policy learns which items
  everybody finishes, not what each user wants: the reward scores only items users have started, and RL extracts the
  per-item part of a reward model that knows more; measured against what is usual for each title, users took to the
  recommendations less.]

== Q2. Is relevance preserved? Does the model recommend the same items to everyone? <sec-q2>

*A frontier, not a verdict.* How much relevance is kept is a choice of operating point (@fig-frontier). At the weight
selected on validation requests, $beta = 1$ on KION, NDCG\@10 falls by 1.8% for an 8.3% higher predicted share and a
3.3% higher logged share (three seeds); at the main run, $beta = 0.3$, it falls by $11.5% plus.minus 1.3$ for 24%. On
Yambda the validation rule selects $beta = 0.05$ (NDCG\@10 $-4.9%$, predicted share $+5.0%$), one grid step from the
weight that collapses the policy in every seed; the main run moves little.

*Relevance by Recall, NDCG and MAP.* At the main run on KION every metric falls at every cut-off, by 8 to 14%
(@tab-ranking). The first item is chosen from the real history alone, so the loss at $K = 1$ ($-7.9%$) is the
policy's own preference; from the second position on, the generated context adds to it (@sec-q4). By outcome, NDCG\@10
on liked titles is kept ($+1.3%$), Recall\@10 on skipped titles falls by 18.2% and NDCG\@10 on titles watched above
their usual share by 9.4%. On Yambda every metric stays within 1.3% of the base recommender. On KION the users with the
least history lose most: $-13.5%$ of NDCG\@10 for 5 to 8 earlier events against $-7.4%$ for 25 or more
(@tab-history); on Yambda no group loses.

#report-table("ranking_metrics", [Ranking metrics of the main run against the base recommender (generated lists,
  final session). At $K = 1$ NDCG and MAP coincide.]) <tab-ranking>

*The same items for everyone?* Not at the main run, but closer to it (@tab-concentration). On KION the base
recommender's generated lists already give the 1% most shown titles 72% of the impressions (62% for ranked lists); the
main run raises this to 76% and the Gini coefficient of impressions from 0.963 to 0.969. With a weaker anchor the lists
converge: 92% at $beta = 0.1$, and without an anchor nearly every user receives the same list (pair overlap 0.98 on
KION, 1.00 on Yambda), which the held-out model still scores 51% above the base recommender. On Yambda the share moves
from 18% to 20%.

*What keeps relevance.* Only the anchor asks for relevance, and its weight does not describe how much it keeps. At
$beta = 0.3$ on KION the leave-one-out estimator ends 6.7 times further from $pi_0$ than the value model (KL
$0.543 plus.minus 0.010$ against $0.081 plus.minus 0.004$), while under a KL budget of 0.1 the two end at the same
distance and relevance (NDCG\@10 $0.1037 plus.minus 0.0021$ and $0.1013 plus.minus 0.0026$) with weights 2.9 times
apart. The effective weight $beta sigma_A$ predicts this if the value model's advantages are about three times larger
before standardisation: on the value model's curve of KL against $beta$, the leave-one-out run at $beta = 0.3$ sits
at $beta approx 0.09$ (a consistency check; $sigma_A$ was not logged). The KL distance describes far more than the
weight but does not pin relevance down: at an equal budget, crediting each item with its own reward is 5.6% more
relevant on KION and 12% on Yambda, at a lower predicted share. Every run shares the frontier of relevance against
reward (@sec-q5). On Yambda $beta = 0.1$ keeps relevance and $beta = 0.03$ collapses the policy in every seed
(NDCG\@10 $0.006 plus.minus 0.004$; @tab-sweep).

#figure(
  image("../figures/frontier_by_distance.svg", width: 88%),
  caption: [Every fine-tuning run of the default reward against the KL distance it ended at from the base
    recommender (logarithmic axis; the base recommender at the left edge; final session).],
) <fig-frontier>

#answer[Q2][Relevance is traded along a frontier: at the validation-selected weight KION loses 1.8% of NDCG\@10, at
  the main run 11.5%, at every cut-off and most for users with short histories, while impressions concentrate further;
  Yambda keeps both. The anchor keeps relevance, the KL distance describes how much far better than the weight, and
  without the anchor every user receives the same list.]

== Q3. Can we get closer to long-term goals? <sec-q3>

*Which signals.* Four signals in the logs describe what follows an item: the session continues, the user returns,
the user likes or dislikes the item, and the user adopts it (the last two on Yambda only). A reward helps a policy only
if its signal differs between the items a user could be shown. The predicted return varies between one user's
candidates four times (KION) and nine times (Yambda) less than between users, and the probability of a like by only
0.003 between a user's candidates. Adoption is the exception: it varies between candidates half as much as between
users (0.077 against 0.165), and it is the only signal that leans towards the items users really consumed next
(AUC 0.56).

*Rewards for return and continued interaction.* We rewarded each signal per item, and the return also for the list
as a whole, after its last item. Rewarding the return of each item raises it by 3 to 7% on KION for 8 to 23% of
NDCG\@10 and does not move it on Yambda (@tab-retention). The _return after the list_ moves the KION policy further
than any per-item reward (KL 0.51 at $beta = 0.3$): the predicted return rises from 0.73 to 0.86 ($+18%$), as the
held-out model also predicts, while NDCG\@10 falls by 69% and the logged share by 64% (@tab-long-term), and the mean
duration of a recommended title falls from 5.7 to 2.2 hours. KION's format, one row per user and title dated by its
last view, does not explain it: in every decile of the duration of an event's own or next title, the return rate is
between 0.68 and 0.77. No group comes back at 0.86: both reward models extrapolate alike, so the held-out model cannot
catch it. With discount 0 the policy hardly moves
(KL 0.006): this is the one reward for which credit along the list matters. On Yambda it does not move.

#report-table("long_term_rewards", [Long-term rewards (review session). "Δ": change of NDCG\@10 in percent;
  "Return": the held-out model's predicted return after the last item; P(like), P(adopt): its predicted probabilities
  (Yambda only); "Cover.": catalogue coverage; "Hours": mean duration of a recommended KION title.]) <tab-long-term>

*Likes and adoption (Yambda).* The like reward at $beta = 0.3$ raises the predicted probability of a like from 1.08%
to 1.17% for 2% of NDCG\@10; at $beta = 0.1$ likes rise by 49% and dislikes by 53%: the policy finds tracks that
provoke reactions, not tracks that are liked more. Adoption is the one long-term reward that moved without a loss of
relevance: at $beta = 0.3$ its predicted probability rises from 0.233 to 0.277 ($+19%$), NDCG\@10 by 1.5% and the
logged share by 2.5%, while coverage falls from 0.52 to 0.41.

*Offline or online.* Offline we can check whether a signal is predictable, whether it varies between the items a
policy chooses from, whether a held-out model agrees, and what it costs in relevance. Whether users return, continue
or adopt more under a new policy is counterfactual: the log shows these outcomes only under the policy that produced
it, adoption needs 14 days after a first play, which a test week does not have, and a prediction can leave the log's
range, as the return after the list did. This needs the online experiment of @sec-online.

#answer[Q3][Rewards for return, likes and adoption can be learned and optimised, but at a moderate distance their
  predictions rise by a few percent; adoption is the most promising signal, the largest gain lies outside the log,
  and only an online test can show whether users return more.]

== Q4. What happens when generating several steps ahead? <sec-q4>

A generated list appends each chosen item to the history as if the user had consumed it, so it plans several steps
ahead under an assumption that is false for most items: at most 17% of the items at any position are hits.

#figure(
  image("../figures/position_hits.svg", width: 88%),
  caption: [Hit rate by position of the list, for all lists, after a miss and after a hit at the position before:
    ranked and generated lists of the base recommender and generated lists of the main run (review session).],
) <fig-positions>

*Effect on the following recommendations.* After a miss, a generated list continues from a context that did not
happen and hits less often than a ranked list: 0.0273 against 0.0312 at position 5 on KION, 0.040 against 0.063 on
Yambda (@fig-positions). After a hit it hits more often: 0.112 against 0.093 at position 2 on KION, and on Yambda,
where listening continues within albums and artists, 0.54 against 0.25 at position 5. Misses dominate on KION, where
generation costs 4% of NDCG\@10; on Yambda generation is ahead at positions 2 to 7. The fine-tuned policy keeps the
pattern.

*Effect on the reward estimate.* On the users' real next five items, with logged responses fed back, the AUC of the
predicted share rises along the list (0.757 to 0.782 on KION, 0.640 to 0.819 on Yambda), while under any simulated
response it stays near its first-step level (@fig-assumptions). The first step itself is the hardest to predict on
Yambda (AUC 0.64), since test requests sit at the start of a week, after a break. Expected responses underestimate the
next share on Yambda by 0.06 to 0.09, sampled ones overestimate slightly (at most $+0.035$), and assuming that earlier
items were finished overestimates by 0.06 to 0.18. The reward model trained with hidden responses and fed "unknown"
is nearly unbiased ($+0.001$ to $+0.013$) and reads observed responses as well as the standard model (AUC 0.755
against 0.758 on KION). A policy fine-tuned against it lands on the same frontier (at $beta = 0.3$ on KION NDCG\@10
0.106 against 0.104 at KL 0.078 against 0.080), as does one fine-tuned with sampled responses: the assumption sets the
level of a multi-step predicted reward, not which policy wins.

#figure(
  image("../figures/feedback_assumptions.svg", width: 60%),
  caption: [The reward model on the users' real next five items: bias and AUC of the predicted share at each step,
    with the earlier responses as logged, expected, sampled, finished, or unknown to the model trained with hidden
    responses (review session).],
) <fig-assumptions>

*Does RL use the steps?* At an equal KL budget, crediting each item with its own reward only is at least as relevant
as crediting it with the rest of the list: NDCG\@10 $0.1070 plus.minus 0.0027$ against $0.1013 plus.minus 0.0026$ on
KION (one-item episodes $0.1043 plus.minus 0.0024$), at predicted shares 1.4 to 1.6 points lower, and
$0.1076 plus.minus 0.0014$ against $0.0960 plus.minus 0.0008$ on Yambda. The myopic policies lie on the same
frontier of relevance against reward. Only the return after the list, paid at the end, needed full credit (@sec-q3).

#answer[Q4][Generation gains after a hit and loses after a miss; simulated responses bias the reward of later steps
  unless the reward model was trained to read unknown ones; and credit along the list helps no per-item reward.]

== Q5. What is the trade-off between relevance and reward? <sec-q5>

*How RL settings and the reward shape it.* Every RL setting we varied, the anchor weight, the KL budget, the
estimator, the credit assignment and the simulated responses, moves the policy along one curve (@fig-frontier,
@sec-q4): *RL settings choose a point on the frontier; only the reward function moves the frontier.* @eq-optimum says
why for per-item rewards: the settings change how far the policy moves towards $pi^*$, not where $pi^*$ lies, and only
$r$ does.

*Can the reward rise without a noticeable loss of accuracy?* @tab-equal-relevance reads each method at equal
relevance, within 1% or 5% of the base recommender's NDCG\@10; between two runs of a method the curve is the straight
line that a mixture of the two policies achieves. On KION, RL on the share gains 4.2% and 14.1% of predicted share and
1.8% and 3.7% of logged share. Ten times logged relevance in the reward moves the frontier: 16.9% and 23.1% of
predicted share. Re-ranking by the same reward model reaches 14.9% and 20.7% with the largest logged gains (15.6%), of
which its start from the ranked list contributes 11.7%. On Yambda the differences at 1% are within the noise between
seeds, and the methods that learn from the log lead in logged share through repeats.

#report-table("equal_relevance", [Methods at equal relevance (final session): the largest change of the predicted
  share (held-out model) reached without losing more than 1% or 5% of the base recommender's NDCG\@10, and the change
  of the logged share at that point, in percent. "–": the method does not reach that relevance.])
<tab-equal-relevance>

*Does it take RL?* Re-ranking with weight $w$ computes the top of the anchored optimum for $beta = w$. At the main
run's predicted share it is more relevant (NDCG\@10 $0.1119 plus.minus 0.0013$ against $0.1041 plus.minus 0.0020$),
has a higher logged share ($0.0201 plus.minus 0.0001$ against $0.0176 plus.minus 0.0002$), and
$8.9% plus.minus 0.3$ of its gain is personal against 0.2% for RL (@tab-alternatives). Most of the relevance gap is a
head start: ranked lists are 4% more relevant than generated ones on KION, and from its own start each method pays
about the same relevance for the same predicted share. Re-ranking also beat RL on the diversity reward, where a greedy
list is not optimal in principle (@tab-list).

*Rewards that move the frontier.* On KION the share plus ten times logged relevance at $beta = 0.3$ improves on the
base recommender's generated lists in relevance (NDCG\@10 $0.1202 plus.minus 0.0007$ against
$0.1177 plus.minus 0.0008$), predicted share ($+10%$) and logged share ($+7%$) at once, though not on its ranked
lists ($0.1221 plus.minus 0.0004$). The relative share moves the RL frontier sideways: the drift towards films
disappears and relevance is kept, but almost nothing is gained (@tab-rewards). Re-ranking by the relative share gives
the one improvement of a length-neutral logged outcome in the study. Against the ranked lists it starts from, at
$w = 0.3$ NDCG\@10 rises from $0.1221 plus.minus 0.0004$ to $0.1234 plus.minus 0.0006$ and the logged surplus by 15%;
at $w = 0.1$ the surplus rises by 38% for 0.5% of NDCG\@10; both are ahead in every seed (@tab-seeds). The personal
reward cannot collapse onto one list even without an anchor (pair overlap 0.33 against 0.98 on KION) but raises the
personal part only slightly (@tab-personal). On Yambda logged relevance alone raises NDCG\@10 by 23% by replaying
histories: relevance on new tracks falls from 0.0260 to 0.0198, and the share of recommendations taken from the
history rises from 51% to 62%.

#answer[Q5][RL settings choose a point on the frontier, and only the reward function moves it. Within 1% of the base
  relevance, re-ranking by the same model, which computes the anchored optimum, gains more than RL; logged relevance
  is the reward term that raises logged outcomes; and re-ranking by the relative share improves relevance and a
  length-neutral logged response at once.]

= Discussion <sec-discussion>

A second reward model, the usual defence against reward hacking, catches the exploitation specific to one set of
weights (0.84 against 0.69 without an anchor) and misses what any model of this kind invites, as Eisenstein et al.
@eisenstein2023helping found for ensembles: both of our models learned, correctly, that some titles are finished more
often than others, and both extrapolate the return after the list past any rate in the log. Scoring the same lists for
another user, splitting the policy's change into a part per item, and counting the logged response to recommended
items expose this cheaply. For a reward that sums per-item predictions, the anchored optimum is a re-ranking of
$pi_0$ (@eq-optimum), and a policy gradient is an expensive way to approximate it; the one time sequential credit
mattered, it chased a prediction outside the log. In practice: report the KL distance a run ended at, not its weight;
re-rank by the same model and by item means before fine-tuning; score lists for another user; and simulate generated
items with a reward model trained to read unknown responses.

== An online experiment <sec-online>

Offline evidence settles the collapse, the role of the anchor and the impersonality of the predicted gain. Whether
users consume more of the fine-tuned lists, return more often or adopt more needs an experiment that serves them. Its
arms are set by KL distance, since near Yambda's cliff a fixed weight does not fix a distance.
- _Arms:_ (A) the base recommender; (B) RL held at the KL distance of the validation-selected run (0.015 nats per item
  on KION, 0.29 on Yambda); (C) re-ranking by the reward model at the predicted share of B; (D) re-ranking by the
  relative share, $w = 0.3$; (E) on the music service, the adoption reward at its KL distance of 0.14.
- _Primary metrics, per user, over four weeks of hashed assignment_ (one week for exposure to settle, fourteen days
  of adoption window, one more for returns): the consumed share of served recommendations and its surplus over the
  items' usual, the return rate, and on Yambda the share of first plays of recommended tracks played again within
  14 days.
- _Guardrails:_ consumption per user, likes and dislikes, coverage and Gini of impressions, the share of new items,
  and every metric for the quartile of users with the shortest histories.
- _Analysis and size:_ per-user differences with a CUPED adjustment on the four weeks before, bootstrap intervals,
  history quartiles as pre-registered segments; a 1% relative change in a binary outcome near 0.7 needs about
  $2 (1.96 + 0.84)^2 dot 0.7 dot 0.3 \/ 0.007^2 approx 67,000$ users per arm ($alpha = 0.05$, power 0.8).

= Limitations, Future Work and Conclusion <sec-limitations>

*Limitations.* All results are offline, and the logged check sees only recommended items that users reached on their
own. Three seeds cover the core of the study; the long-term rewards and the analyses by position, user and item have
one, and adoption could not be verified on a test week. The explanation through $beta sigma_A$ is a consistency check,
since $sigma_A$ was not logged. The models are small (4 layers of width 256), so a larger reward model may know more
about users; Yambda's recommender and its replay baseline read 118 events of a median 2,638; the Yambda grids are
narrower than KION's, and the baselines are untuned. Re-ranking and RL start from different lists, the equal-relevance
curves interpolate between runs, and the personal part is the reward model's own belief.

*Future work.* The online experiment of @sec-online is the decisive next step. Offline, three directions follow. Rewards
that no item can earn on its own: the relative share and its per-user counterpart from OneRec-V2
@zhou2025onerecv2, with re-ranking by them, the one method that improved a length-neutral logged outcome. Learning
from logged rewards, where the logged gains of this study came from: exponential reward-weighted fine-tuning
@chidambaram2026robust is the smooth form of our liked-only baseline. And measurement: logging $sigma_A$ and the
personal share during training, splitting the reward's variance into item and user parts to predict the per-item shift
before running RL, a larger reward model, a longer Yambda context, and a test period long enough to observe adoption.

*Conclusion.* We fine-tuned a generative recommender against a learned reward model in the setting with the fewest
protections. The reward rose and a model trained on other users confirmed it, but the policy had learned which items
everybody finishes, users took to the recommendations less than usual, and relevance was traded along a frontier that
RL settings could only move along. Long-term rewards bought little, and the largest such gain lay outside the range
of the log. What improved logged outcomes came from the log or from rewards no item can earn on its own: logged relevance
in the reward, off-policy learning from logged sessions, and re-ranking by the relative share. For a reward model to
be worth optimising with RL, it has to know something about the user that cannot be looked up per item: on KION that
part was small for every method, on Yambda the reward model held more of it than RL used. Code, configurations and
every table are in `github.com/daniilantonovv/RecSysMaxxing`.

#heading(numbering: none)[References]
#{
  set text(size: 7.5pt)
  set par(justify: false, leading: 0.5em)
  columns(2, gutter: 14pt, bibliography("../references.bib", style: "ieee", title: none))
}

#pagebreak()
#counter(heading).update(0)
#set heading(numbering: "A", supplement: [Appendix])
#set figure(placement: none)

= Supporting tables

#report-table("seeds", [The central comparisons over three seeds (0, 1, 2): mean $plus.minus$ standard deviation.
  A seed fixes the initial weights of every model and the sampling of fine-tuning. "Budget" is a KL budget of 0.1
  nats on KION and 0.5 on Yambda; "–": not run with three seeds on that dataset.], size: 7.2pt) <tab-seeds>

#report-table("history_length", [NDCG\@10 and the change of the logged share per recommended item by the user's
  number of events before the request, in quartiles (review session, main run).], size: 7.2pt) <tab-history>

#report-table("concentration", [Concentration of impressions: catalogue coverage, Gini coefficient of the impressions
  per item, and the share of impressions taken by the 1% most shown items (review session).], size: 7.2pt) <tab-concentration>

#report-table("main_run", [The main run ($beta = 0.3$) against the base recommender, final session, seed 0. Changes
  are relative, in percent, with a 95% paired-bootstrap interval over users where the quantity is defined per user.], size: 7.2pt) <tab-main>

#report-table("anchor_sweep", [Anchor sweep with the value-model estimator, 1,000 iterations per run (final
  session). KL is in nats per generated item; "Training model" is the share predicted by the model that produced the
  rewards.], size: 7.2pt) <tab-sweep>

#report-table("baselines", [Baselines on generated or ranked lists of ten (final session); every baseline reads the
  recommender's window of history. On KION the history's items are blocked, so replay of the history reduces to
  popularity and the two rows coincide. The last two rows are the recommender that is fine-tuned.], size: 7.2pt) <tab-baselines>

#report-table("alternatives", [Ways of raising the reward without on-policy RL (final session). Re-ranking starts
  from the ranked lists of the base recommender, every other method from its generated lists.], size: 7.2pt) <tab-alternatives>

#report-table("list_reward", [A reward for the list as a whole: predicted share plus the mean distance between the
  items of the list in the base recommender's embedding space (final session). Greedy re-ranking is maximal marginal
  relevance over 100 candidates.], size: 7.2pt) <tab-list>

#report-table("rewards", [Rewards that an item cannot earn on its own account, and logged relevance (final
  session). "Rel. share" is the held-out model's predicted share relative to the item's usual; 0.5 is "as usual".], size: 7.2pt) <tab-rewards>

#report-table("retention_rewards", [Rewards for continuing the session and for returning, per item (final session).
  "Continues" and "returns" are the held-out model's predictions; on KION a session is a calendar day.], size: 7.2pt) <tab-retention>

#report-table("personal_rewards", [Rewards with a baseline (final session): the share minus that predicted for the
  same list shown to another user, or minus the item's mean share. "Personal part" is the predicted share for the user
  minus that for another user.], size: 7.2pt) <tab-personal>
