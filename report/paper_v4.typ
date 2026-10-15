// Paper draft, version 4.
// Build: python3 report/tables.py && python3 report/figures.py && typst compile report/paper_v4.typ
// Tables come from results/<session>/*.csv through report/tables.py, the SVG figures through report/figures.py; the
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
    include "tables/" + name + ".typ"
  },
  caption: caption,
)

// The answer to one of the five questions, closing its section.
#let answer(question, body) = block(
  width: 100%,
  inset: (x: 8pt, y: 6pt),
  radius: 2pt,
  stroke: (left: 1.6pt + luma(110)),
  fill: luma(246),
)[*Answer to #question.* #body]

// One box of the pipeline figure.
#let stage(title, body) = rect(width: 100%, height: 3.3cm, inset: 6pt, radius: 2pt, stroke: 0.6pt + luma(90))[
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
  #text(size: 9pt)[Draft, version 4 #h(0.6em) 15 October 2026]
]

#v(0.6em)

#pad(x: 1.1cm)[
  #align(center, text(weight: "bold")[Abstract])
  #set text(size: 9.5pt)
  A recommender that generates a list item by item can be fine-tuned with reinforcement learning (RL) towards feedback
  that next-item training does not see. We study the least protected form of this idea, in which a learned model of
  user feedback is the only source of reward: a GPT-2 recommender is fine-tuned with a policy gradient to raise the
  consumed share of its items as a second GPT-2 predicts it, held near the pretrained recommender by a KL anchor, on a
  video log (KION) and a music log (Yambda-50M), with three seeds for every central comparison. Instruments that the
  optimised model cannot influence judge the result: a reward model fitted on other users, the logged response to the
  recommended items that users went on to consume, the same lists scored for another user, and relevance split by
  outcome, by user and by position in the list. We answer five questions. _Behaviour:_ the policy learns which items
  everybody finishes, not what each user wants; on KION a predicted gain of 24% is almost entirely impersonal, and the
  logged share per recommended item rises by 1%. _Relevance and variety:_ on KION NDCG\@10 falls by 11.5%, most for
  users with short histories, and impressions concentrate further on the most shown titles; on Yambda both are kept.
  _Long-term goals:_ rewards for return, likes and adoption raise their predictions by a few percent at a moderate
  distance from the pretrained model, and the one large gain, the return after the list, comes from drifting to short
  titles. _Several steps ahead:_ a generated list gains after a hit and loses after a miss, and the reward of later
  steps depends on an assumption about responses nobody observed, which a reward model trained with hidden responses
  makes unbiased. _Trade-off:_ RL settings choose a point on one frontier, and only the reward function moves it;
  re-ranking by the same reward model reaches more reward at equal relevance. All results are offline.

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
the only term that does. The study answers five questions, which organise its results:

#block(inset: (left: 0.8em))[
  #set enum(numbering: n => strong[Q#n], indent: 0pt, body-indent: 0.7em)
  + How does behaviour change, and why? (@sec-q1)
  + Is relevance kept? Does everyone get the same items? (@sec-q2)
  + Long-term goals: return and continued interaction. (@sec-q3)
  + Generating several steps ahead. (@sec-q4)
  + Trade-off between relevance and reward. (@sec-q5)
]

@tab-answers answers each question in one sentence; the sections named above give the evidence.

#figure(
  results-table(
    columns: (3.3cm, 1fr),
    align: (left, left),
    inset-x: 4pt,
    header: ([Question], [Answer in one sentence]),
    [*Q1.* How does behaviour change, and why?],
    [The policy learns which titles everybody finishes rather than what each user wants: on KION it moves impressions
      from long series to short films, 78% of its change is one shift per item, and nearly all of the 24% predicted
      gain would be the same for any other user, because the reward model's predictions differ between items far more
      than between users' reactions to one item.],
    [*Q2.* Is relevance kept? Does everyone get the same items?],
    [On KION relevance is not fully kept (NDCG\@10 $-11.5% plus.minus 1.3$, most for users with short histories) and
      impressions concentrate further (the most shown 1% of titles take 76% of them instead of 72%), while on Yambda
      both are kept; in both, the KL distance from the pretrained recommender, not the anchor weight, decides how much
      is kept, and without the anchor every user gets the same list.],
    [*Q3.* Long-term goals: return and continued interaction.],
    [Return, likes and adoption can be predicted and rewarded, but at a moderate distance their predictions rise by a
      few percent (adoption: $+19%$ with no loss of relevance and a fifth less coverage), and the one large gain, the
      return after the list on KION ($+18%$), comes from drifting to short titles at a 69% loss of NDCG\@10; none of
      it can be confirmed offline.],
    [*Q4.* Generating several steps ahead.],
    [A generated list gains after a hit and loses after a miss (on Yambda the next generated item is a hit in 54% of
      lists after a hit against 25% in a ranked list, and in 4% against 6% after a miss), the predicted reward of later
      steps depends on an assumption about responses nobody observed, which a reward model trained with hidden
      responses makes unbiased, and crediting an item with the rest of the list helps no per-item reward.],
    [*Q5.* Trade-off between relevance and reward.],
    [RL settings choose a point on the frontier; only the reward function moves the frontier: within 1% of the base
      recommender's NDCG\@10, RL on the share gains 4% of predicted share on KION, the same RL with logged relevance in
      the reward 17%, and re-ranking by the same reward model 15%.],
  ),
  caption: [The five questions and their answers. Numbers are from KION unless Yambda is named; "$plus.minus$" is the
    standard deviation over three seeds.],
  placement: none,
) <tab-answers>

Every reward gain in this setting is reported by a model. The pipeline therefore carries instruments that do not
depend on the model being optimised, and the study is organised around what they show. Our contributions are the
following.

- *A protocol for judging RL against a learned reward offline.* A second reward model fitted on a disjoint half of
  the users; the logged response to those recommended items that the user went on to consume, which no model stands
  behind; the same lists scored for another user, which separates what the policy learned about the user from what
  it learned about items; relevance split by the logged outcome, by the length of the user's history and by position
  in the list; and the reward model's predictions on real sequences under each assumption a simulation has to make.
  Operating points are selected on validation requests only (@sec-instruments, @sec-setup).
- *Three seeds for every central comparison.* The main run, the anchor sweep, the KL budget, the estimator and
  credit comparisons, the alternatives to RL and logged relevance in the reward were each trained with three seeds,
  and every central conclusion holds in each of them (@sec-seeds).
- *What the policy learns, and why.* On KION 78% of the change in the policy's scores is one shift per item,
  correlated 0.90 with the item's usual consumed share; 0.2% of the predicted gain depends on who receives the list.
  The anchor weight is not a comparable quantity, the KL distance is, and at an equal distance the estimator and the
  credit assignment make no difference (@sec-q1, @sec-q2).
- *Long-term rewards and several steps ahead.* Rewards for return, likes and adoption, a reward paid after the whole
  list, hits by position of generated and ranked lists, and a reward model trained with hidden responses that removes
  the bias of simulated feedback (@sec-q3, @sec-q4).
- *The frontier.* Every RL setting we varied moves the policy along one frontier of relevance against reward; adding
  logged relevance to the reward moves the frontier, and re-ranking by the same reward model reaches more of the
  reward at equal relevance without training (@sec-q5).

All results are offline. @sec-online designs the experiment that would settle what they leave open, and
@sec-limitations lists the rest.

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
        a logged history $h$; each item is appended to the history as if it had been consumed.],
      sym.arrow.r,
      stage[3. Reward][The training reward model predicts the response to each generated item: consumed share
        $hat(rho)$, relative share $hat(q)$, continuation $hat(c)$, return $hat(u)$; on Yambda also like, dislike and
        adoption.],
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
      #sym.dot.c relevance against the user's real future, split by outcome, history length and position #sym.dot.c
      concentration of impressions #sym.dot.c the reward model on real sequences under simulated responses.
    ]
  ],
  caption: [The pipeline. The reward model of step 3 is the only source of reward during fine-tuning; the instruments
    in the dashed box never influence the policy.],
  placement: none,
) <fig-pipeline>

== Setting and models <sec-models>

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
before any outcome for it exists. Heads read the state, each trained with binary cross-entropy: the consumed share
$hat(rho)$ (against $rho$ as a soft label); the _relative share_ $hat(q)$ defined below; the probability $hat(c)$
that the session continues (the user's next event follows within a gap $g$); the probability $hat(u)$ that the user
returns (a new session starts within a window $w$); and a choice head, which an earlier iteration of the study found
useless as a relevance signal and which no reward here uses. A retention outcome counts as known only if its whole
waiting time lies inside the observed data.

*Reactions and adoption.* Where the log records them (Yambda), three more heads predict that the user likes the
item, dislikes it, and _adopts_ it. A like or a dislike is attributed to the user's latest play of the same track
within a day before the reaction. A track is adopted when the user's first play of it is followed by another play
in a later session (more than the session gap after the first) within 14 days; the label is known only for first
plays whose 14 days lie inside the log. These heads enter the loss with the weight of the retention heads, a
quarter of the consumed share's.

*Relative share.* The consumed share pays for items that are easy to finish, whoever receives them: a film over a
series, a short track over a long one. To separate the user's response from the item's, each event also gets its
mid-rank quantile among the training-period events of the same item,
$ q_t = (\#{rho' < rho_t} + 1/2 \#{rho' = rho_t}) / n_(i_t), $ <eq-relative>
where $rho'$ runs over the $n_(i_t)$ training-period consumed shares of item $i_t$. Its mean is exactly one half for
every item, so no item is better than another on average, and a policy can raise it only by matching items to the
users who take to them more than others do.

*Two copies.* Two reward models are trained on disjoint halves of the users. The _training_ reward model produces
the rewards during fine-tuning. The _held-out_ reward model never influences the policy and only scores its lists.

*Hidden responses.* To score the $k$-th item of a generated list, the reward model needs the responses to the items
before it, which nobody observed. A reward model trained only on logged sequences has never read an unknown response.
A second pair of reward models is therefore trained with one more input value, "unknown": in half of the training
windows the responses to the last 1 to 10 events, chosen uniformly, are replaced by it. This is the situation of a
generated list, whose history has known responses and whose generated items do not. The validation loss is computed
with all responses visible.

== Reinforcement fine-tuning <sec-rl>

One episode is the generation of one list for a history sampled from the training period. The state is
$s_k = (h, a_(1:k-1))$, the action is the next item $a_k$, the reward $r_k$ is predicted by the training reward
model (by default $r_k = hat(rho)_k$), and the return is $G_k = sum_(l >= k) gamma^(l-k) r_l$. The reward model is
the environment. To score the $k$-th item it is fed, for the items generated before it, its own expectation
$hat(rho)$ by default; @sec-q4 compares this with a sampled response and with the "unknown" input.

The policy gradient $EE[sum_k A_k nabla_theta log pi_theta (a_k | s_k)]$ is estimated with one of two advantage
estimators. The default uses a learned _value model_ $V_phi$, a copy of the recommender's transformer with a scalar
head, and generalised advantage estimates @schulman2015gae; each batch of lists is used for four gradient steps on
the clipped surrogate of PPO @schulman2017ppo. The alternative is a _leave-one-out_ baseline: several lists are
sampled for each history, and the baseline of a list is the mean return of the others
@williams1992reinforce @kool2019buy. Advantages are standardised to zero mean and unit variance within a batch in
both cases. The clipping of the surrogate touched at most 0.3% of sampled items in any run that did not collapse, so
we do not attribute anything to PPO: in effect the default is an actor--critic policy gradient with several steps
per batch.

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
$ r_k = & w_rho (hat(rho)_k - b_k) + w_q hat(q)_k + w_c hat(c)_k + w_u hat(u)_k + w_L (hat(l)_k - hat(d)_k)
  + w_A hat(alpha)_k \
  & + w_R bb(1)[k = K] hat(u)_K + w_F (bb(1)[a_k in F]) / (log_2 (k + 1))
  + w_D / (K - 1) sum_(l < k) d(a_k, a_l). $ <eq-reward>
The default is $w_rho = 1$ and everything else zero. The _relative-share_ reward is $w_q = 1$ alone, and the
retention rewards use $w_c$ and $w_u$. The _like_ reward pays the predicted probability of a like minus that of a
dislike ($w_L = 1$), the _adoption_ reward the predicted probability $hat(alpha)$ that the item is adopted
($w_A = 1$). The _return after the list_ ($w_R = 1$) pays the predicted return after the last item only, so that the
reward belongs to the list as a whole and reaches each item through the return $G_k$. The term with $w_F$ is the
_logged-relevance_ reward of @petrov2024aligning: for a history from the training period the user's real
continuation $F$ is known, and the term sums to DCG\@K over the list. A baseline $b_k$ makes the reward _personal_: it
subtracts what the same reward model predicts for the same list shown to another user, or the item's mean share in
the training period. The last term makes the reward depend on the _list as a whole_: $d$ is the cosine distance
between the embeddings of two items in $pi_0$, and over a list the term sums to $w_D K \/ 2$ times its mean pairwise
distance.

== Alternatives to on-policy RL <sec-alt-methods>

Four methods raise the same rewards without generating lists in a training loop. *Liked-only fine-tuning* continues
next-item training for one epoch on targets with $rho >= 0.5$; it uses logged feedback and no reward model.
*Off-policy REINFORCE* @matveev2026session raises the log-probability of every logged item in proportion to the
discounted logged reward that followed it in the session, with importance weights against $pi_0$ capped at $kappa$.
*Re-ranking* takes the 100 best candidates of $pi_0$ and keeps the $K$ with the highest
$r(j | h) + w log pi_0 (j | h)$, where $r$ is the reward predicted for $j$ as the next item after the real history;
nothing is trained, and $w$ plays the part of the anchor. For the list reward it becomes greedy maximal marginal
relevance @carbonell1998mmr. *Best of $n$ lists* samples $n$ lists from $pi_0$ and serves the one with the highest
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
  The gain of a method over $pi_0$ splits into the gain other users would give the same lists and the rest, whose
  share of the total is the _personal share of the gain_.
- *Relevance.* NDCG, Recall, MAP and HitRate at $K = 1, 5, 10$ against $F$, and NDCG\@10 against its parts: _liked_
  ($rho >= 0.5$) and _skipped_ items; items consumed _above_ and _below usual_ ($rho$ against the item's mean share in
  the training period); and, where repetition is allowed, _repeats_ from the visible history and _new_ items. Users
  are also grouped into quartiles of the number of events before the request.
- *Concentration.* Catalogue coverage (the share of items in at least one list), the Gini coefficient of the
  impressions per item, the share of impressions taken by the 1% most shown items, and the expected overlap between
  the lists of two users (_pair overlap_).
- *Per-item shift.* For each history the change $log pi_theta (j | h) - log pi_0 (j | h)$ of every item splits into
  the item's mean change and the rest; the share of its variance that the item means explain, weighted by $pi_0$, is
  1 for a policy that applies one preference per item to every user.
- *Position.* The rate at which the item at each position of a list is a hit, overall, after a miss and after a hit
  at the position before.
- *Simulation on real sequences.* For requests with at least five next events, the list is the user's real next five
  items in order, and the reward model predicts the share of each from the history and the items before it, with
  the earlier responses as logged, or as a simulation would have to assume them.

= Experimental Setup <sec-setup>

*Data.* Yambda-50M @ploshkin2025yambda is a listening log of Yandex Music: each event has the track, a timestamp,
the played share, the track length and a flag telling whether the user chose the track or a recommender served it;
separate tables hold likes and dislikes. KION @petrov2022kion is a log of an online cinema with one row per user and
title: the date of the last viewing and the watched share of the title; an item table gives each title's name,
type (film or series), year and genres. Both logs are split along one time axis: the last 7 days are the test
period, the 7 days before them the validation period, the rest the training period; the catalogue and all item
statistics come from the training period only (@tab-data). The consumed share is bimodal in both logs, so the reward
model is in effect a skip classifier. On KION the share depends strongly on the length of the title: films are
watched to 0.5 on average, and the longest tenth of the catalogue, which is series, to 0.22, because the share is
taken of the whole series. A reward built on the share prefers films before any model is involved. On Yambda the
session continues if the user has another event within 30 minutes and the user returns by starting a new session
within a day; KION only has dates, so a session is a calendar day and a return is a viewing on another day within a
week.

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
    [Events with a like / a dislike], [not recorded], [0.77% / 0.16%],
    [First plays adopted (played again in a later session within 14 days)], [--], [24%],
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
measured on generated lists of ten unless ranked lists are named. The change from the base to the fine-tuned
recommender is reported with a 95% interval from a paired bootstrap over users, which measures the noise of the user
sample, and with the standard deviation over three seeds, which measures the noise between training runs.

*Selection.* No model, hyperparameter or operating point was chosen on the test period within this protocol. Model
sizes and learning rates were fixed before the study. Sweeps are reported in full, and where one point of a sweep
has to be named we name the run with the highest held-out predicted share on _validation_ requests among the runs
that keep the validation NDCG\@10 within 5% of the base recommender's. The weight $beta = 0.3$ of the main run is
the exception: it was carried over from earlier iterations of this study, in which test metrics of anchor sweeps had
been seen, so it is an illustration and not a selected optimum. By the validation rule the selected weights are
$beta = 1$ on KION and $beta = 0.05$ on Yambda (@tab-selection), and the review sessions select the same.

*Sessions and seeds.* Each dataset was run in three kinds of sessions on one 96 GB GPU (@tab-config). A _final_
session ran the full grid: 49 fine-tuning runs on KION (6.1 hours) and 25 on Yambda, in a first session and a
continuation that reused its models (12.3 hours together). A _review_ session retrained every model with the same
seed and ran the long-term rewards, the reward models with hidden responses, and the analyses by position, user and
item (1.7 hours on KION, 7.0 on Yambda); it reproduced the base recommenders of the final sessions exactly (NDCG\@10
0.1173 and 0.1172). Two _seed_ sessions repeated the core of the study with seeds 1 and 2: the base recommender,
both reward models, the main run, the anchor sweep, a KL budget, the credit comparison, the alternatives to RL, the
decomposition of the gain and logged relevance in the reward, and on KION also the leave-one-out estimator and the
relative-share reward (1.9 hours per seed on KION, 4.5 on Yambda). Seed 0 is the seed of the final and review sessions, and the numbers in the text are seed 0's unless a
standard deviation is given.

= Results <sec-results>

In the tables, "Pred. share" is the consumed share of the generated lists as the held-out reward model predicts it,
and "Logged" and "Surplus" are the logged share and the logged surplus per recommended item of @sec-instruments.

== Base models <sec-pretrained>

*Recommender.* On KION the recommender is mid-table (@tab-baselines): level with GRU4Rec, 13% above SASRec and 5%
below BERT4Rec for ranked lists, and popularity is a strong floor. Generating a list item by item costs 4% of
NDCG\@10 against ranking (@sec-q4 shows where). On Yambda the recommender is the best of the trained models, and all
of them are far below a replay of the user's own history: NDCG\@10 0.187 against 0.117. The split into repeats and
new items shows where the trained models stand. Replay scores only on repeats; on tracks the user has not played in
the visible history, the recommender is the best ranked model (0.032) and popularity reaches a third of it. Every
statement below about relevance on Yambda is relative to this recommender, and the split is reported wherever a
method could gain by replaying.

#report-table("baselines", [Baselines on generated or ranked lists of ten for the test requests. The last two rows
  are the recommender that is fine-tuned, ranked and generated. On KION items of the history are blocked, so every
  hit is a new item.]) <tab-baselines>

*Reward model.* On held-out events the reward model predicts the consumed share with AUC 0.755 on KION and 0.868 on
Yambda (@tab-reward-model; $0.7545 plus.minus 0.0009$ and $0.8668 plus.minus 0.0006$ over seeds), against 0.656 and
0.607 for the item's mean share and 0.673 and 0.798 for the user's recent mean. The two copies agree (correlation of
predictions 0.94 and 0.95). The right half of the table measures what a signal offers a recommender that chooses
among its own candidates: how much it varies between the 100 candidates of one user, against how much its mean
varies between users. On Yambda the predicted share varies five times less between one user's candidates than
between users: it is mostly a property of the listener, which leaves a policy little to act on. On Yambda the like
and dislike heads reach AUC 0.74 and 0.73 at base rates of 0.75% and 0.12% of events (@tab-reactions).

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
  caption: [The held-out reward model of the final sessions on test events (AUC; the share heads against the
    outcome thresholded at one half), and the standard deviation of each prediction between the 100 candidates of one
    user ("candidates") and of its mean over candidates between users ("users").],
) <tab-reward-model>

== How stable the numbers are <sec-seeds>

#report-table("seeds", [The central comparisons over three seeds (0, 1, 2): mean $plus.minus$ standard deviation.
  A seed fixes the initial weights of every model and the sampling of fine-tuning, so it covers the base
  recommender, both reward models and every run. "Budget" is a KL budget of 0.1 nats on KION and 0.5 on Yambda; "–":
  not run with three seeds on that dataset.], size: 8pt) <tab-seeds>

Every conclusion of the following sections holds in each of the three seeds (@tab-seeds). The direction of every
change of the main run is the same in all seeds, the logged surplus falls by 0.0007 in each of them on KION, and the
order of the methods at equal relevance does not change. Three quantities vary more than the rest: relevance at the
anchor weight 0.1 on KION, which is near the edge where the policy starts to drift (NDCG\@10
$0.079 plus.minus 0.012$); the personal share of the gain on Yambda ($16% plus.minus 16%$), whose denominator is a
gain of one point of share; and the gap between full and myopic credit on KION, about two standard deviations:
crediting each item with its own reward is ahead of full credit in every seed (its lowest NDCG\@10, 0.1039, is above
the highest of full credit, 0.1037), while one-item episodes overlap with full credit. On Yambda both myopic variants
are ahead in every seed. The long-term rewards, the reward models with hidden responses and the
analyses by position, user and item come from the review sessions, with seed 0 only.

== Q1. How does behaviour change, and why? <sec-q1>

#report-table("main_run", [The main fine-tuning run ($beta = 0.3$) against the base recommender, seed 0. Changes
  are relative, in percent, with a 95% paired-bootstrap interval over users where the quantity is defined per user.
  @tab-seeds gives the same changes over three seeds.], size: 7.9pt) <tab-main>

*KION: the models agree, the log hardly does.* The training reward model reports a gain of 20.3% and the held-out one
21.5% ($23.9% plus.minus 2.1$ over seeds): by the standard check for exploitation the gain is real (@tab-main).
Relevance falls by 10.7% overall and by 18.2% on skipped titles, and it is kept on liked ones ($+1.3%$), which looks
like a policy that has learned to avoid what users abandon. The other instruments disagree. The policy recommends
fewer titles that the user went on to watch (hits $-8.3%$), and those are finished more often ($+9.8%$) because they
are titles that everybody finishes more often. The two effects almost cancel: the logged share per recommended item
rises by 0.6% ($[-0.6, +1.7]$), and by $1.1% plus.minus 0.4$ over seeds. Measured against what is usual for each
title, users took to the recommended titles less than before: the logged surplus falls from 0.0012 to 0.0005, by
0.0007 in every seed, and the predicted relative share falls too ($-1.0%$). The same lists shown to _other_ users are
predicted to earn the same gain ($+22.0%$).

*Where the impressions go.* The policy moves impressions from series to films: films take 66% of the impressions
of the base recommender's generated lists and 80% of the main run's, 89% at $beta = 0.1$ and all of them without an
anchor (@tab-content). The five titles that gained most are films of 1.4 to 2.2 hours that viewers usually finish
to 0.54--0.69. Four of the five that lost most are series of 9.5 to 22 hours whose usual share, taken of all their
episodes, is 0.08 to 0.29, and the fifth is a film usually watched to 0.33 (@tab-titles). Popularity is not what the policy follows: one of the five largest
gains is a film that 953 users watched in the training period, recommended in 17,315 lists instead of 7,308.

#report-table("content_types", [KION: impressions of films and series in the generated lists of the test requests,
  for the base recommender and fine-tuned policies (review session).]) <tab-content>

#report-table("titles", [KION: the titles whose number of lists grew and shrank most under the main run.
  "Viewers in training": users who watched the title in the training period; "usual share": the mean consumed share
  of the title in the training period; lists among the 59,997 test requests.]) <tab-titles>

*One shift per item.* For each history, the change of an item's log-probability under the policy splits into a part
that is the same for every history and the rest (@sec-instruments). On KION the per-item part explains 78% of the
change, and an item's shift correlates 0.90 with its usual consumed share, 0.12 with its log popularity and $-0.36$
with its duration (@tab-shift). The policy learned a ranking of titles by how often they are finished and applies it
to every user. On Yambda the per-item part explains 44%, and 11% when every item counts alike: the policy moved less
(KL 0.04 against 0.09 nats per item), and less of what it changed is one preference per item.

#report-table("logit_shift", [The change of the next-item log-probabilities from the base recommender to a
  fine-tuned policy, on 2,048 test histories: the share of its variance that one shift per item explains, weighted by
  the base recommender's probabilities and unweighted, and the correlation of the per-item shift with properties of
  the item (review sessions).], size: 8pt) <tab-shift>

*Why.* The reward makes this the cheapest move. On KION the reward model's prediction of the share varies almost as
much between one user's candidates as between users (0.144 against 0.153, @tab-reward-model), and most of the
variation between candidates is the item: the item's mean share alone predicts the outcome with AUC 0.66. A policy
gradient sees one sampled list per history and averages over histories, so the part of the signal that is the same
for all of them is what accumulates. The held-out model confirms the gain because it has learned the same item means.
The gain is impersonal: 99% of it on KION is what the held-out model predicts for the same lists shown to another
user ($0.2% plus.minus 0.5%$ is personal over seeds), and at a weaker anchor the personal share is negative
(@tab-whose). Re-ranking candidates by $macron(rho)(j) + w log pi_0 (j | h)$, where $macron(rho)(j)$ is the mean
consumed share of item $j$ in the training period, uses no reward model and no RL, and reaches the main run's point:
a predicted share of 0.549 at NDCG\@10 0.109, against 0.553 at 0.105. What RL extracts from the reward model on this
reward is, to a first approximation, a ranking of items by how often they are finished.

#report-table("whose_gain", [The gain in predicted share (held-out model) and the part of it that other users would
  give the same lists. "Personal" is the share of the gain that depends on who receives the list.]) <tab-whose>

*Yambda: little to gain.* The same weight moves the policy less than half as far (KL 0.037 against 0.092 nats per
item) and changes little: $+1.7%$ of predicted share ($1.67 plus.minus 0.01$ over seeds), NDCG\@10 $+0.8% plus.minus
0.7$, and a logged share per recommended item $+1.7% plus.minus 0.7$ over seeds ($+2.5%$, $[+0.3, +4.6]$, in seed 0);
the surplus does not move. The policy has shifted slightly towards popular tracks. The value model explains 78% of
the variance of returns here, against 49% on KION: the reward is mostly determined by the listener, as
@tab-reward-model predicted.

#answer[Q1][The policy learns which items everybody finishes, not what each user wants. On KION it moves impressions
  from long series to short films (films from 66% to 80% of impressions), 78% of its change is one shift per item,
  and the 24% predicted gain is almost entirely what any other user would also give the same lists; the log confirms
  1% of consumed share per recommended item and shows users taking less to the recommended titles than usual. The
  reason is the reward: the reward model's predictions differ between items far more than between users' reactions
  to an item, so a ranking of items is what the policy gradient finds. On Yambda the reward depends on the listener,
  and the policy changes little.]

== Q2. Is relevance kept? Does everyone get the same items? <sec-q2>

*Relevance at every cut-off.* On KION every ranking metric falls at every cut-off, by 8 to 14%: NDCG\@1 by 7.9%,
NDCG\@5 by 11.9%, NDCG\@10 by 10.7%, Recall\@10 by 10.3%, MAP\@10 by 12.2% and HitRate\@10 by 7.9% (@tab-ranking).
The first item is chosen from the real history alone, so the loss at $K = 1$ is the policy's own preference; from
the second position on, the generated context adds to it (@sec-q4). Over three seeds NDCG\@10 falls by
$11.5% plus.minus 1.3$. On Yambda every metric stays within 2.2% of the base recommender at every cut-off, and
NDCG\@10 changes by $+0.8% plus.minus 0.7$. By outcome (@tab-main), the KION policy keeps relevance on liked titles
and loses it on skipped titles ($-18.2%$) and on titles watched above usual ($-9.4%$): it has not learned what each
user likes, only what everyone finishes.

#report-table("ranking_metrics", [Ranking metrics at $K = 1, 5, 10$ for the main run against the base recommender
  (generated lists, seed 0). At $K = 1$ NDCG, MAP and HitRate coincide.], size: 8pt) <tab-ranking>

*Who loses relevance.* On KION the users with the least history lose most (@tab-history): users with 5 to 9 events
before the request lose 13.5% of NDCG\@10 and 2.4% of logged share per item, users with more than 25 events lose 7.4%
and gain 3.3% of logged share. For a user the recommender knows little about, the item-level preference of the
policy replaces most of what the list had been. On Yambda, where every user has at least 50 events, no group loses.

#report-table("history_length", [NDCG\@10 and the change of the logged share per recommended item by the user's
  number of events before the request, in quartiles (review sessions, main run).], size: 8pt) <tab-history>

*Does everyone get the same items?* Not yet at the main run, but closer to it. The base recommender's generated lists
on KION are already concentrated: the 1% most shown titles take 72% of the impressions (62% for ranked lists).
The main run raises this to 76%, the Gini coefficient of impressions from 0.963 to 0.969, and lowers the catalogue
coverage from 0.50 to 0.45 (@tab-concentration). With a weaker anchor the lists converge: 92% at $beta = 0.1$, and
100% without an anchor, where nearly every user receives the same list (pair overlap 0.98 on KION and 1.00 on Yambda,
@tab-sweep). On Yambda the main run moves the share of the most shown 1% from 18% to 20% and the Gini from 0.737 to
0.747; concentration grows markedly only at $beta = 0.05$ (29%).

#report-table("concentration", [Concentration of impressions in the lists for the test requests: catalogue
  coverage, the Gini coefficient of the impressions per item, and the share of impressions taken by the 1% most shown
  items (review sessions).]) <tab-concentration>

*What keeps relevance: the distance, not the weight.* Nothing in the reward asks for relevance, so the anchor is
what keeps it, and the KL divergence from $pi_0$ at which a run ends describes how much it keeps
(@fig-frontier). Without an anchor the policy collapses, and the held-out model partly applauds: the training reward
model reports a share of 0.84 on KION and the held-out one 0.69, still 51% above the base recommender, while NDCG\@10
and the logged share are near zero (@tab-sweep). On KION relevance falls and the predicted share rises steadily as
the anchor weakens; on Yambda both move by a few percent down to $beta = 0.05$, and $beta = 0.03$ collapses the
policy in every seed (NDCG\@10 $0.006 plus.minus 0.004$, KL $3.6 plus.minus 0.9$), while $beta = 0.1$ keeps
relevance in every seed ($0.1172 plus.minus 0.0017$).

#report-table("anchor_sweep", [Anchor sweep with the value-model estimator, 1,000 iterations per run. KL is the
  divergence from the base recommender at the end of fine-tuning, in nats per generated item; "Training model" is
  the share predicted by the reward model that produced the rewards.], size: 8pt) <tab-sweep>

#figure(
  image("figures/frontier_by_distance.svg", width: 100%),
  caption: [Every fine-tuning run of the default reward against the distance it ended at from the base recommender
    (logarithmic axis; the base recommender is drawn at the left edge). Runs that differ in the advantage estimator,
    in how the constraint is set, in credit assignment and in the learning rate lie on one curve. For one-item
    episodes the distance is that of the first position only.],
) <fig-frontier>

The weight itself means different things under different estimators. @tab-estimators changes only the baseline of
the policy gradient, with the number of lists per iteration, the learning rate and the passes over a batch held
equal. At $beta = 0.3$ on KION the leave-one-out estimator ends 6.7 times further from the base recommender than
the value model (KL $0.543 plus.minus 0.010$ against $0.081 plus.minus 0.004$), and at $beta = 0.1$ it collapses
where the value model does not; on Yambda the factor at $beta = 0.1$ is 15. Under a KL budget of 0.1 the two
estimators end at the same distance and on the same frontier (NDCG\@10 $0.1037 plus.minus 0.0021$ against
$0.1013 plus.minus 0.0026$ at predicted shares 0.546 and 0.558), while the weights that hold the budget differ by a
factor of 2.9 ($0.752 plus.minus 0.017$ against $0.256 plus.minus 0.004$). On Yambda budgets of 0.2 and 0.5 nats end
at almost the same weight (0.066 and 0.067): in this region a fixed weight hardly selects an operating point, while a
budget does. One reading of the factor is that standardising the advantages gives the policy-gradient term the same
scale under both estimators while the share of that scale that is signal differs; we did not test it. The practical
point does not depend on it: an anchor weight is not a quantity that can be compared between estimators, rewards,
datasets or papers, and a KL budget is. PPO's clipping touched at most 0.3% of sampled items in every run that did
not collapse, the run with ten times the learning rate included.

#report-table("estimators", [The value model against the leave-one-out baseline at equal anchor weights and at
  equal KL budgets. "$beta$ at end" is the weight the run finished with. "Clipped" is the share of sampled items
  whose probability ratio left PPO's clipping range.]) <tab-estimators>

#answer[Q2][On KION relevance is not fully kept: NDCG\@10 falls by $11.5% plus.minus 1.3$ at the main operating point,
  at every cut-off, most for the users with the shortest histories, and the lists concentrate further on the most
  shown titles (76% of impressions instead of 72%). On Yambda relevance and variety are kept. In both logs only the
  anchor keeps them, the KL distance at which a run ends describes how much, and the anchor weight does not; without
  the anchor every user receives the same list.]

== Q3. Long-term goals: return and continued interaction <sec-q3>

Four signals describe what follows an item: the session continues, the user returns, the user likes or dislikes the
item, and the user adopts it. The first two exist in both logs, the last two only in Yambda. A fifth reward asks for
the return after the whole list, paid at its last position (@sec-rewards).

*What a policy can choose between.* A reward helps a policy only if it differs between the items the user could be
shown. The predicted return varies between one user's candidates a quarter as much as between users on KION (0.033
against 0.125) and a ninth as much on Yambda (0.014 against 0.130); a like varies by 0.003 between candidates
(@tab-spread). Adoption is the exception: it varies between a user's candidates half as much as between users (0.077
against 0.165), relatively more than any other signal, and it is the only one whose ranking of the candidates leans
towards the items the user really consumed next (AUC 0.56; the share 0.53, a like 0.48).

#report-table("signal_spread", [What a predicted signal offers a recommender that chooses among the 100 best
  candidates of the base recommender: its standard deviation between one user's candidates, of its mean between
  users, and the AUC of the signal for telling candidates the user consumed next from the others (held-out reward
  models of the review sessions).]) <tab-spread>

*Return per item and continued sessions.* Rewarding the predicted return of each item raises it by 3 to 7% on KION
for a loss of 8 to 23% of NDCG\@10 and does not move it on Yambda (@tab-retention). The session signal on KION is
derived from calendar days and the order of rows within a day, and we do not interpret its reward.

#report-table("long_term_rewards", [Long-term rewards (review sessions, 1,000 iterations). "Δ" is the change of
  NDCG\@10 against the base recommender in percent; "Return" is the held-out model's predicted return after the last
  item of the list; P(like) and P(adopt) its predicted probability of a like and of adoption, which exist only for
  Yambda; "Cover." the catalogue coverage; "Hours" the mean duration of a recommended KION title.], size: 8pt)
<tab-long-term>

*Return after the list.* On KION this reward moves the policy further than any per-item reward at the same weight
(KL 0.51 at $beta = 0.3$, against 0.08 for the share). The predicted return after the list rises from 0.73 to 0.86
($+18%$), while NDCG\@10 falls by 69% and the logged share by 64% (@tab-long-term). The policy found what the reward
model associates with an early return, short titles: the mean duration of a recommended title falls from 5.7 to 2.2
hours. Short titles are what the reward model associates with an early return; whether they cause one is what the
log cannot say, and a list of ten short films lies far from the lists the model was trained on. With discount 0, which credits only the last item, the policy hardly moves (KL 0.006,
return $+0.7%$): the reward is paid for the list as a whole, and only crediting every item with it reaches it. This is
the one reward of the study for which credit along the list matters (@sec-q4), and what it buys is not worth having.
On Yambda the return after the list does not move at any weight (0.89).

*Likes.* On Yambda the reward $hat(l) - hat(d)$ at $beta = 0.3$ raises the predicted probability of a like from 1.08%
to 1.17% ($+8%$) for 2% of NDCG\@10 (@tab-long-term). At $beta = 0.1$ the policy moves far (KL 0.83): the probability
of a like rises by 49% and that of a dislike by 53%, and NDCG\@10 falls by 28%. The policy found tracks that provoke
reactions of both kinds, not tracks that are liked more.

*Adoption.* The adoption reward is the one long-term reward that moved without a loss of relevance. At $beta = 0.3$
the predicted probability that a first play leads to a comeback within 14 days rises from 0.233 to 0.277 ($+19%$),
NDCG\@10 by 1.5% and the logged share by 2.5%, at KL 0.14; the price is variety, with catalogue coverage falling from
0.52 to 0.41. At $beta = 0.1$ the policy drifts (KL 1.2): adoption $+47%$, NDCG\@10 $-37%$, coverage 0.13. The test
week cannot confirm any of it: a first play in the last seven days of the log has no fourteen days after it, so the
adoption of the recommended tracks is unobserved.

*What offline evidence cannot settle.* Return, adoption and reactions after a change of policy are counterfactual:
the log shows them under the policy that produced it, and the logged check of @sec-instruments covers only items the
user reached. @sec-online designs the experiment that measures them.

#answer[Q3][Return, likes and adoption can be predicted and rewarded, but at a moderate distance from the base
  recommender their predictions rise by a few percent. The best case is adoption on Yambda: $+19%$ predicted at no
  loss of relevance, for a fifth less catalogue coverage. The one large gain, the return after the list on KION
  ($+18%$), comes from drifting to short titles at a 69% loss of NDCG\@10. Return varies four to nine times less
  between a user's candidates than between users, so there is little for a policy to choose. None of these gains can
  be confirmed offline; they need the experiment of @sec-online.]

== Q4. Generating several steps ahead <sec-q4>

A generated list is built step by step: the item chosen at position $k$ is appended to the history as if the user had
consumed it, and the next item is chosen after it. The policy thus plans several steps ahead under an assumption,
that each of its items is taken, which is false for most of them: at most 17% of the items at any position are hits.
Three questions follow: what the assumption does to the list, what it does to the reward, and whether RL makes use of
the steps.

#figure(
  image("figures/position_hits.svg", width: 100%),
  caption: [The rate at which the item at each position is among the user's next items, for all lists, after a miss
    and after a hit at the position before: ranked and generated lists of the base recommender and generated lists of
    the main run (review sessions; @tab-positions has the numbers).],
) <fig-positions>

*What generation does to the list.* At position 1 a generated and a ranked list coincide. On KION, from position 3
on, the generated list hits less often than the ranked one (0.0287 against 0.0324 at position 5, 0.0160 against
0.0213 at position 10, @fig-positions). The split by the position before explains why. After a miss, the generated
list continues from a context that did not happen and hits less often than the ranked list (0.0273 against 0.0312 at
position 5); after a hit it continues from a context that did happen and hits more often (0.112 against 0.093 at
position 2). On Yambda the effect is large in both directions. After a hit the next generated item is again a hit in
about half of the lists (0.54 at position 5) against a quarter for the ranked list (0.25): listening continues in
sequences, an album or an artist, which a generated list follows. After a miss the generated list hits at 0.040
against 0.063. Since hits are frequent on Yambda, generation is ahead overall at positions 2 to 7; on KION misses
dominate, and generation costs 4% of NDCG\@10. Fine-tuning does not change the pattern: the main run's generated
lists follow the base recommender's closely after a miss and after a hit.

*What the assumption does to the reward.* To score the $k$-th item the reward model needs the responses to the items
before it. On lists whose responses are known, the user's real next five items in order (11,140 KION and 7,153
Yambda requests), we compared the assumptions a simulation can make (@fig-assumptions). With the logged responses fed
back, the AUC of the predicted share rises along the list, from 0.757 to 0.782 on KION and from 0.640 to 0.819 on
Yambda: the response to the previous item is the best single predictor of the next. Under every assumption the AUC
stays near 0.76 on KION and rises only to about 0.69 on Yambda: no simulated response carries that information. The
assumptions differ in bias. Feeding back the expected share underestimates the next share on Yambda by 0.06 to 0.09,
presumably because the model was rarely shown a response in the middle of a bimodal distribution; sampling a response overestimates slightly
($+0.003$ to $+0.035$); assuming that every earlier item was finished, which is what generation assumes about
relevance, overestimates by 0.06 to 0.18. The reward model trained with hidden responses and fed "unknown" is nearly
unbiased at every later step ($+0.001$ to $+0.013$), and it reads observed responses as well as the standard model
(AUC 0.755 against 0.758 on KION, 0.867 against 0.868 on Yambda).

#figure(
  image("figures/feedback_assumptions.svg", width: 74%),
  caption: [The reward model on the users' real next five items: bias (predicted minus logged share) and AUC of its
    prediction at each step, when the responses to the earlier items are fed back as logged, as their expectation, as
    a sample, as finished, or as unknown to a reward model trained with hidden responses (review sessions;
    @tab-assumptions has the numbers).],
) <fig-assumptions>

*Does the assumption change the policy?* Fine-tuning against the reward model with hidden responses, which is fed
"unknown" for generated items, lands on the same frontier as fine-tuning with expected responses: at $beta = 0.3$ on
KION NDCG\@10 0.106 against 0.104 at the same distance (KL 0.078 against 0.080) and the same predicted share (0.551
against 0.553); at $beta = 0.1$ it ends nearer to the base recommender (KL 0.45 against 0.63) and correspondingly more
relevant (@tab-masked). Sampling the response gives the same picture (@tab-feedback). The assumption sets the level of
every multi-step predicted reward, a share of 0.55, 0.62 or 0.69 for the same KION policy, but not which policy wins.

*Does RL use the steps?* At an equal KL budget, a policy credited with each item's own reward only ($gamma = 0$), or
trained on one-item episodes, is at least as relevant as one credited with the rest of the list (@tab-credit). On
KION at a budget of 0.1, NDCG\@10 is $0.1070 plus.minus 0.0027$ and $0.1043 plus.minus 0.0024$ against
$0.1013 plus.minus 0.0026$, at predicted shares 1.4 to 1.6 points lower; on Yambda at a budget of 0.5, $0.1076
plus.minus 0.0014$ and $0.1075 plus.minus 0.0038$ against $0.0960 plus.minus 0.0008$, at shares 3 to 4 points lower.
The myopic policies trade a little reward for relevance at the frontier's rate, on it or slightly above it
(@fig-frontier). At a fixed weight they are not comparable: with $gamma = 0$ the policy ends 31 times further from
the base recommender, presumably because per-item rewards standardise to larger advantages than returns that share
most of their terms. For rewards that are sums of per-item predictions, crediting an item with what follows it
contributed nothing we could detect. The exception is the return after the list (@sec-q3), paid only at the end:
there full credit is needed to move at all, and the policy used it to game the reward model.

#report-table("credit", [Credit ablations on the default reward at equal KL budgets and, for KION, at an equal
  anchor weight (seed 0; @tab-seeds has three seeds at one budget).]) <tab-credit>

#answer[Q4][Generating several steps ahead gains after a hit and loses after a miss: on Yambda the next generated
  item is a hit in 54% of lists after a hit, against 25% in a ranked list, and in 4% against 6% after a miss; on KION
  misses dominate and generation costs 4% of NDCG\@10. The predicted reward of later steps depends on an assumption
  about responses nobody observed; feeding back expectations or finished items biases it, and a reward model trained
  with hidden responses is unbiased without losing accuracy. None of the assumptions changes which policy RL finds,
  and for per-item rewards crediting an item with the rest of the list helps nothing; only a reward paid at the end
  of the list needed it.]

== Q5. Trade-off between relevance and reward <sec-q5>

*One frontier for all RL settings.* @fig-frontier places every fine-tuning run of the default reward at the distance
it ended at from the base recommender: different anchor weights, KL budgets, advantage estimators, credit
assignments, a learning rate ten times larger, and sampled or hidden simulated responses (@tab-masked,
@tab-feedback). All lie on one curve of relevance, and one of predicted reward, against distance. *RL settings choose
a point on the frontier; only the reward function moves the frontier.*

*Equal relevance.* @tab-equal-relevance reads each method's curve at a fixed loss of relevance: the largest predicted
gain it reaches while staying within 1% or 5% of the base recommender's NDCG\@10, and the logged share at that point.
Between two runs of a method the curve is taken to be the straight line, which is exactly what a policy achieves
that serves one run's lists to a random part of the users and the other run's to the rest. On KION, RL on the share
gains 4.2% of predicted share within 1% of relevance and 14.1% within 5%, and 1.8% and 3.7% of logged share. Adding
ten times the logged-relevance term to the reward moves the frontier: 16.9% and 23.1% of predicted share, 8.8% and
6.7% of logged share. Re-ranking by the same reward model reaches 14.9% and 20.7%, with the largest logged gains
(15.6%), of which the ranked list it starts from contributes 11.7% with no reward at all. Off-policy REINFORCE and
liked-only fine-tuning, which never consult the reward model, gain least in predicted share. On Yambda the differences
at 1% are within the noise between seeds (the standard deviation of the base recommender's NDCG\@10 over seeds is
0.9% of it), and the methods that learn from the log lead in logged share: $+24%$ for logged relevance and $+10%$ for
off-policy REINFORCE, through repeats (@tab-repeats).

#report-table("equal_relevance", [Methods compared at equal relevance (final sessions, seed 0): the largest change of
  the predicted share (held-out model) that a method reaches without losing more than 1% or 5% of the base
  recommender's NDCG\@10, and the change of the logged share per recommended item at that point, in percent against
  the base recommender's generated lists. Between two runs of a method the curve is the straight line that a mixture
  of the two policies achieves. "–": the method does not reach that relevance.], size: 8pt) <tab-equal-relevance>

*Does it take RL?* At nearly the same predicted share as the main RL run, re-ranking with $w = 0.3$ is more relevant
(NDCG\@10 $0.1119 plus.minus 0.0013$ against $0.1041 plus.minus 0.0020$ over seeds), has a higher logged share per
recommended item ($0.0201 plus.minus 0.0001$ against $0.0176 plus.minus 0.0002$), covers more of the catalogue, and
$9.0% plus.minus 0.3$ of its gain is personal against $0.2% plus.minus 0.5$ for RL (@tab-alternatives,
@fig-methods). Most of the difference in relevance is a head start: re-ranking begins from the ranked lists of the
base recommender and RL from its generated lists, which are 4% less relevant. Measured from its own starting point,
each method pays about the same relevance for the same predicted share ($-7.6%$ of NDCG\@10 for $+0.08$ of share by
re-ranking, $-10.7%$ for $+0.10$ by RL). The logged share is the axis on which re-ranking moves and RL does not. A
reward for the whole list, under which an item's worth depends on its neighbours, does not change the picture: greedy
re-ranking reaches a higher list reward than RL at less cost in relevance (@tab-list). On Yambda off-policy
REINFORCE on logged sessions raises NDCG\@10 by 6% ($0.1256 plus.minus 0.0009$) and the logged share by 10%, and
re-ranking is less relevant than RL ($0.1155 plus.minus 0.0008$ against $0.1190 plus.minus 0.0015$), because the
ranked lists it starts from are less relevant than the generated ones there.

#figure(
  image("figures/methods.svg", width: 100%),
  caption: [Methods compared at the predicted share they reach (held-out model): relevance, the logged share per
    recommended item, and coverage. Points of a line are one method at several strengths. RL and the fine-tuning
    methods start from the generated lists of the base recommender, re-ranking from its ranked lists.],
) <fig-methods>

#report-table("alternatives", [Ways of raising the reward without on-policy RL. Re-ranking starts from the ranked
  lists of the base recommender, every other method from its generated lists.], size: 8pt) <tab-alternatives>

*Rewards that move the frontier.* Logged relevance is the reward term that moves it upwards. On KION the share plus
ten times logged relevance at $beta = 0.3$ improves on the base recommender in relevance (NDCG\@10
$0.1202 plus.minus 0.0007$ against $0.1177 plus.minus 0.0008$ over seeds), predicted share ($+10%$) and logged share
($+7%$) at once; without the anchor its coverage falls from 0.50 to 0.11 (@tab-relevance). On Yambda logged relevance
alone, with no anchor, raises NDCG\@10 by 23% ($0.1453 plus.minus 0.0014$), the largest gain in relevance of the
study, by replaying the user's history: relevance on new tracks falls from 0.0260 to 0.0198 and the share of
recommendations taken from the history rises from 51% to 62% (@tab-repeats). It is a legitimate gain on this metric
and not better recommendation of new music. A reward neutral to item length, the relative share, moves the frontier
sideways: the drift towards films disappears (the mean duration of a recommended title rises from 5.7 to 6.7 hours
where the share reward took it to 3.4), relevance is kept ($-0.4%$ at $beta = 0.3$), and almost nothing is gained,
because the part of the reward that depends on the user is small (@tab-rewards). A reward that subtracts what the
same model predicts for another user cannot collapse onto one list even without an anchor (pair overlap 0.33
against 0.98 on KION), and raises the personal part slightly (@tab-personal).

#report-table("rewards", [Rewards that an item cannot earn on its own account, and logged relevance. "Rel. share" is
  the held-out model's prediction of the share relative to the item's usual (@eq-relative); 0.5 is "as usual".],
  size: 8pt) <tab-rewards>

#answer[Q5][RL settings choose a point on the frontier; only the reward function moves the frontier. The anchor
  weight, the KL budget, the estimator, the credit assignment and the simulated responses all move the policy along
  one curve of relevance against reward. Within 1% of the base recommender's NDCG\@10, RL on the share gains 4% of
  predicted share on KION, the same RL with logged relevance in the reward 17%, and re-ranking by the same reward
  model, which needs no training, 15%, with the largest logged gain. Logged relevance is the reward term that improves
  logged outcomes, and on Yambda it does so by replaying the user's history.]

= Discussion <sec-discussion>

*What the instruments add.* The usual defence against reward hacking is a second model, and here it fails in an
instructive way: it catches the exploitation that is specific to one set of weights (0.84 against 0.69 without an
anchor) and misses the exploitation that any model of this kind invites. Both reward models have learned, correctly,
that some titles are finished more often than others, and a policy that serves those titles to everyone raises both.
The instruments that expose this are cheap: score the same lists for another user, split the policy's change into a
part per item and the rest, and count what the log says about the recommended items the user reached.

*What RL added.* For a reward that is a sum of per-item predictions and is mostly determined by the item, the
optimum of the anchored objective is close to a re-ranking of $pi_0$ by item quality, and a policy gradient is an
expensive way to approximate it: a fine-tuning run costs minutes of GPU time per operating point, a re-ranking
weight costs nothing. The estimator comparison and the credit ablations say the same from the inside: at an equal
distance from $pi_0$, nothing that distinguishes one policy-gradient method from another mattered. The sequential
side of RL mattered once, for a reward paid after the whole list, and there it served to exploit the reward model.

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
    [An anchor weight is not comparable, a KL distance is], [Conclusive for this setup, in three seeds: a statement
      about the optimiser], [Nothing],
    [The predicted gain is impersonal], [Conclusive as a statement about the reward model's predictions],
      [Whether real responses to the fine-tuned lists depend on the user more than predicted],
    [The predicted gain on KION is not a gain in consumed share], [Suggestive: the logged share per recommended item
      rises by 1%, but only for the items users reached on their own], [The consumed share of served
      recommendations, the decisive measurement],
    [Users with short histories lose most], [Measured on logged futures], [Whether they also consume less],
    [Re-ranking is at least as good as RL], [Under the same reward model and the same logged checks],
      [Real responses; serving cost of scoring 100 candidates against one generation pass],
    [Long-term rewards buy little; adoption is the exception], [Weak: return and adoption are counterfactual, and
      adoption is not observable in a test week], [Return and adoption rates under each policy],
    [Generated lists gain after a hit and lose after a miss], [Conclusive for logged futures], [Whether users who
      skip an item are served worse lists in practice],
  ),
  caption: [What the offline evidence settles and what needs an online experiment.],
) <tab-offline>

== An online experiment <sec-online>

The claims that the offline evidence leaves open (@tab-offline) are about responses to lists the log never showed.
They need an experiment that serves the lists.

- *Arms.* (A) The base recommender's generated lists. (B) The RL policy at the operating point selected on validation
  requests ($beta = 1$ on KION, $beta = 0.05$ on Yambda). (C) Re-ranking by the reward model at the predicted share
  of B. (D) On the music service, RL on the adoption reward at $beta = 0.3$. B against A measures what fine-tuning
  does to users; C against B whether RL is needed; D against A whether a long-term reward is real.
- *Unit and assignment.* Users, assigned by a hash of the user identifier, so that each user sees one policy
  throughout; new users are assigned on arrival.
- *Duration.* Four weeks: one week for exposure to settle, then fourteen days of adoption window for the first plays
  of the second week, and one more week for returns.
- *Primary metrics.* Per user: the consumed share of the recommended items that were served, which decides Q1; the
  return rate within the return window (a week on KION, a day on Yambda); on Yambda the share of first plays of
  recommended tracks that are played again in a later session within 14 days.
- *Guardrails.* Recommendation-driven consumption per user, likes and dislikes per recommended item, catalogue
  coverage and the Gini coefficient of impressions, the share of new items, and every metric for the quartile of
  users with the shortest histories, who lose most offline (@tab-history).
- *Analysis.* Differences of per-user means with a CUPED adjustment on the same metric in the four weeks before the
  experiment, intervals by a bootstrap over users, and the history quartiles as pre-registered segments. B is better
  than A if the consumed share of served recommendations rises and the return rate does not fall; D is better than A
  if adoption rises and the consumed share does not fall.
- *Size.* For a per-user binary outcome with a base rate near 0.7, a relative change of 1% (0.7 points) needs about
  $2 (1.96 + 0.84)^2 dot 0.7 dot 0.3 \/ 0.007^2 approx 67,000$ users per arm (two-sided $alpha = 0.05$, power
  0.8). The KION test week alone has 60,000 users with a request.

*Practical implications.* Report the KL distance a fine-tuned policy ended at, not only the weight that was set.
Before fine-tuning against a feedback model, re-rank by the same model and by item means: if the frontier is the
same, the model's contribution is knowledge about items. Score lists for another user, and decompose the policy's
change into a part per item. Prefer a reward that an item cannot earn on its own account, and expect it to be worth
less than the raw signal suggested. Simulate generated items with a reward model trained to read unknown responses.

= Limitations <sec-limitations>

- *Three seeds, for the core only.* The central comparisons have three seeds; the long-term rewards, the reward
  models with hidden responses and the analyses by position, user and item have one. A standard deviation over three
  seeds is itself uncertain.
- *Everything is offline.* @tab-offline lists what this leaves open, and @sec-online how to close it. The logged check
  sees only recommended items that users reached without the recommender's help.
- *Adoption is not verified.* Its label needs fourteen days after a first play, which the test week does not have.
  Training-period first plays of held-out users, or a longer test period, would allow a check.
- *The Yambda grids are narrower than KION's.* Sampled feedback, the list reward, and the retention and personal
  rewards were run at one anchor weight each.
- *The base recommender on Yambda is below a replay heuristic,* and gains in relevance there are gains over this
  recommender.
- *The main run's weight is not a blind choice* (@sec-setup).
- *The explanation of why weights differ between estimators is a hypothesis* that no run isolates.
- *The personal part is the reward model's own belief* about how much of its prediction depends on the user.
- *Re-ranking and RL start from different lists,* ranked and generated; on KION this favours re-ranking by about 4%.
- *Equal-relevance curves interpolate between runs.* The straight line is achievable by mixing two policies, but a
  method may reach more between two of its runs than the line shows.
- *Baselines are untuned.*

= Conclusion

We fine-tuned a generative recommender against a learned reward model in the setting with the fewest protections, on
a video and a music log, with three seeds for every central comparison, and answered five questions.

*Q1. How does behaviour change, and why?* The policy learned which items everybody finishes and serves them to
everyone; the reward model's predictions differ between items more than between users, and a ranking of items is
what a policy gradient finds in them. *Q2. Is relevance kept? Does everyone get the same items?* On KION relevance
falls by a tenth, most for the users with the least history, and impressions concentrate further; on Yambda both are
kept; the KL distance decides, and without an anchor everyone receives the same list. *Q3. Long-term goals.*
Rewards for return, likes and adoption buy a few percent of their predictions at a moderate distance; the large gain
of a reward paid after the list was reward hacking, and none of it can be confirmed offline. *Q4. Generating
several steps ahead.* A generated list gains after a hit and loses after a miss; simulated responses bias the reward
of later steps unless the reward model was trained to read unknown ones, and crediting items with the rest of the
list helps no per-item reward. *Q5. Trade-off between relevance and reward.* RL settings choose a point on the
frontier; only the reward function moves the frontier, and re-ranking by the same reward model reaches more reward
at equal relevance than RL.

What moved logged outcomes came from the log: a logged-relevance term inside the reward, and off-policy learning
from logged sessions. For a reward model to be worth optimising with RL, it has to know something about the user that
cannot be looked up per item, and the reward has to be built so that only that knowledge pays. In our data that part
was small. Whether real users respond to the fine-tuned lists as the log suggests is the question of the experiment
in @sec-online.

#heading(numbering: none, level: 2)[Code and data availability]

The code, the configuration and every table of every session are in the repository
`github.com/daniilantonovv/RecSysMaxxing` (`results/`); the tables of this paper are typeset from those files by
`report/tables.py` and the SVG figures by `report/figures.py`.

#bibliography("references.bib", style: "ieee", title: "References")

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
    [Reactions, adoption], [not recorded], [latest play within a day; 14 days],
    [Items of the history in a list], [blocked], [allowed],
    table.cell(colspan: 3, emph[Final sessions]),
    [Anchor weights], [0, 0.1, 0.2, 0.3, 0.5, 1], [0, 0.03, 0.05, 0.1, 0.3],
    [KL budgets], [0.1, 0.5], [0.2, 0.5],
    [Leave-one-out at weights], [0.1, 0.3], [0.03, 0.1],
    [Credit ablations], [weights 0.1, 0.3; both budgets], [both budgets],
    [Relative-share reward at weights], [0, 0.1, 0.3], [0.03, 0.1],
    [Logged relevance], [alone at 0; with the share at 0, 0.03, 0.1, 0.3], [alone at 0; with the share at 0, 0.1],
    [Sampled feedback, list reward at weights], [0.1, 0.3], [0.1],
    [Retention rewards], [session, return, share + return at 0.1, 0.3], [return at 0.1],
    [Personal rewards at weights], [0, 0.1, 0.3], [0.1],
    table.cell(colspan: 3, emph[Review sessions]),
    [Anchor weights], [0, 0.1, 0.3, 1], [0.05, 0.1, 0.3],
    [Long-term rewards at 0.1, 0.3], [return after the list, also with discount 0], [like − dislike, adoption, return
      after the list, also with discount 0],
    [Hidden responses: RL at weights], [0.1, 0.3], [0.1, 0.3],
    table.cell(colspan: 3, emph[Seed sessions (seeds 1 and 2)]),
    [Anchor weights], [0.1, 0.3, 0.5, 1], [0.03, 0.1, 0.3],
    [KL budget; leave-one-out; credit], [0.1 nats; at 0.3 and at 0.1 nats; at 0.1 nats], [0.5 nats; --; at 0.5
      nats],
    [Alternatives], [liked-only, off-policy REINFORCE ($kappa = 0.5$), re-ranking], [the same],
    [Rewards], [logged relevance alone and with the share at 0.3; relative share at 0.3], [logged relevance alone],
  ),
  caption: [What the sessions ran; for Yambda's final session, a first session and its continuation together.
    Fine-tuning: 512 lists per iteration, 4 gradient steps per batch, 2,000 iterations for the main run and 1,000 for
    every other run.],
) <tab-config>

#report-table("stage_minutes", [Wall-clock minutes by stage of the final sessions on one RTX PRO 6000 (96 GB). One
  fine-tuning run of 1,000 iterations takes 6.6 minutes on KION and 18.5 on Yambda, with its evaluation on test and
  validation requests. The review sessions took 99 and 419 minutes, the seed sessions 112 and about 270 minutes per
  seed.]) <tab-time>

#report-table("selection", [Selection of the anchor weight on validation requests: the highest predicted share
  among the runs within 5% of the base recommender's NDCG\@10.]) <tab-selection>

= Further results

#report-table("feedback", [Fine-tuning with expected and with sampled simulated feedback, and the held-out
  predicted share of each policy under three assumptions about the response to earlier items of the list.])
<tab-feedback>

#report-table("masked_feedback", [Fine-tuning against the reward model with expected responses for generated items
  and against the reward model trained with hidden responses, fed "unknown" for them (review sessions).])
<tab-masked>

#report-table("position_hits", [Hit rate by position of the list, for the base recommender's ranked and generated
  lists (review sessions).]) <tab-positions>

#report-table("feedback_assumptions", [The reward model on the users' real next five items: the logged share and
  the AUC with logged responses, the bias of the prediction under each assumption about earlier responses, and the
  AUC under expected and unknown responses (review sessions).], size: 8pt) <tab-assumptions>

#report-table("reaction_heads", [Yambda: the like and dislike heads of the held-out reward model on test events.])
<tab-reactions>

#report-table("retention_rewards", [Retention rewards. "Continues" and "returns" are the held-out model's
  predictions for the generated lists.]) <tab-retention>

#report-table("relevance_rewards", [Logged relevance in the reward, alone and added to the share.], size: 8pt)
<tab-relevance>

#report-table("repeats", [Yambda: relevance on tracks from the visible history and on new tracks.]) <tab-repeats>

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

The pipeline was completed seven times before the final sessions (version 2 of this draft describes those runs in
full). They differ from the present sessions in the reward models, which had fewer heads, and in the instruments
available, and they agree with them where they overlap. The main run on KION was made three times with the same base
recommender and other reward models: NDCG\@10 changed by $-11.4%$, $-11.4%$ and $-11.1%$ and the held-out predicted
share by $+22.8%$, $+23.1%$ and $+23.1%$, within the spread of the three seeds here ($-11.5% plus.minus 1.3$ and
$+23.9% plus.minus 2.1$). On Yambda two earlier runs gave $+1.1%$ and $+1.8%$ of predicted share, against
$+1.67% plus.minus 0.01$ here. The first two runs used REINFORCE with a leave-one-out baseline on Yambda, where the
policy held at $beta = 0.3$ and collapsed at $beta = 0.1$, while the value-model runs held at $0.1$ and collapsed at
$0.03$; @tab-estimators repeats the comparison with the learning rate and the number of lists per history held equal
and confirms that the estimator alone moves the transition. Two findings of the earlier runs are not repeated here: a
choice head trained to tell logged items from popularity-distributed replacements did not work as a relevance signal
inside the reward, and a reward weighted by item duration moved the policy along the duration axis in the direction
the normalisation favoured.
