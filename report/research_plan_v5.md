# Research plan v5: mechanisms, inference-time alignment, learning from the log

Written 21 October 2026, before the runs. Each hypothesis states the result that would refute it, so the
outcome can be read against a prediction rather than explained after the fact.

## Where v4 left us

1. RL against a learned reward raises the predicted share, but 78% of the policy's change on KION is one shift per
   item. The gain is almost entirely impersonal, and the logged surplus falls.
2. Re-ranking by `r + w·log π0` is the top-K of the closed-form optimum `π* ∝ π0·exp(r/β)`, and it beats RL at equal
   relevance.
3. RL settings only move the policy along one frontier. Only the reward function moves the frontier: logged relevance
   moves it, and re-ranking by the relative share is the only method that improved a length-neutral logged outcome.
4. A generated list gains after a hit and loses after a miss. Simulated responses bias the reward of later steps.
5. Long-term rewards bought little, and KION's return after the list follows its logging format.

All five are stated about *what happened*. The open questions are about *why*, and whether today's methods change
the answer.

## Revised questions

### RQ1. Is what RL learns a re-weighting of the catalogue, and is that set by the reward or by the optimiser? (Q1)

| Test | Method | Prediction | Refuted if |
|---|---|---|---|
| Restricted capacity | RL that may change only one score offset per item (V parameters); RL with rank-1 and rank-8 LoRA updates of the transformer; full RL. All at KL budgets 0.1 and 0.5 | KION: item offsets reach ≥ 80% of full RL's predicted gain at equal KL. Rank-1 LoRA matches full RL, as reported for language models ("LoRA Without Regret") | Offsets reach < 50% of the gain at equal KL |
| Surgery | Average the main run's log-probability change per item. Add it to π0 ("base + item offsets"); subtract it from RL ("RL − offsets") | "base + offsets" reproduces RL's gain and its NDCG loss. "RL − offsets" stays near π0's relevance with a small gain | "RL − offsets" keeps most of the gain |
| Prediction before RL | The share of within-history variation that one effect per item explains, in the reward model's predictions on π0's candidates. By the closed form, that is the optimum's change at any β. Compared with each fine-tuned policy's share | KION: the policy's share exceeds the reward's, i.e. RL amplifies the item part. Yambda: lower shares for both | The policy's share ≤ the reward's on both datasets: then the per-item shift is set by the reward, not the optimiser |
| Advantage scale | σ_A, now logged in every run | value-model σ_A ≈ 3× leave-one-out σ_A, which explains the 2.9× weight gap at equal KL | The ratio is near 1 |

### RQ2. Does reaching RL's optimum need training at all? (Q4, Q5, Q3)

The anchored objective has a closed form per position: `π*(a|s) ∝ π0(a|s)·exp(Q(s,a)/β)`, with
`Q = r + γ·U`. We decode with it directly over π0's top-50 candidates, one position at a time, after the
items already chosen. U is learned from π0's own rollouts: controlled decoding (Mudgal et al., 2024), the first step
of Q♯ (Zhou et al., NeurIPS 2025).

| Variant | What it isolates |
|---|---|
| re-ranking once, real history (v4) | the closed form, γ = 0, no context update |
| guided, γ = 0 | the same, with each position decided after the items already chosen |
| guided, γ = 0.9 | plus look-ahead |

- **Predictions:**
  - Share reward: guided γ = 0 is at least as good as re-ranking, which beats RL at equal NDCG@10, and look-ahead adds
    nothing.
  - Return after the list (KION) and adoption (Yambda): look-ahead reaches more of the reward than γ = 0 at equal
    relevance, and pays less relevance for it than RL on the same reward.
- **Refuted if** RL beats guided decoding at equal relevance on the share reward. Then the training itself adds
  something that inference cannot compute.

### RQ3. Can the log alone move the frontier? (Q5)

Every RL variant moved along one frontier. Logged relevance moved it. We test two current preference-learning
methods that use logged responses only:

- **Exponential reward-weighted fine-tuning** (Chidambaram et al., 2026): next-item training with weights
  `exp(r/τ)`, where r is the logged relative share or the logged share, and τ ∈ {0.3, 1}.
- **Within-user DPO**: pairs of items that one user consumed after the same history, one above and one below the
  usual response for that item. Neither item is preferred for having been started, nor for being short. β ∈ {0.1, 0.5}.

| Prediction | Refuted if |
|---|---|
| DPO and reward weighting by the relative share raise the logged surplus over π0 with NDCG@10 within 2% | The surplus does not move. Then the personal signal in the log is too weak to learn from at this scale |
| Reward weighting by the plain share reproduces RL's drift towards films | It does not drift. Then the drift needs the reward model's extrapolation, not just the share's definition |

### RQ4. Must a list read its own unconfirmed items? (Q4)

One head per position (multi-token prediction, Gloeckle et al. 2024). Head k predicts the item k events ahead from the
real history only.

- **Prediction:** after a miss, the hit rate equals the ranked list's; after a hit, it falls below the generated
  list's. Net NDCG@10 is at least the generated list's on KION, where misses dominate, and below it on Yambda.
- **Refuted if** the heads lose to generated lists after a miss.

### RQ5. Cheaper operating points (Q5)

- **Weight interpolation and extrapolation** of the main run, `θ0 + a(θ_RL − θ0)` for a ∈ {0.25, …, 2} (WiSE-FT,
  WARP, ExPO), against the KL sweep at equal KL along each policy's own lists.
  - **Prediction:** on or above the sweep's frontier.
  - **Refuted if** below it.
- **GRPO** at the same KL budgets as the value model and leave-one-out.
  - **Prediction:** the same frontier.

### Q3 check that needs no model

The return rate after an event, by the duration of its own item and of the next item.

- **Prediction:** on KION it falls with the *next* item's duration far more than with its own. On Yambda neither
  dominates.

## Design rules

- Methods are compared at an equal KL budget or at equal NDCG@10 (equal-relevance reading), never at an equal β.
- Every new run logs σ_A and its KL. Inference-time methods report the KL along their own lists.
- The test period is read only after all runs finish. Operating points for the paper are chosen on validation.
- Any result that changes a claim of the paper is repeated with seeds 1 and 2 before it enters the text.

## Runs

| Notebook | Stages | Estimated time on the RTX PRO 6000 |
|---|---|---|
| `6_frontier_kion` | base models, main run, anchor sweep (0.1, 0.3, 1), KL budgets (0.1, 0.5), leave-one-out, re-ranking and alternatives, RL on the return after the list, then all RQ1–RQ5 stages | about 3.5 h |
| `6_frontier_yambda` | the same with reduced grids; adoption instead of the return after the list | about 8 h |

Run KION first. Then repeat the decisive comparisons with seeds 1 and 2 before Yambda. For RQ1 those are item
offsets against full RL; for RQ2, guided decoding against RL; for RQ3, DPO on the logged surplus.

## What each outcome does to the paper

- **RQ1 confirmed:** Q1 gains a causal answer. RL learns a catalogue re-weighting because the optimiser favours
  directions shared by all histories, and a reward model that knows more about users does not change that.
- **RQ2 confirmed:** Q5 becomes "RL is not needed to reach the anchored optimum". Look-ahead earns its place only for
  rewards of the whole list, which answers Q4's "several steps ahead" from the planning side.
- **RQ3 confirmed:** the paper gains a positive method for the task: a logged-preference objective that improves a
  length-neutral logged outcome at no cost in relevance.
- **RQ4:** the clean answer to "how does generating several steps ahead affect the following recommendations".

## Outcomes on KION, seed 0 (22 October)

Session `6_frontier_kion`, reports in `results/kion_frontier`. Changes are relative to π0 unless stated; "share" is
the held-out reward model's predicted played ratio. The results that change a claim of the paper are repeated with
seeds 1 and 2 in `7_frontier_seeds_kion_1_2`; the return-label check is a count over the log and needs no seeds.

| Test | Result | Verdict |
|---|---|---|
| Restricted capacity | Item offsets reach 94% (budget 0.1) and 91% (budget 0.5, at KL 0.42 against full RL's 0.54) of full RL's share gain, NDCG@10 within 0.002 of it. Rank-1 LoRA: 85% and 100%; rank-8: 89% and 92% | Confirmed |
| Surgery | "base + RL's item offsets": 89% of the gain, NDCG@10 −10.4% (RL −11.0%). "RL − its item offsets": 19% of the gain, NDCG@10 −0.4%, π0's logged surplus, personal part 0.0169 (π0 0.0137, RL 0.0145) | Confirmed. The per-item part lowers personalisation and the rest raises it; in RL the two cancel |
| Prediction before RL | Per-item share of within-history variation: reward model 0.778, RL 0.763, anchor sweep 0.69–0.75, rank-1/8 0.69–0.72, item offsets 0.89 (their ceiling); only GRPO at budget 0.1 exceeds the reward (0.787) | Refuted on KION (Yambda pending): the per-item shift is set by the reward, not amplified by the optimiser |
| Advantage scale | σ_A 0.93–0.96 with the value model against 0.43–0.56 with leave-one-out: 2.2× at budget 0.1 and 1.7× at 0.5, while the weights that hold those budgets differ 2.9× and 2.7× | Partly: σ_A accounts for 55–75% of the weight gap on a log scale |
| GRPO | At equal KL, 7–10% more share gain than the value model with no lower NDCG@10 (0.1051 vs 0.1018 at budget 0.1, 0.0827 vs 0.0806 at 0.5) | On the frontier or slightly above |
| Guided decoding, share | Share at RL's NDCG@10 (0.1044), interpolated along each curve: re-ranking once 0.574, guided γ = 0 0.566, RL 0.555, guided γ = 0.9 0.548 | Partly: guidance beats RL without training, but deciding each position after the chosen items is worse than re-ranking once, and look-ahead does not help |
| Guided decoding, return after the list | 0.729 → 0.742 at γ = 0 and 0.753 at γ = 0.9 (strength 0.1, NDCG@10 −2.3%); RL on this reward 0.859 at −70% | Confirmed, small: look-ahead doubles the gain of γ = 0 for −2.3% NDCG@10, where RL pays −70% for a gain five times larger |
| Reward-weighted SFT, relative ratio | τ = 0.3: NDCG@10 +1.8% (95% CI +1.1 to +2.5%), above-usual NDCG@10 +2.8%, logged surplus +25% (+18 to +31%), personal part +20%, coverage +7%, share −3%. τ = 1: +1.3%, +12%, +8% | Confirmed and exceeded: the first method that raises relevance, the logged surplus, personalisation and coverage together |
| Within-user DPO | β = 0.5: surplus +39%, personal part +35%, NDCG@10 −4.8%, coverage −21%. β = 0.1: NDCG@10 −21%, coverage −58% | Surplus confirmed; "NDCG@10 within 2%" refuted |
| Reward-weighted SFT, played ratio | τ = 0.3: share +20% (RL +22%), NDCG@10 −13.5%, surplus −74%, mean title 10.2k s (RL 12.3k, π0 20.3k) | Confirmed: RL's drift follows from the share's definition, without a reward model |
| One head per position | NDCG@10 0.1248 against generated 0.1173 (+6.4%) and ranked 0.1220 (+2.3%); hit rate after a miss above both at every position, after a hit below the generated list's; best logged share and surplus of the three, coverage 0.48 (ranked 0.57) | Confirmed, and beats ranked lists too |
| Weight interpolation | ×0.25–×2 lie 0.0003–0.0015 NDCG@10 above the anchor sweep at equal share; ×1.5 and ×2 extend the main run without collapse | Confirmed: the frontier from one run, no training |
| Return label | Return rate 0.68–0.77 in every decile of the duration of the event's own title and of the next one; correlation with log duration −0.03 (own) and −0.05 (next) | Refuted: no logging artifact. RL's predicted 0.86 lies above every group; the paper's Q3 is corrected |

What this changes:

- **Q1:** RL learns a re-weighting of the catalogue because the reward's optimum is one: 78% of the within-history
  variation of the predicted reward is one effect per item. The optimiser neither creates nor amplifies it.
- **Q2 and Q5:** the positive result is reward-weighted fine-tuning on the relative ratio: the logged data alone move
  the frontier where every reward-model method only moved along it.
- **Q4:** a list that never reads its own unconfirmed items is better than both generated and ranked lists. Guided
  decoding shows the same from the other side: conditioning on simulated responses loses to re-ranking once.
- **Q3:** the return after the list is an extrapolation of both reward models, not a property of the log. Look-ahead
  is the efficient way to pursue a reward of the whole list.

## KION over three seeds, and Yambda (22 October)

Sessions `7_frontier_seeds_kion_1_2` (`results/kion_frontier_seeds`; seed 0 is `results/kion_frontier`) and
`6_frontier_yambda` (`results/yambda_frontier`, one seed, without the like and dislike labels, which none of its rewards
use). KION: mean ± standard deviation over seeds 0, 1, 2, each quantity computed within its seed.

| Test | KION, three seeds | Yambda, seed 0 |
|---|---|---|
| Item offsets, share of full RL's gain | 93 ± 3% (budget 0.1), 89 ± 4% (0.5); NDCG@10 within 0.002 of full RL | 34% at budget 0.5: the offsets stop at KL 0.25 with the anchor weight at zero, where full RL is about twice as far up |
| Rank-1 LoRA, share of the gain | 87 ± 5% and 93 ± 6%, NDCG@10 0.004–0.005 above full RL | 95% at budget 0.5, on the frontier |
| Surgery | base + item offsets: 88 ± 2% of the gain, NDCG@10 −11.0 ± 1.8% (RL −11.6 ± 1.5%), personal part −16 ± 3%. RL − item offsets: 21 ± 2% of the gain, NDCG@10 −0.1 ± 0.4%, personal part +20 ± 3% | The main run moved little (KL 0.037); its gain splits half and half |
| Per-item share | reward 0.794 ± 0.014, RL 0.771 ± 0.007; every RL run below the reward in every seed | reward 0.423, every RL run above it (0.46–0.55) |
| Guided decoding, share | share at RL's NDCG@10: re-ranking 0.564 ± 0.025 > guided 0.556 ± 0.024 > RL 0.548 ± 0.023, the same order in every seed | strength 0.3: NDCG@10 +2.1% at RL's share; strength 0.1: share +3.7% and logged surplus +10% at the base relevance. Guided beats both RL and re-ranking |
| Guided decoding, long-term reward | return after the list (seed 0): see above | adoption, strength 0.3: +6.5% at NDCG@10 +3.8%; with look-ahead +9.7% at +4.2%. Strength 0.1: +20% and +25% at −0.8% and −3.0% |
| Reward-weighted SFT, relative ratio, τ = 0.3 | NDCG@10 +1.9 ± 0.5%, surplus +28 ± 3%, personal +19 ± 1%, coverage +5 ± 2%, share −3.8%; NDCG@10 and surplus significant in every seed | NDCG@10 +0.5% (n.s.), surplus +13% (significant), the rest n.s.; τ = 1: above-usual NDCG@10 +1.8%, logged share +1.9% (significant). Liked-only fine-tuning does better here: NDCG@10 +2.2% |
| Within-user DPO | β = 0.5: surplus +42 ± 5%, personal +35 ± 6%, NDCG@10 −3.7 ± 1.6%, coverage −22 ± 2% | β = 0.1: surplus +32%, NDCG@10 −24% |
| Reward-weighted SFT, played ratio | share +20.8 ± 0.5% (RL +24.0 ± 2.0%), NDCG@10 −13.4 ± 0.3%, surplus −75 ± 1%, mean title length −49% (RL −40%) | not run |
| One head per position | NDCG@10 +6.3 ± 0.4% vs generated, +2.5 ± 0.1% vs ranked; logged surplus +25 ± 1% vs generated | NDCG@10 +0.4% vs generated, +3.4% vs ranked; after a miss level with ranked, after a hit level with ranked and far below generated, as predicted; personal part +10%, coverage −18% |
| Weight interpolation | ×0.25, ×0.5, ×0.75: +0.0001 ± 0.0002, +0.0006 ± 0.0002, +0.0009 ± 0.0002 NDCG@10 against the anchor sweep at equal share; ×1.5 and ×2: +0.0005 ± 0.0013 and −0.0009 ± 0.0014 | ×0.5 on the sweep; ×1.5 and ×2 about 0.003 below the run at β = 0.1 |
| GRPO | (seed 0) 7–10% more gain than the value model at equal KL | 76% of the value model's gain at budget 0.5, NDCG@10 0.002 above the frontier |
| Return label | (seed 0) refuted | 0.914–0.922 in every decile, correlation −0.007 and −0.005: "neither dominates", as predicted |

Verdicts after both datasets:

- **RQ1:** the per-item re-weighting is set by the reward where the reward is mostly per item (KION) and amplified by
  the optimiser where it is not (Yambda): the policy's per-item share sits at the reward's on KION and above it on
  Yambda. Item offsets suffice exactly where the reward is per item. Rank-1 LoRA matches full RL on both.
- **RQ2:** the closed form at inference matches or beats RL on both datasets. Whether to decide each position after
  the chosen items follows the list format that suits the dataset: on Yambda, where generated lists beat ranked ones,
  guided decoding beats re-ranking; on KION re-ranking wins. Look-ahead helps only rewards that depend on what follows
  (adoption, the return after the list).
- **RQ3:** reward-weighted fine-tuning on the relative ratio is the robust positive result on KION and a small one on
  Yambda; DPO buys a larger surplus with relevance and coverage.
- **RQ4:** one head per position removes the after-miss penalty on both datasets; net it is the best list on KION and
  level with generation on Yambda, where hits after hits matter more.
- **RQ5:** interpolation stays on the frontier; extrapolation is no better than training further.

## Control for continued training (written 22 October, before the run)

Notebook `8_control_kion_0_1_2`, stage `control_for_continued_training`; reports will be in `results/kion_control`.

**Why.** Reward-weighted fine-tuning on the logged relative share (RW-SFT, τ = 0.3) is the paper's positive result:
NDCG@10 +1.9 ± 0.5%, logged surplus +28 ± 3%, personal part +19 ± 1%, coverage +5 ± 2% (KION, generated lists,
three seeds). Two things were not controlled.

1. **The epoch itself.** RW-SFT trains the base recommender for one more epoch, warming up to a learning rate of 1e-4
   and decaying it to zero. The base recommender kept its epoch 9 of 11 (KION, seed 0), taken at about 80% of its peak
   learning rate of 1e-3, so an annealing epoch may raise relevance by itself. Already τ = 1, where the weights differ
   at most e ≈ 2.7-fold, gave +1.3% NDCG@10 against +1.8% at τ = 0.3 (seed 0). RW-SFT's own next-item loss on the
   validation period rose (6.612 against the base recommender's 6.606), so whatever raised NDCG@10 was not a better
   fit of the next item.
2. **The list format.** RW-SFT was compared on generated lists only. The base recommender's ranked lists are 4% more
   relevant than its generated ones (NDCG@10 0.1220 against 0.1173), and more relevant than RW-SFT's generated lists
   (0.1193).

**Test.** *Continued training* is the same epoch with every logged item weighing one: the same code, optimiser,
learning rate, batch size and seed, with only the weights removed. The base recommender, continued training, RW-SFT
at τ = 0.3 and 1, and within-user DPO at β = 0.5 are each judged on ranked and generated lists. Seeds 0, 1 and 2 run in
one session, each with its own base recommender and reward models; every comparison is user by user within a seed.

| Hypothesis | Prediction | Refuted if |
|---|---|---|
| H1. The epoch alone raises relevance | Continued training raises NDCG@10 of generated lists over the base recommender by at least half of RW-SFT's gain (≥ +0.9%), significant in every seed, and lowers the next-item loss on the validation period | Continued training changes NDCG@10 by less than +0.5% |
| H2. The weights add the length-neutral response | RW-SFT τ = 0.3 against continued training: logged surplus ≥ +15% and personal part ≥ +10%, the intervals excluding zero in every seed and on both kinds of list; NDCG@10 within ±1% | The surplus over continued training is not significant in every seed |
| H3. Dose and response | The logged surplus rises from continued training through τ = 1 to τ = 0.3 | Another order |
| H4. Ranked lists | RW-SFT's ranked lists are at least as relevant as the base recommender's ranked lists, with a higher surplus | They lose more than 1% of NDCG@10 against the base recommender's ranked lists |
| H5. DPO | Against continued training, DPO keeps a larger surplus gain than RW-SFT and loses more relevance | DPO's surplus over continued training is not significant |

**What each outcome changes in the paper.**

- *H1 confirmed:* Q5's positive result becomes "at the relevance of continued training, RW-SFT raises the logged surplus
  by X% and the personal part by Y%". The NDCG@10 gain is credited to the extra epoch, and the paper says that the
  base recommender had not converged in its learning rate, so every change relative to π0 carries this slack. The
  base recommender itself stays as it is: every session used it.
- *H1 refuted:* the claim stands, now with "not an effect of continued training".
- *H2 refuted:* the positive result is withdrawn: RW-SFT is then continued training.
- *H4 refuted:* the gain holds over generated lists only. On KION the list format then decides relevance more than
  the fine-tuning does, in line with Q4 (one head per position).

## Control for continued training: outcome (23 October)

Session `8_control_kion_0_1_2`, reports in `results/kion_control` (`seed_0`, `seed_1`, `seed_2`, `over_seeds`); 20 minutes
per seed on the RTX PRO 6000. Mean ± standard deviation over seeds 0, 1 and 2, each change computed within its seed
(`report/tables/control.typ`). "Every seed": the 95% per-user interval of the change excludes zero in all three.

| Hypothesis | Result | Verdict |
|---|---|---|
| H1. The epoch alone raises relevance | Continued training against the base recommender: NDCG@10 +0.7 ± 0.3% on generated lists (significant in 2 seeds of 3) and +0.2 ± 0.3% on ranked lists (1 of 3); next-item loss on the validation period lower in every seed (6.606 → 6.594, 6.607 → 6.589, 6.602 → 6.592); coverage +10 ± 1% and +9 ± 1%; logged surplus +5 ± 2% (1 of 3); personal part −0.4 ± 1.2% | Partly. The epoch explains a third of RW-SFT's NDCG@10 gain on generated lists (+0.7 of +1.9) and all of its coverage gain, none of its surplus or personal part. The prediction (≥ +0.9%) is missed, the refutation (< +0.5%) not reached |
| H2. The weights add the length-neutral response | RW-SFT, τ = 0.3, against continued training: logged surplus +21 ± 4% on generated and +17 ± 0.1% on ranked lists, personal part +19 ± 2% and +15 ± 2%, NDCG@10 +1.3 ± 0.5% and +0.7 ± 0.1%, every seed; coverage −4% and −3%; predicted share −1.8% | Confirmed; the weights also add relevance beyond the epoch |
| H3. Dose and response | Logged surplus over continued training: τ = 1 +7 ± 2% and +6 ± 1%, τ = 0.3 +21% and +17% | Confirmed |
| H4. Ranked lists | RW-SFT's ranked lists against the base recommender's ranked lists: NDCG@10 +0.9 ± 0.3%, logged surplus +23 ± 5%, personal part +13 ± 1%, every seed | Confirmed |
| H5. DPO | Against continued training: logged surplus +35 ± 7% and +34 ± 4%, NDCG@10 −4.3 ± 1.6% and −3.8 ± 0.1%, coverage −30% and −18% | Confirmed |

What this changes in the paper:

- Q5's positive result is now stated against continued training: the weights add 21% of logged surplus, 19% of
  personal part and 1.3% of NDCG@10 on generated lists, and 17%, 15% and 0.7% on ranked lists, in every seed. The
  coverage gain of RW-SFT (+5% against the base recommender) belongs to the extra epoch.
- RW-SFT's ranked lists beat the base recommender's ranked lists, so the result does not depend on the list format.
- The base recommender had not converged in its learning rate: one more epoch of plain training lowers its validation
  loss in every seed and raises NDCG@10 of its generated lists by 0.7%. Every change measured against π0 carries this
  slack; it is small next to RL's −11.5% and a third of the logged methods' relevance gains.

The quick session (`0_quick_kion`, `results/kion_quick`, 55 minutes of stages) reproduced the seed-0 numbers of the
paper: NDCG@10 of generated lists 0.1173 → 0.1046 under the main run (paper 0.1173 → 0.1047), held-out predicted share
+21.5% (21.5%), reward model AUC 0.755 against 0.656 for the item mean, β = 1 chosen on validation requests, and a
personal share of the gain of 0.5%.
