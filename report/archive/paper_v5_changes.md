# Paper v5: what changed from v4

Version 5 (`report/paper_v5.typ`, 22 October) adds the frontier sessions to version 4 (`report/paper_v4.typ`,
18 October; corrected on 22 October, see `paper_v4_cuts.md`). The sessions are `results/kion_frontier`,
`results/kion_frontier_seeds` (seeds 1 and 2) and `results/yambda_frontier`. Their hypotheses and outcomes are in
`research_plan_v5.md`. Length: 19 pages, against 15. The text grew from 10 to 12 pages, the references from about
1 to 1.5, and the supporting tables from 4 to 5.

## New

- **Abstract, Table 1 and the answer boxes** are rewritten around the new answers:
  - Q1: one re-weighting of the catalogue, set by the reward on KION and amplified by the optimiser on Yambda.
  - Q4: one head per position.
  - Q5: reward-weighted fine-tuning on the logged relative share.
- **Introduction:** one sentence naming the probes.
- **Related work:** a paragraph on the methods borrowed from the post-training of language models: LoRA, "LoRA
  Without Regret", controlled decoding, Q♯, AWR, DPO, WiSE-FT, WARP, ExPO, GRPO and multi-token prediction. The
  positioning against Chidambaram et al. now ends on what this study adds.
- **Method:**
  - guided decoding and weight interpolation (under "Alternatives");
  - a paragraph "What RL learns": item offsets, rank-1 LoRA and policy surgery;
  - a paragraph "Learning from the log": reward-weighted fine-tuning, within-user DPO and one head per position;
  - GRPO as a third estimator;
  - the per-item share redefined as the within-history share. A constant per history now explains nothing, so for
    the reward's predictions it is the optimum's share at any β.
- **Setup:** the frontier sessions and their seeds.
- **Q1:**
  - "One shift per item" and "Why" became "One re-weighting of the catalogue" and "The reward or the optimiser?";
  - new Table 2 (`tables/mechanism.typ`) covers restricted fine-tuning, the parts of the main run, and the per-item
    share of the reward and of the policies.
- **Q2:** the frontier session logs σ_A, which accounts for 55–75% of the gap in weights. This replaces "a
  consistency check; σ_A was not logged". There is also a sentence on what makes the lists alike.
- **Q3:** a paragraph "Steering at inference": guided decoding by adoption on Yambda and by the return after the list
  on KION. The long-term table gains two panels (`tables/long_term_rewards_and_guidance.typ`).
- **Q4:**
  - the position figure now shows ranked lists, generated lists and one head per position from the frontier
    sessions (`figures/position_hits_with_heads.svg`; v4 keeps `figures/position_hits.svg`);
  - new text on one head per position and on guided decoding against re-ranking;
  - "Does RL use the steps?" became "Does planning help?" and adds look-ahead at inference.
- **Q5:**
  - GRPO and rank-1 LoRA on the frontier;
  - a paragraph "Does it take training?": weight interpolation and guided decoding;
  - new Figure 4 (`figures/frontier_methods.svg`);
  - a paragraph "Learning from the log";
  - new Table 5 (`tables/logged_methods.typ`).
- **Discussion, online experiment, limitations, future work and conclusion** are rewritten for the new results. The
  online arms are now: base recommender; RL at the validation KL; re-ranking; reward-weighted fine-tuning on the
  relative share; one head per position; and on Yambda guided decoding by adoption.

## Moved to the supporting tables

- the titles table (`tab-titles`);
- the equal-relevance table (`tab-equal-relevance`);
- the figure of the reward model on real sequences (`fig-assumptions`).

## Left out

- The diversity reward: the sentence "Re-ranking also beat RL on the diversity reward…" and its table `list_reward`.
- The personal reward: the sentence on its pair overlap (0.33 against 0.98) and its table `personal_rewards`.
- From "What keeps relevance": at an equal KL budget, own-reward credit is 5.6% (KION) and 12% (Yambda) more
  relevant. The Q4 paragraph "Does planning help?" keeps the numbers.
- From the reward-estimate paragraph:
  - that Yambda's first step is the hardest to predict (AUC 0.64);
  - the masked-feedback run's NDCG@10 and KL (0.106 against 0.104 at KL 0.078 against 0.080).
- From Q1: v4's per-item measure ("78% of its change", "on Yambda the item means explain 44%", review session). It is
  replaced by the frontier sessions' within-history share: RL 0.77 ± 0.01 on KION, 0.50 on Yambda.
- From the conclusion: the sentence on long-term rewards. The answer to Q3 keeps it.

## Numbers to know

- KION frontier numbers are means over seeds 0–2, each quantity computed within its seed (`tables.py`,
  `frontier_lookups`). The paper's main-run numbers (NDCG@10 −11.5 ± 1.3%) come from the core seeds sessions. The
  frontier sessions' main run gives −11.6 ± 1.5%: same configuration, different session.
- The Yambda frontier session ran without likes and dislikes. The bundle it used had no side data; none of its
  rewards needs them.
