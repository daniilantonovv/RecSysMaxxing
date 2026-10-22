# Yambda, final session

The Yambda results of `report/archive/paper_v3` (and of the later versions). This session continued `yambda_diagnostics`: it reused that session's
interaction log, base recommender, reward models, baselines and 12 fine-tuning runs, and trained 13 more runs
(configuration `("yambda", "final")` in `kaggle/build_notebooks.py`).

The CSV files come from two places:

- The reports the session's notebook displayed were read from the notebook: `policy_test_relevance`, `baselines`,
  `rl_vs_pretrained`, `rl_vs_pretrained_significance`, `kl_sweep`, `kl_sweep_selection`, `kl_budget_sweep`,
  `advantage_estimators`, `simulated_feedback_sweep`, `simulated_feedback_cross_judged`,
  `reward_gain_decomposition`, `alternatives_to_rl`, `reward_sweep`, `relevance_reward_sweep`,
  `personal_reward_sweep`, `relative_reward_sweep`, `list_reward_sweep`, `list_reward_methods`,
  `sequence_credit_ablation` and `stage_minutes`. They carry four decimals, and the reports of fine-tuning runs
  carry only the rows the notebook shows for them.
- Every other file is copied from `yambda_diagnostics`. Those reports come from the same models, and the
  re-evaluated reports above match that session to four decimals, except for the best of 8 sampled lists, which
  are drawn again.

`stage_minutes` covers this session only; `yambda_diagnostics/stage_minutes.csv` holds the first one. The session's
own `experiment_config.json` and full-precision reports are in its output archive `yambda_final_reports.zip` and
can replace these files one for one.
