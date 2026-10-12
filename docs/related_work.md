# Reading list

Grouped by the part of the study each paper informs. Every title / arXiv-ID pair below was checked
against arXiv on 2026-10-10. Authors, venues and years were **not** checked: take them from the
arXiv page when citing.

## The two papers this repository builds on

| arXiv | Title | What we take from it |
|---|---|---|
| 2403.04875 | Aligning GPTRec with Beyond-Accuracy Goals with Reinforcement Learning | Two-stage scheme (supervised, then RL), item-by-item "Next-K" generation, PPO with masks re-applied in the loss, dropout off during RL, Top-K vs Next-K degradation as an evaluation axis. |
| 2607.02818 | Session-Level Optimization for Large-Scale Retrieval using REINFORCE with Multi-Step Off-Policy Correction | Yambda setup, the user feedback model (played ratio + like), recall on high- vs low-return targets as a model-free check, doubly robust OPE, test-time scaling, and the warning that model-based gains are not independent evidence. |

## Generative and sequential recommenders

| arXiv | Title |
|---|---|
| 1808.09781 | Self-Attentive Sequential Recommendation (SASRec) |
| 1904.06690 | BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer |
| 2306.11114 | Generative Sequential Recommendation with GPTRec |
| 2305.05065 | Recommender Systems with Generative Retrieval (TIGER) |
| 2402.17152 | Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations |
| 2504.10507 | PinRec: Unified Generative Retrieval for Pinterest Recommender Systems |
| 2502.18965 | OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment |
| 2506.13695 | OneRec Technical Report |
| 2508.20900 | OneRec-V2 Technical Report |
| 2510.11639 | OneRec-Think: In-Text Reasoning for Generative Recommendation |

## RL for recommenders

| arXiv | Title |
|---|---|
| 1812.02353 | Top-K Off-Policy Correction for a REINFORCE Recommender System |
| 1905.12767 | Reinforcement Learning for Slate-based Recommender Systems: A Tractable Decomposition and Practical Methodology (SlateQ) |
| 2006.05779 | Self-Supervised Reinforcement Learning for Recommender Systems (SQN / SAC) |
| 2111.03474 | Supervised Advantage Actor-Critic for Recommender Systems |
| 2301.08632 | Generative Slate Recommendation with Reinforcement Learning |
| 2209.15166 | Reward Shaping for User Satisfaction in a REINFORCE Recommender |

## Reward models, simulators, and exploiting them

Most relevant to research questions 1 and 4: a policy optimised against a learned model drifts to
where the model is wrong.

| arXiv | Title |
|---|---|
| 2005.13239 | MOPO: Model-based Offline Policy Optimization (penalise reward by model uncertainty) |
| 2005.05951 | MOReL: Model-Based Offline Reinforcement Learning |
| 2102.08363 | COMBO: Conservative Offline Model-Based Policy Optimization |
| 1906.08253 | When to Trust Your Model: Model-Based Policy Optimization (short rollouts bound compounding error) |
| 2307.04571 | Alleviating Matthew Effect of Offline Reinforcement Learning in Interactive Recommendation (DORL) |
| 2204.01266 | CIRS: Bursting Filter Bubbles by Counterfactual Interactive Recommender System |
| 2210.10760 | Scaling Laws for Reward Model Overoptimization |
| 2310.02743 | Reward Model Ensembles Help Mitigate Overoptimization |
| 2209.13085 | Defining and Characterizing Reward Hacking |
| 2601.04674 | PROMISE: Process Reward Models Unlock Test-Time Scaling Laws in Generative Recommendations |
| 1902.10730 | Degenerate Feedback Loops in Recommender Systems |
| 1909.04847 | RecSim: A Configurable Simulation Platform for Recommender Systems |
| 2103.08057 | RecSim NG: Toward Principled Uncertainty Modeling for Recommender Ecosystems |
| 1805.10000 | Virtual-Taobao: Virtualizing Real-world Online Retail Environment for Reinforcement Learning |
| 1812.10613 | Generative Adversarial User Model for Reinforcement Learning Based Recommendation System |
| 2309.12645 | KuaiSim: A Comprehensive Simulator for Recommender Systems |

## Staying close to the data: KL-regularised and conservative policy improvement

Background for the KL anchor and for research question 5.

| arXiv | Title |
|---|---|
| 1502.05477 | Trust Region Policy Optimization |
| 1707.06347 | Proximal Policy Optimization Algorithms |
| 1611.02796 | Sequence Tutor: Conservative Fine-Tuning of Sequence Generation Models with KL-control |
| 1907.00456 | Way Off-Policy Batch Deep Reinforcement Learning of Implicit Human Preferences in Dialog |
| 1909.08593 | Fine-Tuning Language Models from Human Preferences |
| 2203.02155 | Training language models to follow instructions with human feedback (InstructGPT) |
| 2305.18290 | Direct Preference Optimization: Your Language Model is Secretly a Reward Model |
| 1612.00563 | Self-critical Sequence Training for Image Captioning (baseline from the model's own samples) |
| 1911.11361 | Behavior Regularized Offline Reinforcement Learning |
| 1812.02900 | Off-Policy Deep Reinforcement Learning without Exploration (BCQ) |
| 1910.00177 | Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning |
| 2006.09359 | AWAC: Accelerating Online Reinforcement Learning with Offline Datasets |
| 2006.04779 | Conservative Q-Learning for Offline Reinforcement Learning |

## Long-term engagement and retention (research question 3)

| arXiv | Title |
|---|---|
| 1902.05570 | Reinforcement Learning to Optimize Long-term User Engagement in Recommender Systems |
| 1807.09387 | Learning from Delayed Outcomes via Proxies with Applications to Recommender Systems |
| 2302.01724 | Reinforcing User Retention in a Billion Scale Short Video Recommender System |
| 2303.06347 | User Retention-oriented Recommendation with Decision Transformer |
| 2406.06043 | Modeling User Retention through Generative Flow Networks |
| 2212.02779 | PrefRec: Recommender Systems with Human Preferences for Reinforcing Long-term User Engagement |
| 2302.01680 | Two-Stage Constrained Actor-Critic for Short Video Recommendation |
| 2305.07764 | Long-Term Value of Exploration: Measurements, Findings and Algorithms |

## Watch-time and duration bias

Why "fraction played" favours short items, and how others correct for it.

| arXiv | Title |
|---|---|
| 2206.06003 | Deconfounding Duration Bias in Watch-time Prediction for Video Recommendation (D2Q) |
| 2306.03392 | Tree based Progressive Regression Model for Watch-Time Prediction in Short-video Recommendation |
| 2308.08120 | Uncovering User Interest from Biased and Noised Watch Time in Video Recommendation |
| 2306.17426 | Leveraging Watch-time Feedback for Short-Video Recommendations: A Causal Labeling Framework |
| 2406.07932 | Counteracting Duration Bias in Video Recommendation via Counterfactual Watch Time |

## Off-policy evaluation and the limits of offline evaluation

| arXiv | Title |
|---|---|
| 1103.4601 | Doubly Robust Policy Evaluation and Learning |
| 1511.03722 | Doubly Robust Off-policy Value Evaluation for Reinforcement Learning |
| 1802.03493 | More Robust Doubly Robust Off-policy Evaluation |
| 1605.04812 | Off-policy evaluation for slate recommendation |
| 2007.12986 | Counterfactual Evaluation of Slate Recommendations with Sequential Reward Interactions |
| 1801.07030 | Offline A/B testing for Recommender Systems |
| 2008.07146 | Open Bandit Dataset and Pipeline: Towards Realistic and Reproducible Off-Policy Evaluation |
| 2301.00993 | Offline Evaluation for Reinforcement Learning-based Recommendation: A Critical Issue and Some Alternatives |
| 2308.11336 | On the Opportunities and Challenges of Offline Reinforcement Learning for Recommender Systems |
| 1907.06902 | Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches |

## Exposure bias and compounding error (research question 4)

| arXiv | Title |
|---|---|
| 1506.03099 | Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks |
| 1011.0686 | A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger) |
| 1511.06732 | Sequence Level Training with Recurrent Neural Networks |
| 1610.09038 | Professor Forcing: A New Algorithm for Training Recurrent Networks |
| 1905.10617 | Exposure Bias versus Self-Recovery: Are Distortions Really Incremental for Autoregressive Text Generation? |

## Datasets

| arXiv | Title |
|---|---|
| 2505.22238 | Yambda-5B -- A Large-Scale Multi-modal Dataset for Ranking And Retrieval |
| 2209.00325 | MTS Kion Implicit Contextualised Sequential Dataset for Movie Recommendation |
| 2202.10842 | KuaiRec: A Fully-observed Dataset and Insights for Evaluating Recommender Systems |
| 2208.08696 | KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos |
