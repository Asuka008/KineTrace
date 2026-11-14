# KineTrace project plan

Deliver a video search system that retrieves clips from a description, explains ranking changes and can improve selection from measured feedback. Current priorities follow the completed LoRA milestone.

## Completed model work

- Frozen CLIP and base Qwen have a recorded MSR-VTT JSFusion 1K-A baseline. Top-5 Qwen improves R@1 from 28.9% to 38.0%.
- The retained hard-negative LoRA improves training-side validation R@1 from 77/128 to 82/128. Its 1K-A result is 365/1,000 compared with base Qwen's 380/1,000.
- Negative-sampling and soft-label controls have recorded results and video review tools. The retained SFT checkpoint is `runs/sft-pilot-zxCXNv5N/adapter`.
- Prior experiments are recorded in `EXPERIMENTS.md`. The model stage supplies a working training and inference component for the service.

## Milestones and acceptance

| Order | Milestone | Implemented locally | School-server acceptance |
| --- | --- | --- | --- |
| 1 | Search service | FastAPI lifecycle, resident CLIP/Qwen, one inference worker, bounded queue, deadlines, CLIP fallback, index-scoped video serving, browser comparison page | Start in one GPU allocation, record actual visible memory, query and play a returned clip |
| 2 | Performance | Serial query replay with client p50/p95, throughput, errors, fallbacks and successful-request recall | Run the 32-query pilot, then the full fixed split, record hardware and cache conditions |
| 3 | Feedback collection | SQLite decisions, selected action, probabilities, features, model/index metadata, idempotent explicit feedback | Collect train-side query judgments, inspect rated coverage and sampled actions |
| 4 | Contextual bandit | Epsilon-greedy selection, weighted ridge reward fitting, query-held-out IPS/SNIPS and effective sample size | Evaluate a candidate against Qwen Top-1 on the same logged actions and deploy a reviewed artifact for a controlled comparison |
| 5 | Temporal quality | Existing chronological frame sampling | Construct verified order-sensitive queries and compare original/reversed frames |

Local verification uses small CPU fixtures. GPU service execution, concurrent throughput and policy uplift await server measurements.

## RL design decision

The first RL component is a contextual bandit operating on the Qwen-scored candidate set. The context consists of query-dependent retrieval/relevance scores and ranks. The action selects a suggested video, and an explicit relevance judgment supplies reward 1 or 0. This formulation matches a single search decision with feedback on the selected clip. Contextual bandit approaches have been evaluated in large recommendation systems, and epsilon-greedy exploration with logged action probabilities supports counterfactual evaluation. [Recommendation research](https://www.microsoft.com/en-us/research/publication/a-contextual-bandit-approach-to-personalized-news-article-recommendation-3/) · [Vowpal Wabbit contextual bandits](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_Contextual_bandits_and_Vowpal_Wabbit.html)

The initial greedy choice is Qwen Top-1. With epsilon 0.1, 90% of selection probability exploits that choice and 10% explores uniformly across the scored prefix. At five candidates, the logged probabilities are 0.92 for the greedy action and 0.02 for each other action. A trained policy uses the same exploration rule around its linear predicted reward. Training is CPU-only and produces a versioned JSON artifact. The model is selected at service startup.

The reward model uses five features, including a bias, CLIP cosine score, Qwen yes probability, reciprocal CLIP rank and reciprocal Qwen rank. It fits selected-action rewards using inverse-propensity-weighted ridge regression with regularization 1. This baseline can learn how to combine the existing model signals. Its effectiveness depends on feedback diversity and feature adequacy.

## Evaluation protocol

1. Collect learning feedback against training-side videos. Keep official test queries and their judgments outside the training database.
2. Record a decision before returning the recommendation. Store every candidate's features, the sampled action, the probability distribution, query text and model/index metadata.
3. Record only explicit binary relevance feedback. Unrated and audio-dependent decisions retain missing rewards. A duplicate identical vote is idempotent, and a conflicting repeat is rejected.
4. Assign normalized query text to a fixed hash split, approximately 80% for fitting and 20% for evaluation. Repeated queries stay in one split. Use the held-out split consistently and reserve fresh data for final assessment after repeated tuning.
5. Report baseline and candidate IPS/SNIPS on logged actions, matched counts, maximum importance weight and effective sample size. Small effective sample size leaves estimates uncertain. Optional feedback can be selectively missing, so report rated coverage and assess that selection bias before interpreting reward estimates as traffic-wide gains.
6. Verify policy improvements through a controlled deployment comparison on fresh feedback. Deploy by explicitly selecting the model artifact. Preserve the prior model artifact to allow rollback.

The first implementation optimizes one recommended clip. Whole-list ranking, longer interaction sequences and user personalization require additional state, feedback and evaluation design. [Offline policy evaluation](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/off_policy_evaluation.html)

## Resource and deployment constraints

Use the existing `~/conda-envs/kinetrace` server environment and dependency specifications. Read GPU properties inside each allocation. The latest measured training allocation exposed 16 GiB, and resident CLIP plus Qwen service memory still needs measurement. A service timeout preserves exclusive ownership of the GPU worker until the running operation finishes. Queue capacity bounds active and waiting requests.

Run outputs, videos, weights, indexes and feedback databases stay outside Git. The service listens on loopback by default and can be reached through the school's authenticated tunnel or proxy. Run a separate service without feedback logging for benchmark replay. The demo and API share the same loaded pipeline.

## Later extensions

After service measurements and contextual-bandit feedback, assess concurrent load, richer context features, learned retrieval and the temporal subset. A generative policy-gradient experiment can follow once a task requires sampled outputs or multi-step actions, with an explicit reward definition and GPU budget. Current deliverables and commands are maintained in the README.
