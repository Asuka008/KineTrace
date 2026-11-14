# KineTrace experiment plan

Develop a reproducible video retrieval and reranking project through command-line experiments on the school server.

## Completed model work

- Frozen CLIP and base Qwen have a recorded MSR-VTT JSFusion 1K-A baseline. Top-5 Qwen improves R@1 from 28.9% to 38.0%.
- The retained hard-negative LoRA improves training-side validation R@1 from 77/128 to 82/128. Its 1K-A result is 365/1,000 compared with base Qwen's 380/1,000.
- The soft-negative control reaches 80/128 on validation. Review of three changed first choices found two ties and one audio-dependent case.
- The retained adapter is `runs/sft-pilot-zxCXNv5N/adapter`. Detailed previous experiments are recorded in `EXPERIMENTS.md`.

## Current experiment

Run `python -m scripts.compare_bandit runs/sft-pilot-zxCXNv5N` inside the server GPU allocation. The driver reads the retained checkpoint settings, scores the saved training candidates once and fits a small contextual-bandit policy on CPU. The five features are a bias, CLIP score, LoRA relevance score, reciprocal retrieval rank and reciprocal LoRA rank.

The initial logging policy uses epsilon-greedy sampling around LoRA Top-1. With epsilon 0.1 over five candidates, its action probabilities are 0.92, 0.02, 0.02, 0.02 and 0.02. Each training query supplies one simulated selected-action reward derived from its known caption targets. Inverse-propensity-weighted ridge regression fits a reward model with regularization 1. Frozen reward scores order the validation candidate prefix. [Contextual-bandit methods](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_Contextual_bandits_and_Vowpal_Wabbit.html)

This pilot evaluates an offline simulation using dataset-derived rewards. The learned parameters belong to the small linear selection policy. Sparse annotations may miss semantically correct videos. Small feature sets and limited exploration coverage may limit improvement.

## Data and evaluation

1. Reuse the pilot's training-side train/validation galleries. Check target and candidate video IDs for overlap before scoring. Original pilot preparation excludes official test IDs.
2. Exclude training queries whose normalized captions occur in validation. Validate the retained training-pair count and the saved pair manifest when present.
3. Use training labels only to assign simulated rewards. Policy features and validation ordering use model scores and ranks. Keep retrieval misses in evaluation and preserve the candidate tail.
4. Run seeds 42, 43 and 44 with fixed epsilon 0.1. Report all seeds, direct validation Recall, paired gains/losses and seed variation. Record sampled positive rewards, non-top-one selections and importance-weight coverage.
5. Compare against cached base Qwen and retained LoRA results on identical validation queries and candidates. The 128-query validation split is a development benchmark that has been used in earlier experiments.
6. Fix settings before final assessment on fresh held-out data. Record actual results before drawing conclusions about policy quality or practical benefit.

## Execution and artifacts

Develop on the Mac using small offline CPU fixtures. The user runs model scoring and full experiments on the school server in `~/conda-envs/kinetrace`. Use the existing SFT dependencies. The driver records GPU memory visible to PyTorch before loading the model and scoring memory peaks after completion.

Each run gets a unique output directory, source revision, environment snapshot, input/adapter hashes, simulated feedback, learned policies, validation rankings and JSON reports. The CPU comparison publishes its directory atomically and rejects an existing output path. Completed training scores can be reused with `bandit-compare` if the later comparison needs rerunning. Videos, models and outputs remain outside Git.

## Subsequent work

Review the three-seed policy results first. Then evaluate temporal order on verified original/reversed-frame pairs and measure offline scoring latency under fixed hardware, frame count and candidate depth. A broader RL experiment should follow an explicit reward specification and a result from this pilot that motivates added complexity.

Commands and artifact names are maintained in the README. GPU scoring and policy quality are pending server execution. Local fixtures verify implementation behavior.
