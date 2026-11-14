# KineTrace experiment plan

Develop a reproducible video retrieval and reranking project through command-line experiments on the school server.

## Completed model work

- Frozen CLIP and base Qwen have a recorded MSR-VTT JSFusion 1K-A baseline. Top-5 Qwen improves R@1 from 28.9% to 38.0%.
- The retained hard-negative LoRA improves training-side validation R@1 from 77/128 to 82/128. Its 1K-A result is 365/1,000 compared with base Qwen's 380/1,000.
- The soft-negative control reaches 80/128 on validation. Review of three changed first choices found two ties and one audio-dependent case.
- The retained adapter is `runs/sft-pilot-zxCXNv5N/adapter`. Detailed previous experiments are recorded in `EXPERIMENTS.md`.

## Completed offline bandit pilot

The server run `runs/sft-pilot-zxCXNv5N/bandit-offline-76ce4o18` completed on 2026-11-14. Four shared captions were excluded, leaving 252 training queries. Validation contains 128 queries and five candidates each. Seeds 42, 43 and 44 reach 82, 82 and 83 Top-1 hits, compared with 82 for retained LoRA and 77 for base Qwen. Mean policy R@1 is 64.32%, with seed standard deviation 0.37 percentage points, versus LoRA's 64.06%.

The best seed gains two hits and loses one. All methods have R@5 of 82.03%. Training importance-weight effective sample size ranges from 29.43 to 34.78, and sampled non-top-one actions range from 18 to 24. The evidence supports completion of the pilot with a small, seed-dependent change. Retain LoRA as the comparison reference.

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

## Completed base-Qwen frame-order diagnostic

The run `runs/sft-pilot-zxCXNv5N/temporal-base-q4snver7` completed on 2026-11-14. Forward Top-1 predictions match the cached base-Qwen baseline on all 128 queries. Forward and reversed presentations hit 77 and 81 annotated targets, respectively. Reversal changes 35 first choices, comprising 13 gains, nine losses and 13 changes where both choices miss the annotated target.

For 105 scored targets, forward scores are higher in 48 cases, reverse scores in 45 and 12 tie. Mean absolute score change is 0.04477. The result establishes sensitivity to frame order. The directional target-score balance and original-caption labels leave temporal correctness unresolved.

Forward scoring p50/p95 is 1,047.3/1,234.2 ms per query and reverse scoring is 1,057.1/1,307.6 ms. Shared decoding is 144.6/182.3 ms. The H200 NVL MIG allocation exposes 16.0 GiB, with post-warm-up peaks of 4.020 GiB allocated and 4.031 GiB reserved. Decoding and scoring are recorded as separate stages.

Run `python -m scripts.compare_temporal runs/sft-pilot-zxCXNv5N` on the server GPU allocation. Use the base Qwen model with the pilot's 128 validation queries and saved CLIP candidates. Decode each candidate once and score chronological and reversed sampled frames using fixed increasing timestamp slots. Alternate which order runs first. The implementation reuses the reranker's prompt, preprocessing and scoring path.

Report forward Recall, reference-label Recall under reversal, all Top-1 changes, candidate/target score deltas and explicit counts of targets outside the scored prefix. Record per-query shared decoding time and scoring p50/p95, actual visible GPU memory and post-warm-up memory peaks. Compare fresh forward predictions with the cached base-Qwen baseline. Save score pairs and source-frame indices for inspection.

This first step measures sensitivity to input frame order. Correct temporal reasoning requires verified order-dependent queries and labels for both original and reversed presentations. General retrieval captions can remain valid under reversal or become invalid, so the reference-label metrics support a diagnostic interpretation. The next dataset work should establish which queries require ordering and whether the sampled frames preserve the relevant events.

Local CPU fixtures cover frame reversal, metadata isolation, alternating execution, stable ties, candidate misses and result publication. The server report supplies the measured values above. The driver and README provide the same command-line entry points.

## Subsequent work

The base-Qwen sensitivity diagnostic is complete. The next evidence gap is a verified temporal subset with order-dependent descriptions and presentation-specific labels. Assess temporal correctness on that subset. The completed LoRA and bandit experiments supply earlier development results. A broader RL experiment should follow an explicit reward specification and evidence that motivates added complexity.
