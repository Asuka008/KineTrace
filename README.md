# KineTrace
**Video Retrieval · Vision-Language Reranking · LoRA Fine-Tuning**

Search short videos from a natural-language description. KineTrace retrieves candidates with frozen CLIP and FAISS, then reranks sampled video frames with Qwen3-VL-Reranker-2B and an optional LoRA adapter. The repository includes training, adapter inference, paired evaluation and portable video review tools.

## Measured results

| Measurement | Result | Evaluation scope |
| --- | --- | --- |
| LoRA validation R@1 | **60.16% → 64.06% (+3.91 percentage points)** | Base Qwen 77/128 → hard-negative LoRA 82/128, on the same 128-query validation split and CLIP Top-5 candidates |
| Trainable LoRA parameters | **1,605,632** | Language attention Q/V adapters, rank 8, alpha 16, with the count recorded in the training pilot and soft-target control |
| Hard-negative LoRA training peak | **4.104 GiB allocated** | 256 updates, four frames per clip, H200 NVL MIG with 16.0 GiB visible to PyTorch |
| Retrieval + base reranker R@1 | **28.9% → 38.0% (+9.1 percentage points)** | Frozen CLIP → base Qwen Top-5, MSR-VTT JSFusion 1K-A, 1,000 queries and 1,000 videos |

The LoRA validation run uses 256 available training pairs, 256 optimizer updates and seed 42. It gains seven first-place hits and loses two relative to the base model. Training and validation videos are disjoint, and the same held-out shortlist is used for both models.

**Generalization result.** On JSFusion 1K-A, the retained LoRA reaches 36.5% R@1 versus 38.0% for base Qwen at Top-5. The validation gain therefore remains specific to the reported development split.

## Implemented pipeline

| Component | Implementation |
| --- | --- |
| Video indexing | Sample chronological frames, encode with frozen CLIP ViT-B/32, normalize and mean-pool, then persist a FAISS CPU IndexFlatIP and ordered video manifest |
| Retrieval | Encode free-text queries or caption batches, preserve cosine scores and original ranks |
| Train-side data | Split by video ID, exclude test IDs, mine CLIP negatives while excluding known positives for each normalized caption |
| LoRA SFT | Freeze the base model, train language attention Q/V adapters with gradient checkpointing and separate positive/negative forwards accumulated into one optimizer step |
| Adapter inference | Save the adapter, reload it in a separate process and score the same candidate prefix with four-frame video input |
| Evaluation | Report Recall@K, pair gains/losses by query ID, and export randomized A/B video reviews with judgment tallying |
| Run artifacts | Save candidates, model settings, sampled training pairs, the last gradient norm and actual visible/peak allocated GPU memory in the server run directory |

Current delivery includes command-line training, inference, paired evaluation and an offline contextual-bandit simulation. The simulation has small CPU fixture coverage. Its quality results await a school-server run.

## Reproduce the LoRA experiment

From the school-server checkout inside a GPU allocation, activate the project environment and install the training dependencies. Initial environment and dataset setup are described below.

```bash
git pull --ff-only origin main
conda activate "$HOME/conda-envs/kinetrace"
python -m pip install -r requirements-sft.txt
SFT_TRAIN_VIDEOS=256 SFT_VAL_VIDEOS=128 SFT_STEPS=256 bash scripts/sft_pilot.sh
```

The driver expects `MSRVTT_Videos.zip` and `MSRVTT_JSFUSION_test.csv` under `MSRVTT_ROOT` (default `~/datasets/MSRVTT`). It downloads train_9k captions once, builds separate train/validation indexes, mines negatives, trains and reloads the adapter, and evaluates both models on the same validation candidates. Set `RUN_ROOT` for the output parent and `QWEN_DIR` for the local Qwen snapshot. The default snapshot path is `~/models/Qwen3-VL-Reranker-2B-9887917`.

The training objective supervises the next answer token, “yes” for the labeled video and “no” for a mined negative. AdamW uses learning rate `1e-4`. Each update averages the positive and negative losses. Only the language attention Q/V LoRA parameters train. The GPU smoke configuration defaults to one update when `SFT_STEPS` is omitted.

Inspect these artifacts in the printed run directory.

| Artifact | Contents |
| --- | --- |
| `commit.txt`, `environment.txt` | Source revision and installed dependencies |
| `adapter/train-report.json` | Training settings, trainable parameter count, last loss/gradient norm, visible GPU memory and peak allocated training memory |
| `adapter/training-pairs.jsonl` | Positive/negative pairs before the training-order shuffle, saved by the current training code |
| `metrics-base.json`, `metrics-adapted.json` | Paired validation Recall |
| `paired.json` | Query-level gains and losses after adapter reload |

The retained school-server checkpoint is `runs/sft-pilot-zxCXNv5N/adapter`. From that checkout, apply it to an existing retrieval run with the following commands.

```bash
RUN=runs/jsfusion1k-20261112-212351
python -m kinetrace rerank --current-sft --input "$RUN/retrieval.jsonl" \
    --videos "$HOME/datasets/MSRVTT/videos" --top-k 5 --frames 4 \
    --out "$RUN/retained-sft.jsonl"
python -m kinetrace metrics --input "$RUN/retained-sft.jsonl"
python -m kinetrace compare-runs --baseline "$RUN/reranked-k5.jsonl" \
    --input "$RUN/retained-sft.jsonl" > "$RUN/retained-paired.json"
```

For a newly trained checkpoint, use `--adapter PATH_TO_ADAPTER --qwen-model PATH_TO_BASE` in place of `--current-sft`. The retained shortcut reads its base snapshot from the saved training report.

<details>
<summary>Additional experiment and review commands</summary>

Run the training controls inside the school GPU allocation.

```bash
python -m scripts.compare_sft_negatives runs/sft-pilot-zxCXNv5N
python -m scripts.compare_soft_sft runs/sft-pilot-zxCXNv5N
```

The soft-target control uses 90% “no” and 10% “yes” target mass on negatives and produces a blind review of different first choices. Upload the browser-exported judgments into its comparison directory and tally on CPU.

```bash
COMPARISON=runs/sft-pilot-zxCXNv5N/soft-negative-ot0fnqdz
python -m scripts.compare_soft_sft "$COMPARISON" \
    --judgments "$COMPARISON/review-judgments.json"
```

For a saved `compare-runs` report, generate a portable review of gains and losses.

```bash
RUN=runs/jsfusion1k-20261112-212351
python -m kinetrace review-changes --input "$RUN/retained-paired.json" \
    --videos "$HOME/datasets/MSRVTT/videos" --seed 42 --per-outcome 10 \
    --out "$RUN/retained-review.html"
```

</details>

## Next experiment — offline contextual bandit

Run the next experiment from the school-server checkout inside a GPU allocation. Stop the previous foreground service with `Ctrl+C` before starting the experiment.

```bash
git pull --ff-only origin main
conda activate "$HOME/conda-envs/kinetrace"
python -m scripts.compare_bandit runs/sft-pilot-zxCXNv5N
```

The command loads the existing LoRA adapter, scores the saved training candidates once and runs three CPU policy fits with seeds 42, 43 and 44. It reads the base model, frame count and device from the retained adapter's training report. Existing base-Qwen and LoRA validation scores supply the comparison on the same validation candidates. The training and validation galleries are checked for overlap, and training captions shared with validation are excluded from policy fitting.

The experiment uses a contextual-bandit simulation. For each eligible training query, epsilon-greedy sampling selects one candidate from the LoRA-scored Top-5. With epsilon 0.1 and five candidates, the first candidate has probability 0.92 and every other candidate has probability 0.02. The sampled reward is 1 for a known training-caption target and 0 otherwise. An inverse-propensity-weighted ridge model learns from these selected-action rewards using CLIP score, LoRA score and reciprocal ranks, plus a bias. The trained model orders the same candidate prefix on validation. [Contextual-bandit formulation and exploration](https://vowpalwabbit.org/docs/vowpal_wabbit/python/latest/tutorials/python_Contextual_bandits_and_Vowpal_Wabbit.html)

This is a small candidate-selection policy trained on frozen model scores. Dataset labels provide simulated rewards. Single-target annotations can assign zero reward to another semantically relevant clip. Validation reports direct labeled Recall and paired gains/losses for every seed, plus the mean and standard deviation of R@1 across seeds. That standard deviation measures sampling variation. Real-user reward improvement requires later independent evidence. The existing 128-query validation split has already informed development, so final claims require a fresh held-out evaluation after settings are fixed.

Outputs go into the printed `bandit-offline-*` directory under the pilot.

| Artifact | Contents |
| --- | --- |
| `commit.txt`, `environment.txt`, `provenance.json` | Revision, dependencies, reference settings and input/adapter hashes |
| `scoring-device.json` | Actual visible GPU memory and scoring allocation/reservation peaks |
| `train-adapted.jsonl` | Training candidate scores from the retained LoRA |
| `summary.json` | Base Qwen, LoRA and every policy seed, with query-level paired changes |
| `comparison/feedback-SEED.jsonl` | Simulated selected actions, exact probabilities and rewards |
| `comparison/policy-SEED.json` | Learned linear reward model |
| `comparison/validation-SEED.jsonl` | Validation rankings, preserving the candidate tail |

The detailed report also records positive rewards, selections outside the original first place and training importance-weight effective sample size. Low coverage and a small pilot limit the evidence. All seeds are reported together.

For a CPU-only rerun using saved training scores, set `EXPERIMENT` to the directory printed by the first command and choose a new output directory.

```bash
EXPERIMENT=runs/sft-pilot-zxCXNv5N/bandit-offline-REPLACE_WITH_RUN_ID
python -m kinetrace bandit-compare \
    --train "$EXPERIMENT/train-adapted.jsonl" \
    --validation runs/sft-pilot-zxCXNv5N/val-adapted.jsonl \
    --top-k 5 --epsilon 0.1 --seeds 42 43 44 \
    --out "$EXPERIMENT/comparison-rerun"
```

## Next milestones

| Stage | Current status | Next acceptance evidence |
| --- | --- | --- |
| LoRA SFT | Training, adapter reload and paired evaluation completed | Reference checkpoint for candidate-policy comparisons |
| Offline contextual bandit | Scoring driver, simulated rewards, policy fitting and paired evaluation implemented | Server results across the three fixed seeds |
| Temporal evaluation | Planned | Verified temporal subset and original/reversed-frame comparison |
| Final evaluation | Planned | Fixed settings and fresh held-out quality measurements |

The project plan describes the experiment sequence and reward limitations. [Project plan](KineTrace-Plan.md)

## Development workflow

Develop and run small offline CPU checks on the Mac. Synchronize source and dependency specifications through GitHub, then pull on the school server for training, inference and benchmarks. Keep videos, weights, indexes and outputs outside Git. Use `~/conda-envs/kinetrace` for the server environment, and size jobs from the memory actually visible to PyTorch inside the allocation. Repository conventions are recorded in [AGENTS.md](AGENTS.md).

## Mac smoke check

From the repository root, using Python 3.9+:

```bash
python3 -m unittest discover -s tests -v
python3 -m kinetrace --help
```

Local checks use the standard library, temporary fixtures and stub model scoring. They exercise the experiment CLI, split checks, sampled rewards, policy fitting and paired evaluation.

## School server setup

Use Python 3.11 and a CUDA-enabled PyTorch environment. One installation example for a Linux cluster supporting the CUDA 12.6 wheel follows. Use the wheel that matches the server driver or the matching torch/torchvision pair provided by the cluster. Activate this environment before submitting a job. [Official PyTorch builds](https://pytorch.org/get-started/previous-versions/)

```bash
mkdir -p "$HOME/conda-envs"
conda create --prefix "$HOME/conda-envs/kinetrace" python=3.11 pip -y
conda activate "$HOME/conda-envs/kinetrace"
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

For later dependency changes, activate the same prefix and run `python -m pip install -r requirements.txt` from the updated checkout.

Prepare the videos as `VIDEO_DIR/video9770.mp4`, etc. Use `MSRVTT_JSFUSION_test.csv` for the official 1K-A evaluation. The loader uses `video_id` and `sentence`, ignores extra CSV columns, and indexes each distinct video ID once. A missing video stops the run with its path. The same format works for a development split. [Split and video acquisition instructions](https://github.com/ArrowLuo/CLIP4Clip#data-preparing) · [Example of the test CSV](https://huggingface.co/datasets/friedrichor/MSR-VTT/blob/main/raw_data/MSRVTT_JSFUSION_test.csv)

```bash
export MSRVTT_CSV=/absolute/path/to/MSRVTT_JSFUSION_test.csv
export VIDEO_DIR=/absolute/path/to/MSRVTT/videos
```

On the login node, check the pilot input before requesting a GPU:

```bash
python -m kinetrace check-data --csv "$MSRVTT_CSV" --videos "$VIDEO_DIR" --limit 32
```

This reads the same CSV prefix as the index builder and checks for an MP4 per distinct video ID. For the official 1K-A pilot expect 32 queries and 32 videos. Multiple captions for one clip in a custom split count as separate queries. Omit `--limit` to check the full split. Frame decoding is first exercised by the GPU job.

If compute nodes are offline, prefetch the CLIP snapshot on a network-enabled server node into a shared cache. Keep the same `HF_HOME` in the GPU allocation. Prefetch Qwen separately before the first rerank.

```bash
export HF_HOME=/absolute/path/to/shared/model-cache
hf download openai/clip-vit-base-patch32 --include '*.json' '*.txt' 'pytorch_model.bin'
# Later, for stage two
hf download Qwen/Qwen3-VL-Reranker-2B --revision 9887917432733561ed27603b14ea9968af048678 --include '*.json' '*.txt' '*.jinja' '*.safetensors'
```

### First GPU job

From the server checkout, with the Python environment active, substitute your cluster's account, partition and GPU resource specification. The submission script inherits `MSRVTT_CSV` and `VIDEO_DIR`. It records the memory visible to PyTorch inside the allocation. The pilot defaults to 32 CSV rows, 4 frames per clip and retrieval depth 20.

```bash
sbatch -A YOUR_ACCOUNT -p YOUR_PARTITION --gres=YOUR_GPU_RESOURCE scripts/baseline.sbatch
```

For a cluster using a generic single GPU resource, the argument is `--gres=gpu:1`. Select the site's MIG resource when required. Output goes to `runs/<job-id>/`. Check `device.txt`, `metrics.json` and one row of `retrieval.jsonl`. For the full baseline set `export QUERY_LIMIT=all` and submit again. With the official 1K-A file, verify 1,000 indexed videos and 1,000 evaluated queries. A pilot on a 32-video corpus is only a plumbing check.

Inspect the pilot after Slurm finishes:

```bash
RUN=runs/YOUR_BASELINE_JOB_ID
cat "$RUN/device.txt" "$RUN/build.json" "$RUN/metrics.json"
head -n 3 "$RUN/target-ranks.jsonl"
```

Each target-rank row gives the query, expected video, first result, saved candidate depth and one-based target rank. A JSON `null` rank means the expected video fell outside that list. To inspect a labeled reranked run, use `python -m kinetrace target-ranks --input "$RUN/reranked.jsonl" --out "$RUN/reranked-target-ranks.jsonl"`.

### Compare rankings on an allocated GPU

For a one-query smoke check, reuse the saved CLIP candidates:

```bash
RUN=runs/pilot32-20261112-204808
head -n 1 "$RUN/retrieval.jsonl" > "$RUN/first-query.jsonl"
python -m kinetrace rerank --input "$RUN/first-query.jsonl" --videos "$VIDEO_DIR" \
    --top-k 5 --frames 4 --out "$RUN/first-query-ranked.jsonl"
python -m kinetrace metrics --input "$RUN/first-query-ranked.jsonl"
```

For a paired comparison across all pilot queries:

```bash
RUN=runs/YOUR_BASELINE_JOB_ID
python -m kinetrace rerank --input "$RUN/retrieval.jsonl" --videos "$VIDEO_DIR" \
    --top-k 5 --frames 4 --out "$RUN/reranked.jsonl"
python -m kinetrace metrics --input "$RUN/reranked.jsonl"
```

Compare with the original `metrics.json`. This command reorders the first five of the saved candidates. At fixed candidate membership, R@1 can change, while R@5 and deeper remain constant. Start the reranker with one query if the visible slice is small. The model loads in a separate process from CLIP.

### Search one description

```bash
python -m kinetrace search --index "$RUN/index" \
    --query 'A red car skids on a wet road and then regains control.' \
    --top-k 10 --out "$RUN/one-query.jsonl"
python -m kinetrace rerank --input "$RUN/one-query.jsonl" --videos "$VIDEO_DIR" \
    --top-k 5 --frames 4 --out "$RUN/one-query-ranked.jsonl"
```

The optional English rendering goes to CLIP. Qwen receives the original query. For an English query, pass only `--query`. The example illustrates the interface. Finding this specific event requires a corpus containing the event. Unlabeled free-text queries can be reranked and inspected. Recall requires a labeled CSV run.

## Source map

| File | Purpose |
| --- | --- |
| `kinetrace/video.py` | Sampling, chronological decoding, timestamp metadata |
| `kinetrace/retrieval.py` | Frame/text encoding, index persistence and candidate retrieval |
| `kinetrace/reranker.py` | Ordered video input and binary relevance scoring |
| `kinetrace/ranking.py` | Candidate-prefix reranking and Recall@K |
| `kinetrace/sft.py` | Training-pair selection and LoRA updates |
| `kinetrace/bandit.py` | Offline reward simulation, policy fitting and paired evaluation |
| `scripts/compare_bandit.py` | Retained-LoRA scoring and the offline bandit experiment |
| `kinetrace/review.py` | Portable, randomized A/B video review pages |
| `kinetrace/__main__.py` | CLI entry points |
| `scripts/baseline.sbatch` | Pilot and full first-stage runs |
| `scripts/sft_pilot.sh` | Train-side preparation, training, adapter reload and paired validation |
| `tests/test_smoke.py` | Local offline checks |

[CLIP checkpoint](https://huggingface.co/openai/clip-vit-base-patch32) · [FAISS inner-product search](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances) · [Qwen reranker model card](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B)
