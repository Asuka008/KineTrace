# KineTrace
**Temporal Video Search and Reranking**

Find a short clip from a description such as “a red car skids on a wet road, then recovers.” A first-stage retriever narrows the corpus. A second-stage vision-language model scores the ordered frames of each shortlisted clip against the original query.

| Stage | Initial implementation |
| --- | --- |
| Offline indexing | Sample frames across each MP4, encode them with frozen CLIP ViT-B/32, normalize, mean-pool and normalize again. Persist a FAISS CPU IndexFlatIP and an ordered video-ID manifest. |
| Retrieval | Encode a free-text query or a batch of captions, then return cosine scores and original ranks. |
| Reranking | Qwen3-VL-Reranker-2B scores one candidate at a time from chronological frames. Keep the first-stage scores, ranks and unscored tail for comparison. |
| Evaluation | Read an MSR-VTT JSFusion-style CSV. Save candidate JSONL and text-to-video Recall@1/5/10 plus recall at the saved depth. Metrics use fractions. |
| Current SFT | Four-frame Top-5 Qwen with the retained hard-negative LoRA. Select it with `rerank --current-sft` on the school server. |

The retained checkpoint is `runs/sft-pilot-zxCXNv5N/adapter`, the SFT version used in the human review. On JSFusion 1K-A the base, retained and random-negative models hit 380, 365 and 374 first places respectively. Human review found individually preferred clips and plausible alternate matches. Overall human preference across all queries is still unmeasured. For an existing candidate run, select the retained checkpoint explicitly from the repository root:

```bash
RUN=runs/jsfusion1k-20261112-212351
python -m kinetrace rerank --current-sft --input "$RUN/retrieval.jsonl" \
    --videos "$HOME/datasets/MSRVTT/videos" --out "$RUN/retained-sft.jsonl"
```

### Test a softer target for mined negatives

From the school-server checkout in a GPU allocation, run:

```bash
git pull --ff-only origin main
conda activate "$HOME/conda-envs/kinetrace"
python -m pip install -r requirements-sft.txt
python -m scripts.compare_soft_sft runs/sft-pilot-zxCXNv5N
```

The experiment trains a fresh adapter on the original pairs, using 90% “no” and 10% “yes” target mass for each weakly labeled negative. Positives retain the “yes” target. This is an experimental loss weight. The driver reuses the reference base, seed, steps, frames and validation candidates. Before training, it checks that training and validation video IDs are disjoint and the eligible pair count matches the reference. When the reference has a saved `training-pairs.jsonl`, it also checks every pair in order. Older pilots support the pair-count check.

The candidate is saved in a new `soft-negative-*` subdirectory of the supplied pilot. Its `commit.txt` records the source version, and `adapter/train-report.json` records the target weight, GPU model, memory visible to PyTorch and peak allocated training memory. The reference pilot used 256 updates and four frames on a slice with 16.0 GiB visible. Check the new allocation's recorded memory when assessing the run.

The console prints the validation recalls and the number of queries with different first choices. Download `review.html` from the new directory when the first choices differ. The page samples up to 20 such queries uniformly regardless of the benchmark label. Rate which clip better satisfies the description and export the judgments. Upload that JSON to the server and tally it with `python -m scripts.compare_soft_sft COMPARISON_DIRECTORY --judgments EXPORTED_JSON`. Ties, missing evidence and unrated pairs are counted separately. Keep the retained SFT as the project choice while assessing the candidate.

School-server run `runs/sft-pilot-zxCXNv5N/soft-negative-ot0fnqdz` completed 256 updates on 256 available pairs with seed 42, four frames and a negative “yes” weight of 0.1. The H200 NVL MIG 1g.18gb exposed 16.0 GiB to PyTorch. Peak allocated training memory was 4.1036 GiB for 1,605,632 trainable parameters. The final pair loss was 0.360574 and the final pre-clipping gradient norm was 3.9183.

| Adapter | Validation R@1 | Validation R@5 |
| --- | ---: | ---: |
| Retained hard-negative LoRA | 82/128 (64.0625%) | 105/128 (82.03125%) |
| Soft-negative LoRA, 10% “yes” | 80/128 (62.5%) | 105/128 (82.03125%) |

The soft candidate lost two net first-place hits, a decrease of 1.5625 percentage points. Three queries selected different first clips, and the generated review includes all three. Human judgments for this comparison are pending. Download `review.html` from that run directory, review the three pairs, export `review-judgments.json`, and upload it into the same run directory on the server. From the server checkout, tally the judgments on CPU with the following command.

```bash
COMPARISON=runs/sft-pilot-zxCXNv5N/soft-negative-ot0fnqdz
python -m scripts.compare_soft_sft "$COMPARISON" \
    --judgments "$COMPARISON/review-judgments.json"
```

In the tally, `current` means the retained hard-negative adapter and `candidate` means the soft-negative adapter. The review covers the three queries whose first choices differ.

**School-server pilot.** Run `pilot32-20261112-204808` used the first 32 JSFusion queries and their 32 videos, with four frames per video and 20 saved candidates. PyTorch reported 16.00 GiB visible on an NVIDIA H200 NVL MIG 1g.18gb. The server's Git ownership check prevented capture of the run's commit hash.

| Metric | Frozen CLIP | Qwen rerank (top 5) |
| --- | ---: | ---: |
| R@1 | 24/32 (75%) | 26/32 (81.25%) |
| R@5 | 29/32 | 29/32 |
| R@10 | 31/32 | 31/32 |
| R@20 | 31/32 | 31/32 |

Qwen revision `9887917` scored four frames per candidate. It promoted two targets to first place and displaced none.

**Full MSR-VTT JSFusion 1K-A baseline.** School-server run `jsfusion1k-20261112-212351` indexed 1,000 videos with four frames each and retrieved 20 candidates per query.

| Metric | Frozen CLIP | CLIP + Qwen (top 5, four frames) |
| --- | ---: | ---: |
| R@1 | 28.9% | 38.0% |
| R@5 | 52.6% | 52.6% |
| R@10 | 60.6% | 60.6% |
| R@20 | 70.2% | 70.2% |

Qwen promoted 136 of the 237 targets originally ranked 2–5 and displaced 45 of the 289 targets originally ranked first. It retained 244 first-place hits and finished with 380/1,000, a net gain of 91. Of the 526 queries whose targets were in the shortlist, 146 still ranked their targets below first place after reranking. End-to-end search latency and peak inference memory await measurement.

To inspect the individual gains and losses, read the original CLIP ranks preserved in the reranked file. This comparison runs on CPU:

```bash
RUN=runs/pilot32-20261112-204808
python -m kinetrace compare-top1 --input "$RUN/reranked-k5.jsonl"
```

On human inspection of q11, the reviewer judged both clips plausible. Qwen's top clip (`video7212`) shows a person walking, singing and walking a dog. The query specifies standing in the road. The labeled target (`video7028`) moved from first to fourth, counting as one benchmark loss.

For a reproducible review of five more losses, run this command in the server terminal:

```bash
RUN=runs/jsfusion1k-20261112-212351
python -m kinetrace review-lost --input "$RUN/compare-k5.json" --videos "$HOME/datasets/MSRVTT/videos" \
    --exclude q11 --seed 42 --limit 5 --out "$RUN/review-lost.html"
```

Download the HTML from JupyterLab and open it in a browser. It embeds both clips, randomizes their left/right order and hides their model identities until you expand the ranking. Rate each pair, then download the JSON judgments before closing. The benchmark metrics use the original annotations.

Choose `audio_required` when the available visuals leave a decisive spoken or acoustic detail unresolved. For q34, walking is visible, but the subject of the conversation (“their culture”) can require speech or captions. A visible mismatch can still rule out a clip. The current reranker receives sampled frames and query text.

The five-case review marked only the labeled target as relevant for q231, q122 and q394, and both clips for q443 and q901. The saved judgment for q901 is `both`; its spoken topic remains unverified from silent playback.

### Compare reranking depths on the same shortlist

On the allocated GPU, reuse the full run and the same four-frame Qwen snapshot:

```bash
RUN=runs/jsfusion1k-20261112-212351
python -m kinetrace compare-depths --input "$RUN/retrieval.jsonl" --videos "$HOME/datasets/MSRVTT/videos" \
    --qwen-model "$HOME/models/Qwen3-VL-Reranker-2B-9887917" --device cuda --frames 4 \
    --out "$RUN/reranked-k10.jsonl" > "$RUN/depths-5-10.json" && cat "$RUN/depths-5-10.json"
```

The command warms up with one query and then measures all queries. For each query it scores ranks 1–5 once and extends to 10, preserving the CLIP order on score ties. The report includes paired Recall, gains and losses versus Top-5, and cumulative reranking time per query (p50/p95 in milliseconds, including decoding and synchronized GPU scoring; model loading, retrieval and file I/O for results are excluded). Both depths use the system's existing filesystem cache.

The server measured these paired results on the H200 NVL MIG slice with 16.0 GiB visible. Increasing depth gained 22 first-place hits and lost 21 relative to Top-5.

| Depth | R@1 | R@5 | R@10 | R@20 | Rerank p50 | Rerank p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 38.0% | 52.6% | 60.6% | 70.2% | 329.6 ms | 565.6 ms |
| 10 | 38.1% | 56.5% | 60.6% | 70.2% | 663.7 ms | 982.2 ms |

### Qwen LoRA training pilot

From the school-server checkout in a GPU allocation:

```bash
conda activate "$HOME/conda-envs/kinetrace"
python -m pip install -r requirements-sft.txt
bash scripts/sft_pilot.sh
```

The driver expects the existing ZIP and test CSV under `MSRVTT_ROOT` (default `~/datasets/MSRVTT`). It downloads the public train_9k captions once, excludes test IDs by checking disjointness, samples 32 training and 16 validation videos by fixed seed, extracts those clips into its run directory, and mines negatives on the training corpus only. Each query provides its labeled clip and the highest-ranked candidate outside the known positives for that caption. Such negatives are weak labels.

One optimizer step accumulates the mean next-token loss for a positive “yes” and a negative “no”, using separate forwards. Only language attention Q/V LoRA parameters train. `SFT_STEPS` defaults to 1 for a backward-pass smoke check. The driver reloads the adapter in a new process and evaluates both versions on the same five candidates per validation query. Inspect `adapter/train-report.json` for loss, gradient norm and peak allocated GPU memory, and `metrics-{base,adapted}.json` for validation Recall. The one-step pilot checks the training path; measuring a quality improvement requires a larger held-out validation corpus and training run. Use `RUN_ROOT` and `QWEN_DIR` to override the output parent and base snapshot.

In server pilot `sft-pilot-88OF3hW6`, one step used one sampled positive/negative pair out of 32 available pairs. The A100-SXM4-80GB MIG 2g.20gb exposed 19.5 GiB. Training reported 1,605,632 trainable parameters, loss 4.8071, pre-clipping gradient norm 14.4618 and peak allocated GPU memory 4.0392 GiB. Reloading the adapter succeeded. On the 16 validation queries, base Qwen hit 15 first places and the adapter hit 14; both retained 16/16 at five.

To inspect exactly which queries changed on the completed pilot, run:

```bash
RUN=runs/sft-pilot-88OF3hW6
python -m kinetrace compare-runs --baseline "$RUN/val-base.jsonl" --input "$RUN/val-adapted.jsonl"
```

For a larger training-side comparison, use the same base, four frames, fixed seed, and Top-5 candidates. The command below samples one caption per video and requests 256 optimizer updates. Inspect `available_pairs` to see how many queries supplied usable negatives. The new validation corpus has 128 candidates per query before CLIP shortlisting; compare the two models within this run. The driver saves the per-query gains and losses to `paired.json`.

```bash
SFT_TRAIN_VIDEOS=256 SFT_VAL_VIDEOS=128 SFT_STEPS=256 bash scripts/sft_pilot.sh
```

Server run `sft-pilot-zxCXNv5N` trained for 256 updates on 256 available positive/negative pairs. PyTorch reported 4.104 GiB peak allocated on the H200 NVL MIG slice with 16.0 GiB visible. On the same 128 validation queries, CLIP ranked 69 targets first, base Qwen ranked 77 first and adapted Qwen ranked 82 first. All three placed 105 targets in the top five. Relative to base Qwen, the adapter gained seven first-place hits and lost two (+3.91 percentage points at R@1).

To evaluate this fixed checkpoint on JSFusion 1K-A, reuse the original CLIP candidates and the existing base-Qwen Top-5 run. Store the evaluation alongside this adapter:

```bash
EVAL=runs/jsfusion1k-20261112-212351
SFT=runs/sft-pilot-zxCXNv5N
python -m kinetrace rerank --input "$EVAL/retrieval.jsonl" --videos "$HOME/datasets/MSRVTT/videos" \
    --qwen-model "$HOME/models/Qwen3-VL-Reranker-2B-9887917" --adapter "$SFT/adapter" \
    --top-k 5 --frames 4 --device cuda --out "$SFT/test1k-k5.jsonl" && \
python -m kinetrace metrics --input "$SFT/test1k-k5.jsonl" > "$SFT/test1k-metrics.json" && \
python -m kinetrace compare-runs --baseline "$EVAL/reranked-k5.jsonl" --input "$SFT/test1k-k5.jsonl" > "$SFT/test1k-paired.json"
```

On JSFusion 1K-A, base Qwen ranked 380 of 1,000 targets first and the adapter ranked 365 first (18 gains and 33 losses). To inspect the changed cases, sample ten from each group directly from the saved comparison on the school server:

```bash
SFT=runs/sft-pilot-zxCXNv5N
python -m kinetrace review-changes --input "$SFT/test1k-paired.json" --videos "$HOME/datasets/MSRVTT/videos" \
    --seed 42 --per-outcome 10 --out "$SFT/test1k-review.html"
```

Download the HTML from JupyterLab and open it locally. It mixes the two groups, shuffles the two clips for each query and keeps ranks and model names off the page. Export the judgments before closing. Match the exported query IDs and A/B video IDs to `test1k-paired.json` for analysis. Each group is sampled separately, so report judgments by group.

### Compare random and CLIP-mined training negatives

From the server checkout in the same GPU environment, run:

```bash
python -m scripts.compare_sft_negatives runs/sft-pilot-zxCXNv5N
```

The driver trains one fresh adapter from the original Qwen base. It reads the frames, step count, device and seed from the saved hard-negative training report. For each query that had a usable CLIP negative, it samples one video uniformly from the distinct targets in the training run, excluding all labeled targets for that normalized caption. A separate random generator keeps the positive examples and their shuffled training order matched. The new adapter includes the sampled `training-pairs.jsonl` for inspection.

The driver scores the new adapter on the saved validation shortlist and reuses the existing base and hard-negative rankings. It prints R@1, R@5 and paired gains and losses, and writes per-query changes to `random-negative-*/validation.json` inside the pilot directory. In `hard_to_random`, a gain means that switching from the hard-negative adapter to the random-negative adapter moved the labeled target to first place. For the 256-step pilot, the reference counts are 77/128 for base Qwen and 82/128 for hard-negative LoRA.

## GitHub workflow

Use this project directory as the repository root. The GitHub remote is https://github.com/Asuka008/KineTrace.git. For an empty remote, run the commands below on the Mac. If you already have a repository with commits, start from its clone and copy these source files into that checkout.

```bash
git init -b main
git add .
git commit -m "Bootstrap KineTrace video retrieval and reranking"
git remote add origin https://github.com/Asuka008/KineTrace.git
git push -u origin main
```

On the school server, clone that URL once. For subsequent changes, run the local checks, commit and push from the Mac, then run `git pull --ff-only` from the server checkout. The project conventions are in [AGENTS.md](AGENTS.md). Datasets, model weights, indexes and run outputs live on the server.

## Mac smoke check

From the repository root, using Python 3.9+:

```bash
python3 -m unittest discover -s tests -v
python3 -m kinetrace --help
```

These commands use the standard library and work offline.

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
    --query '一辆红色汽车在雨天侧滑，随后恢复控制。' \
    --retrieval-text 'A red car skids on a wet road and then regains control.' \
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
| `kinetrace/__main__.py` | CLI entry points |
| `scripts/baseline.sbatch` | Pilot and full first-stage runs |
| `tests/test_smoke.py` | Local offline checks |

[CLIP checkpoint](https://huggingface.co/openai/clip-vit-base-patch32) · [FAISS inner-product search](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances) · [Qwen reranker model card](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B)
