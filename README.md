# KineTrace
**Temporal Video Search and Reranking**

Find a short clip from a description such as “a red car skids on a wet road, then recovers.” A first-stage retriever narrows the corpus. A second-stage vision-language model scores the ordered frames of each shortlisted clip against the original query.

| Stage | Initial implementation |
| --- | --- |
| Offline indexing | Sample frames across each MP4, encode them with frozen CLIP ViT-B/32, normalize, mean-pool and normalize again. Persist a FAISS CPU IndexFlatIP and an ordered video-ID manifest. |
| Retrieval | Encode a free-text query or a batch of captions, then return cosine scores and original ranks. |
| Reranking | Qwen3-VL-Reranker-2B scores one candidate at a time from chronological frames. Keep the first-stage scores, ranks and unscored tail for comparison. |
| Evaluation | Read an MSR-VTT JSFusion-style CSV. Save candidate JSONL and text-to-video Recall@1/5/10 plus recall at the saved depth. Metrics use fractions. |
| Planned experiments | Train a contrastive retrieval head, mine hard negatives, fine-tune a reranker adapter, profile latency and build a Gradio comparison view. See the [project plan](KineTrace-Plan.md). |

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

Qwen promoted 136 of the 237 targets originally ranked 2–5 and displaced 45 of the 289 targets originally ranked first. It retained 244 first-place hits and finished with 380/1,000, a net gain of 91. Of the 526 queries whose targets were in the shortlist, 146 still ranked their targets below first place after reranking. Peak memory usage and latency await measurement.

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
