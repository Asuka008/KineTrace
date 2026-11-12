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

**Verification status.** The local smoke suite exercises ranking, data handling and the CLI with fixtures. Real video decoding, model inference, GPU memory, latency and benchmark scores await the school-server run.

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
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-server.txt
```

Prepare the videos as `VIDEO_DIR/video9770.mp4`, etc. Use `MSRVTT_JSFUSION_test.csv` for the official 1K-A evaluation. The loader uses `video_id` and `sentence`, ignores extra CSV columns, and indexes each distinct video ID once. A missing video stops the run with its path. The same format works for a development split. [Split and video acquisition instructions](https://github.com/ArrowLuo/CLIP4Clip#data-preparing) · [Example of the test CSV](https://huggingface.co/datasets/friedrichor/MSR-VTT/blob/main/raw_data/MSRVTT_JSFUSION_test.csv)

```bash
export MSRVTT_CSV=/absolute/path/to/MSRVTT_JSFUSION_test.csv
export VIDEO_DIR=/absolute/path/to/MSRVTT/videos
```

If compute nodes are offline, prefetch the CLIP snapshot on a network-enabled server node into a shared cache. Keep the same `HF_HOME` in the GPU allocation. Prefetch Qwen separately before the first rerank.

```bash
export HF_HOME=/absolute/path/to/shared/model-cache
hf download openai/clip-vit-base-patch32 --include '*.json' '*.txt' 'pytorch_model.bin'
# Later, for stage two
hf download Qwen/Qwen3-VL-Reranker-2B --include '*.json' '*.txt' '*.jinja' '*.safetensors'
```

### First GPU job

From the server checkout, with the Python environment active, substitute your cluster's account, partition and GPU resource specification. The submission script inherits `MSRVTT_CSV` and `VIDEO_DIR`. It records the memory visible to PyTorch inside the allocation. The pilot defaults to 32 CSV rows, 4 frames per clip and retrieval depth 20.

```bash
sbatch -A YOUR_ACCOUNT -p YOUR_PARTITION --gres=YOUR_GPU_RESOURCE scripts/baseline.sbatch
```

For a cluster using a generic single GPU resource, the argument is `--gres=gpu:1`. Select the site's MIG resource when required. Output goes to `runs/<job-id>/`. Check `device.txt`, `metrics.json` and one row of `retrieval.jsonl`. For the full baseline set `export QUERY_LIMIT=all` and submit again. With the official 1K-A file, verify 1,000 indexed videos and 1,000 evaluated queries. A pilot on a 32-video corpus is only a plumbing check.

### Compare rankings on an allocated GPU

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
