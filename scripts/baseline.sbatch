#!/usr/bin/env bash
#SBATCH --job-name=kinetrace
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the repository root}"
: "${MSRVTT_CSV:?export the absolute path to the split CSV}"
: "${VIDEO_DIR:?export the absolute path to the video directory}"
run="${RUN_ROOT:-runs}/${SLURM_JOB_ID:?}"
mkdir -p "$run"
# The cluster-specific GPU request is supplied on the sbatch command line.
python - <<'PY' | tee "$run/device.txt"
import sys, torch
if not torch.cuda.is_available():
    raise SystemExit("No CUDA device visible inside this allocation.")
p = torch.cuda.get_device_properties(0)
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda)
print("device", p.name, "visible GiB", round(p.total_memory / 2**30, 2))
PY
git rev-parse HEAD > "$run/commit.txt"
python -m pip freeze > "$run/environment.txt"
limit_args=()
if [[ "${QUERY_LIMIT:-32}" != all ]]; then
    limit_args=(--limit "${QUERY_LIMIT:-32}")
fi
python -m kinetrace build --csv "$MSRVTT_CSV" --videos "$VIDEO_DIR" \
    --frames "${FRAMES:-4}" --clip-model "${CLIP_MODEL:-openai/clip-vit-base-patch32}" \
    --index "$run/index" "${limit_args[@]}" | tee "$run/build.json"
python -m kinetrace retrieve --csv "$MSRVTT_CSV" --index "$run/index" \
    --top-k "${RECALL_K:-20}" --out "$run/retrieval.jsonl" "${limit_args[@]}" | tee "$run/metrics.json"
python -m kinetrace target-ranks --input "$run/retrieval.jsonl" --out "$run/target-ranks.jsonl"
