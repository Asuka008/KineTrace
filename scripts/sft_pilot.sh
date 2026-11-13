#!/usr/bin/env bash
# Run from the server checkout inside the activated KineTrace environment.
set -euo pipefail
root="${MSRVTT_ROOT:-$HOME/datasets/MSRVTT}"
model="${QWEN_DIR:-$HOME/models/Qwen3-VL-Reranker-2B-9887917}"
mkdir -p "${RUN_ROOT:-runs}" "$root"
run=$(mktemp -d "${RUN_ROOT:-runs}/sft-pilot-XXXXXXXX")
echo "SFT pilot outputs: $run"
git rev-parse HEAD > "$run/commit.txt"
python -m pip freeze > "$run/environment.txt"
annotations="$root/msrvtt_train_9k.json"
if [[ ! -s "$annotations" ]]; then
    tmp=$(mktemp "$root/.train-json-XXXXXXXX")
    trap 'rm -f "$tmp"' EXIT
    curl -fL --retry 3 https://huggingface.co/datasets/friedrichor/MSR-VTT/resolve/main/msrvtt_train_9k.json -o "$tmp"
    mv "$tmp" "$annotations"
fi
python scripts/prepare_sft_pilot.py --train-json "$annotations" \
    --test-csv "$root/MSRVTT_JSFUSION_test.csv" --archive "$root/MSRVTT_Videos.zip" --out "$run/data"
videos="$run/data/videos"
for split in train val; do
    python -m kinetrace build --csv "$run/data/$split.csv" --videos "$videos" \
        --index "$run/$split-index" --frames 4
    python -m kinetrace retrieve --csv "$run/data/$split.csv" --index "$run/$split-index" \
        --top-k 5 --out "$run/$split.jsonl"
done
python -m kinetrace train-lora --input "$run/train.jsonl" --videos "$videos" --qwen-model "$model" \
    --steps "${SFT_STEPS:-1}" --out "$run/adapter"
for variant in base adapted; do
    opts=()
    if [[ "$variant" == adapted ]]; then opts=(--adapter "$run/adapter"); fi
    python -m kinetrace rerank --input "$run/val.jsonl" --videos "$videos" --qwen-model "$model" \
        --top-k 5 --frames 4 "${opts[@]}" --out "$run/val-$variant.jsonl"
    python -m kinetrace metrics --input "$run/val-$variant.jsonl" | tee "$run/metrics-$variant.json"
done
