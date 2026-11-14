"""Score the retained LoRA training candidates once, then run a CPU bandit simulation."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from kinetrace.__main__ import parser, run
from kinetrace.bandit import compare_scored, validate_rows, validate_splits
from kinetrace.data import output_file, read_run
from kinetrace.ranking import compare_runs, summarize
from kinetrace.sft import training_pairs


def experiment(pilot, epsilon=0.1, seeds=(42, 43, 44)):
    # Validate settings and cached evaluation before allocating a model.
    if not 0 < epsilon <= 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Need epsilon in (0, 1] and distinct seeds")
    report_path = pilot / "adapter/train-report.json"
    reference = json.loads(report_path.read_text())
    train, val, adapted, base = [list(read_run(pilot / name)) for name in
                                ("train.jsonl", "val.jsonl", "val-adapted.jsonl", "val-base.jsonl")]
    compare_runs(val, adapted)
    compare_runs(val, base)
    validate_rows(train, 5, scored=False)
    depth = min(5, len(val[0]["candidates"]))
    validate_rows(adapted, depth)
    validate_splits(train, adapted)
    pairs = training_pairs(train)
    if len(pairs) != reference["available_pairs"]:
        raise ValueError("Training pair count changed from the retained LoRA run")
    pair_file = pilot / "adapter/training-pairs.jsonl"
    if pair_file.exists():
        saved = [(r["query"], r["positive"], r["negative"]) for r in read_run(pair_file)]
        if saved != pairs:
            raise ValueError("Training pairs changed from the retained LoRA run")
    out = Path(tempfile.mkdtemp(prefix="bandit-offline-", dir=pilot))
    print(f"Offline bandit experiment directory: {out}", flush=True)
    (out / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True))
    (out / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    sources = [pilot / name for name in ("train.jsonl", "val.jsonl", "val-adapted.jsonl", "val-base.jsonl")]
    sources.extend(path for path in sorted((pilot / "adapter").iterdir()) if path.is_file())
    provenance = {"pilot": str(pilot), "reference": reference,
                  "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    device = {"device": reference["device"]}
    if reference["device"] == "cuda":
        import torch

        props = torch.cuda.get_device_properties(0)
        device.update(gpu=props.name, visible_gib=props.total_memory / 2**30)
        torch.cuda.reset_peak_memory_stats()
    with output_file(out / "scoring-device.json") as stream:
        json.dump(device, stream, indent=2)
    print(json.dumps(device), flush=True)
    scoring = parser().parse_args([
        "rerank", "--input", str(pilot / "train.jsonl"), "--out", str(out / "train-adapted.jsonl"),
        "--videos", str(pilot / "data/videos"), "--qwen-model", reference["base"],
        "--adapter", str(pilot / "adapter"), "--device", reference["device"],
        "--frames", str(reference["frames"]), "--top-k", str(depth)])
    run(scoring)
    if reference["device"] == "cuda":
        torch.cuda.synchronize()
        device.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    with output_file(out / "scoring-device.json") as stream:
        json.dump(device, stream, indent=2)
    result = compare_scored(SimpleNamespace(train=scoring.out, validation=pilot / "val-adapted.jsonl",
                                            out=out / "comparison", top_k=depth,
                                            epsilon=epsilon, seeds=list(seeds)))
    result["base_qwen"] = summarize(base)
    with output_file(out / "summary.json") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(json.dumps({key: result[key] for key in ("base_qwen", "lora", "r1_across_seeds")}, indent=2))
    print(f"Detailed results: {out / 'summary.json'}", flush=True)
    return out


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("pilot", type=Path)
    cli.add_argument("--epsilon", type=float, default=0.1)
    cli.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    arguments = cli.parse_args()
    experiment(arguments.pilot, arguments.epsilon, arguments.seeds)
