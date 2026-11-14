"""Run the base-Qwen frame-order diagnostic on an existing training-side validation split."""
import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from kinetrace.__main__ import parser
from kinetrace.data import output_file, read_run
from kinetrace.ranking import compare_runs
from kinetrace.temporal import compare_order


def experiment(pilot):
    reference_path = pilot / "adapter/train-report.json"
    reference = json.loads(reference_path.read_text())
    source = pilot / "val.jsonl"
    baseline = pilot / "val-base.jsonl"
    compare_runs(list(read_run(source)), list(read_run(baseline)))
    out = Path(tempfile.mkdtemp(prefix="temporal-base-", dir=pilot))
    print(f"Base-Qwen frame-order directory: {out}", flush=True)
    (out / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True))
    (out / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    provenance = {"base": reference["base"], "frames": reference["frames"],
                  "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in (reference_path, source, baseline)}}
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    args = parser().parse_args([
        "compare-order", "--input", str(source), "--out", str(out / "comparison"),
        "--videos", str(pilot / "data/videos"), "--qwen-model", reference["base"],
        "--device", reference["device"], "--frames", str(reference["frames"]), "--top-k", "5"])
    report = compare_order(args)
    cached = list(read_run(baseline))
    fresh = list(read_run(args.out / "forward.jsonl"))
    report["cached_base_to_fresh_forward"] = compare_runs(cached, fresh)
    cached_top = {r["query_id"]: r["candidates"][0]["video_id"] for r in cached}
    report["cached_base_top1_disagreements"] = sum(
        cached_top[r["query_id"]] != r["candidates"][0]["video_id"] for r in fresh)
    with output_file(out / "summary.json") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    keys = ("forward", "reverse_reference_labels", "top1_disagreements", "scored_annotated_targets")
    print(json.dumps({key: report[key] for key in keys}, indent=2))
    print(f"Detailed results: {out / 'summary.json'}", flush=True)
    return out


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("pilot", type=Path)
    experiment(cli.parse_args().pilot)
