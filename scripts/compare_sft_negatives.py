"""Train the random-negative control for an existing SFT pilot and compare validation runs."""
import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from kinetrace.data import output_file, read_run
from kinetrace.ranking import compare_runs, summarize


def compare_negatives(pilot):
    training = json.loads((pilot / "adapter/train-report.json").read_text())
    if training.get("negative_strategy", "hard") != "hard":
        raise ValueError("Expected a hard-negative reference adapter")
    train = pilot / "train.jsonl"
    val = pilot / "val.jsonl"
    shortlist = list(read_run(val))
    base = list(read_run(pilot / "val-base.jsonl"))
    hard = list(read_run(pilot / "val-adapted.jsonl"))
    depth = min(5, summarize(shortlist)["candidate_depth"])
    compare_runs(shortlist, base)
    base_hard = compare_runs(base, hard)
    train_ids = {row["target"] for row in read_run(train)}
    val_ids = {row["target"] for row in shortlist}
    val_ids.update(c["video_id"] for row in shortlist for c in row["candidates"])
    if train_ids & val_ids:
        raise ValueError("The training and validation galleries overlap")

    out = Path(tempfile.mkdtemp(prefix="random-negative-", dir=pilot))
    print(f"Random-negative outputs: {out}", flush=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
    (out / "commit.txt").write_text(commit)
    shared = ["--qwen-model", training["base"], "--device", training["device"],
              "--videos", str(pilot / "data/videos"), "--frames", str(training["frames"])]
    cli = [sys.executable, "-m", "kinetrace"]
    adapter = out / "adapter"
    subprocess.run([*cli, "train-lora", *shared, "--input", str(train), "--out", str(adapter),
                    "--seed", str(training["seed"]), "--steps", str(training["steps"]),
                    "--negative-strategy", "random"], check=True)
    random_training = json.loads((adapter / "train-report.json").read_text())
    if random_training["available_pairs"] != training["available_pairs"]:
        raise ValueError("The number of eligible training pairs changed from the reference run")
    ranked = out / "val-random.jsonl"
    subprocess.run([*cli, "rerank", *shared, "--input", str(val), "--adapter", str(adapter),
                    "--top-k", str(depth), "--out", str(ranked)], check=True)
    random_run = list(read_run(ranked))
    comparisons = {"base_to_hard": base_hard, "base_to_random": compare_runs(base, random_run),
                   "hard_to_random": compare_runs(hard, random_run)}
    metrics = {name: summarize(rows) for name, rows in
               (("base", base), ("hard", hard), ("random", random_run))}
    with output_file(out / "validation.json") as stream:
        json.dump({"metrics": metrics, "comparisons": comparisons}, stream, indent=2, ensure_ascii=False)
    print(json.dumps({"metrics": metrics, "comparisons": {
        name: {key: value for key, value in report.items() if key != "changes"}
        for name, report in comparisons.items()}}, indent=2))
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pilot", type=Path, help="completed sft_pilot.sh run directory")
    compare_negatives(parser.parse_args().pilot)
