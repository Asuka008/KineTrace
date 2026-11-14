"""Compare one soft-negative SFT candidate with the retained hard-negative pilot on validation."""
import argparse
import json
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from kinetrace.data import output_file, read_run
from kinetrace.ranking import compare_runs, summarize
from kinetrace.review import write_page
from kinetrace.sft import training_pairs


def compare_validation(current, candidate, videos, out):
    paired = compare_runs(current, candidate)
    by_id = {row["query_id"]: row for row in candidate}
    disagreements = []
    for row in current:
        first = row["candidates"][0]["video_id"]
        second = by_id[row["query_id"]]["candidates"][0]["video_id"]
        if first != second:
            disagreements.append({"query_id": row["query_id"], "query": row["query"],
                                  "current": first, "candidate": second})
    rng = random.Random(42)
    sample = rng.sample(disagreements, min(20, len(disagreements)))
    report = {"current": summarize(current), "candidate": summarize(candidate),
              "top1_disagreements": len(disagreements), "paired": paired, "review_sample": sample}
    if sample:
        write_page(out / "review.html", videos, sample, ("current", "candidate"), rng, preference=True)
    with output_file(out / "validation.json") as stream:
        json.dump(report, stream, indent=2)
    return report


def tally(report, judgments):
    sample = {r["query_id"]: r for r in report["review_sample"]}
    counts = Counter()
    for j in judgments:
        r = sample.pop(j["query_id"], None)
        if r is None or {r["current"], r["candidate"]} != {j["video_a"], j["video_b"]}:
            raise ValueError(f"Review does not match this experiment at {j['query_id']}")
        choice = j["judgment"]
        if choice in ("A", "B"):
            selected = j["video_a"] if choice == "A" else j["video_b"]
            counts["current" if selected == r["current"] else "candidate"] += 1
        elif choice in ("both", "neither", "audio_required", "uncertain", "visual_evidence_insufficient"):
            counts[choice] += 1
        elif choice is None:
            counts["unrated"] += 1
        else:
            raise ValueError(f"Unknown judgment: {choice}")
    counts["unrated"] += len(sample)
    return dict(counts)


def experiment(pilot):
    ref = json.loads((pilot / "adapter/train-report.json").read_text())
    if ref.get("negative_strategy", "hard") != "hard" or ref.get("negative_yes_weight", 0) != 0:
        raise ValueError("Expected the original hard-negative SFT pilot")
    current = list(read_run(pilot / "val-adapted.jsonl"))
    val = list(read_run(pilot / "val.jsonl"))
    compare_runs(val, current)
    depth = min(5, summarize(val)["candidate_depth"])
    train = list(read_run(pilot / "train.jsonl"))
    train_ids = {row["target"] for row in train}
    train_ids.update(c["video_id"] for row in train for c in row["candidates"])
    val_ids = {row["target"] for row in val}
    val_ids.update(c["video_id"] for row in val for c in row["candidates"])
    if train_ids & val_ids:
        raise ValueError("The training and validation galleries overlap")
    pairs = training_pairs(train)
    if len(pairs) != ref["available_pairs"]:
        raise ValueError("The number of eligible training pairs changed from the reference run")
    # Older pilots predate the saved pair manifest, so retain their count check.
    saved_pairs = pilot / "adapter/training-pairs.jsonl"
    if saved_pairs.exists():
        original = [(row["query"], row["positive"], row["negative"]) for row in read_run(saved_pairs)]
        if pairs != original:
            raise ValueError("Training pairs changed from the reference run")
    videos = pilot / "data/videos"
    out = Path(tempfile.mkdtemp(prefix="soft-negative-", dir=pilot))
    print(f"SFT comparison directory: {out}", flush=True)
    cli = [sys.executable, "-m", "kinetrace"]
    shared = ["--videos", str(videos), "--device", ref["device"], "--frames", str(ref["frames"]),
              "--qwen-model", ref["base"]]
    (out / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True))
    subprocess.run([*cli, "train-lora", *shared, "--input", str(pilot / "train.jsonl"),
                    "--negative-strategy", "hard", "--negative-yes-weight", "0.1",
                    "--steps", str(ref["steps"]), "--seed", str(ref["seed"]), "--out", str(out / "adapter")],
                   check=True)
    ranked = out / "val-soft.jsonl"
    subprocess.run([*cli, "rerank", *shared, "--input", str(pilot / "val.jsonl"), "--top-k", str(depth),
                    "--adapter", str(out / "adapter"), "--out", str(ranked)], check=True)
    report = compare_validation(current, list(read_run(ranked)), videos, out)
    print(json.dumps({k: report[k] for k in ("current", "candidate", "top1_disagreements")}, indent=2))
    if report["review_sample"]:
        print(f"Blind review: {out / 'review.html'} ({len(report['review_sample'])} pairs)")
    else:
        print("Both checkpoints selected the same first clip for every validation query.")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="original pilot, or comparison directory when tallying")
    parser.add_argument("--judgments", type=Path, help="tally the browser-exported JSON on CPU")
    args = parser.parse_args()
    if args.judgments:
        report = json.loads((args.directory / "validation.json").read_text())
        print(json.dumps(tally(report, json.loads(args.judgments.read_text())), indent=2))
    else:
        experiment(args.directory)
