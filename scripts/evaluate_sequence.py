"""Score the published VITATECS Sequence caption pairs on the original MSR-VTT clips."""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

from kinetrace.data import output_file, write_run
from kinetrace.reranker import QwenScorer
from kinetrace.temporal import SCORE_TOLERANCE
from kinetrace.video import read_clip


REVISION = "58bff59bb1d3ed287983fca8ca7589498f4db6b2"
LABEL_URL = f"https://raw.githubusercontent.com/lscpku/VITATECS/{REVISION}/data/Sequence.jsonl"
LABEL_SHA256 = "b839b664486724ddf90343af5fa674ce6ab81ffd219d8fd520b377c7ad39e8dd"


def load_labels(root, supplied):
    path = supplied or root / f"VITATECS-Sequence-{REVISION[:8]}.jsonl"
    download = supplied is None and not path.exists()
    if download:
        with urlopen(LABEL_URL, timeout=45) as response:
            raw = response.read()
    else:
        raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != LABEL_SHA256:
        raise ValueError(f"Label checksum mismatch. Expected the file from {LABEL_URL}")
    if download:
        with output_file(path) as stream:
            stream.write(raw.decode("utf-8"))
    pairs = []
    for line, record in enumerate(raw.splitlines(), 1):
        if not record.strip():
            continue
        row = json.loads(record)
        if row["src_dataset"] != "MSRVTT":
            continue
        if row["aspect"] != "Sequence" or not re.fullmatch(r"video[0-9]+\.mp4", row["video_name"]):
            raise ValueError(f"Unexpected Sequence annotation at line {line}")
        pairs.append({"line": line, "video_name": row["video_name"], "caption": row["caption"],
                      "counterfactual": row["counterfactual"]})
    if not pairs:
        raise ValueError("No MSRVTT Sequence pairs in the label file")
    return pairs


def video_paths(names, video_dir, archive_path, temporary):
    paths = {name: video_dir / name for name in names}
    missing = {name for name, path in paths.items() if not path.is_file()}
    if missing:
        if not archive_path.is_file():
            raise FileNotFoundError(f"Missing {len(missing)} required clips in {video_dir} and no {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            members = {}
            for member in archive.infolist():
                name = Path(member.filename).name
                if name in missing and not member.is_dir():
                    if name in members:
                        raise ValueError(f"Ambiguous clip filename in {archive_path}: {name}")
                    members[name] = member
            absent = sorted(missing.difference(members))
            if absent:
                raise FileNotFoundError(f"Missing required clips in {video_dir} and {archive_path}: {absent}")
            for name, member in members.items():
                paths[name] = temporary / name
                with archive.open(member) as source, paths[name].open("wb") as target:
                    shutil.copyfileobj(source, target)
    return paths


def evaluate(args):
    pairs = load_labels(args.msrvtt_root, args.labels)
    videos = args.videos or args.msrvtt_root / "videos"
    args.out_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out_root) as temporary:
        paths = video_paths({r["video_name"] for r in pairs}, videos,
                            args.msrvtt_root / "MSRVTT_Videos.zip", Path(temporary))
        device = {"device": args.device}
        if args.device == "cuda":
            import torch

            props = torch.cuda.get_device_properties(0)
            device.update(gpu=props.name, visible_gib=props.total_memory / 2**30)
            torch.cuda.reset_peak_memory_stats()
        print(json.dumps(device), file=sys.stderr, flush=True)
        scorer = QwenScorer(args.qwen_model, args.device, videos, frames=4)
        results = []
        counts = {"correct": 0, "incorrect": 0, "tie": 0}
        for pair in pairs:
            images, metadata = read_clip(paths[pair["video_name"]], 4)
            positive = scorer.score_frames(pair["caption"], images, metadata)
            negative = scorer.score_frames(pair["counterfactual"], images, metadata)
            margin = positive - negative
            if not math.isfinite(margin):
                raise ValueError(f"Non-finite score at label line {pair['line']}")
            if abs(margin) <= SCORE_TOLERANCE:
                outcome = "tie"
            else:
                outcome = "correct" if margin > 0 else "incorrect"
            counts[outcome] += 1
            results.append(dict(pair, sampled_indices=metadata["frames_indices"], caption_score=positive,
                                counterfactual_score=negative, outcome=outcome))
        if args.device == "cuda":
            device.update(peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                          peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    out = Path(tempfile.mkdtemp(prefix="vitatecs-sequence-", dir=args.out_root))
    report = {"task": "VITATECS Sequence / MSRVTT caption discrimination", "labels_url": LABEL_URL,
              "labels_sha256": LABEL_SHA256, "model": args.qwen_model, "frames": 4, "device": device,
              "pairs": len(results), "videos": len(paths), **counts, "tie_atol": SCORE_TOLERANCE,
              "strict_pair_accuracy": counts["correct"] / len(results)}
    (out / "commit.txt").write_text(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True))
    (out / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    write_run(out / "pairs.jsonl", results)
    with output_file(out / "summary.json") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print(json.dumps(report, indent=2))
    print(f"Results: {out}", flush=True)
    return report


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--msrvtt-root", type=Path,
                     default=Path(os.getenv("MSRVTT_ROOT", Path.home() / "datasets/MSRVTT")))
    cli.add_argument("--videos", type=Path, help="flat MP4 directory (default: MSRVTT_ROOT/videos)")
    cli.add_argument("--labels", type=Path, help="local copy of the pinned Sequence.jsonl for offline use")
    cli.add_argument("--qwen-model", default=os.getenv("QWEN_DIR", str(Path.home() / "models/Qwen3-VL-Reranker-2B-9887917")))
    cli.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    cli.add_argument("--out-root", type=Path, default=Path(os.getenv("RUN_ROOT", "runs")))
    return cli


if __name__ == "__main__":
    evaluate(parser().parse_args())
