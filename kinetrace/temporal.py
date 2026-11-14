"""Base-Qwen frame-order sensitivity on fixed retrieved candidates."""
import hashlib
import json
import math
import statistics
import sys
import tempfile
from copy import deepcopy
from pathlib import Path
from time import perf_counter

from .data import read_run, write_run
from .depths import latency_ms
from .ranking import compare_runs, rerank_prefix, summarize
from .reranker import QWEN_MODEL, QWEN_REVISION, QwenScorer
from .video import read_clip


SCORE_TOLERANCE = 1e-6


def paired_query(row, scorer, videos, frames, depth, synchronize, reverse_first=False):
    scores = {"forward": {}, "reverse": {}}
    seconds = {"decode": 0.0, "forward": 0.0, "reverse": 0.0}
    details = []
    for position, candidate in enumerate(row["candidates"][:depth]):
        video_id = candidate["video_id"]
        start = perf_counter()
        images, metadata = read_clip(videos / f"{video_id}.mp4", frames)
        seconds["decode"] += perf_counter() - start
        if len(images) < 2:
            raise ValueError(f"Frame-order comparison needs at least two sampled frames: {video_id}")
        # Keep the increasing timestamp slots fixed while reversing only frame content.
        orders = ("reverse", "forward") if (reverse_first + position) % 2 else ("forward", "reverse")
        for order in orders:
            ordered = images if order == "forward" else list(reversed(images))
            synchronize()
            start = perf_counter()
            value = float(scorer.score_frames(row["query"], ordered, deepcopy(metadata)))
            synchronize()
            seconds[order] += perf_counter() - start
            if not math.isfinite(value):
                raise ValueError(f"Non-finite {order} score for {video_id}")
            scores[order][video_id] = value
        details.append({"query_id": row["query_id"], "video_id": video_id,
                        "is_annotated_target": video_id == row["target"],
                        "forward_score": scores["forward"][video_id],
                        "reverse_score": scores["reverse"][video_id],
                        "forward_minus_reverse": scores["forward"][video_id] - scores["reverse"][video_id],
                        "timestamp_slots": metadata,
                        "reverse_source_indices": list(reversed(metadata["frames_indices"]))})
    forward = rerank_prefix(row, depth, lambda query, video: scores["forward"][video])
    reverse = rerank_prefix(row, depth, lambda query, video: scores["reverse"][video])
    return forward, reverse, details, seconds


def score_deltas(details):
    values = [r["forward_minus_reverse"] for r in details]
    if not values:
        return {"pairs": 0, "mean_forward_minus_reverse": None, "mean_absolute_delta": None,
                "max_absolute_delta": None, "forward_higher": 0, "reverse_higher": 0, "ties": 0}
    return {"pairs": len(values), "mean_forward_minus_reverse": statistics.mean(values),
            "mean_absolute_delta": statistics.mean(abs(v) for v in values),
            "max_absolute_delta": max(abs(v) for v in values),
            "forward_higher": sum(v > SCORE_TOLERANCE for v in values),
            "reverse_higher": sum(v < -SCORE_TOLERANCE for v in values),
            "ties": sum(abs(v) <= SCORE_TOLERANCE for v in values)}


def compare_order(args):
    if args.out.exists():
        raise FileExistsError(f"Choose a new comparison directory: {args.out}")
    if args.frames < 2:
        raise ValueError("Frame-order comparison requires at least two frames")
    rows = list(read_run(args.input))
    summarize(rows)
    compare_runs(rows, rows)  # Validate query IDs before model loading.
    for row in rows:
        if not row["query"].strip() or any(
                c["retrieval_rank"] != rank or "rerank_score" in c
                for rank, c in enumerate(row["candidates"], 1)):
            raise ValueError("Expected raw CLIP retrieval order with nonempty query text")
    device = {"device": args.device}
    synchronize = lambda: None
    cuda = None
    if args.device == "cuda":
        import torch

        cuda = torch.cuda
        props = cuda.get_device_properties(0)
        device.update(gpu=props.name, visible_gib=props.total_memory / 2**30)
        synchronize = cuda.synchronize
    print(json.dumps({"scoring_device": device}), file=sys.stderr, flush=True)
    scorer = QwenScorer(args.qwen_model, args.device, args.videos, args.frames)
    paired_query(rows[0], scorer, args.videos, args.frames, 1, synchronize)
    if cuda is not None:
        cuda.reset_peak_memory_stats()
    forward, reverse, details, query_changes = [], [], [], []
    timings = {"decode": [], "forward": [], "reverse": []}
    for index, row in enumerate(rows):
        first, second, pairs, seconds = paired_query(
            row, scorer, args.videos, args.frames, args.top_k, synchronize, bool(index % 2))
        forward.append(first)
        reverse.append(second)
        details.extend(pairs)
        for name in timings:
            timings[name].append(seconds[name])
        if first["candidates"][0]["video_id"] != second["candidates"][0]["video_id"]:
            query_changes.append({"query_id": row["query_id"], "query": row["query"],
                                  "target": row["target"],
                                  "forward_top": first["candidates"][0]["video_id"],
                                  "reverse_top": second["candidates"][0]["video_id"]})
        print(f"Frame-order queries completed: {index + 1}/{len(rows)}", file=sys.stderr, flush=True)
    if cuda is not None:
        device.update(peak_allocated_gib=cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=cuda.max_memory_reserved() / 2**30)
    target_details = [r for r in details if r["is_annotated_target"]]
    report = {"experiment": "base_qwen_frame_order_sensitivity", "device": device,
              "model": args.qwen_model,
              "revision": QWEN_REVISION if args.qwen_model == QWEN_MODEL else None,
              "frames": args.frames, "rerank_top_k": args.top_k,
              "input": str(args.input), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
              "intervention": "reverse sampled frame content within fixed increasing timestamp slots",
              "evaluation_scope": "original video labels applied to both orders for sensitivity diagnostics",
              "score_tie_atol": SCORE_TOLERANCE, "warmup_candidate_pairs": 1,
              "execution_order": "alternate first order across queries and candidates",
              "forward": summarize(forward), "reverse_reference_labels": summarize(reverse),
              "reference_label_changes": compare_runs(forward, reverse),
              "top1_disagreements": len(query_changes), "top1_changes": query_changes,
              "all_scored_candidates": score_deltas(details),
              "scored_annotated_targets": score_deltas(target_details),
              "targets_outside_scored_prefix": len(rows) - len(target_details),
              "shared_decode_ms_per_query": latency_ms(timings["decode"]),
              "scoring_ms_per_query": {order: latency_ms(timings[order]) for order in ("forward", "reverse")},
              "timing_scope": "scoring includes preprocessing and synchronized model forward; decode recorded separately"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temporary:
        stage = Path(temporary) / "comparison"
        stage.mkdir()
        write_run(stage / "forward.jsonl", forward)
        write_run(stage / "reverse.jsonl", reverse)
        write_run(stage / "score-pairs.jsonl", details)
        (stage / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        stage.rename(args.out)
    return report
