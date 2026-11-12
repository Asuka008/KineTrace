"""Paired Top-5 and Top-10 measurements on a saved CLIP shortlist."""
import math
import statistics
from time import perf_counter

from .data import read_run, write_run
from .ranking import rerank_prefix, summarize
from .reranker import QwenScorer


def paired_query(row, scorer, synchronize):
    candidates = row["candidates"]
    if len(candidates) < 10 or any(item["retrieval_rank"] != i or "rerank_score" in item
                                   for i, item in enumerate(candidates, 1)):
        raise ValueError("Expected raw CLIP retrieval order with at least ten candidates")
    synchronize()
    start = perf_counter()
    five = rerank_prefix(row, 5, scorer)
    synchronize()
    five_seconds = perf_counter() - start

    start = perf_counter()
    remaining = dict(row, candidates=candidates[5:10])
    next_five = rerank_prefix(remaining, 5, scorer)
    # Stable sorting keeps CLIP rank as the tie-breaker within and across groups.
    ten_best = sorted(five["candidates"][:5] + next_five["candidates"],
                      key=lambda item: -item["rerank_score"])
    ten = dict(row, candidates=ten_best + candidates[10:])
    synchronize()
    ten_seconds = five_seconds + (perf_counter() - start)
    return five, ten, five_seconds, ten_seconds


def latency_ms(seconds):
    ordered = sorted(t * 1000 for t in seconds)
    return {"p50": statistics.median(ordered), "p95": ordered[math.ceil(0.95 * len(ordered)) - 1]}


def compare_depths(args):
    if args.out.resolve() == args.input.resolve():
        raise ValueError("Choose a separate output for Top-10 results")
    rows = list(read_run(args.input))
    if not rows:
        raise ValueError("Empty retrieval input")
    scorer = QwenScorer(args.qwen_model, args.device, args.videos, args.frames)
    device = {"device": args.device}
    synchronize = lambda: None
    if args.device == "cuda":
        cuda = scorer.torch.cuda
        synchronize = cuda.synchronize
        props = cuda.get_device_properties(0)
        device.update(name=props.name, visible_gib=props.total_memory / 2**30)

    paired_query(rows[0], scorer, synchronize)  # unmeasured warm-up
    fives, tens, time5, time10 = [], [], [], []
    gained = lost = 0
    for row in rows:
        five, ten, t5, t10 = paired_query(row, scorer, synchronize)
        hit5 = five["candidates"][0]["video_id"] == row["target"]
        hit10 = ten["candidates"][0]["video_id"] == row["target"]
        gained += int(hit10 and not hit5)
        lost += int(hit5 and not hit10)
        fives.append(five)
        tens.append(ten)
        time5.append(t5)
        time10.append(t10)
    metrics5, metrics10 = summarize(fives), summarize(tens)
    write_run(args.out, tens)
    return {"device": device, "model": args.qwen_model, "frames": args.frames,
            "top5": metrics5, "top10": metrics10, "gained_vs_top5": gained, "lost_vs_top5": lost,
            "rerank_ms_top5": latency_ms(time5), "rerank_ms_top10": latency_ms(time10),
            "warmup_queries": 1, "input": str(args.input), "top10_run": str(args.out)}
