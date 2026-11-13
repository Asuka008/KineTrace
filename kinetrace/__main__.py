import argparse
import json
from pathlib import Path

from .data import check_videos, read_queries, read_run, write_run
from .ranking import compare_runs, compare_top1, rerank_prefix, summarize, target_ranks
from .reranker import QWEN_MODEL
from .retrieval import CLIP_MODEL


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def parser():
    cli = argparse.ArgumentParser(description="KineTrace video search")
    commands = cli.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check-data", help="check split rows and MP4 paths before submitting")
    build = commands.add_parser("build", help="encode clips and persist a FAISS index")
    search = commands.add_parser("search", help="search an existing index with free text")
    evaluate = commands.add_parser("retrieve", help="retrieve candidates for a labeled CSV")
    rerank = commands.add_parser("rerank", help="apply Qwen to a saved candidate prefix")
    depths = commands.add_parser("compare-depths", help="measure paired Top-5 and Top-10 reranking")
    train = commands.add_parser("train-lora", help="fit a yes/no Qwen adapter on train-side CLIP candidates")
    train.add_argument("--steps", type=positive, default=1, help="positive/negative pairs of updates (one optimizer step per pair)")
    train.add_argument("--seed", type=int, default=42)
    rerank.add_argument("--adapter", type=Path, help="load a trained PEFT adapter on the selected Qwen base")
    metrics = commands.add_parser("metrics", help="report text-to-video Recall@K")
    compare = commands.add_parser("compare-top1", help="show queries promoted to or displaced from first place")
    paired = commands.add_parser("compare-runs", help="compare first-place hits on identical labeled shortlists")
    paired.add_argument("--baseline", required=True, type=Path)
    paired.add_argument("--input", required=True, type=Path)
    ranks = commands.add_parser("target-ranks", help="export the target rank for each labeled query")

    review = commands.add_parser("review-lost", help="sample top-one losses for paired video review")
    review.add_argument("--limit", type=positive, default=5)
    changes = commands.add_parser("review-changes", help="blind review of gains and losses from compare-runs")
    changes.add_argument("--per-outcome", type=positive, default=10, help="maximum cases per group")
    for cmd in (review, changes):
        cmd.add_argument("--seed", type=int, default=42)
    review.add_argument("--exclude", nargs="+", default=[], metavar="QUERY_ID")

    for cmd in (build, search, evaluate, rerank, depths, train):
        cmd.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    for cmd in (build, search, evaluate):
        cmd.add_argument("--index", type=Path, required=True)
    for cmd in (check, build, evaluate):
        cmd.add_argument("--csv", type=Path, required=True)
        cmd.add_argument("--limit", type=positive, help="read only the first N query rows")
    for cmd in (check, build, rerank, review, changes, depths, train):
        cmd.add_argument("--videos", required=True, type=Path)
    for cmd in (build, rerank, depths, train):
        cmd.add_argument("--frames", type=positive, default=4)
    for cmd in (search, evaluate, rerank):
        cmd.add_argument("--top-k", type=positive, default=5 if cmd is rerank else 20)
    for cmd in (evaluate, rerank, ranks, review, changes, depths, train):
        cmd.add_argument("--out", required=True, type=Path)
    for cmd in (rerank, metrics, ranks, compare, review, changes, depths, train):
        cmd.add_argument("--input", required=True, type=Path)
    build.add_argument("--clip-model", default=CLIP_MODEL, help="HF model ID or local snapshot")
    search.add_argument("--query", required=True)
    search.add_argument("--retrieval-text", help="optional English rendering for the CLIP encoder")
    search.add_argument("--out", type=Path, help="save a single-query candidate run for reranking")
    evaluate.add_argument("--text-batch", default=8, type=positive)
    for cmd in (rerank, depths, train):
        cmd.add_argument("--qwen-model", default=QWEN_MODEL, help="HF model ID or local snapshot")
    return cli


def run(args):
    if args.command == "train-lora":
        from .sft import train_lora

        return train_lora(args)
    if args.command == "compare-depths":
        from .depths import compare_depths

        return compare_depths(args)
    if args.command in ("review-lost", "review-changes"):
        from .review import write_review

        return write_review(args)
    if args.command == "check-data":
        queries = read_queries(args.csv, args.limit)
        ids = check_videos(queries, args.videos)
        return {"queries": len(queries), "videos": len(ids)}
    if args.command == "target-ranks":
        if args.input.resolve() == args.out.resolve():
            raise ValueError("Choose a separate output for target ranks")
        write_run(args.out, target_ranks(read_run(args.input)))
        return {"target_ranks": str(args.out)}
    if args.command == "compare-runs":
        return compare_runs(read_run(args.baseline), read_run(args.input))
    if args.command == "compare-top1":
        return compare_top1(read_run(args.input))
    if args.command == "metrics":
        return summarize(read_run(args.input))
    if args.command == "rerank":
        from .reranker import QwenScorer

        scorer = QwenScorer(args.qwen_model, args.device, args.videos, args.frames)
        if args.adapter:
            from peft import PeftModel

            scorer.model = PeftModel.from_pretrained(scorer.model, str(args.adapter)).eval()
        rows = (rerank_prefix(row, args.top_k, scorer) for row in read_run(args.input))
        write_run(args.out, rows)
        return {"run": str(args.out), "rerank_top_k": args.top_k, "frames": args.frames}

    from .retrieval import Retriever, build_index, retrieve_queries

    if args.command == "build":
        return build_index(args, read_queries(args.csv, args.limit))
    retriever = Retriever(args.index, args.device)
    if args.command == "search":
        text = args.retrieval_text or args.query
        row = {"query": args.query, "retrieval_text": text,
               "candidates": retriever.search([text], args.top_k)[0]}
        if args.out:
            write_run(args.out, [row])
        return row
    queries = read_queries(args.csv, args.limit)
    write_run(args.out, retrieve_queries(retriever, queries, args.top_k, args.text_batch))
    return summarize(read_run(args.out))


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), indent=2, ensure_ascii=False))
