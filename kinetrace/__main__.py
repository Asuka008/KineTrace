import argparse
import json
from pathlib import Path

from .data import check_videos, read_queries, read_run, write_run
from .ranking import compare_top1, rerank_prefix, summarize, target_ranks
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
    metrics = commands.add_parser("metrics", help="report text-to-video Recall@K")
    compare = commands.add_parser("compare-top1", help="show queries promoted to or displaced from first place")
    ranks = commands.add_parser("target-ranks", help="export the target rank for each labeled query")

    for cmd in (build, search, evaluate, rerank):
        cmd.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    for cmd in (build, search, evaluate):
        cmd.add_argument("--index", type=Path, required=True)
    for cmd in (check, build, evaluate):
        cmd.add_argument("--csv", type=Path, required=True)
        cmd.add_argument("--limit", type=positive, help="read only the first N query rows")
    for cmd in (check, build, rerank):
        cmd.add_argument("--videos", required=True, type=Path)
    for cmd in (build, rerank):
        cmd.add_argument("--frames", type=positive, default=4)
    for cmd in (search, evaluate, rerank):
        cmd.add_argument("--top-k", type=positive, default=5 if cmd is rerank else 20)
    for cmd in (evaluate, rerank, ranks):
        cmd.add_argument("--out", required=True, type=Path)
    for cmd in (rerank, metrics, ranks, compare):
        cmd.add_argument("--input", required=True, type=Path)
    build.add_argument("--clip-model", default=CLIP_MODEL, help="HF model ID or local snapshot")
    search.add_argument("--query", required=True)
    search.add_argument("--retrieval-text", help="optional English rendering for the CLIP encoder")
    search.add_argument("--out", type=Path, help="save a single-query candidate run for reranking")
    evaluate.add_argument("--text-batch", default=8, type=positive)
    rerank.add_argument("--qwen-model", default=QWEN_MODEL, help="HF model ID or local snapshot")
    return cli


def run(args):
    if args.command == "check-data":
        queries = read_queries(args.csv, args.limit)
        ids = check_videos(queries, args.videos)
        return {"queries": len(queries), "videos": len(ids)}
    if args.command == "target-ranks":
        if args.input.resolve() == args.out.resolve():
            raise ValueError("Choose a separate output for target ranks")
        write_run(args.out, target_ranks(read_run(args.input)))
        return {"target_ranks": str(args.out)}
    if args.command == "compare-top1":
        return compare_top1(read_run(args.input))
    if args.command == "metrics":
        return summarize(read_run(args.input))
    if args.command == "rerank":
        from .reranker import QwenScorer

        scorer = QwenScorer(args.qwen_model, args.device, args.videos, args.frames)
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
