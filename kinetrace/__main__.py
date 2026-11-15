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
    bandit = commands.add_parser("bandit-compare", help="compare a simulated bandit with scored LoRA validation")
    bandit.add_argument("--train", type=Path, required=True)
    bandit.add_argument("--validation", type=Path, required=True)
    bandit.add_argument("--out", type=Path, required=True)
    bandit.add_argument("--top-k", type=positive, default=5)
    bandit.add_argument("--epsilon", type=float, default=0.1)
    bandit.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    depths = commands.add_parser("compare-depths", help="measure paired Top-5 and Top-10 reranking")
    order = commands.add_parser("compare-order", help="measure base Qwen sensitivity to reversed sampled frames")
    order_review = commands.add_parser("review-order", help="build a blind frame-order review from a compare-order directory")
    order_review.add_argument("--seed", type=int, default=42)
    order_review.add_argument("--limit", type=positive, help="review a random subset of the scored targets")
    tally = commands.add_parser("tally-order", help="evaluate saved order scores against completed human labels")
    tally.add_argument("--review", required=True, type=Path, help="directory produced by review-order")
    tally.add_argument("--judgments", required=True, type=Path, help="JSON exported from the review page")
    train = commands.add_parser("train-lora", help="fit a yes/no Qwen adapter on train-side CLIP candidates")
    train.add_argument("--steps", type=positive, default=1, help="positive/negative pairs of updates (one optimizer step per pair)")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--negative-strategy", choices=("hard", "random"), default="hard",
                       help="use the highest-ranked eligible CLIP candidate or sample a training video uniformly")
    train.add_argument("--negative-yes-weight", default=0.0, type=float,
                       help="soft target mass on yes for a mined negative (0 <= weight < 0.5)")
    adapters = rerank.add_mutually_exclusive_group()
    adapters.add_argument("--adapter", type=Path, help="load a trained PEFT adapter on the selected Qwen base")
    adapters.add_argument("--current-sft", action="store_true", help="use the retained school-server SFT pilot checkpoint")
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

    for cmd in (build, search, evaluate, rerank, depths, train, order):
        cmd.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    for cmd in (build, search, evaluate):
        cmd.add_argument("--index", type=Path, required=True)
    for cmd in (check, build, evaluate):
        cmd.add_argument("--csv", type=Path, required=True)
        cmd.add_argument("--limit", type=positive, help="read only the first N query rows")
    for cmd in (check, build, rerank, review, changes, depths, train, order, order_review):
        cmd.add_argument("--videos", required=True, type=Path)
    for cmd in (build, rerank, depths, train, order):
        cmd.add_argument("--frames", type=positive, default=4)
    for cmd in (search, evaluate, rerank, order):
        cmd.add_argument("--top-k", type=positive, default=5 if cmd in (rerank, order) else 20)
    for cmd in (evaluate, rerank, ranks, review, changes, depths, train, order, order_review, tally):
        cmd.add_argument("--out", required=True, type=Path)
    for cmd in (rerank, metrics, ranks, compare, review, changes, depths, train, order, order_review):
        cmd.add_argument("--input", required=True, type=Path)
    build.add_argument("--clip-model", default=CLIP_MODEL, help="HF model ID or local snapshot")
    search.add_argument("--query", required=True)
    search.add_argument("--retrieval-text", help="optional English rendering for the CLIP encoder")
    search.add_argument("--out", type=Path, help="save a single-query candidate run for reranking")
    evaluate.add_argument("--text-batch", default=8, type=positive)
    for cmd in (rerank, depths, train, order):
        cmd.add_argument("--qwen-model", default=QWEN_MODEL, help="HF model ID or local snapshot")
    return cli


def run(args):
    if args.command == "review-order":
        from .order_review import prepare_review

        return prepare_review(args)
    if args.command == "tally-order":
        from .order_review import tally_review

        return tally_review(args)
    if args.command == "compare-order":
        from .temporal import compare_order

        return compare_order(args)
    if args.command == "bandit-compare":
        from .bandit import compare_scored

        return compare_scored(args)
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

        if args.current_sft:
            args.adapter = Path("runs/sft-pilot-zxCXNv5N/adapter")
            if not args.adapter.is_dir():
                raise FileNotFoundError(f"Retained SFT adapter missing: {args.adapter}. Run from the pilot checkout.")
            if args.qwen_model == QWEN_MODEL:
                args.qwen_model = json.loads((args.adapter / "train-report.json").read_text())["base"]
        scorer = QwenScorer(args.qwen_model, args.device, args.videos, args.frames)
        if args.adapter:
            from peft import PeftModel

            scorer.model = PeftModel.from_pretrained(scorer.model, str(args.adapter)).eval()
        rows = (rerank_prefix(row, args.top_k, scorer) for row in read_run(args.input))
        write_run(args.out, rows)
        return {"run": str(args.out), "rerank_top_k": args.top_k, "frames": args.frames,
                "adapter": str(args.adapter) if args.adapter else None}

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
