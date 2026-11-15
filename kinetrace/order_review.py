"""Blind human review of the scored target presentations in a compare-order run."""
import base64
import hashlib
import io
import json
import random
import re
import tempfile
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from .data import read_run, write_run
from .ranking import compare_runs
from .temporal import SCORE_TOLERANCE
from .video import read_clip


@contextmanager
def new_directory(path):
    if path.exists():
        raise FileExistsError(f"Choose a new output directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        stage = Path(temporary) / "result"
        stage.mkdir()
        yield stage
        stage.rename(path)


def scored_targets(directory, summary):
    forward = list(read_run(directory / "forward.jsonl"))
    reverse = list(read_run(directory / "reverse.jsonl"))
    compare_runs(forward, reverse)
    reverse_by_id = {row["query_id"]: row for row in reverse}
    traces = {}
    for pair in read_run(directory / "score-pairs.jsonl"):
        if pair["is_annotated_target"]:
            qid = pair["query_id"]
            if qid in traces:
                raise ValueError(f"Duplicate scored target for {qid}")
            traces[qid] = pair

    pool = []
    for row in forward:
        qid, target = row["query_id"], row["target"]
        pair = traces.pop(qid, None)
        scores = {}
        for order, ranking in (("forward", row), ("reverse", reverse_by_id[qid])):
            for candidate in ranking["candidates"][:summary["rerank_top_k"]]:
                if candidate["video_id"] == target:
                    scores[order] = candidate["rerank_score"]
                    break
        if not scores and pair is None:
            continue
        if (pair is None or pair["video_id"] != target or len(scores) != 2
                or any(not 0 <= value <= 1 or value != pair[f"{order}_score"]
                       for order, value in scores.items())):
            raise ValueError(f"Target trace disagrees with saved rankings for {qid}")
        pool.append({"query_id": qid, "query": row["query"], "video_id": target,
                     "timestamp_slots": pair["timestamp_slots"],
                     "reverse_source_indices": pair["reverse_source_indices"], "scores": scores})
    if (traces or len(forward) != summary["forward"]["queries"]
            or len(pool) != summary["scored_annotated_targets"]["pairs"]
            or len(forward) - len(pool) != summary["targets_outside_scored_prefix"]):
        raise ValueError("Target coverage disagrees with the comparison summary")
    if not pool:
        raise ValueError("No scored annotated targets to review")
    return pool


def frame_png(image):
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return buffer.getvalue()


def prepare_review(args):
    summary_bytes = (args.input / "summary.json").read_bytes()
    summary = json.loads(summary_bytes)
    if summary["experiment"] != "base_qwen_frame_order_sensitivity":
        raise ValueError("Expected the comparison directory from compare-order")
    pool = scored_targets(args.input, summary)
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    cases = pool[:args.limit]
    manifest = {"source_summary_sha256": hashlib.sha256(summary_bytes).hexdigest(),
                "input_sha256": summary["input_sha256"], "model": summary["model"],
                "queries": summary["forward"]["queries"], "scored_targets": len(pool),
                "seed": args.seed, "cases": cases}
    template = Path(__file__).with_name("order_review.html").read_text(encoding="utf-8")
    before, after = template.split("<!-- REVIEW_DATA -->")
    with new_directory(args.out) as stage, (stage / "review.html").open("w", encoding="utf-8") as page:
        page.write(before + '{"cases":[')
        for index, case in enumerate(cases):
            video = case["video_id"]
            if not re.fullmatch(r"[\w-]+", video):
                raise ValueError(f"Invalid video ID: {video!r}")
            images, metadata = read_clip(args.videos / f"{video}.mp4", summary["frames"])
            indices = metadata["frames_indices"]
            if (metadata != case["timestamp_slots"] or indices[::-1] != case["reverse_source_indices"]
                    or len(images) != len(indices) or len(images) < 2):
                raise ValueError(f"Decoded frames disagree with the recorded presentation for {video}")
            frames = [frame_png(image) for image in images]
            case["forward_frame_sha256"] = [hashlib.sha256(frame).hexdigest() for frame in frames]
            case["a_order"] = rng.choice(("forward", "reverse"))
            if case["a_order"] == "reverse":
                frames.reverse()
            blind = {"query_id": case["query_id"], "query": case["query"],
                     "frames": [base64.b64encode(frame).decode("ascii") for frame in frames]}
            if index:
                page.write(",")
            page.write(json.dumps(blind).replace("<", "\\u003c"))
        manifest_bytes = json.dumps(manifest, indent=2, allow_nan=False).encode("utf-8")
        (stage / "manifest.json").write_bytes(manifest_bytes)
        review_id = hashlib.sha256(manifest_bytes).hexdigest()
        page.write('],"review_id":"' + review_id + '"}' + after)
    return {"page": str(args.out / "review.html"), "questions": len(cases),
            "scored_targets": len(pool), "targets_outside_scored_prefix": summary["targets_outside_scored_prefix"]}


def tally_review(args):
    manifest_bytes = (args.review / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    export = json.loads(args.judgments.read_bytes())
    review_id = hashlib.sha256(manifest_bytes).hexdigest()
    if export["review_id"] != review_id:
        raise ValueError("Judgments belong to a different review")
    answers = {r["query_id"]: r for r in export["answers"]}
    if (len(answers) != len(export["answers"])
            or answers.keys() != {c["query_id"] for c in manifest["cases"]}):
        raise ValueError("Expected exactly one judgment per review case")
    labels, verified = [], []
    exclusions = Counter()
    outcomes = {"correct": 0, "incorrect": 0, "tie": 0}
    for case in manifest["cases"]:
        answer = answers[case["query_id"]]
        if any(answer.get(k) not in ("yes", "no", "uncertain") for k in ("order_dependent", "a", "b")):
            raise ValueError(f"Incomplete or unknown judgment for {case['query_id']}")
        a_first = case["a_order"] == "forward"
        forward, reverse = (answer["a"], answer["b"]) if a_first else (answer["b"], answer["a"])
        label = {k: case[k] for k in ("query_id", "query", "video_id", "timestamp_slots", "reverse_source_indices")}
        label.update(order_dependent=answer["order_dependent"], forward_label=forward, reverse_label=reverse)
        if answer["order_dependent"] != "yes":
            reason = "order_independent" if answer["order_dependent"] == "no" else "order_dependence_uncertain"
        elif "uncertain" in (forward, reverse):
            reason = "presentation_uncertain"
        elif forward == reverse:
            reason = "both_match" if forward == "yes" else "neither_matches"
        else:
            reason = None
        label["exclusion"] = reason
        labels.append(label)
        if reason:
            exclusions[reason] += 1
            continue
        preferred = "forward" if forward == "yes" else "reverse"
        other = "reverse" if forward == "yes" else "forward"
        margin = case["scores"][preferred] - case["scores"][other]
        if abs(margin) <= SCORE_TOLERANCE:
            outcome = "tie"
        else:
            outcome = "correct" if margin > 0 else "incorrect"
        outcomes[outcome] += 1
        verified.append(dict(label, scores=case["scores"], preferred_order=preferred, outcome=outcome))
    report = {"review_id": review_id, "model": manifest["model"], "queries": manifest["queries"],
              "scored_targets": manifest["scored_targets"], "reviewed": len(labels),
              "targets_outside_scored_prefix": manifest["queries"] - manifest["scored_targets"],
              "scored_targets_not_reviewed": manifest["scored_targets"] - len(labels),
              "excluded": dict(exclusions), "verified": len(verified), **outcomes,
              "tie_atol": SCORE_TOLERANCE,
              "strict_pair_accuracy": outcomes["correct"] / len(verified) if verified else None}
    with new_directory(args.out) as stage:
        write_run(stage / "labels.jsonl", labels)
        write_run(stage / "verified.jsonl", verified)
        (stage / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    return report
