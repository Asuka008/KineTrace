"""Offline contextual-bandit simulation using labeled, frozen LoRA candidate scores."""
import hashlib
import json
import math
import random
import statistics
import tempfile
from pathlib import Path

from .data import read_run, write_run
from .ranking import compare_runs, summarize

FEATURES = ["bias", "clip_score", "qwen_score", "inverse_clip_rank", "inverse_qwen_rank"]


def features(candidate, rank):
    values = [1.0, float(candidate["retrieval_score"]), float(candidate["rerank_score"]),
              1.0 / candidate["retrieval_rank"], 1.0 / rank]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Bandit features must be finite")
    return values


def prediction(weights, values):
    return sum(w * x for w, x in zip(weights, values))


def validate_rows(rows, depth, scored=True):
    summarize(rows)  # Checks distinct nonempty candidates and fixed depth.
    query_ids = set()
    for row in rows:
        if row["query_id"] in query_ids or not row["query"].strip():
            raise ValueError("Expected unique query IDs and nonempty queries")
        query_ids.add(row["query_id"])
        candidates = row["candidates"]
        ranks = [c["retrieval_rank"] for c in candidates]
        if any(type(rank) is not int for rank in ranks) or sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError("Invalid retrieval ranks")
        prefix = candidates[:depth]
        if {c["retrieval_rank"] for c in prefix} != set(range(1, len(prefix) + 1)):
            raise ValueError("Scored prefix must contain the original CLIP prefix")
        if scored:
            for rank, candidate in enumerate(prefix, 1):
                features(candidate, rank)
            scores = [float(c["rerank_score"]) for c in prefix]
            if scores != sorted(scores, reverse=True):
                raise ValueError("Expected candidates sorted by LoRA relevance")


def validate_splits(train, validation):
    def gallery(rows):
        return {r["target"] for r in rows} | {c["video_id"] for r in rows for c in r["candidates"]}

    if gallery(train) & gallery(validation):
        raise ValueError("Training and validation galleries overlap")
    val_queries = {r["query"].strip().casefold() for r in validation}
    eligible = [r for r in train if r["query"].strip().casefold() not in val_queries]
    if not eligible:
        raise ValueError("No training queries remain after excluding shared validation captions")
    return eligible


def simulated_feedback(rows, depth, epsilon, seed):
    if not math.isfinite(epsilon) or not 0 < epsilon <= 1:
        raise ValueError("epsilon must be finite and in (0, 1]")
    rng = random.Random(seed)
    positives = {}
    for row in rows:
        positives.setdefault(row["query"].strip().casefold(), set()).add(row["target"])
    decisions = []
    for row in rows:
        candidates = row["candidates"][:depth]
        actions = [{"video_id": c["video_id"], "features": features(c, rank)}
                   for rank, c in enumerate(candidates, 1)]
        probabilities = [epsilon / len(actions)] * len(actions)
        probabilities[0] += 1 - epsilon
        chosen = rng.choices(range(len(actions)), weights=probabilities, k=1)[0]
        known = positives[row["query"].strip().casefold()]
        decisions.append({"query_id": row["query_id"], "actions": actions,
                          "chosen": chosen, "probabilities": probabilities,
                          "reward": int(actions[chosen]["video_id"] in known)})
    return decisions


def fit_reward_model(rows, regularization=1.0):
    """Fit inverse-propensity-weighted ridge by rank-one inverse updates."""
    size = len(FEATURES)
    inverse = [[float(i == j) / regularization for j in range(size)] for i in range(size)]
    b = [0.0] * size
    for row in rows:
        chosen = row["chosen"]
        scale = math.sqrt(1 / row["probabilities"][chosen])
        x = [value * scale for value in row["actions"][chosen]["features"]]
        y = row["reward"] * scale
        product = [sum(a * value for a, value in zip(line, x)) for line in inverse]
        denominator = 1 + sum(value * p for value, p in zip(x, product))
        inverse = [[inverse[i][j] - product[i] * product[j] / denominator
                    for j in range(size)] for i in range(size)]
        b = [old + value * y for old, value in zip(b, x)]
    return {"version": 1, "features": FEATURES,
            "weights": [sum(a * value for a, value in zip(line, b)) for line in inverse]}


def apply_policy(rows, model, depth):
    ranked = []
    for row in rows:
        prefix = [dict(c, policy_score=prediction(model["weights"], features(c, rank)))
                  for rank, c in enumerate(row["candidates"][:depth], 1)]
        prefix.sort(key=lambda c: -c["policy_score"])
        ranked.append(dict(row, candidates=prefix + row["candidates"][depth:]))
    return ranked


def compare_scored(args):
    if args.out.exists():
        raise FileExistsError(f"Choose a new output directory: {args.out}")
    if args.top_k < 1 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Need positive depth and distinct seeds")
    train, validation = [list(read_run(path)) for path in (args.train, args.validation)]
    for rows in (train, validation):
        validate_rows(rows, args.top_k)
    eligible = validate_splits(train, validation)
    baseline = summarize(validation)
    report = {"experiment": "offline_bandit_simulation", "policy_training_device": "cpu",
              "reward": "known training-caption target match; other candidates receive zero",
              "validation_reward": "single annotated target match",
              "epsilon": args.epsilon, "top_k": args.top_k, "seeds": args.seeds,
              "train_queries": len(eligible), "excluded_shared_queries": len(train) - len(eligible),
              "inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in (args.train, args.validation)},
              "lora": baseline, "runs": []}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as temporary:
        stage = Path(temporary) / "results"
        stage.mkdir()
        for seed in args.seeds:
            decisions = simulated_feedback(eligible, args.top_k, args.epsilon, seed)
            model = fit_reward_model(decisions)
            model.update(seed=seed, epsilon=args.epsilon, top_k=args.top_k)
            ranked = apply_policy(validation, model, args.top_k)
            metrics = summarize(ranked)
            paired = compare_runs(validation, ranked)
            write_run(stage / f"feedback-{seed}.jsonl", decisions)
            write_run(stage / f"validation-{seed}.jsonl", ranked)
            (stage / f"policy-{seed}.json").write_text(json.dumps(model, indent=2, allow_nan=False))
            weights = [1 / r["probabilities"][r["chosen"]] for r in decisions]
            report["runs"].append({"seed": seed, "metrics": metrics, "paired": paired,
                                   "r1_delta_pp": 100 * (metrics["R@1"] - baseline["R@1"]),
                                   "sampled_positive_rewards": sum(r["reward"] for r in decisions),
                                   "sampled_non_top1": sum(r["chosen"] != 0 for r in decisions),
                                   "max_training_weight": max(weights),
                                   "training_weight_ess": sum(weights)**2 / sum(w*w for w in weights)})
        values = [r["metrics"]["R@1"] for r in report["runs"]]
        report["r1_across_seeds"] = {"mean": statistics.mean(values), "std": statistics.pstdev(values),
                                    "min": min(values), "max": max(values)}
        (stage / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        stage.rename(args.out)
    return report
