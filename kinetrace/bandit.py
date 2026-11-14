"""Epsilon-greedy video selection, logged propensities and offline policy evaluation."""
import hashlib
import json
import math
import random
import sqlite3
from contextlib import closing
from pathlib import Path

from .data import output_file


FEATURES = ["bias", "clip_score", "qwen_score", "inverse_clip_rank", "inverse_qwen_rank"]


def features(candidate, rank):
    values = [1.0, float(candidate["retrieval_score"]), float(candidate["rerank_score"]),
              1.0 / candidate["retrieval_rank"], 1.0 / rank]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Bandit features must be finite")
    return values


def greedy_action(actions, model=None):
    if model is None:
        return 0  # The Qwen-sorted first result is the initial policy.
    return max(range(len(actions)), key=lambda i: sum(
        w * x for w, x in zip(model["weights"], actions[i]["features"])))


def load_model(path):
    model = json.loads(Path(path).read_text())
    if model.get("features") != FEATURES or model.get("version") != 1:
        raise ValueError("Unsupported bandit model feature schema")
    weights = model.get("weights", [])
    if len(weights) != len(FEATURES) or not all(math.isfinite(w) for w in weights):
        raise ValueError("Invalid bandit model weights")
    return model


class FeedbackStore:
    """Persist each decision before exposure and accept one idempotent binary reward."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS decisions (
                event_id TEXT PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                payload TEXT NOT NULL, reward INTEGER CHECK (reward IN (0, 1)),
                feedback_at TEXT)""")

    def record(self, decision):
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("INSERT INTO decisions(event_id, payload) VALUES (?, ?)",
                       (decision["event_id"], json.dumps(decision, allow_nan=False)))

    def feedback(self, event_id, reward):
        if type(reward) is not int or reward not in (0, 1):
            raise ValueError("Reward must be integer 0 or 1")
        with closing(sqlite3.connect(self.path)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT reward FROM decisions WHERE event_id=?", (event_id,)).fetchone()
            if row is None:
                raise KeyError(event_id)
            if row[0] is not None and row[0] != reward:
                raise ValueError("Feedback already recorded with a different reward")
            db.execute("""UPDATE decisions SET reward=?, feedback_at=CURRENT_TIMESTAMP
                          WHERE event_id=? AND reward IS NULL""", (reward, event_id))

    def snapshot(self):
        with closing(sqlite3.connect(self.path)) as db:
            rows = db.execute("SELECT payload, reward FROM decisions ORDER BY rowid").fetchall()
        return [dict(json.loads(payload), reward=reward) for payload, reward in rows]


class BanditPolicy:
    def __init__(self, database, epsilon=0.1, model_path=None):
        if not 0 < epsilon <= 1:
            raise ValueError("Exploration epsilon must be in (0, 1]")
        self.store = FeedbackStore(database)
        self.epsilon = epsilon
        self.model = load_model(model_path) if model_path else None
        self.version = (hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
                        if model_path else "qwen-top1-v1")
        self.rng = random.SystemRandom()

    def recommend(self, result, depth):
        # Every eligible action has already been scored on the same query.
        candidates = result["candidates"][:depth]
        actions = [{"video_id": c["video_id"], "features": features(c, rank)}
                   for rank, c in enumerate(candidates, 1)]
        best = greedy_action(actions, self.model)
        probabilities = [self.epsilon / len(actions)] * len(actions)
        probabilities[best] += 1 - self.epsilon
        chosen = self.rng.choices(range(len(actions)), weights=probabilities, k=1)[0]
        decision = {"version": 1, "event_id": result["request_id"], "query": result["query"],
                    "retrieval_text": result["retrieval_text"], "actions": actions,
                    "chosen": chosen, "probabilities": probabilities, "policy": self.version,
                    "epsilon": self.epsilon, "model": result["model"]}
        self.store.record(decision)
        return {"event_id": decision["event_id"], "candidate": candidates[chosen],
                "probability": probabilities[chosen], "policy": self.version}


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


def evaluate(rows, model=None):
    weights, weighted_rewards = [], []
    for row in rows:
        chosen = row["chosen"]
        probability = row["probabilities"][chosen]
        weight = float(greedy_action(row["actions"], model) == chosen) / probability
        weights.append(weight)
        weighted_rewards.append(weight * row["reward"])
    total = sum(weights)
    squared = sum(w * w for w in weights)
    return {"events": len(rows), "matched_actions": sum(w > 0 for w in weights),
            "ips": sum(weighted_rewards) / len(rows),
            "snips": sum(weighted_rewards) / total if total else None,
            "effective_sample_size": total * total / squared if squared else 0.0,
            "max_importance_weight": max(weights)}


def train_from_feedback(args):
    if args.out.exists():
        raise FileExistsError(f"Choose a new model output: {args.out}")
    if not args.database.is_file():
        raise FileNotFoundError(args.database)
    rows = FeedbackStore(args.database).snapshot()
    labeled = [row for row in rows if row["reward"] is not None]
    training, heldout = [], []
    for row in labeled:
        # Keep repeated normalized queries in one split, independently of reward/action.
        key = row["query"].strip().casefold()
        bucket = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 5
        (heldout if bucket == 0 else training).append(row)
    if not training or not heldout:
        raise ValueError("Need rated queries in both hash splits (80% training / 20% evaluation)")
    model = fit_reward_model(training)
    model["report"] = {"decisions": len(rows), "rated": len(labeled),
                       "unrated": len(rows) - len(labeled), "train_events": len(training),
                       "split": "sha256(normalized query) % 5 == 0 is evaluation",
                       "baseline_policy": "qwen-top1-greedy",
                       "evaluation_scope": "greedy policies on rated held-out decisions",
                       "baseline": evaluate(heldout), "candidate": evaluate(heldout, model)}
    with output_file(args.out) as stream:
        json.dump(model, stream, indent=2, allow_nan=False)
    return {"model": str(args.out), **model["report"]}
