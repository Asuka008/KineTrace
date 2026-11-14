import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kinetrace.bandit import BanditPolicy, FEATURES, FeedbackStore, evaluate, fit_reward_model, train_from_feedback


def decision(event, query="query", chosen=0, reward=1):
    return {"version": 1, "event_id": event, "query": query,
            "actions": [{"video_id": "a", "features": [1, 1, 0, 0, 0]},
                        {"video_id": "b", "features": [1, 0, 1, 0, 0]}],
            "chosen": chosen, "probabilities": [0.5, 0.5], "reward": reward}


class BanditTests(unittest.TestCase):
    def test_ridge_solution_and_propensity_evaluation(self):
        # Fit one observation x=(1,1,0,0,0), reward1, weight2 with ridge1.
        fitted = fit_reward_model([decision("one")])
        self.assertAlmostEqual(fitted["weights"][0], 0.4)
        self.assertAlmostEqual(fitted["weights"][1], 0.4)
        rows = [decision("a"), decision("b", chosen=1, reward=0)]
        report = evaluate(rows)
        self.assertEqual(report["ips"], 1)
        self.assertEqual(report["snips"], 1)
        self.assertEqual(report["effective_sample_size"], 1)
        unmatched = evaluate([rows[1]])
        self.assertIsNone(unmatched["snips"])
        self.assertEqual(unmatched["matched_actions"], 0)

    def test_feedback_is_idempotent_and_conflicts_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = FeedbackStore(Path(tmp) / "feedback.sqlite3")
            store.record(decision("event"))
            self.assertIsNone(store.snapshot()[0]["reward"])
            store.feedback("event", 1)
            store.feedback("event", 1)
            with self.assertRaises(ValueError):
                store.feedback("event", 0)
            with self.assertRaises(ValueError):
                store.feedback("event", True)
            with self.assertRaises(KeyError):
                store.feedback("unknown", 1)
            self.assertEqual(store.snapshot()[0]["reward"], 1)

    def test_exploration_records_actual_probability_and_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            policy = BanditPolicy(Path(tmp) / "feedback.sqlite3", epsilon=0.2)
            candidates = [{"video_id": v, "retrieval_score": 0.3, "rerank_score": 0.8,
                           "retrieval_rank": i} for i, v in enumerate("ab", 1)]
            result = {"request_id": "one", "query": "running", "retrieval_text": "running",
                      "candidates": candidates, "model": {}}
            with patch.object(policy.rng, "choices", return_value=[1]):
                recommendation = policy.recommend(result, 2)
            self.assertEqual(recommendation["candidate"]["video_id"], "b")
            self.assertAlmostEqual(recommendation["probability"], 0.1)
            row = policy.store.snapshot()[0]
            self.assertEqual(row["probabilities"], [0.9, 0.1])
            self.assertEqual(row["actions"][1]["features"], [1, 0.3, 0.8, 0.5, 0.5])
            model_path = Path(tmp) / "policy.json"
            model_path.write_text(json.dumps({"version": 1, "features": FEATURES,
                                              "weights": [0, 0, 0, -1, 0]}))
            deployed = BanditPolicy(policy.store.path, epsilon=0.2, model_path=model_path)
            with patch.object(deployed.rng, "choices", return_value=[1]):
                recommendation = deployed.recommend(dict(result, request_id="two"), 2)
            self.assertAlmostEqual(recommendation["probability"], 0.9)
            self.assertEqual(recommendation["policy"], hashlib.sha256(model_path.read_bytes()).hexdigest())

    def test_training_uses_query_holdout_and_leaves_unrated_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = FeedbackStore(root / "feedback.sqlite3")
            groups = {0: [], 1: []}
            for i in range(40):
                query = f"q{i}"
                held = int(hashlib.sha256(query.encode()).hexdigest(), 16) % 5 == 0
                groups[int(held)].append(query)
                store.record(decision(str(i), query))
                store.feedback(str(i), i % 2)
            store.record(decision("missing"))
            args = SimpleNamespace(database=store.path, out=root / "policy.json")
            result = train_from_feedback(args)
            self.assertEqual(result["rated"], 40)
            self.assertEqual(result["unrated"], 1)
            self.assertEqual(result["train_events"], len(groups[0]))
            self.assertEqual(result["candidate"]["events"], len(groups[1]))
            model = json.loads(args.out.read_text())
            self.assertEqual(model["features"], FEATURES)
            with self.assertRaises(FileExistsError):
                train_from_feedback(args)
