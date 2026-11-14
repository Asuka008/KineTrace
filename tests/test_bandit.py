import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from kinetrace.bandit import (apply_policy, compare_scored, fit_reward_model,
                              simulated_feedback, validate_rows, validate_splits)
from kinetrace.data import read_run, write_run
from kinetrace.ranking import rerank_prefix
from scripts.compare_bandit import experiment


def rows(prefix):
    candidates = [{"video_id": prefix + str(i), "retrieval_score": 0.8 - i / 10,
                   "rerank_score": 0.9 - i / 10, "retrieval_rank": i + 1} for i in range(3)]
    return [{"query_id": f"q{i}", "query": f"{prefix} action {i}", "target": prefix + str(i),
             "candidates": copy.deepcopy(candidates)} for i in range(3)]


class OfflineBanditTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.train, self.val = rows("t"), rows("v")
        self.args = SimpleNamespace(train=self.root / "train.jsonl", validation=self.root / "val.jsonl",
                                    out=self.root / "comparison", top_k=2, epsilon=0.2, seeds=[42, 43, 44])
        write_run(self.args.train, self.train)
        write_run(self.args.validation, self.val)

    def test_sampled_rewards_probabilities_known_positives_and_misses(self):
        self.train[1]["query"] = self.train[0]["query"]
        with patch("kinetrace.bandit.random.Random.choices", return_value=[1]):
            sampled = simulated_feedback(self.train, 2, 0.2, 42)
        self.assertEqual(sampled[0]["probabilities"], [0.9, 0.1])
        self.assertEqual([row["reward"] for row in sampled], [1, 1, 0])
        for actual, expected in zip(sampled[0]["actions"][1]["features"], [1, 0.7, 0.8, 0.5, 0.5]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(simulated_feedback(self.train, 2, .2, 42),
                         simulated_feedback(self.train, 2, .2, 42))
        for value in [0, -1, 2, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                simulated_feedback(self.train, 2, value, 42)

    def test_ridge_analytic_solution_and_prediction_ignores_target(self):
        model = fit_reward_model([{"chosen": 0, "probabilities": [.5], "reward": 1,
                                  "actions": [{"features": [1, 1, 0, 0, 0]}]}])
        self.assertAlmostEqual(model["weights"][0], .4)
        self.assertAlmostEqual(model["weights"][1], .4)
        model["weights"] = [0, 0, 0, -1, 0]
        ranked = apply_policy(self.val, model, 2)
        changed_labels = [dict(row, target="unknown") for row in self.val]
        reranked = apply_policy(changed_labels, model, 2)
        self.assertEqual([r["candidates"] for r in ranked], [r["candidates"] for r in reranked])
        self.assertEqual(ranked[0]["candidates"][0]["video_id"], "v1")
        self.assertEqual(ranked[0]["candidates"][2:], self.val[0]["candidates"][2:])
        self.assertEqual(self.val[0]["candidates"][0]["video_id"], "v0")

    def test_split_overlap_and_bad_scores_rejected(self):
        contaminated = copy.deepcopy(self.train)
        contaminated[0]["candidates"][2]["video_id"] = "v2"
        with self.assertRaisesRegex(ValueError, "overlap"):
            validate_splits(contaminated, self.val)
        self.train[0]["query"] = self.val[0]["query"].upper() + " "
        self.assertEqual(len(validate_splits(self.train, self.val)), 2)
        for field, value in [("retrieval_rank", 0), ("rerank_score", float("nan"))]:
            broken = copy.deepcopy(self.train)
            broken[0]["candidates"][0][field] = value
            with self.assertRaises(ValueError):
                validate_rows(broken, 2)

    def test_cli_reports_all_seeds_and_preserves_misses(self):
        completed = subprocess.run([
            sys.executable, "-m", "kinetrace", "bandit-compare", "--train", str(self.args.train),
            "--validation", str(self.args.validation), "--out", str(self.args.out), "--top-k", "2"],
            capture_output=True, text=True, check=True)
        report = json.loads(completed.stdout)
        self.assertEqual(report["experiment"], "offline_bandit_simulation")
        self.assertEqual(report["lora"]["queries"], 3)
        self.assertEqual([r["seed"] for r in report["runs"]], [42, 43, 44])
        for seed in [42, 43, 44]:
            result = list(read_run(self.args.out / f"validation-{seed}.jsonl"))
            self.assertEqual(result[2]["candidates"][2]["video_id"], "v2")
        with self.assertRaises(FileExistsError):
            compare_scored(self.args)

    def test_failed_comparison_does_not_publish_partial_output(self):
        with patch("kinetrace.bandit.fit_reward_model", side_effect=RuntimeError("fixture")):
            with self.assertRaises(RuntimeError):
                compare_scored(self.args)
        self.assertFalse(self.args.out.exists())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["train.jsonl", "val.jsonl"])

    def test_driver_scores_training_only_and_reuses_reference(self):
        pilot = self.root / "pilot"
        (pilot / "adapter").mkdir(parents=True)
        reference = {"base": "fixture-model", "device": "cpu", "frames": 3, "available_pairs": 3}
        (pilot / "adapter/train-report.json").write_text(json.dumps(reference))
        unscored = copy.deepcopy(self.train)
        for row in unscored:
            for candidate in row["candidates"]:
                del candidate["rerank_score"]
        for name, records in [("train", unscored), ("val", self.val),
                              ("val-adapted", self.val), ("val-base", self.val)]:
            write_run(pilot / f"{name}.jsonl", records)

        def score(args):
            self.assertEqual(args.input, pilot / "train.jsonl")
            self.assertEqual((args.qwen_model, args.frames, args.device), ("fixture-model", 3, "cpu"))
            self.assertEqual(args.adapter, pilot / "adapter")
            write_run(args.out, [rerank_prefix(r, args.top_k, lambda *_: .8) for r in unscored])

        with patch("scripts.compare_bandit.run", side_effect=score) as scoring, \
                patch("scripts.compare_bandit.subprocess.check_output", return_value="fixture\n"), \
                patch("builtins.print"):
            out = experiment(pilot)
        scoring.assert_called_once()
        report = json.loads((out / "summary.json").read_text())
        self.assertEqual(report["base_qwen"]["queries"], 3)
        self.assertEqual(report["lora"]["R@1"], 1 / 3)
        self.assertEqual(len(report["runs"]), 3)
        self.assertTrue((out / "scoring-device.json").is_file())
