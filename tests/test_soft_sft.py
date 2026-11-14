import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.compare_soft_sft import compare_validation, experiment, tally
from kinetrace.__main__ import parser, run
from kinetrace.data import read_run, write_run
from kinetrace.sft import train_lora, training_pairs


class SoftSftTest(unittest.TestCase):
    def test_preference_includes_two_unlabeled_first_choices(self):
        def row(qid, order):
            return {"query_id": qid, "query": "a runner", "target": "t",
                    "candidates": [{"video_id": v} for v in order]}
        current = [row("1", "abt"), row("2", "tab")]
        candidate = [row("2", "tab"), row("1", "bat")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for v in "ab":
                (root / f"{v}.mp4").write_bytes(b"fixture")
            report = compare_validation(current, candidate, root, root)
            self.assertEqual(report["top1_disagreements"], 1)
            self.assertEqual(report["paired"]["changes"], [])
            page = (root / "review.html").read_text()
            self.assertIn("A 更符合", page)
            self.assertNotIn("<details>", page)
            vote = {"query_id": "1", "video_a": "b", "video_b": "a", "judgment": "A"}
            self.assertEqual(tally(report, [vote]), {"candidate": 1, "unrated": 0})
            self.assertEqual(tally(report, []), {"unrated": 1})
            with self.assertRaises(ValueError):
                tally(report, [dict(vote, video_a="t")])
            with self.assertRaises(ValueError):
                tally(report, [vote, vote])

    def test_existing_training_is_hard_labeled_by_default(self):
        args = parser().parse_args(["train-lora", "--videos", ".", "--input", "train.jsonl", "--out", "adapter"])
        self.assertEqual(args.negative_yes_weight, 0)
        for invalid in [-0.1, 0.5, float("nan")]:
            args.negative_yes_weight = invalid
            with self.assertRaisesRegex(ValueError, "weight"):
                train_lora(args)

    def test_current_sft_loads_recorded_base_and_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter"
            adapter.mkdir()
            (adapter / "train-report.json").write_text(json.dumps({"base": "recorded-base"}))
            source, output = root / "input.jsonl", root / "out.jsonl"
            write_run(source, [{"query": "running", "candidates": [{"video_id": "a"}]}])
            args = parser().parse_args(["rerank", "--current-sft", "--videos", tmp,
                                        "--input", str(source), "--out", str(output)])
            peft = Mock()
            with patch("kinetrace.__main__.Path", return_value=adapter), \
                    patch("kinetrace.reranker.QwenScorer") as scorer, \
                    patch.dict("sys.modules", {"peft": peft}):
                scorer.return_value.return_value = 0.7
                base_model = scorer.return_value.model
                result = run(args)
            scorer.assert_called_once_with("recorded-base", "cuda", root, 4)
            peft.PeftModel.from_pretrained.assert_called_once_with(base_model, str(adapter))
            self.assertEqual(result["adapter"], str(adapter))
            self.assertEqual(list(read_run(output))[0]["candidates"][0]["rerank_score"], 0.7)

    def test_all_review_outcomes_and_missing_votes_are_counted(self):
        choices = ["A", "B", "both", "neither", "audio_required", "uncertain",
                   "visual_evidence_insufficient", None]
        sample = [{"query_id": str(i), "current": "a", "candidate": "b"} for i in range(9)]
        judgments = [{"query_id": str(i), "video_a": "a", "video_b": "b", "judgment": choice}
                     for i, choice in enumerate(choices)]
        counts = tally({"review_sample": sample}, judgments)
        self.assertEqual(counts, {"current": 1, "candidate": 1, "both": 1, "neither": 1,
                                  "audio_required": 1, "uncertain": 1,
                                  "visual_evidence_insufficient": 1, "unrated": 2})


class SoftExperimentTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.pilot = Path(directory.name)
        self.ref = {"base": "fixture", "device": "cpu", "frames": 3, "seed": 17, "steps": 9,
                    "available_pairs": 2}
        (self.pilot / "adapter").mkdir()
        (self.pilot / "adapter/train-report.json").write_text(json.dumps(self.ref))
        self.train = [{"query": v, "target": v,
                       "candidates": [{"video_id": c, "retrieval_rank": i}
                                      for i, c in enumerate(("t1", "t2"), 1)]}
                      for v in ("t1", "t2")]
        write_run(self.pilot / "train.jsonl", self.train)
        self.val = [{"query_id": "1", "query": "a runner", "target": "a",
                     "candidates": [{"video_id": v} for v in ("a", "b")]}]
        for name in ("val", "val-adapted"):
            write_run(self.pilot / f"{name}.jsonl", self.val)

    def test_driver_reuses_reference_settings_and_supports_legacy_pairs(self):
        def fake_model(cmd, check):
            self.assertTrue(check)
            args = parser().parse_args(cmd[3:])
            self.assertEqual((args.qwen_model, args.frames, args.device), ("fixture", 3, "cpu"))
            self.assertEqual(args.videos, self.pilot / "data/videos")
            if args.command == "train-lora":
                self.assertEqual((args.negative_strategy, args.negative_yes_weight), ("hard", 0.1))
                self.assertEqual((args.seed, args.steps), (17, 9))
                self.assertEqual(args.input, self.pilot / "train.jsonl")
            else:
                self.assertEqual((args.command, args.top_k), ("rerank", 2))
                self.assertEqual(args.input, self.pilot / "val.jsonl")
                self.assertEqual(args.adapter, args.out.parent / "adapter")
                write_run(args.out, self.val)

        for manifest in (False, True):
            with self.subTest(pair_manifest=manifest):
                if manifest:
                    write_run(self.pilot / "adapter/training-pairs.jsonl",
                              [{"query": q, "positive": p, "negative": n}
                               for q, p, n in training_pairs(self.train)])
                with patch("scripts.compare_soft_sft.subprocess.check_output", return_value="fixture\n"), \
                        patch("scripts.compare_soft_sft.subprocess.run", side_effect=fake_model) as calls, \
                        patch("builtins.print"):
                    out = experiment(self.pilot)
                self.assertEqual(calls.call_count, 2)
                self.assertEqual((out / "commit.txt").read_text(), "fixture\n")
                report = json.loads((out / "validation.json").read_text())
                self.assertEqual(report["current"], report["candidate"])
                self.assertEqual(report["top1_disagreements"], 0)
                self.assertFalse((out / "review.html").exists())

    def test_rejects_overlap_even_when_only_a_negative_is_in_validation(self):
        self.train[0]["candidates"].append({"video_id": "a", "retrieval_rank": 3})
        write_run(self.pilot / "train.jsonl", self.train)
        with patch("scripts.compare_soft_sft.subprocess.run") as calls:
            with self.assertRaisesRegex(ValueError, "overlap"):
                experiment(self.pilot)
        calls.assert_not_called()
        self.assertEqual(list(self.pilot.glob("soft-negative-*")), [])

    def test_rejects_changed_pair_count_and_saved_pairs(self):
        self.ref["available_pairs"] = 3
        (self.pilot / "adapter/train-report.json").write_text(json.dumps(self.ref))
        with self.assertRaisesRegex(ValueError, "number of eligible"):
            experiment(self.pilot)
        self.ref["available_pairs"] = 2
        (self.pilot / "adapter/train-report.json").write_text(json.dumps(self.ref))
        write_run(self.pilot / "adapter/training-pairs.jsonl",
                  [{"query": "changed", "positive": "t1", "negative": "t2"}])
        with self.assertRaisesRegex(ValueError, "Training pairs changed"):
            experiment(self.pilot)
