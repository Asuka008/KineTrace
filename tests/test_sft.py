import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from kinetrace.__main__ import parser
from kinetrace.data import read_queries, write_run
from kinetrace.ranking import compare_runs
from kinetrace.sft import training_pairs
from scripts.compare_sft_negatives import compare_negatives
from scripts.prepare_sft_pilot import prepare


class SftPilot(unittest.TestCase):
    def test_random_changes_only_negatives_and_samples_from_all_train_targets(self):
        targets = ["a", "b", "c", "d", "e", "f"]
        rows = [{"query": q, "target": t, "candidates": [{"video_id": "c", "retrieval_rank": 1}]}
                for q, t in zip([" Ski ", "ski", "jump", "run", "throw", "lift"], targets)]
        hard = training_pairs(rows)
        sampled = training_pairs(iter(rows), "random", 42)
        self.assertEqual([p[:2] for p in sampled], [p[:2] for p in hard])
        self.assertEqual(len(sampled), 5)  # "jump" never had a usable hard negative
        self.assertEqual(sampled, training_pairs(rows, "random", 42))
        for query, target, negative in sampled:
            self.assertIn(negative, targets)
            self.assertNotEqual(negative, target)
            if query.strip().casefold() == "ski":
                self.assertNotIn(negative, {"a", "b"})
        self.assertTrue(any(neg != "c" for _, _, neg in sampled))

    def test_control_reuses_reference_settings_and_compares_cached_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            pilot = Path(tmp)
            ref = {"base": "fixture", "device": "cpu", "frames": 3, "seed": 17, "steps": 9,
                   "available_pairs": 2}
            (pilot / "adapter").mkdir()
            (pilot / "adapter/train-report.json").write_text(json.dumps(ref))
            write_run(pilot / "train.jsonl", [{"target": "t1"}, {"target": "t2"}])

            def row(qid, order):
                return {"query_id": qid, "query": qid, "target": "a",
                        "candidates": [{"video_id": v} for v in order]}
            base = [row("1", "ab"), row("2", "ba")]
            hard = [row("1", "ab"), row("2", "ab")]
            for file, rows in (("val", base), ("val-base", base), ("val-adapted", hard)):
                write_run(pilot / f"{file}.jsonl", rows)

            def fake_model(cmd, check):
                args = parser().parse_args(cmd[3:])
                self.assertEqual((args.qwen_model, args.frames, args.device), ("fixture", 3, "cpu"))
                self.assertEqual(args.videos, pilot / "data/videos")
                if args.command == "train-lora":
                    self.assertEqual((args.negative_strategy, args.seed, args.steps), ("random", 17, 9))
                    self.assertEqual(args.input, pilot / "train.jsonl")
                    args.out.mkdir()
                    (args.out / "train-report.json").write_text(json.dumps(ref))
                else:
                    self.assertEqual((args.command, args.input, args.top_k), ("rerank", pilot / "val.jsonl", 2))
                    self.assertEqual(args.adapter, args.out.parent / "adapter")
                    write_run(args.out, list(reversed(base)))

            with patch("scripts.compare_sft_negatives.subprocess.check_output", return_value="fixture"), \
                    patch("scripts.compare_sft_negatives.subprocess.run", side_effect=fake_model) as calls, \
                    patch("builtins.print"):
                out = compare_negatives(pilot)
            self.assertEqual(calls.call_count, 2)
            report = json.loads((out / "validation.json").read_text())
            self.assertEqual(report["metrics"]["random"]["R@1"], 0.5)
            self.assertEqual(report["comparisons"]["hard_to_random"]["lost"], 1)
            self.assertEqual(report["comparisons"]["base_to_random"]["net"], 0)

    def test_same_caption_known_positives_excluded_from_negatives(self):
        candidates = [{"video_id": vid, "retrieval_rank": rank}
                      for rank, vid in enumerate(["b", "a", "c"], 1)]
        rows = [{"query": "skiing", "target": "a", "candidates": candidates},
                {"query": "Skiing", "target": "b", "candidates": candidates}]
        self.assertEqual(training_pairs(rows), [("skiing", "a", "c"), ("Skiing", "b", "c")])

    def test_compare_runs_pairs_by_id_and_rejects_changed_candidates(self):
        def row(qid, ids):
            return {"query_id": qid, "query": qid, "target": "a",
                    "candidates": [{"video_id": v} for v in ids]}
        base = [row("q1", "ab"), row("q2", "ba")]
        new = [row("q2", "ab"), row("q1", "ba")]
        report = compare_runs(base, new)
        self.assertEqual((report["gained"], report["lost"], report["net"]), (1, 1, 0))
        self.assertEqual([r["query_id"] for r in report["changes"]], ["q1", "q2"])
        with self.assertRaisesRegex(ValueError, "membership mismatch"):
            compare_runs(base, [row("q1", "ac"), new[0]])

    def test_split_extract_and_reject_test_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, held, archive, out = [root / f for f in ["train.json", "held.csv", "v.zip", "out"]]
            records = [{"video_id": f"video{i}", "caption": ["one", "two"]} for i in range(50)]
            data.write_text(json.dumps(records))
            held.write_text("video_id,sentence\nvideo50,test\n")
            with zipfile.ZipFile(archive, "w") as z:
                for r in records:
                    z.writestr("folder/" + r["video_id"] + ".mp4", b"fixture")
            prepare(data, held, archive, out, 42)
            train, val = [{r["target"] for r in read_queries(out / f"{split}.csv")}
                          for split in ("train", "val")]
            self.assertEqual((len(train), len(val), train & val), (32, 16, set()))
            self.assertEqual({p.stem for p in (out / "videos").iterdir()}, train | val)
            large = root / "custom"
            prepare(data, held, archive, large, 42, train_size=25, val_size=20)
            self.assertEqual([len(read_queries(large / f"{s}.csv")) for s in ("train", "val")], [25, 20])
            held.write_text("video_id,sentence\nvideo0,test\n")
            with self.assertRaisesRegex(ValueError, "test video ID"):
                prepare(data, held, archive, root / "bad", 42)
