import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from kinetrace.__main__ import parser, run
from kinetrace.data import read_run, write_run
from kinetrace.reranker import QwenScorer
from kinetrace.temporal import paired_query, score_deltas
from scripts.compare_temporal import experiment


def fixture_rows():
    candidates = [{"video_id": v, "retrieval_rank": i, "retrieval_score": 1 / i}
                  for i, v in enumerate("abc", 1)]
    return [{"query_id": f"q{i}", "query": query, "target": target,
             "candidates": copy.deepcopy(candidates)}
            for i, (query, target) in enumerate([("first", "a"), ("second", "missing")])]


def clip(*args):
    return [0, 1, 2, 3], {"total_num_frames": 10, "fps": 3, "frames_indices": [0, 3, 6, 9]}


class TemporalTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.rows = fixture_rows()
        self.source = self.root / "val.jsonl"
        write_run(self.source, self.rows)
        self.args = parser().parse_args([
            "compare-order", "--input", str(self.source), "--out", str(self.root / "result"),
            "--videos", str(self.root), "--device", "cpu", "--top-k", "2"])

    def test_frame_logits_preserves_existing_entry_and_processor_metadata(self):
        scorer = QwenScorer.__new__(QwenScorer)
        scorer.video_dir, scorer.frames, scorer.device = self.root, 4, "cpu"
        inputs = MagicMock()
        inputs.__contains__.return_value = True
        inputs.to.return_value = inputs
        scorer.processor = Mock(return_value=inputs)
        scorer.model = MagicMock()
        scorer.model.return_value.logits.__getitem__.return_value.float.return_value = "logits"
        images, metadata = clip()
        original = copy.deepcopy(metadata)

        def process(**kwargs):
            self.assertEqual(kwargs["videos"], [images])
            self.assertFalse(kwargs["do_sample_frames"])
            kwargs["video_metadata"][0]["frames_indices"].append(9)
            return inputs

        scorer.processor.side_effect = process
        with patch.dict("sys.modules", {"numpy": SimpleNamespace(stack=lambda values: list(values))}), \
                patch("kinetrace.reranker.read_clip", return_value=(images, metadata)) as decode:
            self.assertEqual(scorer.next_token_logits("running", "a"), "logits")
        decode.assert_called_once_with(self.root / "a.mp4", 4)
        self.assertEqual(metadata, original)
        self.assertEqual(scorer.model.call_args.kwargs["logits_to_keep"], 1)
        self.assertFalse(scorer.model.call_args.kwargs["use_cache"])

    def test_pair_reuses_decoded_frames_and_fixed_slots_with_alternating_order(self):
        calls = []

        def score(query, images, metadata):
            calls.append((list(images), copy.deepcopy(metadata)))
            metadata["frames_indices"].append(99)
            return images[0] / 3

        scorer = SimpleNamespace(score_frames=score)
        with patch("kinetrace.temporal.read_clip", side_effect=clip) as decode:
            forward, reverse, details, timings = paired_query(
                self.rows[0], scorer, self.root, 4, 2, lambda: None)
        self.assertEqual(decode.call_count, 2)
        self.assertEqual([frames for frames, meta in calls],
                         [[0, 1, 2, 3], [3, 2, 1, 0], [3, 2, 1, 0], [0, 1, 2, 3]])
        self.assertTrue(all(meta["frames_indices"] == [0, 3, 6, 9] for _, meta in calls))
        self.assertEqual(details[0]["reverse_source_indices"], [9, 6, 3, 0])
        self.assertEqual(details[0]["timestamp_slots"]["frames_indices"], [0, 3, 6, 9])
        self.assertEqual(forward["candidates"][2:], self.rows[0]["candidates"][2:])
        self.assertEqual(reverse["candidates"][0]["video_id"], "a")  # Stable ties.
        self.assertEqual(set(timings), {"decode", "forward", "reverse"})

    def test_cli_reports_flips_misses_and_original_label_scope(self):
        # Equal forward scores retain a first; candidate b responds more to reversal.
        scorer = Mock()
        scorer.score_frames.side_effect = [0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        with patch("kinetrace.temporal.QwenScorer", return_value=scorer) as factory, \
                patch("kinetrace.temporal.read_clip", side_effect=clip), \
                patch("builtins.print"):
            report = run(self.args)
        factory.assert_called_once_with(self.args.qwen_model, "cpu", self.root, 4)
        self.assertEqual(report["experiment"], "base_qwen_frame_order_sensitivity")
        self.assertEqual(report["forward"]["R@1"], .5)
        self.assertEqual(report["reverse_reference_labels"]["R@1"], 0)
        self.assertEqual(report["reference_label_changes"]["lost"], 1)
        self.assertEqual(report["top1_disagreements"], 2)
        self.assertEqual(report["targets_outside_scored_prefix"], 1)
        self.assertEqual(report["all_scored_candidates"]["pairs"], 4)
        self.assertEqual(report["scored_annotated_targets"]["pairs"], 1)
        self.assertEqual(len(list(read_run(self.args.out / "score-pairs.jsonl"))), 4)
        with self.assertRaises(FileExistsError):
            run(self.args)

    def test_bad_input_or_scoring_failure_leaves_no_output(self):
        self.args.frames = 1
        with patch("kinetrace.temporal.QwenScorer") as factory:
            with self.assertRaises(ValueError):
                run(self.args)
        factory.assert_not_called()
        self.args.frames = 4
        scorer = SimpleNamespace(score_frames=lambda *_: float("nan"))
        with patch("kinetrace.temporal.QwenScorer", return_value=scorer), \
                patch("kinetrace.temporal.read_clip", side_effect=clip), patch("builtins.print"):
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                run(self.args)
        self.assertFalse(self.args.out.exists())
        self.rows[0]["candidates"][0]["rerank_score"] = .5
        write_run(self.source, self.rows)
        with patch("kinetrace.temporal.QwenScorer") as factory:
            with self.assertRaisesRegex(ValueError, "raw CLIP"):
                run(self.args)
        factory.assert_not_called()

    def test_tolerance_empty_targets_and_short_video(self):
        result = score_deltas([{"forward_minus_reverse": v} for v in [1e-8, .1, -.2]])
        self.assertEqual((result["forward_higher"], result["reverse_higher"], result["ties"]), (1, 1, 1))
        self.assertIsNone(score_deltas([])["mean_absolute_delta"])
        with patch("kinetrace.temporal.read_clip", return_value=([0], {})):
            with self.assertRaisesRegex(ValueError, "at least two"):
                paired_query(self.rows[0], Mock(), self.root, 4, 2, lambda: None)

    def test_driver_uses_base_and_compares_cached_forward(self):
        (self.root / "adapter").mkdir()
        (self.root / "adapter/train-report.json").write_text(
            json.dumps({"base": "local-base", "device": "cpu", "frames": 4}))
        write_run(self.root / "val-base.jsonl", self.rows)
        scorer = SimpleNamespace(score_frames=lambda *_: .5)
        with patch("kinetrace.temporal.QwenScorer", return_value=scorer) as factory, \
                patch("kinetrace.temporal.read_clip", side_effect=clip), \
                patch("scripts.compare_temporal.subprocess.check_output", return_value="fixture\n"), \
                patch("builtins.print"):
            out = experiment(self.root)
        factory.assert_called_once_with("local-base", "cpu", self.root / "data/videos", 4)
        report = json.loads((out / "summary.json").read_text())
        self.assertEqual(report["cached_base_to_fresh_forward"]["net"], 0)
        self.assertEqual(report["top1_disagreements"], 0)
        self.assertTrue((out / "provenance.json").is_file())
