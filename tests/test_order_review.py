import base64
import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kinetrace.__main__ import parser, run
from kinetrace.data import read_run, write_run


class OrderReviewTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "comparison"
        self.source.mkdir()
        self.review = self.root / "review"
        self.result = self.root / "result"
        self.judgments = self.root / "judgments.json"
        self.prepare = parser().parse_args([
            "review-order", "--input", str(self.source), "--videos", str(self.root),
            "--out", str(self.review)])
        self.tally = parser().parse_args([
            "tally-order", "--review", str(self.review), "--judgments", str(self.judgments),
            "--out", str(self.result)])
        self.meta = {"frames_indices": [0, 4], "total_num_frames": 5, "fps": 2.0}
        forward, reverse, traces = [], [], []
        for i in range(8):
            video = f"v{i}"
            query = {"query_id": f"q{i}", "query": f"动作 {i} </script>",
                     "target": video if i < 7 else "miss"}
            fs, rs = 0.8, (0.8000005 if i == 2 else 0.2)
            for rows, score in ((forward, fs), (reverse, rs)):
                rows.append(dict(query, candidates=[{"video_id": video, "rerank_score": score}]))
            traces.append({"query_id": f"q{i}", "video_id": video, "is_annotated_target": i < 7,
                           "forward_score": fs, "reverse_score": rs, "timestamp_slots": self.meta,
                           "reverse_source_indices": [4, 0]})
        for name, records in (("forward", forward), ("reverse", reverse[::-1]), ("score-pairs", traces)):
            write_run(self.source / f"{name}.jsonl", records)
        (self.source / "summary.json").write_text(json.dumps({
            "experiment": "base_qwen_frame_order_sensitivity", "input_sha256": "fixture",
            "model": "fixture", "rerank_top_k": 1, "frames": 2, "forward": {"queries": 8},
            "scored_annotated_targets": {"pairs": 7}, "targets_outside_scored_prefix": 1}))
        decode = patch("kinetrace.order_review.read_clip", return_value=([b"first", b"last"], self.meta))
        self.decode = decode.start()
        self.addCleanup(decode.stop)
        png = patch("kinetrace.order_review.frame_png", side_effect=bytes)
        png.start()
        self.addCleanup(png.stop)

    def prepare_page(self):
        run(self.prepare)
        self.manifest = json.loads((self.prepare.out / "manifest.json").read_text())
        html = (self.prepare.out / "review.html").read_text()
        return json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>', html)[1])

    def test_page_and_tally_decode_blinding_in_both_directions(self):
        page = self.prepare_page()
        self.assertEqual(self.decode.call_count, 7)
        self.assertEqual({c["a_order"] for c in self.manifest["cases"]}, {"forward", "reverse"})
        choices = [("yes", "yes", "no"), ("yes", "no", "yes"), ("yes", "yes", "no"),
                   ("no", "yes", "yes"), ("yes", "uncertain", "no"), ("yes", "no", "no"),
                   ("uncertain", "no", "yes")]
        answers = []
        for blind, case in zip(page["cases"], self.manifest["cases"]):
            self.assertEqual(set(blind), {"query_id", "query", "frames"})
            self.assertEqual(blind["query"], case["query"])
            a_first = case["a_order"] == "forward"
            self.assertEqual([base64.b64decode(b) for b in blind["frames"]],
                             [b"first", b"last"] if a_first else [b"last", b"first"])
            qid = case["query_id"]
            depends, f, r = choices[int(qid[1:])]
            answers.append({"query_id": qid, "order_dependent": depends,
                            "a": f if a_first else r, "b": r if a_first else f})
        self.judgments.write_text(json.dumps({"review_id": page["review_id"], "answers": answers[::-1]}))
        summary = run(self.tally)
        self.assertEqual(summary, json.loads((self.result / "summary.json").read_text()))
        self.assertEqual([summary[k] for k in ("queries", "scored_targets", "reviewed", "verified")],
                         [8, 7, 7, 3])
        self.assertEqual([summary[k] for k in ("correct", "incorrect", "tie")], [1, 1, 1])
        self.assertEqual(summary["strict_pair_accuracy"], 1/3)
        self.assertEqual(summary["excluded"], {"order_independent": 1, "presentation_uncertain": 1,
                                              "neither_matches": 1, "order_dependence_uncertain": 1})
        subset = {r["query_id"]: r for r in read_run(self.result / "verified.jsonl")}
        self.assertEqual(subset["q1"]["preferred_order"], "reverse")
        self.assertEqual(subset["q2"]["outcome"], "tie")
        self.assertEqual(len(list(read_run(self.result / "labels.jsonl"))), 7)
        with self.assertRaises(FileExistsError):
            run(self.tally)

    def test_review_identity_completeness_and_empty_subset(self):
        self.prepare.limit = 2
        page = self.prepare_page()
        self.prepare.out = self.root / "repeat"
        self.assertEqual(self.prepare_page(), page)
        answers = [{"query_id": c["query_id"], "order_dependent": "yes", "a": "yes", "b": "yes"}
                   for c in page["cases"]]
        for identity, rows in [
                ("other", answers), (page["review_id"], answers[:1]),
                (page["review_id"], answers + answers[:1]),
                (page["review_id"], [answers[0], dict(answers[1], a=None)])]:
            self.judgments.write_text(json.dumps({"review_id": identity, "answers": rows}))
            with self.assertRaises(ValueError):
                run(self.tally)
            self.assertFalse(self.result.exists())
        self.judgments.write_text(json.dumps({"review_id": page["review_id"], "answers": answers}))
        report = run(self.tally)
        self.assertIsNone(report["strict_pair_accuracy"])
        self.assertEqual(report["excluded"], {"both_match": 2})
        self.assertEqual(report["scored_targets_not_reviewed"], 5)
        self.assertEqual((self.result / "verified.jsonl").read_text(), "")

    def test_mismatched_frames_or_trace_leaves_no_review(self):
        self.decode.return_value = ([b"first", b"last"], dict(self.meta, fps=3.0))
        with self.assertRaisesRegex(ValueError, "Decoded frames"):
            run(self.prepare)
        self.assertFalse(self.review.exists())
        path = self.source / "score-pairs.jsonl"
        traces = list(read_run(path))
        traces[0]["reverse_score"] = 0.9
        write_run(path, traces)
        with self.assertRaisesRegex(ValueError, "rankings"):
            run(self.prepare)
        self.assertFalse(self.review.exists())
