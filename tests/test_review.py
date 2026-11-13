import json
import tempfile
import unittest
from pathlib import Path

from kinetrace.__main__ import parser, run
from kinetrace.ranking import compare_runs, compare_top1


class ReviewTest(unittest.TestCase):
    def test_blinded_sample_from_saved_runs_and_legacy_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for video in ("a", "b"):
                (root / f"{video}.mp4").write_bytes(video.encode())

            def row(i, order):
                return {"query_id": str(i), "query": "A < B", "target": "a",
                        "candidates": [{"video_id": v, "retrieval_rank": "ab".index(v) + 1}
                                       for v in order]}

            before = [row(i, "ab" if i < 4 else "ba") for i in range(8)]
            after = [row(i, "ba" if i < 4 else "ab") for i in range(8)]
            report = root / "report.json"
            page = root / "review.html"
            paths = ["--videos", tmp, "--out", str(page), "--input", str(report)]
            report.write_text(json.dumps(compare_runs(before, after)))
            args = parser().parse_args(["review-changes", "--per-outcome", "2", *paths])
            result = run(args)
            self.assertEqual(result["eligible"], {"lost": 4, "gained": 4})
            ids = result["query_ids"]
            self.assertEqual(len(set(ids)), 4)
            self.assertEqual(sum(int(i) < 4 for i in ids), 2)
            html = page.read_text()
            self.assertNotIn("<details>", html)
            self.assertEqual(html.count("data:video/mp4;base64,"), 8)
            self.assertIn("A &lt; B", html)
            self.assertEqual(run(args), result)
            self.assertEqual(page.read_text(), html)
            report.write_text(json.dumps(compare_top1(after)))
            legacy = run(parser().parse_args(["review-lost", *paths, "--exclude", "0"]))
            self.assertEqual(set(legacy["query_ids"]), {"1", "2", "3"})
            self.assertIn("<details>", page.read_text())
