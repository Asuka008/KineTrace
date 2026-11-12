import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from kinetrace.__main__ import parser, run
from kinetrace.data import read_run, write_run
from kinetrace.ranking import rerank_prefix


class PairedDepths(unittest.TestCase):
    def test_cli_reports_paired_changes_and_preserves_tail_and_ties(self):
        candidates = [{"video_id": str(i), "retrieval_rank": i} for i in range(1, 13)]
        rows = [{"query": q, "target": t, "candidates": candidates}
                for q, t in [("gain", "6"), ("loss", "1"), ("tie", "1")]]

        def score(query, video):
            return int(query != "tie" and video == "6")

        with tempfile.TemporaryDirectory() as tmp:
            raw, out = Path(tmp) / "raw.jsonl", Path(tmp) / "ten.jsonl"
            write_run(raw, rows)
            args = parser().parse_args(["compare-depths", "--device", "cpu", "--input", str(raw),
                                        "--out", str(out), "--videos", tmp])
            with patch("kinetrace.depths.QwenScorer", return_value=score):
                report = run(args)
            self.assertEqual(report["gained_vs_top5"], 1)
            self.assertEqual(report["lost_vs_top5"], 1)
            self.assertEqual(report["top5"]["R@1"], 2/3)
            self.assertEqual(report["top10"]["R@1"], 2/3)
            self.assertEqual(list(read_run(out)), [rerank_prefix(r, 10, score) for r in rows])
