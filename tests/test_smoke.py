"""Offline smoke checks. The synthetic rankings are fixtures, not model predictions."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from kinetrace.__main__ import parser, run
from kinetrace.data import read_queries, read_run, write_run
from kinetrace.ranking import rerank_prefix, summarize
from kinetrace.video import sample_indices


class Smoke(unittest.TestCase):
    def test_retrieve_rerank_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv_path, raw, ranked = (root / name for name in ("split.csv", "raw.jsonl", "ranked.jsonl"))
            csv_path.write_text("video_id,sentence\nrecover,Skids then recovers\nskid,A car skids\n")
            retriever = Mock(ids=["skid", "recover"])
            candidates = [{"video_id": v, "retrieval_rank": i, "retrieval_score": 1/i}
                          for i, v in enumerate(retriever.ids, 1)]
            retriever.search.side_effect = lambda queries, depth: [candidates for q in queries]
            with patch("kinetrace.retrieval.Retriever", return_value=retriever):
                before = run(parser().parse_args(["retrieve", "--index", tmp, "--csv", str(csv_path),
                                                   "--out", str(raw)]))
            self.assertEqual(before["R@1"], 0.5)
            scores = {("Skids then recovers", "recover"), ("A car skids", "skid")}
            with patch("kinetrace.reranker.QwenScorer", return_value=lambda q, v: (q, v) in scores):
                run(parser().parse_args(["rerank", "--input", str(raw), "--out", str(ranked),
                                         "--videos", tmp, "--top-k", "2"]))
            after = run(parser().parse_args(["metrics", "--input", str(ranked)]))
            self.assertEqual(after["R@1"], 1)
            self.assertEqual(after["R@2"], before["R@2"])
            first = next(read_run(ranked))["candidates"][0]
            self.assertEqual((first["video_id"], first["retrieval_rank"]), ("recover", 2))

    def test_tail_and_candidate_miss(self):
        row = {"query": "recover", "target": "missing",
               "candidates": [{"video_id": v} for v in ["a", "b", "c"]]}
        result = rerank_prefix(row, 2, lambda q, v: int(v == "b"))
        self.assertEqual([i["video_id"] for i in result["candidates"]], ["b", "a", "c"])
        self.assertNotIn("rerank_score", result["candidates"][2])
        self.assertEqual(summarize([result])["R@3"], 0)

    def test_indices(self):
        self.assertEqual(sample_indices(2, 8), [0, 1])
        self.assertEqual(sample_indices(9, 3), [0, 4, 8])
        self.assertEqual(sample_indices(1, 1), [0])

    def test_free_query_preserves_original_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "query.jsonl"
            retriever = Mock()
            retriever.search.return_value = [[{"video_id": "v1", "retrieval_rank": 1}]]
            with patch("kinetrace.retrieval.Retriever", return_value=retriever):
                row = run(parser().parse_args(["search", "--index", tmp, "--query", "红车侧滑",
                                               "--retrieval-text", "a red car skids", "--out", str(output)]))
            retriever.search.assert_called_once_with(["a red car skids"], 20)
            self.assertEqual(row["query"], "红车侧滑")
            self.assertEqual(list(read_run(output)), [row])

    def test_partial_output_is_not_published(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.jsonl"
            write_run(path, [{"old": True}])
            with self.assertRaises(ValueError):
                write_run(path, [{"new": 1}, {"new": float("nan")}])
            self.assertEqual(list(read_run(path)), [{"old": True}])

    def test_invalid_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "split.csv"
            csv_path.write_text("video_id,sentence\n../bad,caption\n")
            with self.assertRaises(ValueError):
                read_queries(csv_path)
