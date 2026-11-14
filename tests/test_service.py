import asyncio
import importlib.util
import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from kinetrace.__main__ import parser
from kinetrace.service import InferenceWorker, SearchEngine, ServiceBusy
from scripts.benchmark_service import benchmark


def configuration(root):
    return parser().parse_args(["serve", "--index", str(root), "--videos", str(root), "--device", "cpu"])


def fixture_engine(root, scorer=None):
    (root / "manifest.json").write_text(json.dumps({"model": "fixture"}))
    (root / "a.mp4").write_bytes(b"video-a")
    (root / "b.mp4").write_bytes(b"video-b")
    retriever = Mock(ids=["a", "b"])
    retriever.search.return_value = [{"video_id": "a", "retrieval_rank": 1, "retrieval_score": 0.8},
                                   {"video_id": "b", "retrieval_rank": 2, "retrieval_score": 0.7}]
    retriever.search.return_value = [retriever.search.return_value]
    scorer = scorer or (lambda query, video: 1.0 if video == "b" else 0.0)
    return SearchEngine(configuration(root), lambda *_: retriever, lambda *_: scorer)


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_keeps_running_inference_slot_until_cuda_work_finishes(self):
        started, finish = threading.Event(), threading.Event()

        def slow(payload, progress, cancelled, submitted, timeout_s, request_id):
            progress.publish({"timings_ms": {}, "candidates": ["a"]})
            started.set()
            finish.wait(2)
            return progress.fallback("timeout", submitted)

        engine = SimpleNamespace(metadata={}, search=slow)
        worker = InferenceWorker(lambda: engine, max_pending=1, timeout_s=0.03)
        await worker.start()
        try:
            result = await worker.search({})
            self.assertTrue(started.is_set())
            self.assertTrue(result["fallback"])
            self.assertEqual(worker.pending, 1)
            with self.assertRaises(ServiceBusy):
                await worker.search({})
        finally:
            finish.set()
            await worker.close()
        self.assertEqual(worker.pending, 0)

    async def test_pending_timeout_cancels_queued_request(self):
        finish = threading.Event()
        calls = []

        def slow(*args):
            calls.append(1)
            finish.wait(2)
            return {"timings_ms": {}, "fallback": False, "fallback_reason": None}

        worker = InferenceWorker(lambda: SimpleNamespace(metadata={}, search=slow), 2, 0.03)
        await worker.start()
        try:
            results = await asyncio.gather(worker.search({}), worker.search({}), return_exceptions=True)
            self.assertTrue(all(isinstance(result, TimeoutError) for result in results))
        finally:
            finish.set()
            await worker.close()
        self.assertEqual(len(calls), 1)


class BenchmarkTests(unittest.TestCase):
    def test_report_counts_failures_and_keeps_fallback_in_successful_recall(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            csv = root / "queries.csv"
            csv.write_text("video_id,sentence\na,first\nb,second\n")
            candidate = {"video_id": "a"}
            result = {"candidates": [candidate], "retrieval": [candidate], "fallback": True}
            args = SimpleNamespace(url="http://fixture", csv=csv, limit=None, retrieval_k=1,
                                   rerank_k=1, timeout=1, out=root / "report.json")
            with patch("scripts.benchmark_service.request", side_effect=[
                    {"bandit_enabled": False}, result, result, TimeoutError("fixture timeout")]):
                report = benchmark(args)
            self.assertEqual(report["queries"], 2)
            self.assertEqual(report["csv"], str(csv))
            self.assertEqual(len(report["csv_sha256"]), 64)
            self.assertEqual(report["rerank_k"], 1)
            self.assertEqual(report["successful"], 1)
            self.assertEqual(report["fallbacks"], 1)
            self.assertEqual(len(report["errors"]), 1)
            self.assertEqual(report["reranked"]["R@1"], 1)
            self.assertIsNone(report["peak_allocated_gib"])


@unittest.skipUnless(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"),
                     "Install requirements-service.txt for HTTP fixtures")
class HttpTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        from kinetrace.service import create_app

        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.args = configuration(self.root)
        self.args.bandit_log = self.root / "feedback.sqlite3"
        self.engine = fixture_engine(self.root)
        self.factory = Mock(return_value=self.engine)
        self.client = TestClient(create_app(self.args, self.factory))
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def test_search_reload_once_media_and_feedback_roundtrip(self):
        self.assertEqual({route.path for route in self.client.app.routes},
                         {"/", "/healthz", "/readyz", "/search", "/feedback",
                          "/videos/{video_id}", "/docs", "/openapi.json"})
        self.assertEqual(self.client.get("/healthz").status_code, 200)
        self.assertEqual(self.client.get("/readyz").json()["model"]["videos"], 2)
        self.assertIn("KineTrace", self.client.get("/").text)
        for _ in range(2):
            response = self.client.post("/search", json={"query": "running"})
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            self.assertEqual(body["retrieval"][0]["video_id"], "a")
            self.assertEqual(body["candidates"][0]["video_id"], "b")
        self.factory.assert_called_once()
        self.assertEqual(self.client.get("/videos/a").content, b"video-a")
        self.assertEqual(self.client.get("/videos/unknown").status_code, 404)
        event = body["recommendation"]["event_id"]
        payload = {"event_id": event, "reward": 1}
        self.assertEqual(self.client.post("/feedback", json=payload).status_code, 200)
        self.assertEqual(self.client.post("/feedback", json=payload).status_code, 200)
        self.assertEqual(self.client.post("/feedback", json=dict(payload, reward=0)).status_code, 409)
        self.assertEqual(self.client.post("/feedback", json=dict(payload, reward=True)).status_code, 422)

    def test_input_validation_and_reranker_failure_preserve_retrieval(self):
        for payload in ({"query": " "}, {"query": "x", "retrieval_k": 2, "rerank_k": 3},
                        {"query": "x", "retrieval_k": True}, {"query": "x", "unknown": 1}):
            self.assertEqual(self.client.post("/search", json=payload).status_code, 422)
        self.engine.scorer = Mock(side_effect=RuntimeError("fixture failure"))
        with self.assertLogs("uvicorn.error", level="ERROR"):
            body = self.client.post("/search", json={"query": "x"}).json()
        self.assertTrue(body["fallback"])
        self.assertEqual(body["candidates"], body["retrieval"])
        self.assertNotIn("recommendation", body)

    def test_symlink_escape_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp) / "secret.mp4"
            outside.write_bytes(b"not-a-gallery-video")
            (self.root / "a.mp4").unlink()
            (self.root / "a.mp4").symlink_to(outside)
            self.assertEqual(self.client.get("/videos/a").status_code, 404)
