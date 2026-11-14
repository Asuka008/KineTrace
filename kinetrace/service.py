"""Serve one resident retrieval/reranking pipeline with bounded GPU admission."""
import asyncio
import hashlib
import json
import logging
import math
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Optional
from urllib.parse import quote

from .ranking import rerank_prefix
from .reranker import QWEN_MODEL, QWEN_REVISION, QwenScorer
from .retrieval import Retriever


LOGGER = logging.getLogger("uvicorn.error")


class ServiceBusy(Exception):
    """The active request and waiting requests fill the admission limit."""


class SearchProgress:
    """Publish a completed retrieval result for a timed-out HTTP request."""

    def __init__(self):
        self.lock = threading.Lock()
        self.retrieved = None
        self.retrieved_at = None

    def publish(self, result):
        with self.lock:
            self.retrieved = result
            self.retrieved_at = perf_counter()

    def fallback(self, reason, submitted):
        with self.lock:
            if self.retrieved is None:
                return None
            result = deepcopy(self.retrieved)
            now = perf_counter()
            result.update(fallback=True, fallback_reason=reason)
            result["timings_ms"].update(rerank=(now - self.retrieved_at) * 1000,
                                        total=(now - submitted) * 1000)
            return result


class SearchEngine:
    """Load once on the inference thread, then reuse the existing pipeline."""

    def __init__(self, args, retriever_factory=Retriever, scorer_factory=QwenScorer):
        self.videos = args.videos.resolve()
        if not self.videos.is_dir():
            raise FileNotFoundError(f"Video directory missing: {self.videos}")
        self.synchronize = lambda: None
        self.cuda = None
        device = {"device": args.device}
        if args.device == "cuda":
            import torch

            props = torch.cuda.get_device_properties(0)
            device.update(gpu=props.name, visible_gib=props.total_memory / 2**30)
            self.synchronize = torch.cuda.synchronize
            self.cuda = torch.cuda
        # Report the allocation before loading either model, including on startup OOM.
        LOGGER.info("service_device %s", json.dumps(device))

        model, adapter = args.qwen_model, args.adapter
        if args.current_sft:
            if adapter:
                raise ValueError("Choose either --current-sft or --adapter")
            adapter = Path("runs/sft-pilot-zxCXNv5N/adapter")
            if not adapter.is_dir():
                raise FileNotFoundError(f"Retained SFT adapter missing: {adapter}")
            if model == QWEN_MODEL:
                model = json.loads((adapter / "train-report.json").read_text())["base"]
        manifest_bytes = (args.index / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        self.retriever = retriever_factory(args.index, args.device)
        self.video_ids = set(self.retriever.ids)
        if any(not re.fullmatch(r"[\w-]+", video_id) for video_id in self.video_ids):
            raise ValueError("Index contains an invalid video ID")
        self.scorer = scorer_factory(model, args.device, args.videos, args.frames)
        if adapter:
            from peft import PeftModel

            self.scorer.model = PeftModel.from_pretrained(self.scorer.model, str(adapter)).eval()
        self.metadata = {**device, "qwen_model": model, "adapter": str(adapter) if adapter else None,
                         "qwen_revision": QWEN_REVISION if model == QWEN_MODEL else None,
                         "frames": args.frames, "index": str(args.index),
                         "index_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                         "clip_model": manifest["model"], "clip_revision": manifest.get("revision"),
                         "videos": len(self.video_ids)}
        if adapter:
            report = adapter / "train-report.json"
            self.metadata["adapter_report_sha256"] = (hashlib.sha256(report.read_bytes()).hexdigest()
                                                       if report.is_file() else None)

    def video_path(self, video_id):
        if video_id not in self.video_ids or not re.fullmatch(r"[\w-]+", video_id):
            return None
        path = (self.videos / f"{video_id}.mp4").resolve()
        if path.parent != self.videos or not path.is_file():
            return None
        return path

    def search(self, payload, progress, cancelled, submitted, timeout_s, request_id):
        deadline = submitted + timeout_s
        if cancelled.is_set() or perf_counter() >= deadline:
            raise TimeoutError("Search expired while waiting for inference")
        self.synchronize()
        if self.cuda is not None:
            self.cuda.reset_peak_memory_stats()
        started = perf_counter()
        text = payload.get("retrieval_text") or payload["query"]
        candidates = self.retriever.search([text], payload["retrieval_k"])[0]
        candidates = [dict(item, video_url=f"/videos/{quote(item['video_id'], safe='')}")
                      for item in candidates]
        self.synchronize()
        retrieved = perf_counter()
        result = {"request_id": request_id, "query": payload["query"], "retrieval_text": text,
                  "retrieval": candidates, "candidates": candidates,
                  "fallback": False, "fallback_reason": None, "model": self.metadata,
                  "timings_ms": {"queue": (started - submitted) * 1000,
                                 "retrieval": (retrieved - started) * 1000,
                                 "rerank": 0.0, "total": (retrieved - submitted) * 1000}}
        progress.publish(result)

        def score(query, video_id):
            if cancelled.is_set() or perf_counter() >= deadline:
                raise TimeoutError("Reranking deadline exceeded")
            value = self.scorer(query, video_id)
            self.synchronize()
            if cancelled.is_set() or perf_counter() >= deadline:
                raise TimeoutError("Reranking deadline exceeded")
            return value

        try:
            ranked = rerank_prefix({"query": payload["query"], "candidates": candidates},
                                   payload["rerank_k"], score)
        except TimeoutError:
            return progress.fallback("timeout", submitted)
        except Exception:
            LOGGER.exception("reranker_failed request_id=%s", request_id)
            return progress.fallback("reranker_error", submitted)
        completed = perf_counter()
        memory = {}
        if self.cuda is not None:
            memory = {"peak_allocated_gib": self.cuda.max_memory_allocated() / 2**30,
                      "peak_reserved_gib": self.cuda.max_memory_reserved() / 2**30}
        return dict(result, candidates=ranked["candidates"], memory=memory, timings_ms={
            **result["timings_ms"], "rerank": (completed - retrieved) * 1000,
            "total": (completed - submitted) * 1000})


class InferenceWorker:
    """Keep running CUDA calls serialized even after an HTTP timeout."""

    def __init__(self, engine_factory, max_pending, timeout_s):
        if max_pending < 1 or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("max_pending and timeout_s must be positive")
        self.engine_factory = engine_factory
        self.max_pending = max_pending
        self.timeout_s = timeout_s
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kinetrace-inference")
        self.lock = threading.Lock()
        self.pending = 0
        self.engine = None
        self.closing = False

    async def start(self):
        self.engine = await asyncio.get_running_loop().run_in_executor(self.executor, self.engine_factory)
        LOGGER.info("service_ready %s", json.dumps(self.engine.metadata))

    async def close(self):
        self.closing = True
        await asyncio.to_thread(self.executor.shutdown, wait=True, cancel_futures=True)
        self.engine = None

    async def search(self, payload):
        with self.lock:
            if self.closing or self.engine is None or self.pending >= self.max_pending:
                raise ServiceBusy("Search queue is full or the service is not ready")
            self.pending += 1
        progress = SearchProgress()
        cancelled = threading.Event()
        submitted = perf_counter()
        request_id = uuid.uuid4().hex

        def release(future):
            with self.lock:
                self.pending -= 1

        try:
            future = self.executor.submit(self.engine.search, payload, progress, cancelled,
                                          submitted, self.timeout_s, request_id)
        except Exception:
            release(None)
            raise
        future.add_done_callback(release)
        wrapped = asyncio.wrap_future(future)
        # A timed-out request can leave a running future. Consume its eventual exception.
        wrapped.add_done_callback(lambda task: None if task.cancelled() else task.exception())
        try:
            result = await asyncio.wait_for(asyncio.shield(wrapped), self.timeout_s)
        except asyncio.TimeoutError:
            cancelled.set()
            future.cancel()
            result = progress.fallback("timeout", submitted)
            if result is None:
                raise TimeoutError("Search timed out before retrieval completed") from None
        except asyncio.CancelledError:
            cancelled.set()
            future.cancel()
            raise
        LOGGER.info("search_completed %s", json.dumps({
            "request_id": request_id, "timings_ms": result["timings_ms"],
            "fallback": result["fallback"], "fallback_reason": result["fallback_reason"]}))
        return result


def create_app(args, engine_factory=None):
    """Build the API without importing ML libraries or loading weights."""
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse
    from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

    from .bandit import BanditPolicy

    if args.port > 65535:
        raise ValueError("Port must be at most 65535")
    if args.bandit_model and not args.bandit_log:
        raise ValueError("--bandit-model requires --bandit-log")
    policy = BanditPolicy(args.bandit_log, args.epsilon, args.bandit_model) if args.bandit_log else None

    class SearchRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        query: str = Field(min_length=1, max_length=4096)
        retrieval_text: Optional[str] = Field(default=None, min_length=1, max_length=4096)
        retrieval_k: int = Field(default=20, ge=1, le=100, strict=True)
        rerank_k: int = Field(default=5, ge=1, le=100, strict=True)

        @field_validator("query", "retrieval_text")
        @classmethod
        def strip_text(cls, value):
            if value is None:
                return value
            value = value.strip()
            if not value:
                raise ValueError("Text must contain a non-whitespace character")
            return value

        @model_validator(mode="after")
        def check_depth(self):
            if self.rerank_k > self.retrieval_k:
                raise ValueError("rerank_k must not exceed retrieval_k")
            return self

    class FeedbackRequest(BaseModel):
        model_config = ConfigDict(extra="forbid")
        event_id: str = Field(pattern=r"^[a-f0-9]{32}$")
        reward: int = Field(ge=0, le=1, strict=True)

    worker = InferenceWorker(engine_factory or (lambda: SearchEngine(args)),
                             args.max_pending, args.timeout_s)

    @asynccontextmanager
    async def lifespan(app):
        try:
            await worker.start()
            yield
        finally:
            await worker.close()

    app = FastAPI(title="KineTrace", lifespan=lifespan, redoc_url=None,
                  swagger_ui_oauth2_redirect_url=None)
    app.state.worker = worker

    @app.get("/")
    async def demo():
        return FileResponse(Path(__file__).with_name("demo.html"), media_type="text/html")

    @app.get("/healthz")
    async def health():
        return {"status": "ok"}

    @app.get("/readyz")
    async def ready():
        if worker.engine is None or worker.closing:
            raise HTTPException(status_code=503, detail="Models are not ready")
        return {"status": "ready", "model": worker.engine.metadata,
                "pending": worker.pending, "max_pending": worker.max_pending,
                "bandit_enabled": policy is not None}

    @app.post("/search")
    async def search(payload: SearchRequest):
        try:
            result = await worker.search(payload.model_dump())
        except ServiceBusy as exc:
            raise HTTPException(status_code=503, detail=str(exc), headers={"Retry-After": "1"}) from exc
        except TimeoutError as exc:
            raise HTTPException(status_code=504, detail=str(exc)) from exc
        except Exception as exc:
            LOGGER.exception("retrieval_failed")
            raise HTTPException(status_code=500, detail="Search retrieval failed") from exc
        if policy is not None and not result["fallback"]:
            result["recommendation"] = await asyncio.to_thread(policy.recommend, result, payload.rerank_k)
        return result

    @app.post("/feedback")
    async def feedback(payload: FeedbackRequest):
        if policy is None:
            raise HTTPException(status_code=404, detail="Feedback logging is disabled")
        try:
            await asyncio.to_thread(policy.store.feedback, payload.event_id, payload.reward)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown recommendation") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "recorded", "event_id": payload.event_id}

    @app.get("/videos/{video_id}")
    async def video(video_id: str):
        if worker.engine is None or worker.closing:
            raise HTTPException(status_code=503, detail="Models are not ready")
        path = worker.engine.video_path(video_id)
        if path is None:
            raise HTTPException(status_code=404, detail="Video not found")
        return FileResponse(path, media_type="video/mp4")

    return app


def run_server(args):
    import uvicorn

    uvicorn.run(create_app(args), host=args.host, port=args.port, workers=1)
