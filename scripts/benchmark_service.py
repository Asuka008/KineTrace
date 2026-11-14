"""Replay labeled queries serially through the search API and record client latency."""
import argparse
import hashlib
import json
import time
import urllib.request

from kinetrace.data import output_file, read_queries
from kinetrace.depths import latency_ms
from kinetrace.ranking import summarize


def request(base_url, endpoint, payload=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base_url.rstrip("/") + endpoint, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def benchmark(args):
    ready = request(args.url, "/readyz", timeout=args.timeout)
    if ready["bandit_enabled"]:
        raise ValueError("Run benchmark with feedback collection disabled to keep replay out of the feedback log")
    rows = read_queries(args.csv, args.limit)
    def payload(row):
        return {"query": row["query"], "retrieval_k": args.retrieval_k, "rerank_k": args.rerank_k}

    request(args.url, "/search", payload(rows[0]), args.timeout)
    elapsed, results, retrieval, errors, fallbacks, memory = [], [], [], [], 0, []
    started = time.perf_counter()
    for row in rows:
        begin = time.perf_counter()
        try:
            result = request(args.url, "/search", payload(row), args.timeout)
        except Exception as exc:
            errors.append({"query_id": row["query_id"], "error": str(exc)})
            continue
        elapsed.append(time.perf_counter() - begin)
        results.append(dict(row, candidates=result["candidates"]))
        retrieval.append(dict(row, candidates=result["retrieval"]))
        fallbacks += result["fallback"]
        if result.get("memory"):
            memory.append(result["memory"])
    total = time.perf_counter() - started
    report = {"ready": ready, "csv": str(args.csv),
              "csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
              "retrieval_k": args.retrieval_k, "rerank_k": args.rerank_k,
              "queries": len(rows), "successful": len(results),
              "errors": errors, "fallbacks": fallbacks, "concurrency": 1, "warmup_queries": 1,
              "client_latency_ms_successful": latency_ms(elapsed) if elapsed else None,
              "successful_queries_per_second": len(results) / total,
              "metrics_scope": "successful requests including CLIP fallbacks",
              "retrieval": summarize(retrieval) if retrieval else None,
              "reranked": summarize(results) if results else None,
              "peak_allocated_gib": max((m["peak_allocated_gib"] for m in memory), default=None),
              "peak_reserved_gib": max((m["peak_reserved_gib"] for m in memory), default=None),
              "cache_condition": "existing server filesystem/model caches after one warmup"}
    with output_file(args.out) as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report


if __name__ == "__main__":
    from pathlib import Path
    from kinetrace.__main__ import positive, positive_seconds

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=positive)
    parser.add_argument("--retrieval-k", type=positive, default=20)
    parser.add_argument("--rerank-k", type=positive, default=5)
    parser.add_argument("--timeout", type=positive_seconds, default=60)
    print(json.dumps(benchmark(parser.parse_args()), indent=2))
