import math


def rerank_prefix(row, depth, score_video):
    """score_video sees only the text and candidate ID. Keep the untouched tail."""
    candidates = row["candidates"]
    scored = []
    for item in candidates[:depth]:
        score = float(score_video(row["query"], item["video_id"]))
        if not math.isfinite(score):
            raise ValueError(f"Non-finite score for {item['video_id']}")
        scored.append(dict(item, rerank_score=score))
    scored.sort(key=lambda item: -item["rerank_score"])  # preserve retrieval order on ties
    return dict(row, candidates=scored + candidates[depth:])


def target_ranks(rows):
    """Locate the labeled target in saved candidate order, including after reranking."""
    for row in rows:
        ids = [item["video_id"] for item in row["candidates"]]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Expected a nonempty list of distinct candidates")
        target = row["target"]
        yield {**{k: row[k] for k in ("query_id", "query") if k in row},
               "target": target, "top_video": ids[0], "candidate_depth": len(ids),
               "target_rank": ids.index(target) + 1 if target in ids else None}


def summarize(rows):
    """Text-to-video recall, one labeled target per query. Scores are fractions."""
    hits = {}
    n = 0
    depth = None
    for row in target_ranks(rows):
        if depth is None:
            depth = row["candidate_depth"]
            hits = {k: 0 for k in sorted({1, 5, 10, depth}) if k <= depth}
        if row["candidate_depth"] != depth:
            raise ValueError("Candidate depth varies across this run")
        for k in hits:
            hits[k] += int(row["target_rank"] is not None and row["target_rank"] <= k)
        n += 1
    if not n:
        raise ValueError("Cannot evaluate an empty run")
    return {"queries": n, "candidate_depth": depth, **{f"R@{k}": v/n for k, v in hits.items()}}
