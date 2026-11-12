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


def summarize(rows):
    """Text-to-video recall, one labeled target per query. Scores are fractions."""
    hits = {}
    n = 0
    depth = None
    for row in rows:
        ids = [item["video_id"] for item in row["candidates"]]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("Expected a nonempty list of distinct candidates")
        if depth is None:
            depth = len(ids)
            hits = {k: 0 for k in sorted({1, 5, 10, depth}) if k <= depth}
        if len(ids) != depth:
            raise ValueError("Candidate depth varies across this run")
        for k in hits:
            hits[k] += int(row["target"] in ids[:k])
        n += 1
    if not n:
        raise ValueError("Cannot evaluate an empty run")
    return {"queries": n, "candidate_depth": depth, **{f"R@{k}": v/n for k, v in hits.items()}}
