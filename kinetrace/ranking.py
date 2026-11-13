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


def compare_top1(rows):
    """Compare saved retrieval ranks with the current order of labeled candidates."""
    totals = {"gained": 0, "lost": 0, "stayed_correct": 0, "stayed_incorrect": 0}
    changes = []
    for row in rows:
        current = next(target_ranks([row]))
        candidates = row["candidates"]
        original = sorted(candidates, key=lambda item: item["retrieval_rank"])
        if [item["retrieval_rank"] for item in original] != list(range(1, len(original) + 1)):
            raise ValueError(f"Invalid retrieval ranks for {row['query_id']}")
        original_top = original[0]["video_id"]
        before = original_top == row["target"]
        after = current["target_rank"] == 1
        if before == after:
            totals["stayed_correct" if after else "stayed_incorrect"] += 1
            continue
        outcome = "gained" if after else "lost"
        totals[outcome] += 1
        old_rank = next(item["retrieval_rank"] for item in candidates if item["video_id"] == row["target"])
        changes.append({"outcome": outcome, "query_id": row["query_id"], "query": row["query"],
                        "target": row["target"], "clip_rank": old_rank, "qwen_rank": current["target_rank"],
                        "clip_top": original_top, "qwen_top": current["top_video"]})
    if not sum(totals.values()):
        raise ValueError("Cannot compare an empty run")
    return {"queries": sum(totals.values()), **totals, "changes": changes}


def compare_runs(baseline, evaluated):
    """Pair by query ID even if the two result files have different row order."""
    def index(rows):
        result = {}
        for row in rows:
            qid = row["query_id"]
            if qid in result:
                raise ValueError(f"Duplicate query ID: {qid}")
            ranks = next(target_ranks([row]))
            candidates = frozenset(item["video_id"] for item in row["candidates"])
            result[qid] = ranks, candidates
        return result

    before, after = index(baseline), index(evaluated)
    if not before or before.keys() != after.keys():
        raise ValueError("Expected identical, nonempty query ID sets")
    changes = []
    base_hits = new_hits = gained = lost = 0
    for qid, (old, old_ids) in before.items():
        new, new_ids = after[qid]
        if old_ids != new_ids or old["target"] != new["target"] or old["query"] != new["query"]:
            raise ValueError(f"Query, label or candidate membership mismatch for {qid}")
        was_hit = old["target_rank"] == 1
        is_hit = new["target_rank"] == 1
        base_hits += was_hit
        new_hits += is_hit
        if was_hit != is_hit:
            gained += is_hit
            lost += was_hit
            changes.append({"query_id": qid, "query": old["query"], "target": old["target"],
                            "outcome": "gained" if is_hit else "lost",
                            "baseline_top": old["top_video"], "evaluated_top": new["top_video"],
                            "baseline_rank": old["target_rank"], "evaluated_rank": new["target_rank"]})
    return {"queries": len(before), "baseline_hits": base_hits, "evaluated_hits": new_hits,
            "gained": gained, "lost": lost, "net": new_hits - base_hits, "changes": changes}
