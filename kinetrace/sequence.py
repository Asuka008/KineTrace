"""Paired caption scoring on nested chronological frame samples."""
import math
from collections import Counter

from .temporal import SCORE_TOLERANCE
from .video import sample_indices


def score_pair(scorer, pair, images, metadata):
    caption = scorer.score_frames(pair["caption"], images, metadata)
    counterfactual = scorer.score_frames(pair["counterfactual"], images, metadata)
    margin = caption - counterfactual
    if not math.isfinite(margin):
        raise ValueError(f"Non-finite score at label line {pair['line']}")
    if abs(margin) <= SCORE_TOLERANCE:
        outcome = "tie"
    else:
        outcome = "correct" if margin > 0 else "incorrect"
    return {"caption_score": caption, "counterfactual_score": counterfactual, "outcome": outcome}


def outcome_counts(records):
    counts = Counter(r["outcome"] for r in records)
    return {"correct": counts["correct"], "incorrect": counts["incorrect"], "tie": counts["tie"],
            "strict_pair_accuracy": counts["correct"] / sum(counts.values())}


def frame_conditions(scorer, pair, images, metadata):
    slots = metadata["frames_indices"]
    sparse = sample_indices(metadata["total_num_frames"], 4)
    if (len(images) != 10 or len(slots) != 10 or slots[::3] != sparse
            or slots != sample_indices(metadata["total_num_frames"], 10)):
        raise ValueError(f"Expected ten frames containing the original four for {pair['video_name']}")
    by_index = dict(zip(slots, images))
    # Use the closest original frame at each dense slot; sparse is sorted, so earlier wins ties.
    repeated = [min(sparse, key=lambda original: abs(original - slot)) for slot in slots]
    conditions = {}
    for name, sources, timestamps in (("four", sparse, sparse), ("four_repeated", repeated, slots),
                                      ("ten", slots, slots)):
        ordered = [by_index[i] for i in sources]
        scores = score_pair(scorer, pair, ordered, dict(metadata, frames_indices=list(timestamps)))
        conditions[name] = dict(scores, source_indices=sources, timestamp_indices=timestamps)
    return conditions


def compare_conditions(rows):
    summary = {name: outcome_counts(row["conditions"][name] for row in rows)
               for name in ("four", "four_repeated", "ten")}
    comparisons = {}
    for baseline, evaluated in (("four", "ten"), ("four_repeated", "ten"), ("four", "four_repeated")):
        gains, losses = [], []
        transitions = Counter()
        for row in rows:
            before = row["conditions"][baseline]["outcome"]
            after = row["conditions"][evaluated]["outcome"]
            transitions[f"{before}_to_{after}"] += 1
            if before != "correct" and after == "correct":
                gains.append(row["video_name"])
            elif before == "correct" and after != "correct":
                losses.append(row["video_name"])
        comparisons[f"{baseline}_to_{evaluated}"] = {
            "gained": gains, "lost": losses, "transitions": dict(transitions),
            "net_correct": len(gains) - len(losses),
            "accuracy_change_pp": 100 * (len(gains) - len(losses)) / len(rows)}
    return {"conditions": summary, "paired_changes": comparisons}


def save_frame_sheet(path, images, indices, sparse):
    """Keep a compact visual record of the inputs and mark the original four frames."""
    from PIL import Image, ImageDraw

    sheet = Image.new("RGB", (5 * 224, 2 * 164), "white")
    draw = ImageDraw.Draw(sheet)
    for position, (image, index) in enumerate(zip(images, indices)):
        x, y = position % 5 * 224, position // 5 * 164
        thumb = Image.fromarray(image)
        thumb.thumbnail((220, 138))
        sheet.paste(thumb, (x + (224 - thumb.width) // 2, y))
        label = f"Frame {index}" + ("  (original 4)" if index in sparse else "")
        draw.text((x + 4, y + 141), label, fill="black")
    sheet.save(path, quality=85)
