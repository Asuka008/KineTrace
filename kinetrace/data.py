"""Read the MSR-VTT JSFusion CSV or a small corpus with the same columns."""
import csv
import json
import os
import re
import tempfile
from contextlib import contextmanager
from pathlib import Path


def read_queries(path, limit=None):
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"video_id", "sentence"}.issubset(reader.fieldnames or ()):
            raise ValueError("Expected CSV columns: video_id,sentence")
        for line, row in enumerate(reader, 2):
            video_id = (row["video_id"] or "").strip()
            text = (row["sentence"] or "").strip()
            if not re.fullmatch(r"[\w-]+", video_id) or not text:
                raise ValueError(f"Invalid video_id or sentence at CSV line {line}")
            rows.append({"query_id": f"q{line-1}", "query": text, "target": video_id})
            if limit is not None and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"Empty query CSV: {path}")
    return rows


def read_run(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


@contextmanager
def output_file(path):
    """Publish completed output atomically, keeping an earlier run on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temp = Path(stream.name)
        try:
            yield stream
            stream.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


def write_run(path, rows):
    with output_file(path) as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
