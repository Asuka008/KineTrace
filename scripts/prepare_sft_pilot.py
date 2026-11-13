"""Prepare disjoint training and validation CSVs and clips from the MSR-VTT training release."""
import argparse
import csv
import json
import random
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


def prepare(train_json, test_csv, archive_path, out, seed):
    if out.exists():
        raise FileExistsError(f"Choose a new pilot directory: {out}")
    records = json.loads(train_json.read_text(encoding="utf-8"))
    with test_csv.open(encoding="utf-8-sig", newline="") as stream:
        excluded = {row["video_id"].strip() for row in csv.DictReader(stream)}
    if not excluded:
        raise ValueError("Empty held-out test split")
    captions = {}
    for row in records:
        vid = row["video_id"]
        if not re.fullmatch(r"video[0-9]+", vid) or vid in captions or vid in excluded:
            raise ValueError(f"Unexpected, duplicated or test video ID: {vid}")
        if not isinstance(row["caption"], list):
            raise ValueError(f"Missing list of captions for {vid}")
        captions[vid] = sorted({s.strip() for s in row["caption"] if s.strip()})
        if not captions[vid]:
            raise ValueError(f"Empty captions for {vid}")
    rng = random.Random(seed)
    selected = rng.sample(sorted(captions), 48)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp, zipfile.ZipFile(archive_path) as archive:
        stage = Path(tmp) / "pilot"
        videos = stage / "videos"
        videos.mkdir(parents=True)
        members = {Path(item.filename).name: item for item in archive.infolist()}
        for split, ids in (("train", selected[:32]), ("val", selected[32:])):
            with (stage / f"{split}.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["video_id", "sentence"])
                for vid in ids:
                    name = f"{vid}.mp4"
                    with archive.open(members[name]) as src, (videos / name).open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    writer.writerow([vid, rng.choice(captions[vid])])
        stage.rename(out)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("train-json", "test-csv", "archive", "out"):
        cli.add_argument("--" + key, required=True, type=Path)
    cli.add_argument("--seed", default=42, type=int)
    args = cli.parse_args()
    prepare(args.train_json, args.test_csv, args.archive, args.out, args.seed)
    print(f"Prepared 32 training and 16 validation videos in {args.out}")
