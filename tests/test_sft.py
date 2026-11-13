import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from kinetrace.data import read_queries
from kinetrace.sft import training_pairs
from scripts.prepare_sft_pilot import prepare


class SftPilot(unittest.TestCase):
    def test_same_caption_known_positives_excluded_from_negatives(self):
        candidates = [{"video_id": vid, "retrieval_rank": rank}
                      for rank, vid in enumerate(["b", "a", "c"], 1)]
        rows = [{"query": "skiing", "target": "a", "candidates": candidates},
                {"query": "Skiing", "target": "b", "candidates": candidates}]
        self.assertEqual(training_pairs(rows), [("skiing", "a", "c"), ("Skiing", "b", "c")])

    def test_split_extract_and_reject_test_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, held, archive, out = [root / f for f in ["train.json", "held.csv", "v.zip", "out"]]
            records = [{"video_id": f"video{i}", "caption": ["one", "two"]} for i in range(50)]
            data.write_text(json.dumps(records))
            held.write_text("video_id,sentence\nvideo50,test\n")
            with zipfile.ZipFile(archive, "w") as z:
                for r in records:
                    z.writestr("folder/" + r["video_id"] + ".mp4", b"fixture")
            prepare(data, held, archive, out, 42)
            train, val = [{r["target"] for r in read_queries(out / f"{split}.csv")}
                          for split in ("train", "val")]
            self.assertEqual((len(train), len(val), train & val), (32, 16, set()))
            self.assertEqual({p.stem for p in (out / "videos").iterdir()}, train | val)
            held.write_text("video_id,sentence\nvideo0,test\n")
            with self.assertRaisesRegex(ValueError, "test video ID"):
                prepare(data, held, archive, root / "bad", 42)
