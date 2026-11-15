import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from kinetrace.data import read_run
from scripts import evaluate_sequence as sequence


class SequenceTests(unittest.TestCase):
    def test_published_pairs_on_same_frames_using_folder_and_zip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [{"src_dataset": "MSRVTT", "aspect": "Sequence", "video_name": f"video{i}.mp4",
                     "caption": f"first {i}", "counterfactual": f"last {i}"} for i in range(3)]
            rows.append(dict(rows[0], src_dataset="VATEX", video_name="unneeded.mp4"))
            data = "\n".join(json.dumps(row) for row in rows).encode()
            (root / "videos").mkdir()
            (root / "videos/video0.mp4").write_bytes(b"video0")
            with zipfile.ZipFile(root / "MSRVTT_Videos.zip", "w") as archive:
                for i in (1, 2):
                    archive.writestr(f"nested/video{i}.mp4", f"video{i}".encode())
            args = sequence.parser().parse_args(["--msrvtt-root", tmp, "--device", "cpu",
                                                 "--out-root", str(root / "runs")])
            model = Mock()
            model.score_frames.side_effect = [.8, .1, .2, .7, .5, .5000001]

            def decode(path, frames):
                self.assertEqual(path.read_bytes(), path.stem.encode())
                self.assertEqual(frames, 4)
                return [path.stem] * 4, {"frames_indices": [0, 1, 2, 3]}

            with patch.object(sequence, "LABEL_SHA256", hashlib.sha256(data).hexdigest()), \
                    patch.object(sequence, "urlopen", return_value=io.BytesIO(data)) as get, \
                    patch.object(sequence, "read_clip", side_effect=decode) as read, \
                    patch.object(sequence, "QwenScorer", return_value=model), \
                    patch("subprocess.check_output", return_value="fixture\n"), patch("builtins.print"):
                report = sequence.evaluate(args)
                self.assertEqual(len(sequence.load_labels(root, supplied=None)), 3)
            get.assert_called_once()
            self.assertEqual(read.call_count, 3)
            self.assertEqual([report[key] for key in ("pairs", "correct", "incorrect", "tie")], [3, 1, 1, 1])
            self.assertEqual(report["strict_pair_accuracy"], 1/3)
            for i in range(3):
                positive, negative = model.score_frames.call_args_list[i*2:i*2+2]
                self.assertEqual((positive.args[0], negative.args[0]), (f"first {i}", f"last {i}"))
                self.assertIs(positive.args[1], negative.args[1])
            out, = args.out_root.iterdir()
            self.assertEqual(list(read_run(out / "pairs.jsonl"))[-1]["outcome"], "tie")
            self.assertEqual(json.loads((out / "summary.json").read_bytes()), report)

    def test_reject_changed_labels_and_incomplete_video_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "Sequence.jsonl"
            labels.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                sequence.load_labels(root, labels)
            with zipfile.ZipFile(root / "videos.zip", "w"):
                pass
            with self.assertRaisesRegex(FileNotFoundError, "video1.mp4"):
                sequence.video_paths({"video1.mp4"}, root, root / "videos.zip", root)
