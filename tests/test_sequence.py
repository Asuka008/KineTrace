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
    def test_paired_frame_comparison_same_slots_and_published_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = sequence.parser().parse_args(["--compare-frames", "--msrvtt-root", tmp,
                                                 "--device", "cpu", "--out-root", str(root / "runs")])
            rows = [{"line": n + 1, "video_name": f"video{n}.mp4", "caption": f"true {n}",
                     "counterfactual": f"false {n}"} for n in range(2)]
            videos = root / "videos"
            videos.mkdir()
            for row in rows:
                (videos / row["video_name"]).touch()
            slots = list(range(0, 19, 2))
            sparse = [0, 6, 12, 18]
            repeated = [0, 0, 6, 6, 6, 12, 12, 12, 18, 18]
            metadata = {"frames_indices": slots, "total_num_frames": 19, "fps": 2}
            model = Mock()
            model.score_frames.side_effect = [.3, .7, .4, .4, .8, .2, .7, .3, .3, .7, .4, .4]

            def sheet(path, images, indices, original):
                self.assertEqual((images, indices, original), (slots, slots, sparse))
                path.write_bytes(b"fixture")

            with patch.object(sequence, "load_labels", return_value=rows), \
                    patch.object(sequence, "read_clip", return_value=(slots, metadata)) as read, \
                    patch.object(sequence, "QwenScorer", return_value=model) as factory, \
                    patch.object(sequence, "save_frame_sheet", side_effect=sheet), \
                    patch("subprocess.check_output", return_value="fixture\n"), patch("builtins.print"):
                report = sequence.evaluate(args)
            factory.assert_called_once_with(args.qwen_model, "cpu", videos, frames=10)
            self.assertEqual([call.args for call in read.call_args_list],
                             [(videos / row["video_name"], 10) for row in rows])
            out, = args.out_root.iterdir()
            self.assertEqual(json.loads((out / "summary.json").read_text()), report)
            self.assertEqual(report["conditions"]["ten"]["strict_pair_accuracy"], .5)
            changes = report["paired_changes"]
            self.assertEqual(changes["four_to_ten"]["gained"], ["video0.mp4"])
            self.assertEqual(changes["four_to_ten"]["lost"], ["video1.mp4"])
            self.assertEqual(changes["four_to_ten"]["transitions"], {"incorrect_to_correct": 1, "correct_to_tie": 1})
            self.assertEqual(changes["four_repeated_to_ten"]["accuracy_change_pp"], 50)
            self.assertEqual(changes["four_to_four_repeated"]["net_correct"], -1)
            cases = list(read_run(out / "pairs.jsonl"))
            for i, case in enumerate(cases):
                self.assertTrue((out / case["frame_sheet"]).is_file())
                expected = (("four", sparse, sparse), ("four_repeated", repeated, slots), ("ten", slots, slots))
                for j, (name, sources, timestamps) in enumerate(expected):
                    condition = case["conditions"][name]
                    self.assertEqual((condition["source_indices"], condition["timestamp_indices"]), (sources, timestamps))
                    start = i * 6 + j * 2
                    for call, key in zip(model.score_frames.call_args_list[start:start + 2], ("caption", "counterfactual")):
                        self.assertEqual(call.args, (rows[i][key], sources, dict(metadata, frames_indices=timestamps)))

    def test_short_clip_leaves_no_partial_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = sequence.parser().parse_args(["--compare-frames", "--videos", tmp, "--device", "cpu",
                                                 "--out-root", str(root / "runs")])
            (root / "video1.mp4").touch()
            row = {"line": 1, "video_name": "video1.mp4", "caption": "open", "counterfactual": "close"}
            clip = (list(range(9)), {"total_num_frames": 9, "frames_indices": list(range(9))})
            with patch.object(sequence, "load_labels", return_value=[row]), patch.object(sequence, "QwenScorer"), \
                    patch.object(sequence, "read_clip", return_value=clip), patch("builtins.print"):
                with self.assertRaisesRegex(ValueError, "video1.mp4"):
                    sequence.evaluate(args)
            self.assertEqual(list(args.out_root.iterdir()), [])

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
