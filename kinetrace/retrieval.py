"""GPU frame/text encoding, exact normalized inner-product search on CPU."""
import hashlib
import json
import tempfile
from pathlib import Path

from .video import read_clip

CLIP_MODEL = "openai/clip-vit-base-patch32"


class ClipEncoder:
    def __init__(self, model_path, device, revision=None):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = device
        self.model = CLIPModel.from_pretrained(model_path, revision=revision).eval().to(device)
        self.revision = getattr(self.model.config, "_commit_hash", None) or revision
        self.processor = CLIPProcessor.from_pretrained(model_path, revision=self.revision)
        self.torch = torch

    def text(self, queries):
        inputs = self.processor(text=queries, return_tensors="pt", padding=True, truncation=True)
        with self.torch.inference_mode():
            vectors = self.model.get_text_features(**inputs.to(self.device)).float()
            return self.torch.nn.functional.normalize(vectors).cpu().numpy()

    def video(self, path, frames):
        images, _ = read_clip(path, frames)
        pixels = self.processor(images=images, return_tensors="pt").pixel_values.to(self.device)
        with self.torch.inference_mode():
            vectors = self.model.get_image_features(pixel_values=pixels).float()
            normalize = self.torch.nn.functional.normalize
            pooled = normalize(vectors).mean(dim=0, keepdim=True)
            return normalize(pooled).cpu().numpy()


def build_index(args, queries):
    import faiss

    index_dir = args.index
    if index_dir.exists():
        raise FileExistsError(f"Choose a new index directory: {index_dir}")
    ids = list(dict.fromkeys(row["target"] for row in queries))
    for video_id in ids:
        path = args.videos / f"{video_id}.mp4"
        if not path.is_file():
            raise FileNotFoundError(path)
    encoder = ClipEncoder(args.clip_model, args.device)
    index = faiss.IndexFlatIP(encoder.model.config.projection_dim)
    for video_id in ids:
        index.add(encoder.video(args.videos / f"{video_id}.mp4", args.frames))
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"video_ids": ids, "model": args.clip_model, "revision": encoder.revision,
                "frames": args.frames, "csv_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
                "indexed_query_rows": len(queries)}
    with tempfile.TemporaryDirectory(dir=index_dir.parent) as tmp:
        staged = Path(tmp) / "index"
        staged.mkdir()
        faiss.write_index(index, str(staged / "vectors.faiss"))
        (staged / "manifest.json").write_text(json.dumps(metadata), encoding="utf-8")
        staged.rename(index_dir)
    return {"videos": len(ids), "frames": args.frames, "index": str(index_dir)}


class Retriever:
    def __init__(self, index_dir, device):
        import faiss

        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        self.ids = manifest["video_ids"]
        self.index = faiss.read_index(str(index_dir / "vectors.faiss"))
        if self.index.ntotal != len(self.ids) or not self.ids:
            raise ValueError("Index size and video manifest disagree")
        self.encoder = ClipEncoder(manifest["model"], device, revision=manifest.get("revision"))

    def search(self, queries, depth):
        scores, indices = self.index.search(self.encoder.text(queries), min(depth, len(self.ids)))
        results = []
        for batch_scores, batch_indices in zip(scores, indices):
            results.append([
                {"video_id": self.ids[int(idx)], "retrieval_rank": rank, "retrieval_score": float(score)}
                for rank, (idx, score) in enumerate(zip(batch_indices, batch_scores), 1)
            ])
        return results


def retrieve_queries(retriever, rows, depth, batch_size):
    indexed = set(retriever.ids)
    missing = {row["target"] for row in rows} - indexed
    if missing:
        raise ValueError(f"Targets outside the indexed corpus: {sorted(missing)[:5]}")
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        results = retriever.search([r["query"] for r in batch], depth)
        for row, candidates in zip(batch, results):
            yield dict(row, candidates=candidates)
