"""Score a sampled video with the binary head of the Qwen3-VL reranker."""
from .video import read_clip

QWEN_MODEL = "Qwen/Qwen3-VL-Reranker-2B"


class QwenScorer:
    def __init__(self, model_path, device, video_dir, frames):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch = torch
        self.device = device
        self.video_dir = video_dir
        self.frames = frames
        dtype = torch.float32
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, dtype=dtype, attn_implementation="sdpa"
        ).eval().to(device)
        vocab = self.processor.tokenizer.get_vocab()
        self.answer_ids = [vocab["no"], vocab["yes"]]

    def __call__(self, query, video_id):
        import numpy as np

        frames, metadata = read_clip(self.video_dir / f"{video_id}.mp4", self.frames)
        messages = [
            {"role": "system", "content": [{"type": "text", "text":
                "Judge whether the Document satisfies the Query and Instruct. Answer yes or no."}]},
            {"role": "user", "content": [
                {"type": "text", "text": "<Instruct>: Match the scene, actions and their order."},
                {"type": "text", "text": "<Query>: " + query},
                {"type": "text", "text": "<Document>:"},
                {"type": "video"},
            ]},
        ]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[prompt], videos=[np.stack(frames)], video_metadata=[metadata],
            do_sample_frames=False, return_tensors="pt"
        ).to(self.device)
        if "pixel_values_videos" not in inputs:
            raise ValueError(f"Missing visual input for {video_id}")
        with self.torch.inference_mode():
            outputs = self.model(**inputs, logits_to_keep=1, use_cache=False)
            logits = outputs.logits[0, -1, self.answer_ids].float()
            return logits.softmax(dim=0)[1].item()
