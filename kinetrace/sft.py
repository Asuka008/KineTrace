"""Supervise the next answer token on train-side query/video pairs."""
import json
import random
import tempfile
from pathlib import Path

from .data import read_run, write_run
from .reranker import QwenScorer


def training_pairs(rows, strategy="hard", seed=42):
    """Use the same eligible positive rows in input order for both training arms."""
    if strategy not in ("hard", "random"):
        raise ValueError(f"Unknown negative strategy: {strategy}")
    rows = list(rows)
    positives = {}
    for row in rows:
        positives.setdefault(row["query"].strip().casefold(), set()).add(row["target"])
    # Pilot train targets enumerate the training gallery. Its video directory also contains validation clips.
    gallery = sorted({row["target"] for row in rows})
    negative_rng = random.Random(seed)
    pairs = []
    for row in rows:
        known = positives[row["query"].strip().casefold()]
        negative = next((c["video_id"] for c in sorted(row["candidates"], key=lambda c: c["retrieval_rank"])
                         if c["video_id"] not in known), None)
        if negative is None:
            continue
        if strategy == "random":
            if len(known) == len(gallery):
                raise ValueError(f"No eligible training video for random sampling: {row['query']}")
            negative = negative_rng.choice(gallery)
            while negative in known:
                negative = negative_rng.choice(gallery)
        pairs.append((row["query"], row["target"], negative))
    if not pairs:
        raise ValueError("Need train queries with at least one distinct CLIP negative")
    return pairs


def train_lora(args):
    import torch
    from peft import LoraConfig, get_peft_model

    if args.out.exists():
        raise FileExistsError(f"Choose a new adapter directory: {args.out}")
    pairs = training_pairs(read_run(args.input), args.negative_strategy, args.seed)
    # Keep the pre-shuffle pairs for auditing the sampled negatives after training.
    input_pairs = pairs.copy()
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    scorer = QwenScorer(args.qwen_model, args.device, args.videos, args.frames)
    scorer.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                        target_modules=r".*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)")
    scorer.model = get_peft_model(scorer.model, config).train()
    params = [p for p in scorer.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=0)
    use_cuda = args.device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda and not torch.cuda.is_bf16_supported())
    report = {"base": args.qwen_model, "input": str(args.input), "frames": args.frames,
              "negative_strategy": args.negative_strategy,
              "steps": args.steps, "seed": args.seed, "available_pairs": len(pairs),
              "trainable_parameters": sum(p.numel() for p in params), "device": args.device}
    if use_cuda:
        props = torch.cuda.get_device_properties(0)
        report.update(gpu=props.name, visible_gib=props.total_memory / 2**30)
        torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        offset = step % len(pairs)
        if offset == 0:
            rng.shuffle(pairs)
        query, positive, negative = pairs[offset]
        optimizer.zero_grad(set_to_none=True)
        pair_loss = 0.0
        for video, label in ((positive, 1), (negative, 0)):
            logits = scorer.next_token_logits(query, video)
            target = torch.tensor([scorer.answer_ids[label]], device=logits.device)
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target) / 2
            pair_loss += loss.item()
            scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True).item()
        if step == 0 and grad_norm == 0:
            raise RuntimeError("No gradient reached the LoRA parameters")
        scaler.step(optimizer)
        scaler.update()
        report.update(last_loss=pair_loss, last_grad_norm=grad_norm)
    if use_cuda:
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.out.parent) as tmp:
        stage = Path(tmp) / "adapter"
        scorer.model.save_pretrained(stage)
        write_run(stage / "training-pairs.jsonl", ({"query": query, "positive": pos, "negative": neg}
                                                  for query, pos, neg in input_pairs))
        (stage / "train-report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        stage.rename(args.out)
    return dict(report, adapter=str(args.out))
