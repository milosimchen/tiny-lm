"""Train a tiny GPT from scratch.

Examples:
    python train.py                         # char-level Shakespeare, small preset
    python train.py --preset tiny --max-iters 300   # 1-minute smoke test on a laptop CPU
    python train.py --tokenizer bpe --vocab-size 1024
    python train.py --data data/my_corpus.txt --out checkpoints/mine
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from tinylm.data import get_batch, load_text, train_val_split
from tinylm.model import GPT, GPTConfig
from tinylm.tokenizer import BPETokenizer, CharTokenizer, save_tokenizer

PRESETS = {
    # ~0.1M params: runs on any laptop CPU in about a minute. For checking things work.
    "tiny":  dict(n_layer=2, n_head=2, n_embd=64,  block_size=64,  batch_size=16, max_iters=500,  lr=1e-3),
    # ~0.8M params: a few minutes on a laptop GPU (or Apple M-series); readable Shakespeare-ish text.
    "small": dict(n_layer=4, n_head=4, n_embd=128, block_size=128, batch_size=32, max_iters=3000, lr=1e-3),
    # ~10M params: use a real GPU (e.g. free Colab T4). Noticeably better text.
    "base":  dict(n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=64, max_iters=5000, lr=3e-4),
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    return "cpu"


def lr_at(it: int, max_iters: int, base_lr: float, warmup: int) -> float:
    """Linear warmup, then cosine decay to 10% of base_lr."""
    if it < warmup:
        return base_lr * (it + 1) / warmup
    progress = (it - warmup) / max(1, max_iters - warmup)
    return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def estimate_loss(model, splits, args, rng, device) -> dict[str, float]:
    """Average loss over several batches; a single batch is too noisy to trust."""
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = []
        for _ in range(args.eval_iters):
            x, y = get_batch(data, args.block_size, args.batch_size, rng)
            _, loss = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
            losses.append(loss.item())
        out[name] = float(np.mean(losses))
    model.train()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/shakespeare.txt")
    ap.add_argument("--out", default="checkpoints/run")
    ap.add_argument("--preset", choices=PRESETS, default="small")
    ap.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    ap.add_argument("--vocab-size", type=int, default=512, help="BPE only")
    ap.add_argument("--n-layer", type=int); ap.add_argument("--n-head", type=int)
    ap.add_argument("--n-embd", type=int); ap.add_argument("--block-size", type=int)
    ap.add_argument("--batch-size", type=int); ap.add_argument("--max-iters", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=50)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (auto-detected by default)")
    args = ap.parse_args()

    # Preset values fill in anything not given explicitly on the command line.
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = args.device or pick_device()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Data and tokenizer
    text = load_text(args.data)
    print(f"loaded {len(text):,} characters from {args.data}")
    t0 = time.time()
    tok = CharTokenizer.train(text) if args.tokenizer == "char" else BPETokenizer.train(text, args.vocab_size)
    ids = tok.encode(text)
    print(f"{args.tokenizer} tokenizer: vocab {tok.vocab_size}, {len(ids):,} tokens ({time.time() - t0:.1f}s)")
    save_tokenizer(tok, out / "tokenizer.json")
    train_data, val_data = train_val_split(ids)
    splits = {"train": train_data, "val": val_data}

    # 2. Model
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=args.block_size, n_layer=args.n_layer,
                    n_head=args.n_head, n_embd=args.n_embd, dropout=args.dropout)
    model = GPT(cfg).to(device)
    print(f"model: {model.num_params() / 1e6:.2f}M parameters on {device}")

    # Weight decay on matrices only, not on biases or LayerNorm parameters.
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

    # 3. Training loop
    log_path = out / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["iter", "train_loss", "val_loss", "lr"])
    best_val = float("inf")
    t0 = time.time()
    for it in range(args.max_iters + 1):
        lr = lr_at(it, args.max_iters, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr

        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, splits, args, rng, device)
            print(f"iter {it:5d} | train {losses['train']:.3f} | val {losses['val']:.3f} "
                  f"| lr {lr:.2e} | {time.time() - t0:.0f}s")
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([it, losses["train"], losses["val"], lr])
            if losses["val"] < best_val:
                best_val = losses["val"]
                torch.save({"model": model.state_dict(), "config": asdict(cfg), "iter": it,
                            "val_loss": best_val}, out / "model.pt")
        if it == args.max_iters:
            break

        x, y = get_batch(train_data, args.block_size, args.batch_size, rng)
        _, loss = model(torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # guards against loss spikes
        opt.step()

    (out / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"done. best val loss {best_val:.3f}. checkpoint: {out / 'model.pt'}")
    print(f"try it:  python sample.py --ckpt {out} --prompt \"ROMEO:\"")


if __name__ == "__main__":
    main()
