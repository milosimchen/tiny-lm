"""Generate text from a trained checkpoint.

    python sample.py --ckpt checkpoints/run --prompt "ROMEO:" --tokens 500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tinylm.model import GPT, GPTConfig
from tinylm.tokenizer import load_tokenizer
from train import pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/run", help="folder written by train.py")
    ap.add_argument("--prompt", default="\n")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8,
                    help="<1 = safer and more repetitive, >1 = more random")
    ap.add_argument("--top-k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
    folder = Path(args.ckpt)
    device = pick_device()
    ckpt = torch.load(folder / "model.pt", map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    tok = load_tokenizer(folder / "tokenizer.json")

    try:
        ids = tok.encode(args.prompt)
    except KeyError as e:  # char tokenizer only knows characters seen in training
        raise SystemExit(f"prompt contains a character the tokenizer never saw: {e}")
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = model.generate(x, args.tokens, temperature=args.temperature, top_k=args.top_k)
    print(tok.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
