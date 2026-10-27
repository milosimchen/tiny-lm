"""Plot the train/val loss curves from a run.

    python scripts/plot_loss.py checkpoints/run

Train loss well below val loss = the model is memorizing (overfitting):
add dropout, use more data, or make the model smaller.
"""

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt  # pip install matplotlib

run = Path(sys.argv[1] if len(sys.argv) > 1 else "checkpoints/run")
rows = list(csv.DictReader(open(run / "log.csv")))
its = [int(r["iter"]) for r in rows]
plt.plot(its, [float(r["train_loss"]) for r in rows], label="train")
plt.plot(its, [float(r["val_loss"]) for r in rows], label="val")
plt.xlabel("iteration"); plt.ylabel("cross-entropy loss"); plt.legend(); plt.title(run.name)
out = run / "loss.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"saved {out}")
