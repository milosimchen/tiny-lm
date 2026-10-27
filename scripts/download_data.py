"""Download Tiny Shakespeare (~1 MB, all of Shakespeare's plays in one file).

    python scripts/download_data.py

Want to train on something else? Any UTF-8 .txt file works:
    python train.py --data path/to/your.txt
"""

from pathlib import Path
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEST = Path(__file__).resolve().parent.parent / "data" / "shakespeare.txt"

if __name__ == "__main__":
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists():
        print(f"already have {DEST}")
    else:
        print(f"downloading {URL}")
        DEST.write_bytes(urlopen(URL).read())
        print(f"saved {DEST.stat().st_size:,} bytes to {DEST}")
