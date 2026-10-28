"""Download Tiny Shakespeare (~1 MB, all of Shakespeare's plays in one file).

    python scripts/download_data.py

Want to train on something else? Any UTF-8 .txt file works:
    python train.py --data path/to/your.txt
"""

import shutil
import subprocess
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DEST = Path(__file__).resolve().parent.parent / "data" / "shakespeare.txt"


def download(url: str, dest: Path) -> None:
    try:
        dest.write_bytes(urlopen(url).read())
    except URLError as e:
        # python.org installs on macOS ship without CA certificates until you run
        # "Install Certificates.command". curl uses the system's certificates.
        if "CERTIFICATE_VERIFY_FAILED" not in str(e) or not shutil.which("curl"):
            raise
        print("Python couldn't verify the certificate; retrying with curl")
        subprocess.run(["curl", "-fsSL", "-o", str(dest), url], check=True)


if __name__ == "__main__":
    DEST.parent.mkdir(exist_ok=True)
    if DEST.exists():
        print(f"already have {DEST}")
        sys.exit(0)
    print(f"downloading {URL}")
    download(URL, DEST)
    print(f"saved {DEST.stat().st_size:,} bytes to {DEST}")
