from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path


DATA_URL = "https://s3-us-west-2.amazonaws.com/telemanom/data.zip"
LABELS_URL = "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv"


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the public SMAP/MSL Telemanom dataset.")
    parser.add_argument("--output-dir", type=str, default="data/telemanom")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "data.zip"
    download(DATA_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    extracted = out_dir / "data"
    if extracted.exists():
        for child in extracted.iterdir():
            target = out_dir / child.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(child), str(target))
        shutil.rmtree(extracted)
    labels_path = out_dir / "labeled_anomalies.csv"
    download(LABELS_URL, labels_path)
    zip_path.unlink(missing_ok=True)
    print(f"Done. Dataset ready at: {out_dir}")


if __name__ == "__main__":
    main()
