from __future__ import annotations

import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

FMA_SMALL_URL = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
FMA_METADATA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"

RAW_DIR = Path("data/raw")
AUDIO_DIR = Path("data/fma_small")
META_DIR = Path("data/fma_metadata")

CHUNK_SIZE = 1024 * 1024  # 1MB


def download(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"✓ Already downloaded: {out_path}")
        return

    print(f"↓ Downloading: {out_path.name}")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))

        with open(out_path, "wb") as f, tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=out_path.name,
        ) as pbar:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    print(f"✓ Downloaded: {out_path}")


def extract(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"→ Extracting {zip_path.name} -> {out_dir}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)
    print(f"✓ Extracted: {out_dir}")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    audio_zip = RAW_DIR / "fma_small.zip"
    meta_zip = RAW_DIR / "fma_metadata.zip"

    download(FMA_SMALL_URL, audio_zip)
    download(FMA_METADATA_URL, meta_zip)

    extract(audio_zip, AUDIO_DIR)
    extract(meta_zip, META_DIR)

    # Quick verification
    tracks = list(META_DIR.rglob("tracks.csv"))
    genres = list(META_DIR.rglob("genres.csv"))
    mp3s = list(AUDIO_DIR.rglob("*.mp3"))

    print("\nVerification")
    print("tracks.csv found:", tracks[0] if tracks else "NOT FOUND")
    print("genres.csv found:", genres[0] if genres else "NOT FOUND")
    print("mp3 count:", len(mp3s))


if __name__ == "__main__":
    main()
