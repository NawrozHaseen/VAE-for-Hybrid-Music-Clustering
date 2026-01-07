from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

MASTER_MANIFEST = Path("data/fma_manifest_3k_6genres_lyrics_whisper.csv")

# If you want to use Whisper directory scanning as a fallback:
WHISPER_DIR = Path("data/whisper_transcriptions")

OUT_DIR = Path("data/lyrics_combined")
OUT_MANIFEST_ALL = Path("data/fma_manifest_combined.csv")
OUT_MANIFEST_TEXT_ONLY = Path("data/fma_manifest_combined_text_only.csv")

# "prefer_whisper" -> use whisper text if exists else genius
# "concat_both"    -> if both exist, whisper + genius
COMBINE_MODE = "concat_both"


def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""


def safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\-_\. ()]", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:160] if s else "unknown"


def build_whisper_map(folder: Path) -> dict[int, Path]:
    """
    Map track_id -> whisper transcript path by scanning whisper_transcriptions folder.
    Looks for a number (track_id) in the filename.
    """
    mapping: dict[int, Path] = {}
    if not folder.exists():
        return mapping

    for p in folder.rglob("*.txt"):
        m = re.search(r"\b(\d{3,7})\b", p.stem)
        if not m:
            continue
        tid = int(m.group(1))
        mapping.setdefault(tid, p)
    return mapping


def main():
    if not MASTER_MANIFEST.exists():
        raise FileNotFoundError(f"Missing master manifest: {MASTER_MANIFEST}")

    df = pd.read_csv(MASTER_MANIFEST)

    if "track_id" not in df.columns:
        raise ValueError("Manifest must contain 'track_id' column.")

    # Normalize columns
    for col in ["lyrics_path", "lyrics_source", "artist", "title"]:
        if col not in df.columns:
            df[col] = ""

    df["lyrics_path"] = df["lyrics_path"].fillna("").astype(str)
    df["lyrics_source"] = df["lyrics_source"].fillna("").astype(str).str.lower()
    df["artist"] = df["artist"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)

    whisper_map = build_whisper_map(WHISPER_DIR)
    print(f"Found whisper files in folder (map): {len(whisper_map)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Create output columns
    df["lyrics_path_genius"] = ""
    df["lyrics_path_whisper"] = ""
    df["text_path_combined"] = ""
    df["text_source_combined"] = ""

    have_any = 0
    have_both = 0
    have_whisper = 0
    have_genius = 0

    for i, row in df.iterrows():
        tid = int(row["track_id"])
        source = row["lyrics_source"]
        path_str = row["lyrics_path"]

        genius_path = None
        whisper_path = None

        #lyrics_path if it matches source
        if source == "genius" and path_str:
            genius_path = Path(path_str)
        elif source == "whisper" and path_str:
            whisper_path = Path(path_str)

        #if whisper not found from lyrics_path, try whisper dir map
        if whisper_path is None:
            whisper_path = whisper_map.get(tid)

        genius_ok = bool(genius_path) and genius_path.exists()
        whisper_ok = bool(whisper_path) and whisper_path.exists()

        if genius_ok:
            df.at[i, "lyrics_path_genius"] = str(genius_path)
            have_genius += 1
        if whisper_ok:
            df.at[i, "lyrics_path_whisper"] = str(whisper_path)
            have_whisper += 1

        if not (genius_ok or whisper_ok):
            continue

        have_any += 1
        if genius_ok and whisper_ok:
            have_both += 1

        genius_text = read_text(genius_path) if genius_ok else ""
        whisper_text = read_text(whisper_path) if whisper_ok else ""

        # Combine (simple)
        if COMBINE_MODE == "prefer_whisper":
            combined = whisper_text if whisper_text else genius_text
            combined_source = "whisper" if whisper_text else "genius"
        else:  # concat_both
            if whisper_text and genius_text:
                combined = whisper_text + "\n\n---\n\n" + genius_text
                combined_source = "both"
            elif whisper_text:
                combined = whisper_text
                combined_source = "whisper"
            else:
                combined = genius_text
                combined_source = "genius"

        # Save combined file
        artist = row["artist"].strip()
        title = row["title"].strip()
        fname = safe_filename(f"{artist} - {title} ({tid}).txt")
        out_path = OUT_DIR / fname
        out_path.write_text(combined, encoding="utf-8")

        df.at[i, "text_path_combined"] = str(out_path)
        df.at[i, "text_source_combined"] = combined_source

    # Save manifests
    OUT_MANIFEST_ALL.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_MANIFEST_ALL, index=False)

    df_text = df[df["text_path_combined"].astype(str).str.len() > 0].copy()
    df_text.to_csv(OUT_MANIFEST_TEXT_ONLY, index=False)

    print("\nCombined manifest created")
    print(f"Master rows:              {len(df)}")
    print(f"Tracks with ANY text:     {have_any}/{len(df)}")
    print(f"Tracks with Whisper text: {have_whisper}/{len(df)}")
    print(f"Tracks with Genius text:  {have_genius}/{len(df)}")
    print(f"Tracks with BOTH:         {have_both}/{len(df)}")
    print(f"Combined folder:          {OUT_DIR}")
    print(f"Wrote manifest (all):     {OUT_MANIFEST_ALL}")
    print(f"Wrote manifest (text):    {OUT_MANIFEST_TEXT_ONLY}")


if __name__ == "__main__":
    main()
