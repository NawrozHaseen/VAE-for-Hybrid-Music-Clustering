# scripts/02_fetch_lyrics.py
"""
Fetch lyrics for the 3k/5-genre FMA subset using:
1) Genius (lyricsgenius) if GENIUS_ACCESS_TOKEN is set in .env
2) LRCLIB fallback (lrclibapi package provides module 'lrclib' with class LrcLibAPI)

Input:
  data/fma_manifest_3k_5genres.csv

Outputs:
  data/lyrics/                             (saved .txt lyrics files)
  data/fma_manifest_3k_5genres_lyrics.csv  (updated manifest with lyrics_path + lyrics_source)

Install:
  pip install python-dotenv lyricsgenius lrclibapi pandas tqdm requests

.env (project root):
  GENIUS_ACCESS_TOKEN=your_token_here

Run:
  python scripts\\02_fetch_lyrics.py
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

import lyricsgenius
from lrclib import LrcLibAPI  # <-- correct class name per lrclibapi docs :contentReference[oaicite:1]{index=1}

MANIFEST_IN = Path("data/fma_manifest_3k_6genres_test.csv")
MANIFEST_OUT = Path("data/fma_manifest_3k_6genres_lyrics_test.csv")

LYRICS_DIR = Path("data/lyrics")

SLEEP_SECONDS = 0.45          # increase if you get rate-limited
MIN_CHARS = 80                # ignore very short/garbage results
OVERWRITE_EXISTING = False    # True = re-fetch even if lyrics_path exists

# Optional: limit rows for testing (e.g., 50). Keep None for full run.
MAX_TO_PROCESS = 1000

# User agent is recommended by LRCLIB docs :contentReference[oaicite:2]{index=2}
LRCLIB_USER_AGENT = "fma-lyrics-fetcher/1.0"


def safe_filename(s: str) -> str:
    s = re.sub(r"[^\w\-_\. ]", "_", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return (s[:150] if s else "unknown") + ".txt"


def normalize_query(s: str) -> str:
    """
    Remove common noisy parts in titles/artists to improve hit-rate.
    """
    s = str(s).strip()
    s = re.sub(r"\s*\(.*?\)\s*", " ", s)  # remove (...) e.g. (Live)
    s = re.sub(r"\s*\[.*?\]\s*", " ", s)  # remove [...]
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fetch_from_genius(genius: lyricsgenius.Genius, artist: str, title: str) -> str | None:
    song = genius.search_song(title=title, artist=artist)
    if not song or not song.lyrics:
        return None
    text = song.lyrics.strip()
    if len(text) < MIN_CHARS:
        return None
    return text


def _extract_plain_lyrics(res: dict) -> str | None:
    """
    LRCLIB responses typically contain plainLyrics and/or syncedLyrics.
    We'll prefer plainLyrics for simplicity.
    """
    if not isinstance(res, dict):
        return None
    text = (res.get("plainLyrics") or "").strip()
    if len(text) >= MIN_CHARS:
        return text
    # fallback to syncedLyrics (strip timestamps lightly)
    synced = (res.get("syncedLyrics") or "").strip()
    if len(synced) >= MIN_CHARS:
        # remove [mm:ss.xx] timestamps if present
        synced = re.sub(r"\[\d+:\d+(?:\.\d+)?\]\s*", "", synced).strip()
        if len(synced) >= MIN_CHARS:
            return synced
    return None


def fetch_from_lrclib(api: LrcLibAPI, artist: str, title: str) -> str | None:
    """
    1 get_lyrics directly (track_name, artist_name) directly (docs example). :contentReference[oaicite:3]{index=3}
    2 if not found, try search_lyrics then get_lyrics_by_id.
    """
    # 1get
    try:
        res = api.get_lyrics(track_name=title, artist_name=artist)
        text = _extract_plain_lyrics(res) if res else None
        if text:
            return text
    except Exception:
        pass

    # 2 search then get by id
    try:
        results = api.search_lyrics(track_name=title, artist_name=artist)
        if not results:
            return None

        # results can be a list of dicts; pick the first plausible one
        first = results[0]
        if isinstance(first, dict):
            # different shapes exist; commonly "id"
            lyr_id = first.get("id")
            if lyr_id:
                res2 = api.get_lyrics_by_id(lyr_id)
                text2 = _extract_plain_lyrics(res2) if res2 else None
                if text2:
                    return text2

        #if search already returns lyrics fields directly
        text3 = _extract_plain_lyrics(first) if isinstance(first, dict) else None
        return text3
    except Exception:
        return None


def main() -> None:
    if not MANIFEST_IN.exists():
        raise FileNotFoundError(f"Missing {MANIFEST_IN}. Run 01_build_fma_manifest_3k_5genres.py first.")

    load_dotenv()
    genius_token = os.getenv("GENIUS_ACCESS_TOKEN", "").strip()

    genius = None
    if genius_token:
        genius = lyricsgenius.Genius(
            genius_token,
            timeout=20,
            retries=2,
            remove_section_headers=True,
            skip_non_songs=True,
            excluded_terms=["(Remix)", "(Live)"],
        )
        genius.verbose = False
    else:
        print("⚠ GENIUS_ACCESS_TOKEN not found. Will use LRCLIB only (lower hit rate).")

    lrclib_api = LrcLibAPI(user_agent=LRCLIB_USER_AGENT)

    df = pd.read_csv(MANIFEST_IN)

    # Ensure these columns exist
    if "lyrics_path" not in df.columns:
        df["lyrics_path"] = ""
    if "lyrics_source" not in df.columns:
        df["lyrics_source"] = ""

    LYRICS_DIR.mkdir(parents=True, exist_ok=True)

    df_iter = df.head(MAX_TO_PROCESS) if isinstance(MAX_TO_PROCESS, int) and MAX_TO_PROCESS > 0 else df

    found_this_run = 0

    for idx, row in tqdm(df_iter.iterrows(), total=len(df_iter), desc="Fetching lyrics"):
        track_id = int(row["track_id"])
        artist = normalize_query(row.get("artist", ""))
        title = normalize_query(row.get("title", ""))

        # Skip if already fetched and file exists
        existing_path = str(row.get("lyrics_path", "")).strip()
        if existing_path and not OVERWRITE_EXISTING and Path(existing_path).exists():
            continue

        lyrics_text = None
        source = ""

        # 1) Genius
        if genius is not None:
            try:
                lyrics_text = fetch_from_genius(genius, artist, title)
                if lyrics_text:
                    source = "genius"
            except Exception:
                lyrics_text = None

        # 2) LRCLIB fallback
        if not lyrics_text:
            lyrics_text = fetch_from_lrclib(lrclib_api, artist, title)
            if lyrics_text:
                source = "lrclib"

        if lyrics_text:
            fname = safe_filename(f"{artist} - {title} ({track_id})")
            out_path = LYRICS_DIR / fname
            out_path.write_text(lyrics_text, encoding="utf-8")

            df.at[idx, "lyrics_path"] = str(out_path)
            df.at[idx, "lyrics_source"] = source
            found_this_run += 1

        time.sleep(SLEEP_SECONDS)

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFEST_OUT, index=False)

    total_with_lyrics = int((df["lyrics_path"].astype(str).str.len() > 0).sum())

    print("\nLyrics fetching finished")
    print(f"Attempted rows this run: {len(df_iter)}")
    print(f"Lyrics found this run:   {found_this_run}")
    print(f"Total tracks with lyrics in manifest now: {total_with_lyrics}/{len(df)}")
    print(f"Updated manifest: {MANIFEST_OUT}")
    print(f"Lyrics folder:    {LYRICS_DIR}")


if __name__ == "__main__":
    main()
