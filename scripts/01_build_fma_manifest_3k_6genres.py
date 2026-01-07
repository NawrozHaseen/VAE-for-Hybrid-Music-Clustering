from __future__ import annotations

from pathlib import Path
import pandas as pd

META_DIR = Path("data/fma_metadata")
AUDIO_DIR = Path("data/fma_small")

# NEW: 6-genre manifest output (matches your new approach)
OUT_MANIFEST = Path("data/fma_manifest_3k_6genres.csv")

# NEW: production values for the actual Step 5 dataset build
TOTAL_TRACKS = 3000
N_GENRES = 6
SEED = 42


def find_file(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    return hits[0]


def find_audio_root() -> Path:
    # Usually: data/fma_small/fma_small/000/000002.mp3
    for candidate in [AUDIO_DIR / "fma_small", AUDIO_DIR]:
        if candidate.exists() and list(candidate.rglob("*.mp3")):
            return candidate
    raise FileNotFoundError("Could not locate extracted mp3 files under data/fma_small")


def build_audio_path(audio_root: Path, track_id: int) -> Path:
    tid_str = f"{track_id:06d}"
    return audio_root / tid_str[:3] / f"{tid_str}.mp3"


def main():
    if TOTAL_TRACKS % N_GENRES != 0:
        raise ValueError(
            f"TOTAL_TRACKS must be divisible by N_GENRES for a balanced dataset. "
            f"Got TOTAL_TRACKS={TOTAL_TRACKS}, N_GENRES={N_GENRES}."
        )

    tracks_csv = find_file(META_DIR, "tracks.csv")
    _ = find_file(META_DIR, "genres.csv")  # kept for parity with your original script
    audio_root = find_audio_root()

    print("Using:")
    print(" tracks.csv:", tracks_csv)
    print(" audio_root:", audio_root)
    print(f" Target: {TOTAL_TRACKS} tracks, {N_GENRES} genres, {TOTAL_TRACKS // N_GENRES} per genre")

    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])

    # Filter to FMA-small
    df = tracks[tracks[("set", "subset")] == "small"].copy()

    # Required fields
    df = df[[("track", "title"), ("artist", "name"), ("track", "genre_top")]].copy()
    df.columns = ["title", "artist", "genre_top"]
    df = df.dropna(subset=["title", "artist", "genre_top"])

    # Keep only sensible strings
    df = df[df["title"].apply(lambda x: isinstance(x, str))]
    df = df[df["artist"].apply(lambda x: isinstance(x, str))]
    df["genre_top"] = df["genre_top"].astype(str).str.strip()

    print(f"Eligible tracks (small subset) with title/artist/genre_top: {len(df)}")

    # Pick top N genres by frequency in FMA-small
    top_genres = df["genre_top"].value_counts().head(N_GENRES).index.tolist()
    print(f"Top {N_GENRES} genres selected:")
    for g in top_genres:
        print(" ", g)

    df = df[df["genre_top"].isin(top_genres)].copy()

    per_genre = TOTAL_TRACKS // N_GENRES
    rows = []

    # Robust balanced selection: iterate through shuffled genre pool, skip missing mp3s,
    # and keep going until we reach per_genre for each genre.
    for g in top_genres:
        gdf = df[df["genre_top"] == g].copy()

        # Shuffle deterministically
        gdf = gdf.sample(frac=1.0, random_state=SEED)

        picked = 0
        for track_id, r in gdf.iterrows():
            tid = int(track_id)
            audio_path = build_audio_path(audio_root, tid)
            if not audio_path.exists():
                continue

            rows.append(
                {
                    "track_id": tid,
                    "title": r["title"].strip(),
                    "artist": r["artist"].strip(),
                    "genre": g,
                    "audio_path": str(audio_path),
                    "lyrics_path": "",
                    "lyrics_source": "",
                }
            )
            picked += 1
            if picked >= per_genre:
                break

        if picked < per_genre:
            raise RuntimeError(
                f"Genre '{g}' could only supply {picked}/{per_genre} usable tracks with existing mp3 files. "
                f"Try reducing TOTAL_TRACKS, choosing different genres, or verify extraction paths."
            )

    out_df = pd.DataFrame(rows)

    # Shuffle final output deterministically (optional but nice)
    out_df = out_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    # Final validation
    if len(out_df) != TOTAL_TRACKS:
        raise RuntimeError(f"Built {len(out_df)} rows, expected exactly {TOTAL_TRACKS}.")

    counts = out_df["genre"].value_counts()
    if not all(counts == per_genre):
        raise RuntimeError(
            "Output is not perfectly balanced. Counts:\n" + counts.to_string()
        )

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_MANIFEST, index=False)

    print("\nWrote manifest:", OUT_MANIFEST)
    print("Total tracks:", len(out_df))
    print("Tracks per genre:")
    print(out_df["genre"].value_counts())


if __name__ == "__main__":
    main()
