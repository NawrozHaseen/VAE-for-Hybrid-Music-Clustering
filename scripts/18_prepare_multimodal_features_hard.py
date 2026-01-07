from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import librosa

#text embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    HAS_ST = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_TFIDF = True
except Exception:
    HAS_TFIDF = False


#config
DEFAULT_MANIFEST_CANDIDATES = [
    "data/fma_manifest_combined_text_only.csv",
    "data/fma_manifest_combined.csv",
    "data/fma_manifest_3k_5genres_lyrics.csv",
    "data/fma_manifest_5k_5genres_lyrics_whisper.csv",
]

OUT_DIR = Path("data/hard")
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def detect_language_simple(text: str) -> str:
    """
    Very lightweight language heuristic:
    - If Bengali Unicode block present => 'bn'
    - Else if has letters => 'en_or_other'
    - Else => 'none'
    """
    if not isinstance(text, str) or not text.strip():
        return "none"
    if re.search(r"[\u0980-\u09FF]", text):
        return "bn"
    if re.search(r"[A-Za-z]", text):
        return "en"
    return "other"


def safe_read_text(path: Optional[str]) -> str:
    if not path or not isinstance(path, str):
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return p.read_text(errors="ignore")
        except Exception:
            return ""


def extract_mfcc_stats(
    audio_path: str,
    sr: int = 22050,
    duration: float = 20.0,
    n_mfcc: int = 40,
    hop_length: int = 512,
) -> Optional[np.ndarray]:
    """
    Returns MFCC stats vector: [mean(mfcc_1..n), std(mfcc_1..n)] => shape (2*n_mfcc,)
    """
    p = Path(audio_path)
    if not p.exists():
        return None
    try:
        y, sr_ = librosa.load(p.as_posix(), sr=sr, mono=True, duration=duration)
        if y is None or len(y) < sr:  # < 1s
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr_, n_mfcc=n_mfcc, hop_length=hop_length)
        mu = mfcc.mean(axis=1)
        sd = mfcc.std(axis=1)
        feat = np.concatenate([mu, sd], axis=0).astype(np.float32)
        return feat
    except Exception:
        return None


def pick_manifest(user_path: Optional[str]) -> Path:
    if user_path:
        p = Path(user_path)
        if not p.exists():
            raise FileNotFoundError(f"Manifest not found: {p}")
        return p
    for c in DEFAULT_MANIFEST_CANDIDATES:
        p = Path(c)
        if p.exists():
            return p
    raise FileNotFoundError(
        "No default manifest found. Please pass --manifest path/to/manifest.csv"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=None, help="Path to manifest CSV")
    ap.add_argument("--max_tracks", type=int, default=None, help="Limit number of tracks (debug)")
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--n_mfcc", type=int, default=40)
    ap.add_argument("--text_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--force", action="store_true", help="Recompute and overwrite cached outputs")
    args = ap.parse_args()

    manifest_path = pick_manifest(args.manifest)
    print("Using manifest:", manifest_path)

    df = pd.read_csv(manifest_path)

    # Expected columns:
    # - track_id
    # - audio_path
    # - genre OR genre_top
    # - lyrics_path (optional)
    # - lyrics (optional)
    # - language (optional)
    if "audio_path" not in df.columns:
        raise ValueError("Manifest must contain 'audio_path' column.")

    if "track_id" not in df.columns:
        # If missing, create stable ids
        df["track_id"] = np.arange(len(df), dtype=int)

    if "genre" not in df.columns and "genre_top" in df.columns:
        df["genre"] = df["genre_top"]
    if "genre" not in df.columns:
        df["genre"] = "unknown"

    # If lyrics_path missing, try to proceed with empty lyrics
    if "lyrics_path" not in df.columns:
        df["lyrics_path"] = ""

    if args.max_tracks is not None:
        df = df.head(args.max_tracks).copy()

    # Output files
    out_audio = OUT_DIR / "audio_mfcc_stats.npy"
    out_text = OUT_DIR / "lyrics_emb.npy"
    out_ids = OUT_DIR / "track_ids.npy"
    out_genres = OUT_DIR / "genres.npy"
    out_genre_idx = OUT_DIR / "genre_idx.npy"
    out_langs = OUT_DIR / "languages.npy"
    out_lang_idx = OUT_DIR / "lang_idx.npy"
    out_meta = OUT_DIR / "hard_metadata.csv"
    out_info = OUT_DIR / "build_info.json"

    if (not args.force) and all(p.exists() for p in [out_audio, out_text, out_ids, out_genre_idx, out_lang_idx]):
        print("Outputs already exist. Re-run with --force to rebuild.")
        return

    #extract mfcc
    audio_feats: List[np.ndarray] = []
    kept_rows = []

    print("Extracting audio MFCC stats...")
    for _, r in tqdm(df.iterrows(), total=len(df)):
        feat = extract_mfcc_stats(
            r["audio_path"],
            sr=args.sr,
            duration=args.duration,
            n_mfcc=args.n_mfcc,
        )
        if feat is None:
            continue
        audio_feats.append(feat)
        kept_rows.append(r)

    if not kept_rows:
        raise RuntimeError("No usable audio files were processed. Check your audio_path values.")

    kept = pd.DataFrame(kept_rows).reset_index(drop=True)
    X_audio = np.stack(audio_feats, axis=0)  # (N, 2*n_mfcc)

    #lyrics text
    texts = []
    languages = []
    for _, r in kept.iterrows():
        txt = ""
        if "lyrics" in kept.columns and isinstance(r.get("lyrics"), str) and r["lyrics"].strip():
            txt = r["lyrics"]
        else:
            txt = safe_read_text(r.get("lyrics_path", ""))
        texts.append(txt)
        languages.append(detect_language_simple(txt))

    kept["language_detected"] = languages

    #text embeddings (cached)
    print("Building lyrics embeddings...")
    if HAS_ST:
        model = SentenceTransformer(args.text_model)
        # sentence-transformers handles empty strings; but we keep them to preserve alignment
        X_text = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True).astype(np.float32)
    else:
        if not HAS_TFIDF:
            raise RuntimeError(
                "Neither sentence-transformers nor scikit-learn TF-IDF is available. "
                "Install sentence-transformers or scikit-learn."
            )
        # TF-IDF fallback (dim depends on data; keep capped)
        vect = TfidfVectorizer(max_features=2000, stop_words="english")
        X_text = vect.fit_transform([t if t.strip() else " " for t in texts]).toarray().astype(np.float32)

    #genre/lang indexing
    genres = kept["genre"].astype(str).fillna("unknown").tolist()
    uniq_genres = sorted(set(genres))
    genre_to_i = {g: i for i, g in enumerate(uniq_genres)}
    y_genre = np.array([genre_to_i[g] for g in genres], dtype=np.int64)

    uniq_langs = sorted(set(languages))
    lang_to_i = {l: i for i, l in enumerate(uniq_langs)}
    y_lang = np.array([lang_to_i[l] for l in languages], dtype=np.int64)

    np.save(out_audio, X_audio)
    np.save(out_text, X_text)
    np.save(out_ids, kept["track_id"].to_numpy(dtype=np.int64))
    np.save(out_genres, np.array(genres, dtype=object))
    np.save(out_genre_idx, y_genre)
    np.save(out_langs, np.array(languages, dtype=object))
    np.save(out_lang_idx, y_lang)

    kept.to_csv(out_meta, index=False)

    info = {
        "manifest_used": str(manifest_path),
        "num_tracks_input": int(len(df)),
        "num_tracks_kept": int(len(kept)),
        "audio_feature_shape": list(X_audio.shape),
        "text_feature_shape": list(X_text.shape),
        "unique_genres": uniq_genres,
        "unique_languages": uniq_langs,
        "text_embedding_backend": "sentence-transformers" if HAS_ST else "tfidf",
        "text_model": args.text_model if HAS_ST else "tfidf(max_features=2000)",
    }
    out_info.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print("\nDone. Wrote:")
    print(" ", out_audio)
    print(" ", out_text)
    print(" ", out_meta)
    print(" ", out_info)


if __name__ == "__main__":
    main()
