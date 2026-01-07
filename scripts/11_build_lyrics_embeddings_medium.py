from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

from sentence_transformers import SentenceTransformer


def read_lyrics_row(r: pd.Series) -> str:
    """
    Priority:
      1) lyrics column if present and non-empty
      2) lyrics_path file if present and exists
    """
    if "lyrics" in r and isinstance(r["lyrics"], str):
        txt = r["lyrics"].strip()
        if txt:
            return txt

    if "lyrics_path" in r and isinstance(r["lyrics_path"], str) and r["lyrics_path"].strip():
        p = Path(r["lyrics_path"])
        if p.exists():
            try:
                return p.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                return ""

    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="data/fma_manifest_combined_text_only_clean.csv")
    ap.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out_emb", type=str, default="data/lyrics_embeddings.npy")
    ap.add_argument("--out_ids", type=str, default="data/lyrics_track_ids.npy")
    ap.add_argument("--report_csv", type=str, default="results/lyrics_embedding_report.csv")

    ap.add_argument("--max_items", type=int, default=0, help="0 = all rows")
    ap.add_argument("--min_chars", type=int, default=30, help="skip lyrics shorter than this")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    df = pd.read_csv(manifest_path)

    if "track_id" not in df.columns:
        raise ValueError("Manifest must include 'track_id' column.")

    if args.max_items and args.max_items > 0:
        df = df.head(args.max_items).copy()

    texts: list[str] = []
    ids: list[int] = []
    report_rows = []

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Reading lyrics"):
        tid = int(r["track_id"])
        txt = read_lyrics_row(r)

        row_info = {"track_id": tid, "status": "", "reason": "", "n_chars": len(txt)}

        if len(txt) < args.min_chars:
            row_info["status"] = "skipped"
            row_info["reason"] = "too_short_or_missing"
            report_rows.append(row_info)
            continue

        texts.append(txt)
        ids.append(tid)
        row_info["status"] = "ok"
        row_info["reason"] = ""
        report_rows.append(row_info)

    if not texts:
        raise RuntimeError(
            "No lyrics found to embed. Ensure the manifest has a usable 'lyrics' column "
            "or valid 'lyrics_path' files."
        )

    print(f"\nEmbedding {len(texts)} lyric texts with model: {args.model}")
    model = SentenceTransformer(args.model)

    emb = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # good for cosine similarity / fusion
    ).astype(np.float32)

    ids_arr = np.array(ids, dtype=np.int64)

    Path(args.out_emb).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_ids).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_csv).parent.mkdir(parents=True, exist_ok=True)

    np.save(args.out_emb, emb)
    np.save(args.out_ids, ids_arr)

    pd.DataFrame(report_rows).to_csv(args.report_csv, index=False)

    print("\nDONE")
    print(f"Manifest: {manifest_path}")
    print(f"Embedded: {emb.shape[0]}  (dim={emb.shape[1]})")
    print(f"Saved:    {args.out_emb}  shape={emb.shape} dtype={emb.dtype}")
    print(f"Saved:    {args.out_ids} shape={ids_arr.shape} dtype={ids_arr.dtype}")
    print(f"Report:   {args.report_csv}")


if __name__ == "__main__":
    main()
