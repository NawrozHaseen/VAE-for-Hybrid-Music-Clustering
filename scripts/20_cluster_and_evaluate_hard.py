from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score


DATA_DIR = Path("data/hard")
RES_DIR = Path("results/hard")
RES_DIR.mkdir(parents=True, exist_ok=True)


def tagged_path(path: Path, tag: str) -> Path:
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def save_and_snapshot(path: Path, tag: str | None) -> None:
    if tag:
        shutil.copy2(path, tagged_path(path, tag))


def cluster_purity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    df = pd.DataFrame({"true": y_true, "pred": y_pred})
    total = len(df)
    if total == 0:
        return 0.0
    purity_sum = 0
    for _, g in df.groupby("pred"):
        purity_sum += g["true"].value_counts().max()
    return float(purity_sum) / float(total)


def safe_silhouette(X: np.ndarray, y_pred: np.ndarray) -> float:
    uniq = np.unique(y_pred)
    if len(uniq) < 2:
        return float("nan")
    try:
        return float(silhouette_score(X, y_pred))
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=None, help="Number of clusters (default: #genres)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default=None, help="Optional tag to snapshot outputs.")
    ap.add_argument("--latents_path", type=str, default=None,
                    help="Optional path to latents .npy (default: data/hard/latents_mu.npy)")
    args = ap.parse_args()

    lat_path = Path(args.latents_path) if args.latents_path else (DATA_DIR / "latents_mu.npy")
    Z = np.load(lat_path)

    y_genre = np.load(DATA_DIR / "genre_idx.npy")
    genres = np.load(DATA_DIR / "genres.npy", allow_pickle=True)

    k = args.k if args.k is not None else int(y_genre.max() + 1)
    print("Clustering with K =", k)

    km = KMeans(n_clusters=k, random_state=args.seed, n_init=20)
    y_pred = km.fit_predict(Z)

    metrics = {
        "feature_space": str(lat_path),
        "k": int(k),
        "silhouette": safe_silhouette(Z, y_pred),
        "nmi": float(normalized_mutual_info_score(y_genre, y_pred)),
        "ari": float(adjusted_rand_score(y_genre, y_pred)),
        "purity": float(cluster_purity(y_genre, y_pred)),
    }

    out_metrics = RES_DIR / "hard_metrics_vae_latents.json"
    out_metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Saved:", out_metrics)
    save_and_snapshot(out_metrics, args.tag)

    # composition table: cluster -> genre counts
    df = pd.DataFrame({"true": y_genre, "pred": y_pred})
    tab = pd.crosstab(df["pred"], df["true"])

    # best-effort rename columns with genre strings
    new_cols = {}
    for c in tab.columns:
        ci = int(c)
        if 0 <= ci < len(genres):
            new_cols[c] = str(genres[ci])
    tab = tab.rename(columns=new_cols)

    out_tab = RES_DIR / "cluster_composition_by_genre.csv"
    tab.to_csv(out_tab)
    print("Saved:", out_tab)
    save_and_snapshot(out_tab, args.tag)

    out_labels = RES_DIR / "cluster_labels_kmeans.npy"
    np.save(out_labels, y_pred.astype(np.int64))
    print("Saved:", out_labels)
    save_and_snapshot(out_labels, args.tag)


if __name__ == "__main__":
    main()
