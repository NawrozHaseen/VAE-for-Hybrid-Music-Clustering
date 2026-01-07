from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

try:
    import umap.umap_ as umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


DATA_DIR = Path("data/hard")
RES_DIR = Path("results/hard")
PLOTS_DIR = RES_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def tagged_path(path: Path, tag: str) -> Path:
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def save_and_snapshot(path: Path, tag: str | None) -> None:
    if tag:
        shutil.copy2(path, tagged_path(path, tag))


def reduce_2d(Z: np.ndarray, seed: int = 42) -> np.ndarray:
    if HAS_UMAP:
        reducer = umap.UMAP(n_neighbors=20, min_dist=0.15, random_state=seed)
        return reducer.fit_transform(Z)
    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto", perplexity=30)
    return tsne.fit_transform(Z)


def scatter_plot(X2: np.ndarray, labels: np.ndarray, title: str, out_path: Path, tag: str | None):
    fig = plt.figure(figsize=(7, 6))
    plt.scatter(X2[:, 0], X2[:, 1], s=8, c=labels)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    save_and_snapshot(out_path, tag)


def stacked_bar(counts: pd.DataFrame, title: str, out_path: Path, tag: str | None):
    fig = plt.figure(figsize=(10, 5))
    counts_norm = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)
    counts_norm.plot(kind="bar", stacked=True, ax=plt.gca())
    plt.title(title)
    plt.xlabel("Cluster")
    plt.ylabel("Fraction")
    plt.legend(loc="best", fontsize=7)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    save_and_snapshot(out_path, tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default=None, help="Optional tag to snapshot outputs.")
    ap.add_argument("--latents_path", type=str, default=None,
                    help="Optional path to latents .npy (default: data/hard/latents_mu.npy)")
    args = ap.parse_args()

    lat_path = Path(args.latents_path) if args.latents_path else (DATA_DIR / "latents_mu.npy")
    Z = np.load(lat_path)

    y_genre = np.load(DATA_DIR / "genre_idx.npy")
    y_lang = np.load(DATA_DIR / "lang_idx.npy")
    genres = np.load(DATA_DIR / "genres.npy", allow_pickle=True)
    langs = np.load(DATA_DIR / "languages.npy", allow_pickle=True)

    pred_path = RES_DIR / "cluster_labels_kmeans.npy"
    if not pred_path.exists():
        raise FileNotFoundError("Run 20_cluster_and_evaluate_hard.py first to create cluster labels.")
    y_pred = np.load(pred_path)

    print("Reducing to 2D using", "UMAP" if HAS_UMAP else "t-SNE")
    Z2 = reduce_2d(Z, seed=args.seed)

    # canonical latent_2d.npy
    out_lat2d = PLOTS_DIR / "latent_2d.npy"
    np.save(out_lat2d, Z2.astype(np.float32))
    save_and_snapshot(out_lat2d, args.tag)

    scatter_plot(Z2, y_pred, "Latent space colored by KMeans cluster", PLOTS_DIR / "latent_by_cluster.png", args.tag)
    scatter_plot(Z2, y_genre, "Latent space colored by true genre", PLOTS_DIR / "latent_by_genre.png", args.tag)
    scatter_plot(Z2, y_lang, "Latent space colored by detected language", PLOTS_DIR / "latent_by_language.png", args.tag)

    df = pd.DataFrame({
        "cluster": y_pred,
        "genre": [str(genres[i]) if 0 <= i < len(genres) else str(i) for i in y_genre],
        "language": [str(langs[i]) if 0 <= i < len(langs) else str(i) for i in y_lang],
    })

    tab_genre = pd.crosstab(df["cluster"], df["genre"])
    tab_lang = pd.crosstab(df["cluster"], df["language"])

    out_genre_counts = RES_DIR / "cluster_distribution_genre_counts.csv"
    out_lang_counts = RES_DIR / "cluster_distribution_language_counts.csv"
    tab_genre.to_csv(out_genre_counts)
    tab_lang.to_csv(out_lang_counts)
    save_and_snapshot(out_genre_counts, args.tag)
    save_and_snapshot(out_lang_counts, args.tag)

    stacked_bar(tab_genre, "Cluster distribution over genres (fraction)", PLOTS_DIR / "cluster_dist_over_genres.png", args.tag)
    stacked_bar(tab_lang, "Cluster distribution over languages (fraction)", PLOTS_DIR / "cluster_dist_over_languages.png", args.tag)

    print("Saved plots to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
