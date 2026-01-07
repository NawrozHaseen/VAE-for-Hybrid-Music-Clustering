from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Optional UMAP
try:
    import umap.umap_ as umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

# Optional t-SNE
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except Exception:
    HAS_TSNE = False


def load_label_map(manifest_csv: str):
    df = pd.read_csv(manifest_csv)
    if "track_id" not in df.columns:
        return None, None

    label_col = None
    if "genre" in df.columns:
        label_col = "genre"
    elif "genre_top" in df.columns:
        label_col = "genre_top"

    if label_col is None:
        return None, None

    mp = {}
    for tid, lab in zip(df["track_id"].values, df[label_col].values):
        if pd.isna(tid) or pd.isna(lab):
            continue
        mp[int(tid)] = str(lab)
    return mp, label_col


def labels_for_ids(label_map, ids: np.ndarray):
    if label_map is None:
        return None
    y = []
    ok = 0
    for tid in ids:
        lab = label_map.get(int(tid), None)
        if lab is None:
            y.append("__MISSING__")
        else:
            y.append(lab)
            ok += 1
    if ok == 0:
        return None
    return np.array(y, dtype=object)


def run_clustering(X: np.ndarray, method: str, n_clusters: int, eps: float, min_samples: int):
    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        yhat = model.fit_predict(X)
        return yhat
    if method == "agglomerative":
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        yhat = model.fit_predict(X)
        return yhat
    if method == "dbscan":
        model = DBSCAN(eps=eps, min_samples=min_samples)
        yhat = model.fit_predict(X)
        return yhat
    raise ValueError("method must be one of: kmeans, agglomerative, dbscan")


def project_pca(X: np.ndarray, seed: int = 42):
    pca = PCA(n_components=2, random_state=seed)
    return pca.fit_transform(X)


def project_umap(X: np.ndarray, seed: int = 42):
    if not HAS_UMAP:
        raise RuntimeError("UMAP not installed. Install: pip install umap-learn")
    reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1)
    return reducer.fit_transform(X)


def project_tsne(X: np.ndarray, seed: int = 42):
    if not HAS_TSNE:
        raise RuntimeError("t-SNE not available in your sklearn install.")
    tsne = TSNE(n_components=2, random_state=seed, perplexity=30, init="pca", learning_rate="auto")
    return tsne.fit_transform(X)


def scatter_by_clusters(P2: np.ndarray, yhat: np.ndarray, title: str, out_png: Path):
    plt.figure(figsize=(10, 7))
    uniq = np.unique(yhat)

    # Noise points for DBSCAN
    if -1 in uniq:
        mask_noise = (yhat == -1)
        plt.scatter(P2[mask_noise, 0], P2[mask_noise, 1], s=10, alpha=0.35, marker="x", label="noise (-1)")
        uniq = np.array([u for u in uniq if u != -1])

    for u in uniq:
        m = (yhat == u)
        plt.scatter(P2[m, 0], P2[m, 1], s=12, alpha=0.7, label=f"cluster {int(u)}")

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    # For many clusters legend becomes huge; keep only if <= 12 clusters
    if len(uniq) <= 12:
        plt.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def scatter_by_true_labels(P2: np.ndarray, y_true: np.ndarray, title: str, out_png: Path, max_classes: int = 25):
    plt.figure(figsize=(10, 7))

    # Limit legend if too many labels
    classes, counts = np.unique(y_true, return_counts=True)
    order = np.argsort(-counts)
    classes = classes[order]

    shown = set(classes[:max_classes].tolist())
    for c in classes:
        m = (y_true == c)
        if c in shown:
            plt.scatter(P2[m, 0], P2[m, 1], s=12, alpha=0.7, label=str(c))
        else:
            # lump the rest
            plt.scatter(P2[m, 0], P2[m, 1], s=12, alpha=0.15)

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()

    # Choose representation
    ap.add_argument("--repr", type=str, required=True,
                    help="Path to .npy representation. Examples: data/vae_mm_latents_mu.npy, data/audio_cnn_mel_X.npy, data/lyrics_embeddings.npy")
    ap.add_argument("--ids", type=str, required=True,
                    help="Path to matching track_ids .npy")

    # Clustering
    ap.add_argument("--method", type=str, default="kmeans", choices=["kmeans", "agglomerative", "dbscan"])
    ap.add_argument("--n_clusters", type=int, default=6)
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--min_samples", type=int, default=5)

    # Projection
    ap.add_argument("--proj", type=str, default="pca", choices=["pca", "umap", "tsne"])
    ap.add_argument("--standardize", action="store_true", help="Recommended for clustering/projection.")
    ap.add_argument("--pre_pca_dim", type=int, default=50,
                    help="For very high-dim inputs (e.g., flattened mel), reduce to this dim before UMAP/t-SNE. Set 0 to disable.")
    ap.add_argument("--seed", type=int, default=42)

    # Labels
    ap.add_argument("--manifest", type=str, default="data/fma_manifest_combined_text_only_clean.csv")

    # Output tag
    ap.add_argument("--tag", type=str, default="run")

    args = ap.parse_args()

    out_dir = Path("results/cluster_viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    X = np.load(args.repr).astype(np.float32)
    ids = np.load(args.ids).astype(np.int64)

    if X.ndim > 2:
        # flatten CNN tensors
        X = X.reshape(X.shape[0], -1)

    if args.standardize:
        X = StandardScaler().fit_transform(X)

    # Optional pre-PCA before UMAP/tSNE to avoid slowdown
    X_for_proj = X
    if args.pre_pca_dim and args.pre_pca_dim > 0 and X.shape[1] > args.pre_pca_dim and args.proj in ("umap", "tsne"):
        X_for_proj = PCA(n_components=args.pre_pca_dim, random_state=args.seed).fit_transform(X)

    # Cluster in the standardized/original feature space (X), not in 2D
    yhat = run_clustering(X, args.method, args.n_clusters, args.eps, args.min_samples)

    # 2D projection
    if args.proj == "pca":
        P2 = project_pca(X_for_proj, seed=args.seed)
    elif args.proj == "umap":
        P2 = project_umap(X_for_proj, seed=args.seed)
    else:
        P2 = project_tsne(X_for_proj, seed=args.seed)

    # True labels (optional)
    label_map, label_col = load_label_map(args.manifest)
    y_true = labels_for_ids(label_map, ids)

    # Filenames
    base = f"{args.tag}_{Path(args.repr).stem}_{args.method}_{args.proj}"
    out_clusters = out_dir / f"{base}_clusters.png"
    out_true = out_dir / f"{base}_truegenre.png"
    out_txt = out_dir / f"{base}_summary.txt"

    # Save plots
    scatter_by_clusters(
        P2, yhat,
        title=f"{args.tag}: {Path(args.repr).stem} | {args.method} | {args.proj}",
        out_png=out_clusters
    )

    if y_true is not None and label_col is not None:
        scatter_by_true_labels(
            P2, y_true,
            title=f"{args.tag}: TRUE LABELS ({label_col}) | {args.proj}",
            out_png=out_true
        )

    # Save a short summary
    uniq = np.unique(yhat)
    n_noise = int(np.sum(yhat == -1)) if -1 in uniq else 0
    n_clusters_found = len([u for u in uniq.tolist() if u != -1])

    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"repr={args.repr}\n")
        f.write(f"ids={args.ids}\n")
        f.write(f"method={args.method}\n")
        if args.method in ("kmeans", "agglomerative"):
            f.write(f"n_clusters={args.n_clusters}\n")
        else:
            f.write(f"eps={args.eps}\n")
            f.write(f"min_samples={args.min_samples}\n")
        f.write(f"proj={args.proj}\n")
        f.write(f"standardize={args.standardize}\n")
        f.write(f"pre_pca_dim={args.pre_pca_dim}\n")
        f.write(f"n_clusters_found={n_clusters_found}\n")
        f.write(f"n_noise={n_noise}\n")
        if label_col is not None:
            f.write(f"label_col={label_col}\n")

    print("\nSaved visualizations:")
    print(" ", out_clusters)
    if y_true is not None and label_col is not None:
        print(" ", out_true)
    print(" ", out_txt)


if __name__ == "__main__":
    main()
