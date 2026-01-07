from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:
    import umap.umap_ as umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


def project_pca(X: np.ndarray, seed: int = 42) -> np.ndarray:
    return PCA(n_components=2, random_state=seed).fit_transform(X)


def project_umap(X: np.ndarray, seed: int = 42) -> np.ndarray:
    if not HAS_UMAP:
        raise RuntimeError("UMAP not installed. Install: pip install umap-learn")
    return umap.UMAP(n_components=2, random_state=seed, n_neighbors=15, min_dist=0.1).fit_transform(X)


def scatter(ax, P2: np.ndarray, y: np.ndarray, title: str):
    uniq = np.unique(y)
    if -1 in uniq:
        m = (y == -1)
        ax.scatter(P2[m, 0], P2[m, 1], s=8, alpha=0.25, marker="x")
        uniq = np.array([u for u in uniq if u != -1])

    for u in uniq:
        m = (y == u)
        ax.scatter(P2[m, 0], P2[m, 1], s=10, alpha=0.7)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def ensure_2d(X: np.ndarray) -> np.ndarray:
    if X.ndim > 2:
        return X.reshape(X.shape[0], -1)
    return X


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--vae", type=str, default="data/vae_mm_latents_mu.npy")
    ap.add_argument("--mel", type=str, default="data/audio_cnn_mel_X.npy")
    ap.add_argument("--lyrics", type=str, default="data/lyrics_embeddings.npy")

    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--dbscan_eps_list", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--dbscan_min_samples", type=int, default=5)

    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default="medium")

    args = ap.parse_args()

    out_dir = Path("results/cluster_viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load and prepare
    X_vae = ensure_2d(np.load(args.vae).astype(np.float32))
    X_mel = ensure_2d(np.load(args.mel).astype(np.float32))
    X_lyr = ensure_2d(np.load(args.lyrics).astype(np.float32))

    # Standardize (recommended)
    if args.standardize:
        X_vae = StandardScaler().fit_transform(X_vae)
        X_mel = StandardScaler().fit_transform(X_mel)
        X_lyr = StandardScaler().fit_transform(X_lyr)

    # Cluster assignments
    y_vae = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed).fit_predict(X_vae)
    y_mel = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed).fit_predict(X_mel)

    # Lyrics: choose DBSCAN at eps=0.4 as "illustration" plus KMeans for comparison
    y_lyr_db = DBSCAN(eps=0.4, min_samples=args.dbscan_min_samples).fit_predict(X_lyr)
    y_lyr_km = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed).fit_predict(X_lyr)

    # Projections (PCA always, UMAP optional)
    P_vae_pca = project_pca(X_vae, seed=args.seed)
    P_mel_pca = project_pca(X_mel, seed=args.seed)
    P_lyr_pca = project_pca(X_lyr, seed=args.seed)

    if HAS_UMAP:
        P_vae_umap = project_umap(X_vae, seed=args.seed)
        P_mel_umap = project_umap(X_mel, seed=args.seed)
        P_lyr_umap = project_umap(X_lyr, seed=args.seed)
    else:
        P_vae_umap = P_mel_umap = P_lyr_umap = None

    # ---- Side-by-side figure ----
    if HAS_UMAP:
        fig, axes = plt.subplots(3, 2, figsize=(12, 16))
        scatter(axes[0, 0], P_vae_pca, y_vae, f"VAE latents + KMeans(k={args.k}) | PCA")
        scatter(axes[0, 1], P_vae_umap, y_vae, f"VAE latents + KMeans(k={args.k}) | UMAP")

        scatter(axes[1, 0], P_mel_pca, y_mel, f"Mel(flat) + KMeans(k={args.k}) | PCA")
        scatter(axes[1, 1], P_mel_umap, y_mel, f"Mel(flat) + KMeans(k={args.k}) | UMAP")

        scatter(axes[2, 0], P_lyr_pca, y_lyr_db, "Lyrics + DBSCAN(eps=0.4) | PCA (noise likely)")
        scatter(axes[2, 1], P_lyr_umap, y_lyr_db, "Lyrics + DBSCAN(eps=0.4) | UMAP (noise likely)")
    else:
        fig, axes = plt.subplots(3, 1, figsize=(10, 16))
        scatter(axes[0], P_vae_pca, y_vae, f"VAE latents + KMeans(k={args.k}) | PCA")
        scatter(axes[1], P_mel_pca, y_mel, f"Mel(flat) + KMeans(k={args.k}) | PCA")
        scatter(axes[2], P_lyr_pca, y_lyr_db, "Lyrics + DBSCAN(eps=0.4) | PCA (noise likely)")

    plt.tight_layout()
    side_png = out_dir / f"side_by_side_{args.tag}.png"
    plt.savefig(side_png, dpi=220)
    plt.close()

    # ---- DBSCAN eps sweep for lyrics ----
    eps_list = [float(x.strip()) for x in args.dbscan_eps_list.split(",") if x.strip()]
    n_clusters = []
    n_noise = []

    for eps in eps_list:
        y = DBSCAN(eps=eps, min_samples=args.dbscan_min_samples).fit_predict(X_lyr)
        uniq = np.unique(y)
        n_noise.append(int(np.sum(y == -1)) if -1 in uniq else 0)
        n_clusters.append(len([u for u in uniq.tolist() if u != -1]))

    plt.figure(figsize=(10, 6))
    plt.plot(eps_list, n_clusters, marker="o")
    plt.xlabel("DBSCAN eps")
    plt.ylabel("Clusters found (excluding noise)")
    plt.title("Lyrics DBSCAN: eps vs clusters found")
    plt.tight_layout()
    sweep1 = out_dir / f"lyrics_dbscan_eps_sweep_clusters_{args.tag}.png"
    plt.savefig(sweep1, dpi=220)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(eps_list, n_noise, marker="o")
    plt.xlabel("DBSCAN eps")
    plt.ylabel("Noise points (-1)")
    plt.title("Lyrics DBSCAN: eps vs number of noise points")
    plt.tight_layout()
    sweep2 = out_dir / f"lyrics_dbscan_eps_sweep_noise_{args.tag}.png"
    plt.savefig(sweep2, dpi=220)
    plt.close()

    print("\nSaved:")
    print(" ", side_png)
    print(" ", sweep1)
    print(" ", sweep2)
    if not HAS_UMAP:
        print("\nNote: UMAP plots skipped (install with: pip install umap-learn).")
    print("\nExtra: Lyrics KMeans comparison exists internally (y_lyr_km) if you want me to add it to the figure.")


if __name__ == "__main__":
    main()
