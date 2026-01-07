from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Optional UMAP
try:
    import umap.umap_ as umap  # type: ignore
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


@dataclass
class Config:
    vae_out_dir: str
    kmeans_out_dir: str
    out_dir: str
    reducer: str
    seed: int

    # UMAP
    umap_n_neighbors: int
    umap_min_dist: float

    # TSNE
    tsne_perplexity: float
    tsne_learning_rate: float
    tsne_n_iter: int


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Visualize VAE latent clusters via UMAP or t-SNE.")
    p.add_argument("--vae_out_dir", type=str, default="results/vae_basic")
    p.add_argument("--kmeans_out_dir", type=str, default="results/kmeans_vae")
    p.add_argument("--out_dir", type=str, default="results/viz_vae")
    p.add_argument("--reducer", type=str, choices=["umap", "tsne"], default="umap")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--umap_n_neighbors", type=int, default=30)
    p.add_argument("--umap_min_dist", type=float, default=0.1)

    p.add_argument("--tsne_perplexity", type=float, default=30.0)
    p.add_argument("--tsne_learning_rate", type=float, default=200.0)
    p.add_argument("--tsne_n_iter", type=int, default=1500)

    a = p.parse_args()
    return Config(
        vae_out_dir=a.vae_out_dir,
        kmeans_out_dir=a.kmeans_out_dir,
        out_dir=a.out_dir,
        reducer=a.reducer,
        seed=a.seed,
        umap_n_neighbors=a.umap_n_neighbors,
        umap_min_dist=a.umap_min_dist,
        tsne_perplexity=a.tsne_perplexity,
        tsne_learning_rate=a.tsne_learning_rate,
        tsne_n_iter=a.tsne_n_iter,
    )


def load_inputs(vae_out: Path, km_out: Path) -> Tuple[np.ndarray, np.ndarray]:
    Z_path = vae_out / "latent_mu.npy"
    L_path = km_out / "labels_vae_kmeans.npy"
    if not Z_path.exists():
        raise FileNotFoundError(f"Missing: {Z_path}")
    if not L_path.exists():
        raise FileNotFoundError(f"Missing: {L_path}")

    Z = np.load(Z_path).astype(np.float32)
    labels = np.load(L_path).astype(np.int64)
    if len(labels) != Z.shape[0]:
        raise ValueError("labels size does not match latents rows.")
    return Z, labels


def reduce_2d(Zs: np.ndarray, cfg: Config) -> np.ndarray:
    if cfg.reducer == "umap":
        if not HAS_UMAP:
            raise RuntimeError("UMAP not installed. Use --reducer tsne or install: pip install umap-learn")
        reducer = umap.UMAP(
            n_neighbors=cfg.umap_n_neighbors,
            min_dist=cfg.umap_min_dist,
            n_components=2,
            random_state=cfg.seed,
        )
        return reducer.fit_transform(Zs)

    tsne = TSNE(
        n_components=2,
        perplexity=cfg.tsne_perplexity,
        learning_rate=cfg.tsne_learning_rate,
        n_iter=cfg.tsne_n_iter,
        init="pca",
        random_state=cfg.seed,
    )
    return tsne.fit_transform(Zs)


def main() -> None:
    cfg = parse_args()
    vae_out = Path(cfg.vae_out_dir)
    km_out = Path(cfg.kmeans_out_dir)
    out_dir = Path(cfg.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    Z, labels = load_inputs(vae_out, km_out)

    # Standardize before embedding (recommended)
    Zs = StandardScaler().fit_transform(Z)

    emb = reduce_2d(Zs, cfg)

    plt.figure(figsize=(9, 7))
    plt.scatter(emb[:, 0], emb[:, 1], c=labels, s=10, alpha=0.85)
    plt.title(f"VAE Latents clustered — {cfg.reducer.upper()}")
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.tight_layout()

    out_path = plots_dir / f"vae_{cfg.reducer}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print("Saved plot:", out_path)


if __name__ == "__main__":
    main()
