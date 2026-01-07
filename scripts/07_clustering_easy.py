from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


@dataclass
class Config:
    vae_out_dir: str
    out_dir: str
    k: int
    seed: int
    n_init: int


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="KMeans clustering on VAE latent features.")
    p.add_argument("--vae_out_dir", type=str, default="results/vae_basic", help="Output directory from Script 06.")
    p.add_argument("--out_dir", type=str, default="results/kmeans_vae", help="Where to save labels and artifacts.")
    p.add_argument("--k", type=int, default=5, help="Number of clusters.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_init", type=int, default=20)
    a = p.parse_args()
    return Config(
        vae_out_dir=a.vae_out_dir,
        out_dir=a.out_dir,
        k=a.k,
        seed=a.seed,
        n_init=a.n_init,
    )


def load_latents(vae_out: Path) -> Tuple[np.ndarray, np.ndarray]:
    lat_path = vae_out / "latent_mu.npy"
    ids_path = vae_out / "track_ids.npy"
    if not lat_path.exists():
        raise FileNotFoundError(f"Missing: {lat_path}")
    if not ids_path.exists():
        raise FileNotFoundError(f"Missing: {ids_path}")

    Z = np.load(lat_path).astype(np.float32)
    track_ids = np.load(ids_path).astype(np.int64)

    if Z.ndim != 2:
        raise ValueError(f"Expected latents 2D, got {Z.shape}")
    if len(track_ids) != Z.shape[0]:
        raise ValueError(f"track_ids len {len(track_ids)} != latents rows {Z.shape[0]}")
    return Z, track_ids


def main() -> None:
    cfg = parse_args()
    vae_out = Path(cfg.vae_out_dir)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Z, track_ids = load_latents(vae_out)

    # Standardize before KMeans (recommended)
    scaler = StandardScaler()
    Zs = scaler.fit_transform(Z)

    km = KMeans(n_clusters=cfg.k, random_state=cfg.seed, n_init=cfg.n_init)
    labels = km.fit_predict(Zs)

    np.save(out_dir / "labels_vae_kmeans.npy", labels)
    np.save(out_dir / "kmeans_vae_centers.npy", km.cluster_centers_.astype(np.float32))
    np.save(out_dir / "track_ids.npy", track_ids)

    # Simple label distribution
    unique, counts = np.unique(labels, return_counts=True)
    dist = {int(u): int(c) for u, c in zip(unique, counts)}

    summary: Dict = {
        "config": asdict(cfg),
        "vae_latent_shape": list(Z.shape),
        "label_distribution": dist,
        "note": "Labels correspond to rows in track_ids.npy.",
    }
    with open(out_dir / "kmeans_vae_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Saved:")
    print(" ", out_dir / "labels_vae_kmeans.npy")
    print(" ", out_dir / "kmeans_vae_centers.npy")
    print(" ", out_dir / "track_ids.npy")
    print(" ", out_dir / "kmeans_vae_summary.json")


if __name__ == "__main__":
    main()
