from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, adjusted_rand_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = Path("data/hard")
RES_DIR = Path("results/hard")
PLOTS_DIR = RES_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


class AE(nn.Module):
    def __init__(self, in_dim: int, z_dim: int = 16, hidden: int = 256):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, z_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x):
        z = self.enc(x)
        x_hat = self.dec(z)
        return x_hat, z


def eval_kmeans(name: str, X: np.ndarray, y_true: np.ndarray, k: int, seed: int) -> Dict:
    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    y_pred = km.fit_predict(X)
    return {
        "method": name,
        "silhouette": safe_silhouette(X, y_pred),
        "nmi": float(normalized_mutual_info_score(y_true, y_pred)),
        "ari": float(adjusted_rand_score(y_true, y_pred)),
        "purity": float(cluster_purity(y_true, y_pred)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=None, help="clusters (default: #genres)")
    ap.add_argument("--pca_dim", type=int, default=32)
    ap.add_argument("--ae_latent", type=int, default=16)
    ap.add_argument("--ae_epochs", type=int, default=30)
    ap.add_argument("--ae_batch", type=int, default=256)
    ap.add_argument("--ae_lr", type=float, default=1e-3)
    ap.add_argument("--tag", type=str, default=None, help="Optional tag to snapshot outputs.")
    ap.add_argument("--latents_path", type=str, default=None,
                    help="Optional path to latents .npy (default: data/hard/latents_mu.npy)")
    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    X_audio = np.load(DATA_DIR / "audio_mfcc_stats.npy")   # (N, A)
    X_text = np.load(DATA_DIR / "lyrics_emb.npy")          # (N, T)
    y = np.load(DATA_DIR / "genre_idx.npy")
    k = args.k if args.k is not None else int(y.max() + 1)

    X_fused = np.concatenate([X_audio, X_text], axis=1).astype(np.float32)

    # Latents
    lat_path = Path(args.latents_path) if args.latents_path else (DATA_DIR / "latents_mu.npy")
    Z = np.load(lat_path)

    rows = []
    rows.append(eval_kmeans("VAE/CVAE latents + KMeans", Z, y, k=k, seed=args.seed))
    rows.append(eval_kmeans("Direct spectral (MFCC stats) + KMeans", X_audio, y, k=k, seed=args.seed))

    pca = PCA(n_components=min(args.pca_dim, X_audio.shape[1]), random_state=args.seed)
    X_pca = pca.fit_transform(X_audio)
    rows.append(eval_kmeans(f"PCA({X_pca.shape[1]}) + KMeans (audio)", X_pca, y, k=k, seed=args.seed))

    # Autoencoder baseline
    ae = AE(in_dim=X_fused.shape[1], z_dim=args.ae_latent).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=args.ae_lr)

    ds = TensorDataset(torch.from_numpy(X_fused).float())
    dl = DataLoader(ds, batch_size=args.ae_batch, shuffle=True, drop_last=False)

    ae.train()
    for epoch in range(1, args.ae_epochs + 1):
        tot = 0.0
        n = 0
        for (xb,) in dl:
            xb = xb.to(device)
            x_hat, _ = ae(xb)
            loss = F.mse_loss(x_hat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.ae_epochs:
            print(f"AE epoch {epoch:03d}/{args.ae_epochs} loss={tot/n:.6f}")

    ae.eval()
    with torch.no_grad():
        Z_ae = []
        for (xb,) in DataLoader(ds, batch_size=512, shuffle=False):
            xb = xb.to(device)
            _, z = ae(xb)
            Z_ae.append(z.cpu().numpy())
        Z_ae = np.concatenate(Z_ae, axis=0).astype(np.float32)

    rows.append(eval_kmeans(f"Autoencoder(z={args.ae_latent}) + KMeans (fused)", Z_ae, y, k=k, seed=args.seed))

    out_csv = RES_DIR / "baseline_comparison.csv"
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print("Saved:", out_csv)
    print(df)
    save_and_snapshot(out_csv, args.tag)

    # Bar plot (canonical)
    fig = plt.figure(figsize=(10, 5))
    x = np.arange(len(df))
    width = 0.2

    plt.bar(x - 1.5 * width, df["silhouette"], width, label="silhouette")
    plt.bar(x - 0.5 * width, df["nmi"], width, label="nmi")
    plt.bar(x + 0.5 * width, df["ari"], width, label="ari")
    plt.bar(x + 1.5 * width, df["purity"], width, label="purity")

    plt.xticks(x, df["method"], rotation=25, ha="right")
    plt.ylabel("Score")
    plt.title("Hard Task: Baseline Comparison")
    plt.legend(fontsize=8)

    out_plot = PLOTS_DIR / "baseline_bars.png"
    fig.savefig(out_plot, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_plot)
    save_and_snapshot(out_plot, args.tag)


if __name__ == "__main__":
    main()
