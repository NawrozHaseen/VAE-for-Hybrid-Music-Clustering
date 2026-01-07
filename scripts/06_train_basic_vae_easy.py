from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import librosa

import joblib

#seed utilities
def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device, non_blocking=True)


@dataclass
class TrainConfig:
    manifest: str
    out_dir: str
    sample_rate: int
    duration_sec: float
    n_mfcc: int
    hop_length: int
    n_fft: int
    batch_size: int
    epochs: int
    lr: float
    latent_dim: int
    hidden_dim: int
    beta: float
    seed: int
    num_workers: int
    cache_features: bool


#mfcc features
def extract_mfcc_feature(
    audio_path: str,
    sample_rate: int,
    duration_sec: float,
    n_mfcc: int,
    n_fft: int,
    hop_length: int
) -> Optional[np.ndarray]:
    """
    Returns fixed-length feature vector:
      [mfcc_mean (n_mfcc), mfcc_std (n_mfcc)] -> shape (2*n_mfcc,)
    """
    try:
        y, sr = librosa.load(audio_path, sr=sample_rate, mono=True)
        if y.size == 0:
            return None

        # Trim/pad to fixed duration
        target_len = int(sample_rate * duration_sec)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode="constant")
        else:
            y = y[:target_len]

        mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        # mfcc shape: (n_mfcc, T)

        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)

        feat = np.concatenate([mfcc_mean, mfcc_std], axis=0).astype(np.float32)
        return feat
    except Exception:
        return None


def build_feature_matrix(
    df: pd.DataFrame,
    sample_rate: int,
    duration_sec: float,
    n_mfcc: int,
    n_fft: int,
    hop_length: int,
    cache_path: Path,
    cache_features: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds X (N, D) and track_ids (N,).
    If cache_features and cache exists, loads it.
    """
    if cache_features and cache_path.exists():
        blob = np.load(cache_path, allow_pickle=True).item()
        return blob["X"], blob["track_ids"]

    feats: List[np.ndarray] = []
    tids: List[int] = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting MFCC features"):
        audio_path = str(row["audio_path"])
        tid = int(row["track_id"]) if "track_id" in df.columns else int(i)

        f = extract_mfcc_feature(
            audio_path=audio_path,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
            n_mfcc=n_mfcc,
            n_fft=n_fft,
            hop_length=hop_length,
        )
        if f is None:
            continue

        feats.append(f)
        tids.append(tid)

    if len(feats) == 0:
        raise RuntimeError("No features extracted. Check audio paths and your manifest.")

    X = np.stack(feats, axis=0)
    track_ids = np.array(tids, dtype=np.int64)

    if cache_features:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, {"X": X, "track_ids": track_ids}, allow_pickle=True)

    return X, track_ids


#vae model
class MLPVAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        # Encoder
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # z = mu + std * eps
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.dec1(z))
        h = F.relu(self.dec2(h))
        return self.out(h)  # reconstruction

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar


def vae_loss(xhat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float) -> Tuple[torch.Tensor, Dict[str, float]]:
    # Reconstruction loss (MSE)
    recon = F.mse_loss(xhat, x, reduction="mean")
    # KL divergence (per batch mean)
    kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
    loss = recon + beta * kl
    return loss, {"recon": float(recon.item()), "kl": float(kl.item()), "total": float(loss.item())}


class NumpyDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray):
        self.X = X.astype(np.float32)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.from_numpy(self.X[idx])


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Train a basic MLP VAE on MFCC features and export latent vectors.")
    p.add_argument("--manifest", type=str, default="data/fma_manifest_combined_text_only_clean.csv", help="Path to manifest CSV with audio_path.")
    p.add_argument("--out_dir", type=str, default="results/vae_basic", help="Output directory.")
    p.add_argument("--sample_rate", type=int, default=22050)
    p.add_argument("--duration_sec", type=float, default=30.0, help="Seconds per track used for features.")
    p.add_argument("--n_mfcc", type=int, default=40)
    p.add_argument("--n_fft", type=int, default=2048)
    p.add_argument("--hop_length", type=int, default=512)

    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--latent_dim", type=int, default=16)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--beta", type=float, default=1.0, help="Beta for KL term (beta-VAE style; keep 1.0 for basic VAE).")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cache_features", action="store_true", help="Cache extracted features to speed up reruns.")

    a = p.parse_args()
    return TrainConfig(
        manifest=a.manifest,
        out_dir=a.out_dir,
        sample_rate=a.sample_rate,
        duration_sec=a.duration_sec,
        n_mfcc=a.n_mfcc,
        hop_length=a.hop_length,
        n_fft=a.n_fft,
        batch_size=a.batch_size,
        epochs=a.epochs,
        lr=a.lr,
        latent_dim=a.latent_dim,
        hidden_dim=a.hidden_dim,
        beta=a.beta,
        seed=a.seed,
        num_workers=a.num_workers,
        cache_features=bool(a.cache_features),
    )


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    manifest_path = Path(cfg.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if "audio_path" not in df.columns:
        raise ValueError("Manifest must contain an 'audio_path' column.")

    # Keep only rows with existing audio
    df = df.dropna(subset=["audio_path"]).copy()
    df["audio_path"] = df["audio_path"].astype(str)
    df = df[df["audio_path"].apply(lambda p: Path(p).exists())].copy()
    if len(df) == 0:
        raise RuntimeError("No valid audio_path entries exist on disk.")

    # Ensure track_id exists (helpful downstream)
    if "track_id" not in df.columns:
        df = df.reset_index(drop=False).rename(columns={"index": "track_id"})

    # Feature cache
    cache_path = out_dir / "mfcc_features_cache.npy"

    X_raw, track_ids = build_feature_matrix(
        df=df,
        sample_rate=cfg.sample_rate,
        duration_sec=cfg.duration_sec,
        n_mfcc=cfg.n_mfcc,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        cache_path=cache_path,
        cache_features=cfg.cache_features,
    )

    print(f"Feature matrix: X shape = {X_raw.shape}")

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw).astype(np.float32)

    joblib.dump(scaler, out_dir / "scaler.joblib")
    np.save(out_dir / "track_ids.npy", track_ids)

    # DataLoader
    ds = NumpyDataset(X)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    input_dim = X.shape[1]
    model = MLPVAE(input_dim=input_dim, hidden_dim=cfg.hidden_dim, latent_dim=cfg.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = {"epoch": [], "recon": [], "kl": [], "total": []}

    # Train loop
    model.train()
    for epoch in range(1, cfg.epochs + 1):
        running = {"recon": 0.0, "kl": 0.0, "total": 0.0}
        n_batches = 0

        for batch in dl:
            batch = to_device(batch, device)
            opt.zero_grad(set_to_none=True)

            xhat, mu, logvar = model(batch)
            loss, parts = vae_loss(xhat, batch, mu, logvar, beta=cfg.beta)
            loss.backward()
            opt.step()

            for k in running:
                running[k] += parts[k]
            n_batches += 1

        for k in running:
            running[k] /= max(1, n_batches)

        history["epoch"].append(epoch)
        history["recon"].append(running["recon"])
        history["kl"].append(running["kl"])
        history["total"].append(running["total"])

        print(
            f"Epoch {epoch:03d}/{cfg.epochs} | "
            f"loss={running['total']:.6f} recon={running['recon']:.6f} kl={running['kl']:.6f}"
        )

    # Save model + config + history
    torch.save(model.state_dict(), out_dir / "vae_basic.pt")
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # Extract latent means (mu) for ALL tracks
    model.eval()
    with torch.no_grad():
        X_tensor = torch.from_numpy(X).to(device)
        mu, logvar = model.encode(X_tensor)
        latent_mu = mu.detach().cpu().numpy().astype(np.float32)

    np.save(out_dir / "latent_mu.npy", latent_mu)
    print("\nSaved outputs to:", out_dir)
    print("  - vae_basic.pt")
    print("  - scaler.joblib")
    print("  - track_ids.npy")
    print("  - latent_mu.npy")
    print("  - history.json")
    print("Next step: KMeans on latent_mu.npy + TSNE/UMAP visualization.")


if __name__ == "__main__":
    main()
