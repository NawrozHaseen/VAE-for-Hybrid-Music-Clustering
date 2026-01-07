from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DATA_DIR = Path("data/hard")
MODEL_DIR = Path("models/hard")
RES_DIR = Path("results/hard")
PLOTS_DIR = RES_DIR / "plots"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tagged_path(path: Path, tag: str) -> Path:
    # Insert _{tag} before suffix
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def save_and_snapshot(path: Path, tag: Optional[str] = None) -> None:
    """
    Assumes the canonical file already exists at `path`.
    If tag is provided, make a copy next to it with suffix _tag.
    """
    if tag:
        snap = tagged_path(path, tag)
        shutil.copy2(path, snap)


class NpyDataset(Dataset):
    def __init__(self, X: np.ndarray, cond: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X).float()
        self.cond = None if cond is None else torch.from_numpy(cond).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if self.cond is None:
            return self.X[idx]
        return self.X[idx], self.cond[idx]


class MLPVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        cond_dim: int = 0,
        conditional: bool = False,
    ):
        super().__init__()
        self.conditional = conditional
        self.cond_dim = cond_dim

        enc_in = input_dim + (cond_dim if conditional else 0)

        self.enc = nn.Sequential(
            nn.Linear(enc_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

        dec_in = latent_dim + (cond_dim if conditional else 0)
        self.dec = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x, c=None):
        if self.conditional:
            if c is None:
                raise ValueError("Conditional model requires conditioning vector c.")
            x = torch.cat([x, c], dim=1)
        h = self.enc(x)
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c=None):
        if self.conditional:
            if c is None:
                raise ValueError("Conditional model requires conditioning vector c.")
            z = torch.cat([z, c], dim=1)
        return self.dec(z)

    def forward(self, x, c=None):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z, c)
        return x_hat, mu, logvar


def kl_divergence(mu, logvar):
    # KL(q(z|x) || N(0, I))
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)


def make_onehot(idx: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((idx.shape[0], num_classes), dtype=np.float32)
    out[np.arange(idx.shape[0]), idx] = 1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_dim", type=int, default=16)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--beta", type=float, default=4.0, help="Beta-VAE KL weight (higher -> more disentangled)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--use_cvae", action="store_true", help="Train CVAE (conditional on genre+language)")
    ap.add_argument("--cond_on", type=str, default="genre_lang", choices=["genre", "lang", "genre_lang"])
    ap.add_argument("--include_genre_in_input", action="store_true",
                    help="Append genre one-hot into X (explicit 'genre information' fusion).")
    ap.add_argument("--include_lang_in_input", action="store_true",
                    help="Append language one-hot into X (explicit language info fusion).")

    ap.add_argument("--tag", type=str, default=None,
                    help="Optional tag to snapshot outputs (e.g., beta, cvae, beta_ld32_b8).")

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Load prepared arrays
    X_audio = np.load(DATA_DIR / "audio_mfcc_stats.npy")      # (N, A)
    X_text = np.load(DATA_DIR / "lyrics_emb.npy")             # (N, T)
    y_genre = np.load(DATA_DIR / "genre_idx.npy")             # (N,)
    y_lang = np.load(DATA_DIR / "lang_idx.npy")               # (N,)

    n_genres = int(y_genre.max() + 1) if y_genre.size else 1
    n_langs = int(y_lang.max() + 1) if y_lang.size else 1

    # Base multimodal fusion: audio + text
    X = np.concatenate([X_audio, X_text], axis=1).astype(np.float32)

    # Optional explicit "genre/language information" fusion into the input vector
    if args.include_genre_in_input:
        X = np.concatenate([X, make_onehot(y_genre, n_genres)], axis=1)
    if args.include_lang_in_input:
        X = np.concatenate([X, make_onehot(y_lang, n_langs)], axis=1)

    # Conditioning vector (for CVAE)
    cond = None
    cond_dim = 0
    if args.use_cvae:
        if args.cond_on == "genre":
            cond = make_onehot(y_genre, n_genres)
        elif args.cond_on == "lang":
            cond = make_onehot(y_lang, n_langs)
        else:
            cond = np.concatenate([make_onehot(y_genre, n_genres), make_onehot(y_lang, n_langs)], axis=1)
        cond_dim = cond.shape[1]

    ds = NpyDataset(X, cond=cond)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = MLPVAE(
        input_dim=X.shape[1],
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        cond_dim=cond_dim,
        conditional=args.use_cvae,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    losses = []
    recon_losses = []
    kl_losses = []

    print(f"Training {'CVAE' if args.use_cvae else 'Beta-VAE'} | input_dim={X.shape[1]} latent_dim={args.latent_dim} beta={args.beta}")
    model.train()
    for epoch in range(1, args.epochs + 1):
        tot = 0.0
        tot_rec = 0.0
        tot_kl = 0.0
        n = 0

        for batch in dl:
            if args.use_cvae:
                xb, cb = batch
                xb = xb.to(device)
                cb = cb.to(device)
                x_hat, mu, logvar = model(xb, cb)
            else:
                xb = batch.to(device)
                x_hat, mu, logvar = model(xb)

            rec = F.mse_loss(x_hat, xb, reduction="none").sum(dim=1)
            kl = kl_divergence(mu, logvar)
            loss = (rec + args.beta * kl).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = xb.size(0)
            tot += float(loss.item()) * bs
            tot_rec += float(rec.mean().item()) * bs
            tot_kl += float(kl.mean().item()) * bs
            n += bs

        losses.append(tot / n)
        recon_losses.append(tot_rec / n)
        kl_losses.append(tot_kl / n)

        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch:03d}/{args.epochs} | loss={losses[-1]:.4f} rec={recon_losses[-1]:.4f} kl={kl_losses[-1]:.4f}")

    # Save model (canonical)
    model_path = MODEL_DIR / ("cvae_multimodal.pt" if args.use_cvae else "beta_vae_multimodal.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(X.shape[1]),
            "latent_dim": int(args.latent_dim),
            "hidden_dim": int(args.hidden_dim),
            "beta": float(args.beta),
            "use_cvae": bool(args.use_cvae),
            "cond_dim": int(cond_dim),
            "include_genre_in_input": bool(args.include_genre_in_input),
            "include_lang_in_input": bool(args.include_lang_in_input),
            "cond_on": args.cond_on,
            "seed": int(args.seed),
        },
        model_path,
    )
    print("Saved model:", model_path)
    save_and_snapshot(model_path, args.tag)

    # Export latent means (mu) (canonical)
    model.eval()
    with torch.no_grad():
        all_mu = []
        for batch in DataLoader(ds, batch_size=512, shuffle=False):
            if args.use_cvae:
                xb, cb = batch
                xb = xb.to(device)
                cb = cb.to(device)
                mu, _ = model.encode(xb, cb)
            else:
                xb = batch.to(device)
                mu, _ = model.encode(xb)
            all_mu.append(mu.cpu().numpy())
        Z = np.concatenate(all_mu, axis=0).astype(np.float32)

    lat_path = DATA_DIR / "latents_mu.npy"
    np.save(lat_path, Z)
    print("Saved latents:", lat_path, "shape=", Z.shape)
    save_and_snapshot(lat_path, args.tag)

    # Plot training curve (canonical)
    fig = plt.figure()
    plt.plot(losses, label="total")
    plt.plot(recon_losses, label="recon")
    plt.plot(kl_losses, label="kl")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss (Beta-VAE/CVAE)")
    out_curve = PLOTS_DIR / "training_curve.png"
    fig.savefig(out_curve, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved plot:", out_curve)
    save_and_snapshot(out_curve, args.tag)

    # Reconstruction examples (canonical)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(X.shape[0], size=min(6, X.shape[0]), replace=False)

    with torch.no_grad():
        xb = torch.from_numpy(X[idx]).float().to(device)
        if args.use_cvae:
            cb = torch.from_numpy(cond[idx]).float().to(device)
            x_hat, _, _ = model(xb, cb)
        else:
            x_hat, _, _ = model(xb)
        xb = xb.cpu().numpy()
        x_hat = x_hat.cpu().numpy()

    fig = plt.figure(figsize=(10, 6))
    dims = min(80, X.shape[1])
    for i in range(len(idx)):
        plt.subplot(3, 2, i + 1)
        plt.plot(xb[i, :dims], label="x", linewidth=1)
        plt.plot(x_hat[i, :dims], label="x_hat", linewidth=1)
        plt.title(f"Reconstruction sample {i}")
        plt.xticks([])
        plt.yticks([])
        if i == 0:
            plt.legend(fontsize=8)
    out_recon = PLOTS_DIR / "recon_examples.png"
    fig.tight_layout()
    fig.savefig(out_recon, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Saved recon examples:", out_recon)
    save_and_snapshot(out_recon, args.tag)


if __name__ == "__main__":
    main()
