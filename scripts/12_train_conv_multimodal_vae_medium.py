from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


@dataclass
class TrainCfg:
    z_dim: int = 32
    beta: float = 1.0
    lr: float = 2e-3
    batch_size: int = 64
    epochs: int = 25
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


#data 

def build_lyrics_lookup(lyr_ids: np.ndarray, lyr_emb: np.ndarray) -> Dict[int, np.ndarray]:
    lookup: Dict[int, np.ndarray] = {}
    for i in range(len(lyr_ids)):
        lookup[int(lyr_ids[i])] = lyr_emb[i]
    return lookup


def align_lyrics_to_audio(
    audio_ids: np.ndarray,
    lyr_ids: np.ndarray,
    lyr_emb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      lyr_aligned: (N, D) float32
      lyr_mask:    (N, 1) float32  (1 if lyrics exist else 0)
    """
    D = lyr_emb.shape[1]
    lookup = build_lyrics_lookup(lyr_ids, lyr_emb)

    lyr_aligned = np.zeros((len(audio_ids), D), dtype=np.float32)
    mask = np.zeros((len(audio_ids), 1), dtype=np.float32)

    found = 0
    for i, tid in enumerate(audio_ids):
        v = lookup.get(int(tid), None)
        if v is not None:
            lyr_aligned[i] = v
            mask[i, 0] = 1.0
            found += 1

    print(f"Lyrics aligned: {found}/{len(audio_ids)} tracks have embeddings.")
    return lyr_aligned, mask


class MultiModalDataset(Dataset):
    def __init__(self, X: np.ndarray, ids: np.ndarray, lyr: np.ndarray, lyr_mask: np.ndarray):
        self.X = X
        self.ids = ids
        self.lyr = lyr
        self.lyr_mask = lyr_mask

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx])                 # (1, F, T)
        tid = torch.tensor(self.ids[idx], dtype=torch.long)
        lyr = torch.from_numpy(self.lyr[idx])             # (D,)
        m = torch.from_numpy(self.lyr_mask[idx])          # (1,)
        return x, lyr, m, tid


#model 

class AudioEncoder(nn.Module):
    def __init__(self, in_ch: int, z_dim: int, feat_hw: Tuple[int, int]):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_ch, feat_hw[0], feat_hw[1])
            h = self.conv(dummy)
            self.h_shape = h.shape[1:]  # (C, H, W)
            flat = int(np.prod(self.h_shape))

        self.fc = nn.Sequential(
            nn.Linear(flat, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, z_dim)
        self.logvar = nn.Linear(256, z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(x).flatten(1)
        h = self.fc(h)
        return self.mu(h), self.logvar(h)


class LyricsProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, out_dim), nn.ReLU()
        )

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e)


class AudioDecoder(nn.Module):
    def __init__(self, z_dim: int, out_ch: int, h_shape: Tuple[int, int, int]):
        super().__init__()
        C, H, W = h_shape
        self.h_shape = (C, H, W)

        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256), nn.ReLU(),
            nn.Linear(256, C * H * W), nn.ReLU()
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(C, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, out_ch, 4, stride=2, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), *self.h_shape)
        return self.deconv(h)


class ConvMultiModalVAE(nn.Module):
    """
    Fusion design:
      - Audio encoder produces (mu_a, logvar_a)
      - Lyrics embedding -> projector -> l_feat
      - Mask m in {0,1} gates lyrics: l_feat = l_feat * m
      - Fusion MLP combines [mu_a, l_feat, m] -> (mu, logvar) for the actual latent
      - Decoder reconstructs audio
    """
    def __init__(self, in_ch: int, feat_hw: Tuple[int, int], z_dim: int, lyr_dim: int):
        super().__init__()
        self.audio_enc = AudioEncoder(in_ch, z_dim, feat_hw)
        self.lyr_proj = LyricsProjector(lyr_dim, out_dim=128)

        self.fuse = nn.Sequential(
            nn.Linear(z_dim + 128 + 1, 256),
            nn.ReLU(),
        )
        self.mu = nn.Linear(256, z_dim)
        self.logvar = nn.Linear(256, z_dim)

        self.audio_dec = AudioDecoder(z_dim, in_ch, self.audio_enc.h_shape)

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor, lyr: torch.Tensor, m: torch.Tensor):
        mu_a, lv_a = self.audio_enc(x)

        l = self.lyr_proj(lyr)          # (B, 128)
        l = l * m                       # gate lyrics when missing (m is (B,1))

        h = self.fuse(torch.cat([mu_a, l, m], dim=1))
        mu = self.mu(h)
        lv = self.logvar(h)

        z = self.reparam(mu, lv)
        xhat = self.audio_dec(z)
        return xhat, mu, lv


def kl_div(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--x", type=str, default="data/audio_cnn_mel_X.npy")
    ap.add_argument("--ids", type=str, default="data/audio_cnn_mel_track_ids.npy")
    ap.add_argument("--lyr_emb", type=str, default="data/lyrics_embeddings.npy")
    ap.add_argument("--lyr_ids", type=str, default="data/lyrics_track_ids.npy")

    ap.add_argument("--z_dim", type=int, default=32)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)

    ap.add_argument("--out_latents", type=str, default="data/vae_mm_latents_mu.npy")
    ap.add_argument("--out_latent_ids", type=str, default="data/vae_mm_latents_track_ids.npy")
    ap.add_argument("--out_dir", type=str, default="results/vae_conv_mm_medium")

    args = ap.parse_args()

    cfg = TrainCfg(z_dim=args.z_dim, beta=args.beta, epochs=args.epochs, batch_size=args.batch, lr=args.lr)

    # Load audio tensors
    X = np.load(args.x).astype(np.float32)                 # (N, 1, F, T)
    audio_ids = np.load(args.ids).astype(np.int64)         # (N,)

    # Load lyrics embeddings
    lyr_emb = np.load(args.lyr_emb).astype(np.float32)     # (M, D)
    lyr_ids = np.load(args.lyr_ids).astype(np.int64)       # (M,)

    lyr_aligned, lyr_mask = align_lyrics_to_audio(audio_ids, lyr_ids, lyr_emb)
    lyr_dim = lyr_aligned.shape[1]

    ds = MultiModalDataset(X, audio_ids, lyr_aligned, lyr_mask)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False, num_workers=cfg.num_workers)

    device = torch.device(cfg.device)
    feat_hw = (X.shape[2], X.shape[3])  # (F, T)

    model = ConvMultiModalVAE(in_ch=X.shape[1], feat_hw=feat_hw, z_dim=cfg.z_dim, lyr_dim=lyr_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    mse = nn.MSELoss()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.csv"
    with log_path.open("w", encoding="utf-8") as f:
        f.write("epoch,loss,recon,kl\n")

    print(f"\nDevice: {device}")
    print(f"Audio X: {X.shape}  Lyrics aligned: {lyr_aligned.shape}  z_dim={cfg.z_dim}  beta={cfg.beta}")

    # ---- Train ----
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = total_recon = total_kl = 0.0
        n_batches = 0

        for x, lyr, m, _tid in tqdm(dl, desc=f"Epoch {epoch}/{cfg.epochs}"):
            x = x.to(device)
            lyr = lyr.to(device)
            m = m.to(device)

            opt.zero_grad()
            xhat, mu, lv = model(x, lyr, m)

            # decoder may overshoot by a few frames; crop to exact
            xhat = xhat[:, :, : x.shape[2], : x.shape[3]]

            recon = mse(xhat, x)
            kl = kl_div(mu, lv)
            loss = recon + cfg.beta * kl

            loss.backward()
            opt.step()

            total_loss += float(loss.item())
            total_recon += float(recon.item())
            total_kl += float(kl.item())
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_recon = total_recon / max(1, n_batches)
        avg_kl = total_kl / max(1, n_batches)

        with log_path.open("a", encoding="utf-8") as f:
            f.write(f"{epoch},{avg_loss:.6f},{avg_recon:.6f},{avg_kl:.6f}\n")

        ckpt_path = out_dir / f"ckpt_epoch_{epoch:03d}.pt"
        torch.save(
            {"model": model.state_dict(), "cfg": cfg.__dict__, "feat_hw": feat_hw, "lyr_dim": lyr_dim},
            ckpt_path
        )

        print(f"Epoch {epoch}: loss={avg_loss:.4f} recon={avg_recon:.4f} kl={avg_kl:.4f}  saved={ckpt_path.name}")

    # extract latents (mu) for ALL samples 
    model.eval()
    dl_eval = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False, num_workers=cfg.num_workers)

    mu_list = []
    id_list = []
    with torch.no_grad():
        for x, lyr, m, tid in tqdm(dl_eval, desc="Extracting latents"):
            x = x.to(device)
            lyr = lyr.to(device)
            m = m.to(device)

            _xhat, mu, _lv = model(x, lyr, m)
            mu_list.append(mu.cpu().numpy())
            id_list.append(tid.numpy())

    Z = np.concatenate(mu_list, axis=0).astype(np.float32)     # (N, z_dim)
    Z_ids = np.concatenate(id_list, axis=0).astype(np.int64)   # (N,)

    np.save(args.out_latents, Z)
    np.save(args.out_latent_ids, Z_ids)

    print("\nDONE")
    print(f"Saved latents: {args.out_latents}  shape={Z.shape} dtype={Z.dtype}")
    print(f"Saved ids:     {args.out_latent_ids} shape={Z_ids.shape} dtype={Z_ids.dtype}")
    print(f"Logs/ckpt:     {out_dir}")


if __name__ == "__main__":
    main()
