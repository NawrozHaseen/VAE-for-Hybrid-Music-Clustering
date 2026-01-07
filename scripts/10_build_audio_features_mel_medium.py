from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm


@dataclass
class MelCfg:
    sample_rate: int = 22050
    seconds: float = 15.0
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128
    power: float = 2.0
    center: bool = True
    top_db: Optional[float] = None  # e.g., 80.0; None keeps librosa default


def read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"track_id", "audio_path"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}. Found: {list(df.columns)}")
    return df


def pad_or_trunc_1d(y: np.ndarray, n_samples: int) -> np.ndarray:
    if len(y) == n_samples:
        return y
    if len(y) > n_samples:
        return y[:n_samples]
    return np.pad(y, (0, n_samples - len(y)))


def pad_or_trunc_2d(X: np.ndarray, T: int) -> np.ndarray:
    # X: (F, t)
    F, t = X.shape
    if t == T:
        return X
    if t > T:
        return X[:, :T]
    out = np.zeros((F, T), dtype=X.dtype)
    out[:, :t] = X
    return out


def extract_logmel_db(y: np.ndarray, cfg: MelCfg) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        power=cfg.power,
        center=cfg.center,
    )
    X = librosa.power_to_db(S, ref=np.max, top_db=cfg.top_db)  # (n_mels, T)
    return X


def per_sample_standardize(X: np.ndarray) -> np.ndarray:
    mu = float(X.mean())
    sd = float(X.std()) + 1e-6
    return (X - mu) / sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default="data/fma_manifest_combined_text_only_clean.csv")
    ap.add_argument("--out_x", type=str, default="data/audio_cnn_mel_X.npy")
    ap.add_argument("--out_ids", type=str, default="data/audio_cnn_mel_track_ids.npy")
    ap.add_argument("--report_csv", type=str, default="results/audio_cnn_mel_build_report.csv")

    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--sr", type=int, default=22050)
    ap.add_argument("--n_mels", type=int, default=128)
    ap.add_argument("--n_fft", type=int, default=2048)
    ap.add_argument("--hop_length", type=int, default=512)
    ap.add_argument("--top_db", type=float, default=-1.0, help="Set to e.g. 80; -1 means 'None'")

    ap.add_argument("--max_items", type=int, default=0, help="0 = all rows")
    ap.add_argument("--strict", action="store_true",
                    help="If set, raise on the first error instead of skipping bad files.")
    args = ap.parse_args()

    cfg = MelCfg(
        sample_rate=args.sr,
        seconds=args.seconds,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        top_db=None if args.top_db < 0 else float(args.top_db),
    )

    manifest_path = Path(args.manifest)
    df = read_manifest(manifest_path)
    if args.max_items and args.max_items > 0:
        df = df.head(args.max_items).copy()

    n_samples = int(cfg.sample_rate * cfg.seconds)

    X_list: list[np.ndarray] = []
    id_list: list[int] = []
    report_rows = []

    # We enforce consistent time frames T using first successful sample.
    fixed_T: Optional[int] = None

    out_x_path = Path(args.out_x)
    out_ids_path = Path(args.out_ids)
    report_path = Path(args.report_csv)
    out_x_path.parent.mkdir(parents=True, exist_ok=True)
    out_ids_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    skipped = 0
    loaded = 0

    for _, r in tqdm(df.iterrows(), total=len(df), desc="Building log-mel tensors"):
        track_id = int(r["track_id"])
        audio_path = Path(str(r["audio_path"]))

        row_info = {
            "track_id": track_id,
            "audio_path": str(audio_path),
            "status": "",
            "reason": "",
        }

        if not audio_path.exists():
            skipped += 1
            row_info["status"] = "skipped"
            row_info["reason"] = "missing_file"
            report_rows.append(row_info)
            if args.strict:
                raise FileNotFoundError(f"Missing audio file: {audio_path}")
            continue

        try:
            # Load mono audio at target sr, then pad/trunc to exact length.
            y, _ = librosa.load(audio_path, sr=cfg.sample_rate, mono=True, duration=cfg.seconds)
            y = pad_or_trunc_1d(y, n_samples)

            mel_db = extract_logmel_db(y, cfg)   # (n_mels, T)
            if fixed_T is None:
                fixed_T = mel_db.shape[1]

            mel_db = pad_or_trunc_2d(mel_db, fixed_T)
            mel_db = per_sample_standardize(mel_db)

            X_list.append(mel_db.astype(np.float32))
            id_list.append(track_id)

            loaded += 1
            row_info["status"] = "ok"
            row_info["reason"] = ""
            report_rows.append(row_info)

        except Exception as e:
            skipped += 1
            row_info["status"] = "skipped"
            row_info["reason"] = f"error:{type(e).__name__}"
            report_rows.append(row_info)
            if args.strict:
                raise
            continue

    if not X_list:
        raise RuntimeError(
            "No features were extracted. Check: (1) manifest audio_path values, "
            "(2) audio files exist, (3) librosa backend works."
        )

    X = np.stack(X_list, axis=0)            # (N, n_mels, T)
    X = X[:, None, :, :]                    # (N, 1, n_mels, T)
    ids = np.array(id_list, dtype=np.int64)

    np.save(out_x_path, X)
    np.save(out_ids_path, ids)

    report_df = pd.DataFrame(report_rows)
    report_df.to_csv(report_path, index=False)

    print("\nDONE")
    print(f"Manifest: {manifest_path}")
    print(f"Loaded:   {loaded}")
    print(f"Skipped:  {skipped}")
    print(f"Tensor:   {out_x_path}  shape={X.shape} dtype={X.dtype}")
    print(f"IDs:      {out_ids_path} shape={ids.shape} dtype={ids.dtype}")
    print(f"Report:   {report_path}")
    print(f"Fixed T frames: {X.shape[-1]}  (n_mels={X.shape[-2]})")


if __name__ == "__main__":
    main()
