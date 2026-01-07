from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler


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


def encode_str_labels(y_str: np.ndarray):
    uniq = {v: i for i, v in enumerate(sorted(set(y_str.tolist())))}
    return np.array([uniq[v] for v in y_str], dtype=int)


def safe_silhouette(X: np.ndarray, yhat: np.ndarray):
    uniq = np.unique(yhat)
    uniq_eff = [u for u in uniq.tolist() if u != -1]
    if len(uniq_eff) < 2:
        return None

    if -1 in uniq:
        mask = (yhat != -1)
        if mask.sum() < 3:
            return None
        try:
            return float(silhouette_score(X[mask], yhat[mask]))
        except Exception:
            return None

    try:
        return float(silhouette_score(X, yhat))
    except Exception:
        return None


def safe_dbi(X: np.ndarray, yhat: np.ndarray):
    uniq = np.unique(yhat)
    uniq_eff = [u for u in uniq.tolist() if u != -1]
    if len(uniq_eff) < 2:
        return None

    if -1 in uniq:
        mask = (yhat != -1)
        if mask.sum() < 3:
            return None
        try:
            return float(davies_bouldin_score(X[mask], yhat[mask]))
        except Exception:
            return None

    try:
        return float(davies_bouldin_score(X, yhat))
    except Exception:
        return None


def safe_ari(y_true_str: np.ndarray | None, yhat: np.ndarray):
    if y_true_str is None:
        return None
    try:
        y_true = encode_str_labels(y_true_str)
        return float(adjusted_rand_score(y_true, yhat))
    except Exception:
        return None


def conservative_score(sil, dbi, ari, noise_frac):
    """
    Conservative: penalize high DBI and high noise fraction.
    """
    sil_v = sil if sil is not None else -1.0
    dbi_v = dbi if dbi is not None else 10.0
    ari_v = ari if ari is not None else 0.0
    nf = float(noise_frac)
    return float(sil_v) + float(ari_v) - 0.2 * float(dbi_v) - 0.8 * nf


def load_repr(path: str) -> np.ndarray:
    X = np.load(path).astype(np.float32)
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    return X


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", type=str, default="data/fma_manifest_combined_text_only_clean.csv")
    ap.add_argument("--standardize", action="store_true")

    ap.add_argument("--vae_latents", type=str, default="data/vae_mm_latents_mu.npy")
    ap.add_argument("--vae_ids", type=str, default="data/vae_mm_latents_track_ids.npy")

    ap.add_argument("--mel_x", type=str, default="data/audio_cnn_mel_X.npy")
    ap.add_argument("--mel_ids", type=str, default="data/audio_cnn_mel_track_ids.npy")

    ap.add_argument("--lyrics_emb", type=str, default="data/lyrics_embeddings.npy")
    ap.add_argument("--lyrics_ids", type=str, default="data/lyrics_track_ids.npy")

    ap.add_argument("--k_list", type=str, default="4,5,6,7,8")
    ap.add_argument("--eps_list", type=str, default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--min_samples_list", type=str, default="3,5,8")

    ap.add_argument("--out_csv", type=str, default="results/medium_full_sweep_metrics.csv")

    args = ap.parse_args()

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_map, label_col = load_label_map(args.manifest)
    if label_map is None:
        print("Note: No genre/genre_top labels found; ARI will be None.")
    else:
        print(f"Using label column: {label_col}")

    k_list = [int(x.strip()) for x in args.k_list.split(",") if x.strip()]
    eps_list = [float(x.strip()) for x in args.eps_list.split(",") if x.strip()]
    ms_list = [int(x.strip()) for x in args.min_samples_list.split(",") if x.strip()]

    reps = [
        ("vae_mm_latents", args.vae_latents, args.vae_ids),
        ("baseline_mel_flat", args.mel_x, args.mel_ids),
        ("baseline_lyrics_only", args.lyrics_emb, args.lyrics_ids),
    ]

    rows = []

    for rep_name, x_path, id_path in reps:
        X = load_repr(x_path)
        ids = np.load(id_path).astype(np.int64)
        y_true = labels_for_ids(label_map, ids)

        X_use = X
        if args.standardize:
            X_use = StandardScaler().fit_transform(X_use)

        # ----- KMeans / Agglomerative sweep -----
        for k in k_list:
            # KMeans
            yhat = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X_use)
            sil = safe_silhouette(X_use, yhat)
            dbi = safe_dbi(X_use, yhat)
            ari = safe_ari(y_true, yhat)
            rows.append({
                "representation": rep_name,
                "algo": "kmeans",
                "params": f"k={k}",
                "n_clusters_found": len(np.unique(yhat)),
                "n_noise": 0,
                "noise_frac": 0.0,
                "silhouette": sil,
                "davies_bouldin": dbi,
                "ari": ari,
                "score": conservative_score(sil, dbi, ari, 0.0),
            })

            # Agglomerative
            yhat = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_use)
            sil = safe_silhouette(X_use, yhat)
            dbi = safe_dbi(X_use, yhat)
            ari = safe_ari(y_true, yhat)
            rows.append({
                "representation": rep_name,
                "algo": "agglomerative",
                "params": f"k={k},ward",
                "n_clusters_found": len(np.unique(yhat)),
                "n_noise": 0,
                "noise_frac": 0.0,
                "silhouette": sil,
                "davies_bouldin": dbi,
                "ari": ari,
                "score": conservative_score(sil, dbi, ari, 0.0),
            })

        # dbscan sweep
        for eps in eps_list:
            for ms in ms_list:
                yhat = DBSCAN(eps=eps, min_samples=ms).fit_predict(X_use)
                uniq = np.unique(yhat)
                n_noise = int(np.sum(yhat == -1)) if -1 in uniq else 0
                n_found = len([u for u in uniq.tolist() if u != -1])
                noise_frac = float(n_noise) / float(len(yhat))

                sil = safe_silhouette(X_use, yhat)
                dbi = safe_dbi(X_use, yhat)
                ari = safe_ari(y_true, yhat)

                rows.append({
                    "representation": rep_name,
                    "algo": "dbscan",
                    "params": f"eps={eps},min={ms}",
                    "n_clusters_found": n_found,
                    "n_noise": n_noise,
                    "noise_frac": noise_frac,
                    "silhouette": sil,
                    "davies_bouldin": dbi,
                    "ari": ari,
                    "score": conservative_score(sil, dbi, ari, noise_frac),
                })

        print(f"Done sweep for: {rep_name}  X={X_use.shape}")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print("\nWrote:", out_path)

    # Best per representation
    best_rep = df.sort_values("score", ascending=False).groupby("representation").head(1)
    best_rep_path = out_path.parent / "medium_full_sweep_best_by_representation.csv"
    best_rep.to_csv(best_rep_path, index=False)

    # Best overall
    best_all = df.sort_values("score", ascending=False).head(20)
    best_all_path = out_path.parent / "medium_full_sweep_best_overall.csv"
    best_all.to_csv(best_all_path, index=False)

    print("Wrote:", best_rep_path)
    print("Wrote:", best_all_path)

    #short preview
    cols = ["representation","algo","params","n_clusters_found","noise_frac","silhouette","davies_bouldin","ari","score"]
    print("\nTop 12 overall (conservative score):")
    print(best_all[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
