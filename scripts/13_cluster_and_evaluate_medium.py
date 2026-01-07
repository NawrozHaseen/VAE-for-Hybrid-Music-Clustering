from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler


#label handling

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

    # map track_id -> label (string)
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


#metrics 

def safe_silhouette(X: np.ndarray, yhat: np.ndarray):
    uniq = np.unique(yhat)
    # Must have at least 2 clusters (excluding noise label -1)
    uniq_eff = [u for u in uniq.tolist() if u != -1]
    if len(uniq_eff) < 2:
        return None
    # For silhouette, drop noise points if DBSCAN
    if -1 in uniq:
        mask = yhat != -1
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


def safe_db_index(X: np.ndarray, yhat: np.ndarray):
    uniq = np.unique(yhat)
    uniq_eff = [u for u in uniq.tolist() if u != -1]
    if len(uniq_eff) < 2:
        return None
    # Davies–Bouldin does not support noise nicely; drop noise points
    if -1 in uniq:
        mask = yhat != -1
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
        # For DBSCAN, keep noise; ARI supports it (noise becomes its own label -1)
        return float(adjusted_rand_score(y_true, yhat))
    except Exception:
        return None


#clustering

def run_cluster_suite(X: np.ndarray, y_true_str: np.ndarray | None, n_clusters: int, tag: str):
    rows = []

    # KMeans
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    y_km = km.fit_predict(X)
    rows.append(("kmeans", f"k={n_clusters}", y_km))

    # Agglomerative (Ward)
    ag = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    y_ag = ag.fit_predict(X)
    rows.append(("agglomerative", f"k={n_clusters},ward", y_ag))

    # DBSCAN sweep
    for eps in [0.4, 0.6, 0.8, 1.0, 1.2]:
        db = DBSCAN(eps=eps, min_samples=5)
        y_db = db.fit_predict(X)
        rows.append(("dbscan", f"eps={eps},min=5", y_db))

    out = []
    for algo, params, yhat in rows:
        uniq = set(yhat.tolist())
        n_noise = int(np.sum(yhat == -1)) if -1 in uniq else 0
        n_found = len([u for u in uniq if u != -1])

        out.append({
            "representation": tag,
            "algo": algo,
            "params": params,
            "n_clusters_found": n_found,
            "n_noise": n_noise,
            "silhouette": safe_silhouette(X, yhat),
            "davies_bouldin": safe_db_index(X, yhat),
            "ari": safe_ari(y_true_str, yhat),
        })
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", type=str, default="data/fma_manifest_combined_text_only_clean.csv")
    ap.add_argument("--n_clusters", type=int, default=6)
    ap.add_argument("--out_csv", type=str, default="results/medium_clustering_metrics_all.csv")

    # Representations
    ap.add_argument("--vae_latents", type=str, default="data/vae_mm_latents_mu.npy")
    ap.add_argument("--vae_ids", type=str, default="data/vae_mm_latents_track_ids.npy")

    ap.add_argument("--mel_x", type=str, default="data/audio_cnn_mel_X.npy")
    ap.add_argument("--mel_ids", type=str, default="data/audio_cnn_mel_track_ids.npy")

    ap.add_argument("--lyrics_emb", type=str, default="data/lyrics_embeddings.npy")
    ap.add_argument("--lyrics_ids", type=str, default="data/lyrics_track_ids.npy")

    ap.add_argument("--standardize", action="store_true", help="Recommended (uses StandardScaler).")
    ap.add_argument("--pca_dim", type=int, default=0,
                    help="Optional PCA reduction for huge baseline vectors. 0 = no PCA (recommended to keep simple).")

    args = ap.parse_args()

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)

    label_map, label_col = load_label_map(args.manifest)
    if label_map is None:
        print("Note: No genre labels found in manifest (genre/genre_top). ARI will be None.")
    else:
        print(f"Using labels from manifest column: {label_col}")

    all_rows = []

    #vae latents
    Z = np.load(args.vae_latents).astype(np.float32)
    z_ids = np.load(args.vae_ids).astype(np.int64)
    y_true_z = labels_for_ids(label_map, z_ids)

    Xz = Z
    if args.standardize:
        Xz = StandardScaler().fit_transform(Xz)

    all_rows += run_cluster_suite(Xz, y_true_z, args.n_clusters, tag="vae_mm_latents")

    #baseline mel
    mel = np.load(args.mel_x).astype(np.float32)  # (N,1,F,T)
    mel_ids = np.load(args.mel_ids).astype(np.int64)
    y_true_m = labels_for_ids(label_map, mel_ids)

    Xm = mel.reshape(mel.shape[0], -1)  # flatten
    if args.standardize:
        Xm = StandardScaler().fit_transform(Xm)

    all_rows += run_cluster_suite(Xm, y_true_m, args.n_clusters, tag="baseline_mel_flat")

    #lyrics-only baseline on m tracks
    lyr = np.load(args.lyrics_emb).astype(np.float32)
    lyr_ids = np.load(args.lyrics_ids).astype(np.int64)
    y_true_l = labels_for_ids(label_map, lyr_ids)

    Xl = lyr
    if args.standardize:
        Xl = StandardScaler().fit_transform(Xl)

    all_rows += run_cluster_suite(Xl, y_true_l, args.n_clusters, tag="baseline_lyrics_only")

    out_df = pd.DataFrame(all_rows)
    out_df.to_csv(args.out_csv, index=False)

    print("\nWrote:", args.out_csv)

    # Print a quick ranking view
    def score_row(r):
        sil = r["silhouette"] if pd.notna(r["silhouette"]) else -1.0
        dbi = r["davies_bouldin"] if pd.notna(r["davies_bouldin"]) else 10.0
        ari = r["ari"] if pd.notna(r["ari"]) else 0.0
        return float(sil) + float(ari) - 0.2 * float(dbi)

    tmp = out_df.copy()
    tmp["score"] = tmp.apply(score_row, axis=1)
    print("\nTop results (heuristic score):")
    print(tmp.sort_values("score", ascending=False).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
