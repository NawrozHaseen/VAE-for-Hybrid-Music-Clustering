from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="results/medium_full_sweep_metrics.csv")
    ap.add_argument("--out_dir", type=str, default="results/report_medium")
    ap.add_argument("--max_noise", type=float, default=0.30,
                    help="Filter out DBSCAN runs with noise_frac > this. (0.30 recommended)")
    ap.add_argument("--min_clusters", type=int, default=2,
                    help="Require at least this many clusters (excluding noise)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    # Replace NaNs to avoid plot crashes
    for c in ["silhouette", "davies_bouldin", "ari", "noise_frac"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # For kmeans/agglomerative, noise_frac is 0 anyway.
    filt = df.copy()

    # Enforce minimum clusters for all
    filt = filt[filt["n_clusters_found"] >= args.min_clusters].copy()

    # Enforce noise constraint for DBSCAN
    is_db = filt["algo"] == "dbscan"
    filt = pd.concat([
        filt[~is_db],
        filt[is_db & (filt["noise_frac"] <= args.max_noise)]
    ], ignore_index=True)

    # Sort by score descending (already in csv)
    filt = filt.sort_values("score", ascending=False).reset_index(drop=True)

    # Save filtered best tables
    best_filtered_path = out_dir / "best_filtered.csv"
    filt.to_csv(best_filtered_path, index=False)

    best_by_rep = filt.groupby("representation").head(1).copy()
    best_by_rep_path = out_dir / "best_filtered_by_representation.csv"
    best_by_rep.to_csv(best_by_rep_path, index=False)

    print("\nSaved:")
    print(" ", best_filtered_path)
    print(" ", best_by_rep_path)

    print(f"\n=== BEST PER REPRESENTATION (noise<= {args.max_noise}, clusters>= {args.min_clusters}) ===")
    cols = ["representation","algo","params","n_clusters_found","noise_frac","silhouette","davies_bouldin","ari","score"]
    print(best_by_rep[cols].to_string(index=False))

    def plot_metric(metric: str, fname: str, title: str):
        d = filt.dropna(subset=[metric]).copy()
        if len(d) == 0:
            print(f"Skipping plot for {metric}: no data after filtering.")
            return

        # Pick top run per (representation, algo) to keep plot readable
        d2 = d.sort_values("score", ascending=False).groupby(["representation","algo"]).head(1)

        labels = (d2["representation"] + " | " + d2["algo"] + " | " + d2["params"]).tolist()
        vals = d2[metric].tolist()

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(vals)), vals)
        plt.xticks(range(len(vals)), labels, rotation=30, ha="right")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=220)
        plt.close()

    plot_metric("silhouette", "plot_silhouette.png", "Best (filtered) Silhouette by Representation/Algorithm")
    plot_metric("davies_bouldin", "plot_davies_bouldin.png", "Best (filtered) Davies–Bouldin (lower is better)")
    plot_metric("ari", "plot_ari.png", "Best (filtered) Adjusted Rand Index (ARI)")

    #dbscan diagnostics noise vse eps, clustering vs eps

    db = df[df["algo"] == "dbscan"].copy()
    if len(db) > 0:
        # parse eps from params
        def parse_eps(p: str):
            # params like "eps=0.6,min=5"
            try:
                part = p.split(",")[0]
                return float(part.split("=")[1])
            except Exception:
                return np.nan

        def parse_min(p: str):
            try:
                part = p.split(",")[1]
                return int(part.split("=")[1])
            except Exception:
                return np.nan

        db["eps"] = db["params"].apply(parse_eps)
        db["min_samples"] = db["params"].apply(parse_min)

        for rep, g in db.groupby("representation"):
            g = g.dropna(subset=["eps"]).copy()
            if len(g) == 0:
                continue

            # Use min_samples=5 line if exists, else the smallest min_samples line
            if (g["min_samples"] == 5).any():
                g2 = g[g["min_samples"] == 5].copy()
                ms_used = 5
            else:
                ms_used = int(g["min_samples"].min())
                g2 = g[g["min_samples"] == ms_used].copy()

            g2 = g2.sort_values("eps")

            # noise vs eps
            plt.figure(figsize=(8, 5))
            plt.plot(g2["eps"], g2["noise_frac"], marker="o")
            plt.xlabel("eps")
            plt.ylabel("noise_frac")
            plt.title(f"DBSCAN noise fraction vs eps ({rep}, min_samples={ms_used})")
            plt.tight_layout()
            plt.savefig(out_dir / f"dbscan_noise_vs_eps_{rep}.png", dpi=220)
            plt.close()

            # clusters vs eps
            plt.figure(figsize=(8, 5))
            plt.plot(g2["eps"], g2["n_clusters_found"], marker="o")
            plt.xlabel("eps")
            plt.ylabel("clusters_found (excluding noise)")
            plt.title(f"DBSCAN clusters found vs eps ({rep}, min_samples={ms_used})")
            plt.tight_layout()
            plt.savefig(out_dir / f"dbscan_clusters_vs_eps_{rep}.png", dpi=220)
            plt.close()

        print("\nWrote DBSCAN diagnostic plots per representation.")

    print("\nWrote report plots to:", out_dir)


if __name__ == "__main__":
    main()
