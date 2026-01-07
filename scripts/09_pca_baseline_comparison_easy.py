from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler


@dataclass
class Config:
    vae_out_dir: str
    kmeans_out_dir: str
    out_dir: str
    k: int
    seed: int
    n_init: int


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Baseline PCA+KMeans and metric comparison vs VAE+KMeans.")
    p.add_argument("--vae_out_dir", type=str, default="results/vae_basic")
    p.add_argument("--kmeans_out_dir", type=str, default="results/kmeans_vae")
    p.add_argument("--out_dir", type=str, default="results/compare_metrics")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_init", type=int, default=20)
    a = p.parse_args()
    return Config(
        vae_out_dir=a.vae_out_dir,
        kmeans_out_dir=a.kmeans_out_dir,
        out_dir=a.out_dir,
        k=a.k,
        seed=a.seed,
        n_init=a.n_init,
    )


def safe_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Calculate clustering metrics with error handling."""
    out: Dict[str, float] = {}
    try:
        out["silhouette"] = float(silhouette_score(X, labels))
    except Exception:
        out["silhouette"] = float("nan")
    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    except Exception:
        out["calinski_harabasz"] = float("nan")
    return out


def load_original_mfcc_features(vae_out_dir: Path) -> Optional[np.ndarray]:
    """
    Try to load original MFCC features from cache.
    """
    cache_path = vae_out_dir / "mfcc_features_cache.npy"
    if cache_path.exists():
        try:
            blob = np.load(cache_path, allow_pickle=True).item()
            return blob["X"]  # Original raw MFCC features
        except Exception:
            return None
    return None


def run_pca_kmeans(X: np.ndarray, k: int, latent_dim: int, seed: int, n_init: int) -> Tuple[np.ndarray, np.ndarray, PCA]:
    """Run PCA + KMeans and return labels, PCA components, and PCA object."""
    # Standardize
    X_scaled = StandardScaler().fit_transform(X)
    
    # PCA to target dimension
    pca = PCA(n_components=min(latent_dim, X_scaled.shape[1]), random_state=seed)
    X_pca = pca.fit_transform(X_scaled)
    
    # KMeans
    km = KMeans(n_clusters=k, random_state=seed, n_init=n_init)
    labels = km.fit_predict(X_pca)
    
    return labels, X_pca, pca


def create_pca_variance_plot(pca: PCA, latent_dim: int, plot_path: Path, title: str):
    """Create and save PCA variance plot."""
    plt.figure(figsize=(10, 6))
    
    # Cumulative variance
    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    
    # Bar plot for individual variances
    plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
            pca.explained_variance_ratio_, 
            alpha=0.5, 
            label='Individual explained variance')
    
    # Line plot for cumulative variance
    plt.plot(range(1, len(cumulative_var) + 1), 
             cumulative_var, 
             'ro-', 
             linewidth=2, 
             markersize=6,
             label='Cumulative explained variance')
    
    # Highlight target dimension
    if latent_dim <= len(cumulative_var):
        plt.axvline(x=latent_dim, color='g', linestyle='--', alpha=0.7, 
                    label=f'Target ({latent_dim}D)')
        plt.axhline(y=cumulative_var[latent_dim-1], 
                    color='g', linestyle='--', alpha=0.7)
    
    plt.title(title)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    plt.close()
    
    return float(cumulative_var[latent_dim-1]) if latent_dim <= len(cumulative_var) else float(cumulative_var[-1])


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    else:
        return obj


def main() -> None:
    cfg = parse_args()
    vae_out = Path(cfg.vae_out_dir)
    km_out = Path(cfg.kmeans_out_dir)
    out_dir = Path(cfg.out_dir)
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("CLUSTERING COMPARISON: VAE vs PCA")
    print("="*60)
    
    # Load VAE outputs
    print("\nLoading VAE outputs...")
    Z = np.load(vae_out / "latent_mu.npy").astype(np.float32)
    labels_vae = np.load(km_out / "labels_vae_kmeans.npy").astype(np.int64)
    
    print(f"  VAE latents shape: {Z.shape}")
    print(f"  VAE labels shape: {labels_vae.shape}")
    
    # Standardize VAE latents
    Z_scaled = StandardScaler().fit_transform(Z)
    
    # Calculate metrics for VAE
    vae_metrics = safe_metrics(Z_scaled, labels_vae)
    vae_sil = vae_metrics["silhouette"]
    
    # Get latent dimension from VAE
    latent_dim = Z.shape[1]
    
    #vae vs pca on mfcc feaures
    print("\n" + "="*60)
    print("COMPARISON 1: VAE vs PCA on MFCC features")
    print("="*60)
    
    # Load original MFCC features
    X_mfcc = load_original_mfcc_features(vae_out)
    
    if X_mfcc is None:
        print("  Warning: Could not load original MFCC features from cache.")
        print("  Run 06_train_basic_vae.py with --cache_features flag first.")
        print("  Skipping comparison with MFCC features.")
        comparison1_results = None
        mfcc_available = False
    else:
        print(f"  MFCC features shape: {X_mfcc.shape}")
        
        # Run PCA + KMeans on MFCC features
        print(f"  Running PCA on {X_mfcc.shape[1]}D MFCC -> {latent_dim}D...")
        labels_pca_mfcc, X_pca_mfcc, pca_mfcc = run_pca_kmeans(
            X=X_mfcc, 
            k=cfg.k, 
            latent_dim=latent_dim, 
            seed=cfg.seed, 
            n_init=cfg.n_init
        )
        
        # Save labels
        np.save(out_dir / "labels_pca_mfcc.npy", labels_pca_mfcc)
        
        # Calculate metrics for PCA on MFCC
        pca_mfcc_metrics = safe_metrics(X_pca_mfcc, labels_pca_mfcc)
        
        # Create PCA variance plot
        explained_mfcc = create_pca_variance_plot(
            pca_mfcc, 
            latent_dim, 
            plots_dir / "pca_variance_mfcc.png",
            f"PCA Explained Variance Ratio (MFCC features -> {latent_dim}D)"
        )
        
        comparison1_results = {
            "labels": labels_pca_mfcc,
            "metrics": pca_mfcc_metrics,
            "explained_variance": explained_mfcc,
        }
        mfcc_available = True
    
    #vae vs pca on vae latents
    print("\n" + "="*60)
    print("COMPARISON 2: VAE vs PCA on VAE latents")
    print("="*60)
    
    # Run PCA + KMeans on VAE latents
    print(f"  Running PCA on {Z.shape[1]}D VAE latents -> {latent_dim}D...")
    labels_pca_latents, X_pca_latents, pca_latents = run_pca_kmeans(
        X=Z, 
        k=cfg.k, 
        latent_dim=latent_dim, 
        seed=cfg.seed, 
        n_init=cfg.n_init
    )
    
    # Save labels
    np.save(out_dir / "labels_pca_latents.npy", labels_pca_latents)
    
    # Calculate metrics for PCA on VAE latents
    pca_latents_metrics = safe_metrics(X_pca_latents, labels_pca_latents)
    
    # Create PCA variance plot
    explained_latents = create_pca_variance_plot(
        pca_latents, 
        latent_dim, 
        plots_dir / "pca_variance_latents.png",
        f"PCA Explained Variance Ratio (VAE latents -> {latent_dim}D)"
    )
    
    comparison2_results = {
        "labels": labels_pca_latents,
        "metrics": pca_latents_metrics,
        "explained_variance": explained_latents,
    }
    
    #results table
    print("\n" + "="*60)
    print("COMPARISON RESULTS")
    print("="*60)
    
    # Build results table
    rows = []
    
    # VAE results
    rows.append({
        "method": "VAE+KMeans",
        "input": "VAE latents",
        "input_dim": int(Z.shape[1]),
        "k": cfg.k,
        "silhouette": float(vae_metrics["silhouette"]),
        "calinski_harabasz": float(vae_metrics["calinski_harabasz"]),
        "pca_variance": float("nan"),
    })
    
    # PCA on MFCC features (if available)
    if mfcc_available:
        rows.append({
            "method": f"PCA({latent_dim})+KMeans",
            "input": f"MFCC features ({int(X_mfcc.shape[1])}D)",
            "input_dim": int(latent_dim),
            "k": cfg.k,
            "silhouette": float(comparison1_results["metrics"]["silhouette"]),
            "calinski_harabasz": float(comparison1_results["metrics"]["calinski_harabasz"]),
            "pca_variance": float(comparison1_results["explained_variance"]),
        })
    
    # PCA on VAE latents
    rows.append({
        "method": f"PCA({latent_dim})+KMeans",
        "input": f"VAE latents ({int(Z.shape[1])}D)",
        "input_dim": int(latent_dim),
        "k": cfg.k,
        "silhouette": float(comparison2_results["metrics"]["silhouette"]),
        "calinski_harabasz": float(comparison2_results["metrics"]["calinski_harabasz"]),
        "pca_variance": float(comparison2_results["explained_variance"]),
    })
    
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics.csv", index=False)
    
    #display rsults
    if mfcc_available:
        print("\nCOMPARISON RESULTS (VAE vs PCA on MFCC features):")
        print("-" * 60)
        
        # Filter for VAE and PCA on MFCC
        mfcc_rows = [rows[0], rows[1]]  # VAE and PCA on MFCC
        mfcc_df = pd.DataFrame(mfcc_rows)
        print(mfcc_df.to_string(index=False))
        
        # Show interpretation for MFCC comparison
        pca_mfcc_sil = comparison1_results["metrics"]["silhouette"]
        if not np.isnan(vae_sil) and not np.isnan(pca_mfcc_sil):
            print(f"\nInterpretation (MFCC comparison):")
            if vae_sil > pca_mfcc_sil:
                diff = ((vae_sil - pca_mfcc_sil) / pca_mfcc_sil) * 100
                print(f"  • VAE outperforms PCA by {diff:+.1f}% on Silhouette Score")
                print(f"  • VAE's non-linear compression is better for clustering")
            elif pca_mfcc_sil > vae_sil:
                diff = ((pca_mfcc_sil - vae_sil) / vae_sil) * 100
                print(f"  • PCA outperforms VAE by {diff:+.1f}% on Silhouette Score")
                print(f"  • Linear PCA is sufficient for this data")
            else:
                print(f"  • VAE and PCA perform similarly")
            print(f"  • PCA captures {comparison1_results['explained_variance']:.1%} of MFCC variance")
    else:
        print("\nCOMPARISON RESULTS (VAE vs PCA on MFCC features):")
        print("-" * 60)
        print("  MFCC comparison not available (run with --cache_features)")
    
    print("\n" + "="*60)
    print("COMPARISON RESULTS (VAE vs PCA on VAE latents):")
    print("-" * 60)
    
    # Filter for VAE and PCA on VAE latents
    latents_rows = [rows[0], rows[-1]]  # VAE and PCA on VAE latents
    latents_df = pd.DataFrame(latents_rows)
    print(latents_df.to_string(index=False))
    
    # Show interpretation for VAE latents comparison
    print(f"\nInterpretation (VAE latents comparison):")
    pca_latents_sil = comparison2_results["metrics"]["silhouette"]
    if not np.isnan(vae_sil) and not np.isnan(pca_latents_sil):
        identical = abs(vae_sil - pca_latents_sil) < 0.0001
        print(f"  • Scores are {'identical' if identical else 'similar'}")
        print(f"  • PCA captures {comparison2_results['explained_variance']:.1%} of VAE latent variance")
        if comparison2_results["explained_variance"] > 0.99:
            print(f"  • PCA({latent_dim}) on {latent_dim}D data ≈ identity transformation")
    
    #save report
    report = {
        "config": asdict(cfg),
        "data_info": {
            "vae_latent_shape": [int(dim) for dim in Z.shape],
            "mfcc_features_available": mfcc_available,
            "mfcc_features_shape": [int(dim) for dim in X_mfcc.shape] if mfcc_available else None,
        },
        "comparisons": {
            "vae_kmeans": {
                "metrics": {
                    "silhouette": float(vae_metrics["silhouette"]),
                    "calinski_harabasz": float(vae_metrics["calinski_harabasz"])
                },
                "note": "Baseline VAE method"
            },
            "pca_mfcc_kmeans": {
                "metrics": {
                    "silhouette": float(comparison1_results["metrics"]["silhouette"]) if mfcc_available else None,
                    "calinski_harabasz": float(comparison1_results["metrics"]["calinski_harabasz"]) if mfcc_available else None
                } if mfcc_available else None,
                "explained_variance": float(comparison1_results["explained_variance"]) if mfcc_available else None,
                "note": "PCA applied to original MFCC features"
            },
            "pca_latents_kmeans": {
                "metrics": {
                    "silhouette": float(comparison2_results["metrics"]["silhouette"]),
                    "calinski_harabasz": float(comparison2_results["metrics"]["calinski_harabasz"])
                },
                "explained_variance": float(comparison2_results["explained_variance"]),
                "note": "PCA applied to VAE latents (shows why original results were identical)"
            }
        },
        "outputs": {
            "metrics_csv": str(out_dir / "metrics.csv"),
            "labels_pca_mfcc": str(out_dir / "labels_pca_mfcc.npy") if mfcc_available else None,
            "labels_pca_latents": str(out_dir / "labels_pca_latents.npy"),
            "plot_pca_mfcc": str(plots_dir / "pca_variance_mfcc.png") if mfcc_available else None,
            "plot_pca_latents": str(plots_dir / "pca_variance_latents.png"),
        }
    }
    
    # Convert to serializable format
    report = convert_to_serializable(report)
    
    with open(out_dir / "metrics_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    
    #outputs log
    print("\n" + "="*60)
    print("OUTPUTS SAVED:")
    print("="*60)
    print(f"Metrics: {out_dir / 'metrics.csv'}")
    print(f"Report: {out_dir / 'metrics_report.json'}")
    
    if mfcc_available:
        print(f"Labels (PCA on MFCC): {out_dir / 'labels_pca_mfcc.npy'}")
        print(f"Plot (PCA on MFCC): {plots_dir / 'pca_variance_mfcc.png'}")
    
    print(f"Labels (PCA on VAE latents): {out_dir / 'labels_pca_latents.npy'}")
    print(f"Plot (PCA on VAE latents): {plots_dir / 'pca_variance_latents.png'}")
    
    print("\n" + "="*60)
    print("KEY FINDINGS:")
    print("="*60)
    
    if mfcc_available:
        vae_sil = vae_metrics["silhouette"]
        pca_mfcc_sil = comparison1_results["metrics"]["silhouette"]
        
        if vae_sil > pca_mfcc_sil:
            diff = ((vae_sil - pca_mfcc_sil) / pca_mfcc_sil) * 100
            print(f"-->VAE outperforms PCA by {diff:+.1f}% on MFCC features")
            print(f"  → VAE's non-linear dimensionality reduction is more effective")
        elif pca_mfcc_sil > vae_sil:
            diff = ((pca_mfcc_sil - vae_sil) / vae_sil) * 100
            print(f"--> PCA outperforms VAE by {diff:+.1f}% on MFCC features")
            print(f"  → Linear compression (PCA) is sufficient for this data")
        else:
            print("✓ VAE and PCA perform similarly on MFCC features")
    
    print("✓ Original identical results explained: PCA on VAE latents is redundant")
    print(f"  → PCA captures {comparison2_results['explained_variance']:.1%} of VAE latent variance")


if __name__ == "__main__":
    main()