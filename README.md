# VAE-for-Hybrid-Music-Clustering
This project aims to perform hybrid music clustering using Variational Autoencoders (VAE) and other clustering techniques like KMeans, Agglomerative, and DBSCAN. The pipeline processes audio and lyrics data, extracts features, trains models, and evaluates the results.
## Prerequisites

Make sure you have the following installed:

- **Python** (Version 3.9+ recommended)
- **Required Libraries**:
  - `librosa`: For audio feature extraction
  - `torch`: For model training (VAE, CVAE, Beta-VAE)
  - `sentence-transformers`: For lyrics embeddings
  - `scikit-learn`: For clustering algorithms and metrics
  - `umap-learn` (optional): For dimensionality reduction
  - `tqdm`: For progress bars
  - `matplotlib`: For visualizations

You can install the required dependencies by running:

``` bash
pip install -r requirements.txt
```
## Directory Structure

```bash
project/
├── data/
│   ├── hard/               # Processed data and features
│   ├── fma_metadata/       # FMA metadata (e.g., tracks.csv)
│   ├── fma_small/          # FMA audio files
│   └── fma_manifest_3k_5genres.csv  # Music manifest
├── models/                 # Saved model files (e.g., VAE)
├── results/                # Results and plots
├── scripts/                # Python scripts for each task
├── requirements.txt        # Required Python libraries
└── README.md               # Project description
```

## Setup Instructions
Create and activate a Virtual Environment:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
```

## Install Dependencies:

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Download and Prepare Data:

Download the FMA dataset and other required data using:

```bash
python scripts/00_download_fma.py
```
This will download the audio and metadata files, and prepare the data/fma_manifest_3k_6genres.csv manifest.

## Task Types Overview

The project is divided into three main task types: Easy, Medium, and Hard. These tasks represent different stages of data processing, feature extraction, model training, and evaluation. Below are the task scripts for each type and the corresponding data/results/model files generated.

## Easy Tasks (Scripts 00-09)

**Tasks**:

- `Data Download and Manifest Creation`: Download audio files, lyrics, transcriptions and metadata, and create a manifest.

- `Feature Extraction`: Extract basic audio features like MFCC and lyrics embeddings.

- `Basic VAE Training and Clustering`: Train a basic VAE model and perform clustering using the learned latents.

- `Script Range`: Scripts 00-09

**Data Files**:

- `Audio files`: data/fma_small/ (e.g., 000/000002.mp3)

- `Metadata`: data/fma_metadata/ (e.g., tracks.csv, genres.csv)

- `Manifest`: data/fma_manifest_3k_6genres.csv

- `Lyrics files`: data/lyrics/

**Results**:

`data/fma_manifest_3k_6genres.csv`: Manifest with audio and genre metadata

`data/fma_manifest_3k_6genres_lyrics.csv`: Updated manifest with lyrics

`results/kmeans_vae/labels_vae_kmeans.npy`: KMeans clustering labels

`results/viz_vae/plots/vae_tsne.png`: t-SNE visualization of clusters

`results/viz_vae/plots/vae_umap.png`: UMAP visualization of clusters

**Model**:

`results/vae_basic/vae_basic.pt`: Trained VAE model

## Medium Tasks (Scripts 10-17)

**Tasks**:

- `Feature Extraction`: Extract audio features for CNN-based models (e.g., log-mel spectrograms).

- `Lyrics Embedding`: Generate lyrics embeddings using Sentence-Transformers or fallback methods.

- `Multimodal VAE Training`: Train a multimodal VAE (combining audio and lyrics embeddings).

- `Clustering and Evaluation`: Perform clustering (e.g., KMeans, Agglomerative) on multimodal features and evaluate the results.

- `Script Range`: Scripts 10-17

**Data Files**:

`data/audio_cnn_mel_X.npy`: Log-mel spectrogram features (audio)

`data/lyrics_embeddings.npy`: Lyrics embeddings

`data/fma_manifest_combined_text_only_clean.csv`: Combined manifest

**Results**:

`results/medium_clustering_metrics_all.csv`: Clustering metrics for all representations

`results/report_medium/plot_silhouette.png`: Silhouette score plot

`results/report_medium/plot_davies_bouldin.png`: Davies–Bouldin index plot

`results/report_medium/plot_ari.png`: ARI plot

`results/report_medium/dbscan_noise_vs_eps_<rep>.png`: DBSCAN noise vs. eps plot

`results/report_medium/dbscan_clusters_vs_eps_<rep>.png`: DBSCAN clusters vs. eps plot

**Model**:

`models/vae_conv_mm_medium/vae_conv_mm_medium.pt`: Trained multimodal VAE model

## Hard Tasks (Scripts 18-22)

**Tasks**:

- `Multimodal Feature Preparation`: Prepare and extract multimodal features (audio + lyrics).

- `Multimodal VAE Training`: Train Beta-VAE or CVAE on fused multimodal features.

- `Clustering and Evaluation`: Perform clustering on latent vectors and evaluate metrics.

- `Visualizations`: Visualize the results and latent space using dimensionality reduction techniques (e.g., UMAP, t-SNE).

- `Baseline Comparison`: Compare the VAE-based models with baseline clustering methods (PCA, Autoencoder).

- `Script Range`: Scripts 18-22

**Data Files**:

`data/hard/audio_mfcc_stats.npy`: MFCC audio features

`data/hard/lyrics_emb.npy`: Lyrics embeddings

`data/hard/track_ids.npy`: Track IDs

`data/hard/genre_idx.npy`: Genre indices

`data/hard/lang_idx.npy`: Language indices

`data/hard/latents_mu.npy`: Latent vectors from trained models

**Results**:

`results/hard/baseline_comparison.csv`: Comparison of clustering methods (Beta VAE vs. Baselines)

`results/hard/plots/training_curve.png`: Training curve plot

`results/hard/plots/recon_examples.png`: Reconstruction examples plot

`results/hard/plots/baseline_bars.png`: Baseline comparison bar plot

`results/hard/cluster_labels_kmeans.npy`: Cluster labels from KMeans

`results/hard/cluster_composition_by_genre.csv`: Cluster composition table

`results/hard/plots/latent_by_cluster.png`: Latent space projection colored by KMeans clusters

`results/hard/plots/latent_by_genre.png`: Latent space projection colored by true genre

`results/hard/plots/latent_by_language.png`: Latent space projection colored by detected language

`results/hard/plots/cluster_dist_over_genres.png`: Cluster distribution over genres

`results/hard/plots/cluster_dist_over_languages.png`: Cluster distribution over languages

**Note**: 
There are other, model-specific (BETA VAE or CVAE) or hyperparameter specific (e.g. b4l16) csv and png files for each for further model and hyperparameter testing as well inside the `results/hard/` directory.

**Model**:

`models/hard/beta_vae_multimodal.pt`: Trained Beta-VAE model (if used. also used BY DEFAULT if not model specified in terminal)

`models/hard/cvae_multimodal.pt`: Trained CVAE model (if used)

## How to Use the Scripts

Each script comes with configurable options via command-line arguments. Below is a general guide to some key options:

- `Manifest`: Specify the path to your music manifest CSV file.

- `Latents Path`: Path to the latent features (e.g., data/hard/latents_mu.npy).

- `Tag`: An optional tag for saving outputs with a suffix.

Example commands:

```bash
python scripts/20_cluster_and_evaluate_hard.py # To run the file as-is using default values
```

```bash
python scripts/20_cluster_and_evaluate_hard.py --latents_path data/hard/latents_mu.npy --k 5 --tag "experiment_1"   # To tweak parameters
```

## Note:

For script 09, you need to re-run script 06 if you haven't to generate MFCC features as this is necessary to use MFCC features in their original form to use as comparison to show how the VAE model and baseline methods like PCA perform. To re-extract, simply use:

```bash
python scripts/06_train_basic_vae_easy.py
```

For scripts 19-22, if you want to use CVAE instead of the default BETA VAE used here, you need to specifically mention using:

```bash
python scripts/19_train_beta_cvae_multimodal_hard.py --use_cvae --tag "cvae"
```

or you wanted to use some hyperparameter tuning for the files, you can specify those parameters as well, for example:

```bash
python scripts/19_train_beta_cvae_multimodal_hard.py --use_cvae --tag "cvae" --beta 4.0 --latent_dim 16
```

or if you wanted to do hyperparameter tuning using the default BETA VAE model instead, use:

```bash
python scripts/19_train_beta_cvae_multimodal_hard.py --tag "beta" --beta 4.0 --latent_dim 16
```


## Additional Notes
- `Training and Evaluation`: Make sure to train your models first (VAE, Beta-VAE, or CVAE) before performing clustering.

- `Known Issues`: If UMAP or t-SNE is not installed, the projection step will default to PCA.

- `Important to Note`: If no model specified, BETA VAE will be used as model by default in scripts 19-22, 

- `Future Work`: You may want to explore hyperparameter tuning for clustering and model training.
