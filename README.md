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


pip install -r requirements.txt

# Directory Structure

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

# Setup Instructions
Create and activate a Virtual Environment:

python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

# Install Dependencies:

Install the required Python packages:

pip install -r requirements.txt

# Download and Prepare Data:

Download the FMA dataset and other required data using:


python scripts/00_download_fma.py
This will download the audio and metadata files, and prepare the data/fma_manifest_3k_5genres.csv manifest.

# Data Files
The following files are used in the project:

Audio Files: Located in data/fma_small/ (e.g., 000/000002.mp3)

Metadata: Located in data/fma_metadata/ (e.g., tracks.csv, genres.csv)

Manifest: Located in data/fma_manifest_3k_6genres.csv

Lyrics Files: Located in data/lyrics/

# Training Models
To train the models, use the following scripts:

# Basic VAE:
Train a basic VAE model using:


python scripts/06_train_basic_vae_easy.py

Multimodal VAE (Audio + Lyrics):

# Train a multimodal VAE with audio and lyrics embeddings using:

python scripts/19_train_beta_cvae_multimodal_hard.py

# Clustering:
Run clustering (e.g., KMeans) on the learned latent representations:

python scripts/20_cluster_and_evaluate_hard.py

# Visualizations:
Visualize the clustering results:


python scripts/21_visualize_latent_and_distributions_hard.py

# Comparison with Baselines:
Compare the VAE-based models with baseline methods (PCA, Autoencoder):

python scripts/22_compare_with_baselines_hard.py

# Result Files
The project generates several important result files:

Clustering Metrics: Located in results/hard/baseline_comparison.csv

Model Latents: Located in data/hard/latents_mu.npy

Training Plots: Located in results/hard/plots/

training_curve.png: Shows training loss over epochs

recon_examples.png: Shows reconstruction examples

# Visualizations
The following visualizations are generated:

Latent Space 2D Projection: Using UMAP or t-SNE.

latent_by_cluster.png

latent_by_genre.png

latent_by_language.png

# Cluster Distributions:

cluster_dist_over_genres.png

cluster_dist_over_languages.png

# How to Use the Scripts
Each script comes with configurable options via command-line arguments. Below is a general guide to some key options:

Manifest: Specify the path to your music manifest CSV file.

Latents Path: Path to the latent features (e.g., data/hard/latents_mu.npy).

Tag: An optional tag for saving outputs with a suffix.

Example command:

python scripts/20_cluster_and_evaluate_hard.py --latents_path data/hard/latents_mu.npy --k 5 --tag "experiment_1"

# Additional Notes
Training and Evaluation: Make sure to train your models first (VAE, Beta-VAE, or CVAE) before performing clustering.

Known Issues: If UMAP or t-SNE is not installed, the projection step will default to PCA.

Future Work: You may want to explore hyperparameter tuning for clustering and model training.
