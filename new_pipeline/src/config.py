import os
from pathlib import Path

import torch

# ── Environment detection ──────────────────────────────────────────────
# Set ECON_ENV=mac (default) for local development on Mac
# Set ECON_ENV=kaggle for Kaggle notebooks (different file paths)
ENV = os.getenv("ECON_ENV", "mac")

# ── Project root (always defined, regardless of environment) ───────────
# PROJECT_ROOT is the directory containing new_pipeline/ — used by
# build_csv.py and other modules that need to resolve file paths.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── Device detection ───────────────────────────────────────────────────
# Priority: MPS (Apple GPU) > CUDA (NVIDIA GPU) > CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"[config] Environment: {ENV}  Device: {DEVICE}")


# ── Paths ──────────────────────────────────────────────────────────────
if ENV == "kaggle":
    # ── Kaggle paths ────────────────────────────────────────────────
    # Outputs (models, results, figures, cache, CSVs) go to /kaggle/working/data/
    # — this is writeable and persists as Kaggle "Output" after the run.
    KAGGLE_WORKING_DIR = Path("/kaggle/working/data")
    DATA_DIR = KAGGLE_WORKING_DIR
    MODELS_DIR = KAGGLE_WORKING_DIR / "models"
    RESULTS_DIR = KAGGLE_WORKING_DIR / "results"
    FIGURES_DIR = KAGGLE_WORKING_DIR / "figures"
    CACHE_DIR = KAGGLE_WORKING_DIR / "cache"

    # Inputs (pre-existing cache .pt files, CSVs) come from a Kaggle Dataset
    # you upload. Set ECON_KAGGLE_DATASET to the name of your Kaggle Dataset.
    KAGGLE_DATASET_NAME = os.getenv("ECON_KAGGLE_DATASET", "economic-satellite-data")
    KAGGLE_INPUT_DIR = Path(f"/kaggle/input/{KAGGLE_DATASET_NAME}")

    # Image directory — on Kaggle, images are bundled in the input dataset
    IMAGE_DIR = KAGGLE_INPUT_DIR / "raw" / "images"
else:
    # ── Mac (local) paths — relative to project root ────────────────
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"
    IMAGE_DIR = RAW_DIR / "images"
    CACHE_DIR = DATA_DIR / "cache"
    RESULTS_DIR = DATA_DIR / "results"
    FIGURES_DIR = DATA_DIR / "figures"
    MODELS_DIR = DATA_DIR / "models"

# Create output directories if they don't exist (only on local Mac, not on Kaggle)
if ENV != "kaggle":
    for d in [IMAGE_DIR, CACHE_DIR, RESULTS_DIR, FIGURES_DIR, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


# ── DataLoader defaults ────────────────────────────────────────────────
# Override with ECON_NUM_WORKERS env var if needed
# Mac can handle more workers; Kaggle often limits to 2
if ENV == "kaggle":
    _DEFAULT_WORKERS = 2
else:
    _DEFAULT_WORKERS = 4

NUM_WORKERS = int(os.getenv("ECON_NUM_WORKERS", str(_DEFAULT_WORKERS)))
