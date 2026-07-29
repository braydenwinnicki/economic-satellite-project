# New Pipeline — Instructions

## Overview

The `new_pipeline/` directory contains the complete ML pipeline for predicting median household income from satellite imagery. It supports:

- Single-tile (one image per census tract) and multi-tile (multiple images per tract)
- 4 model architectures: CNN, ResNet (frozen), ResNet (unfrozen layer 3), ResNet (unfrozen layer 4)
- MPS (Apple GPU), CUDA (NVIDIA GPU), and CPU
- Local Mac development and Kaggle notebooks with automatic path switching

---

## Directory Structure

```
new_pipeline/
├── INSTRUCTIONS.md              ← This file
├── run_experiment.py            ← Main entry point for training/evaluation
├── get_new_data.py              ← Download satellite imagery + census data
├── build_csv.py                 ← Build CSV from shapefile + census API
├── data/                        ← All data (created automatically)
│   ├── cache/                   ← Pre-computed .pt cache files
│   ├── models/                  ← Saved model weights (.pth)
│   ├── results/                 ← Evaluation results CSVs
│   ├── figures/                 ← Generated charts
│   └── raw/images/              ← Downloaded satellite images
├── src/
│   ├── config.py                ← Central config (paths, device, workers)
│   ├── train.py                 ← Training loop
│   ├── evaluate.py              ← Evaluation loop
│   ├── make_charts.py           ← Chart generation
│   ├── build_cache.py           ← Pre-process images into .pt cache
│   ├── preprocessing.py         ← CSV cleaning
│   ├── satellite.py             ← Google Maps API downloader
│   ├── get_incomes.py           ← Census API data fetcher
│   └── models/
│       ├── cnn.py               ← Single-tile CNN
│       ├── resnet.py            ← Single-tile ResNet
│       ├── cnn_multi.py         ← Multi-tile CNN
│       ├── resnet_multi.py      ← Multi-tile ResNet
│       ├── dataset.py           ← Single-tile dataset
│       ├── dataset_multi.py     ← Multi-tile dataset
│       └── collate.py           ← Custom batching for multi-tile
└── experiment_logs/             ← Auto-generated experiment logs
```

---

## Quick Start (Mac)

### 1. Get Data

```
python3 new_pipeline/get_new_data.py --shapefile /path/to/tracts.shp --fips 09 --census-api-link "https://api.census.gov/data/2023/acs/acs5"
```

This downloads satellite images (Google Maps API), fetches median income (Census API), builds a CSV, preprocesses it, and creates a .pt cache.

### 2. Run an Experiment

Single-tile:
```
python3 new_pipeline/run_experiment.py --cache new_pipeline/data/cache/09_tracts.pt --model resnet_frozen --mode both --epochs 10 --batch-size 32 --lr 0.001
```

Multi-tile (requires --csv):
```
python3 new_pipeline/run_experiment.py --cache new_pipeline/data/cache/09_tracts_multi.pt --csv new_pipeline/data/09_tracts.csv --model resnet_frozen --mode both --epochs 10 --batch-size 8
```

---

## CLI Arguments

- `--cache` (required) — Path to .pt cache file
- `--csv` — Path to CSV (required for multi-tile mode)
- `--model` (required) — Model architecture: cnn, resnet_frozen, resnet_l3, or resnet_l4
- `--mode` — Train, eval, or both (default: both)
- `--epochs` — Number of training epochs (default: 10)
- `--batch-size` — Batch size (default: 32)
- `--lr` — Learning rate (default: 0.001)
- `--random-state` — Random seed for train/test split (default: 42)
- `--test-size` — Fraction of data for testing (default: 0.2)
- `--device` — Device to use: mps, cuda, cpu, or auto-detect (default: auto)
- `--num-workers` — DataLoader worker processes (default: 4 on Mac, 2 on Kaggle)

---

## Device Support

The pipeline auto-detects the best available device: MPS (Apple Silicon GPU) on Mac, CUDA (NVIDIA GPU) on Kaggle, or CPU as fallback.

Override with `--device`:
```
python3 new_pipeline/run_experiment.py --device cpu --cache ...
```

---

## Environment Switching (Mac vs Kaggle)

### On Mac (default)

ECON_ENV defaults to "mac" — no setup needed:
```
python3 new_pipeline/run_experiment.py --cache data/cache/09_tracts.pt --model resnet_frozen
```

### On Kaggle

Step 1: Zip your local `new_pipeline/data/` folder and upload it as a Kaggle Dataset.

Step 2: In your Kaggle notebook, add that Dataset as an input and set the environment variables:
```
import os
os.environ["ECON_ENV"] = "kaggle"
os.environ["ECON_KAGGLE_DATASET"] = "economic-satellite-data"
```

Step 3: Run the experiment:
```
!python3 new_pipeline/run_experiment.py --cache /kaggle/input/economic-satellite-data/cache/09_tracts.pt --model resnet_frozen --mode both --epochs 10
```

What happens on Kaggle:
- Inputs (cache, CSVs) are read from `/kaggle/input/<dataset-name>/`
- Outputs (models, results, figures) are written to `/kaggle/working/data/`
- Device auto-detects CUDA GPU
- Workers default to 2

You can download outputs from the Kaggle "Output" tab after the run.

---

## Environment Variables

- `ECON_ENV` — Set to "mac" or "kaggle" to switch file paths (default: mac)
- `ECON_KAGGLE_DATASET` — Name of your Kaggle Dataset (default: economic-satellite-data)
- `ECON_NUM_WORKERS` — DataLoader worker count (default: 4 on Mac, 2 on Kaggle)

---

## Model Options

- `cnn` — Small CNN from scratch, no pretrained weights. Good for baselines and quick tests.
- `resnet_frozen` — ResNet18 with all convolutional layers frozen. Fast training with transfer learning.
- `resnet_l3` — ResNet18 with layers 3, 4, and FC unfrozen. More adaptation to your data.
- `resnet_l4` — ResNet18 with layer 4 and FC unfrozen. Middle ground.

---

## Output Files

After an experiment, you'll find:
- Model weights at `data/models/<model>_multi.pth` (if multi-tile)
- Results CSV at `data/results/<model>_results.csv` (per-tract predictions)
- Performance chart at `data/figures/<model>_performance.png`
- Predicted vs actual scatter plot at `data/figures/<model>_predicted_vs_actual.png`
- Residual distribution at `data/figures/<model>_residuals.png`
- Experiment log at `experiment_logs/<timestamp>_<model>.json` (full experiment record)

---

## Experiment Logs

Every run automatically saves a detailed JSON log to `experiment_logs/`. Each log contains the timestamp, all parameters (model, cache, epochs, batch size, learning rate, etc.), environment info (mac/kaggle, device, workers), data info (single vs multi-tile, train/test sizes, income stats), training history (loss per epoch), evaluation metrics (MAE, RMSE, R2, test loss), and output paths. This makes it easy to compare experiments and track what you have run.