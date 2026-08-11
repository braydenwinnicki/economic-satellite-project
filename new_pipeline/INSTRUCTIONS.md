  # New Pipeline — Instructions

## Overview

The `new_pipeline/` directory contains the complete ML pipeline for predicting median household income from satellite imagery. It supports:

- **Single-tile** (one image per census tract) and **multi-tile** (multiple images per tract, grid-sampled)
- **4 model architectures**: CNN, ResNet (frozen), ResNet (unfrozen layer 3), ResNet (unfrozen layer 4)
- **Automatic single/multi-tile detection** from cache format
- **Z-score normalization** of income labels (with per-tract splitting for multi-tile)
- **MPS** (Apple GPU), **CUDA** (NVIDIA GPU), and **CPU**
- **Local Mac development** and **Kaggle notebooks** with automatic path switching
- **Automatic experiment logging** to JSON files

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
│   ├── experiment_log.py        ← Auto-logging for experiments
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

Single-tile (one centroid image per tract):
```
python3 new_pipeline/get_new_data.py \
    --shapefile /path/to/tracts.shp \
    --fips 09 \
    --census-api-link "https://api.census.gov/data/2023/acs/acs5"
```

Multi-tile (multiple grid-sampled images per tract):
```
python3 new_pipeline/get_new_data.py \
    --shapefile /path/to/tracts.shp \
    --fips 09 \
    --mode multi \
    --census-api-link "https://api.census.gov/data/2023/acs/acs5"
```

This downloads satellite images (Google Maps API), fetches median income (Census API), builds a CSV, preprocesses it, and creates a .pt cache. Multi-tile mode uses grid sampling where tile count scales with tract area.

### 2. Run an Experiment

Single-tile:
```
python3 new_pipeline/run_experiment.py \
    --cache new_pipeline/data/cache/09_tracts.pt \
    --model resnet_frozen \
    --mode both \
    --epochs 10 \
    --batch-size 32 \
    --lr 0.001
```

Multi-tile (requires --csv):
```
python3 new_pipeline/run_experiment.py \
    --cache new_pipeline/data/cache/09_tracts_multi.pt \
    --csv new_pipeline/data/09_tracts_multi.csv \
    --model resnet_frozen \
    --mode both \
    --epochs 10 \
    --batch-size 8
```

The pipeline auto-detects single-tile vs multi-tile from the cache format (dict = multi-tile, flat tensor = single-tile).

---

## CLI Arguments

### `run_experiment.py`

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--cache` | Yes | — | Path to .pt cache file (produced by `build_cache.py` or `build_tract_cache.py`) |
| `--csv` | For multi-tile | None | Path to tile-level CSV (required when cache is multi-tile format) |
| `--model` | Yes | — | Model architecture: `cnn`, `resnet_frozen`, `resnet_l3`, or `resnet_l4` |
| `--mode` | No | `both` | `train`, `eval`, or `both` |
| `--epochs` | No | 10 | Number of training epochs |
| `--batch-size` | No | 32 | Batch size for DataLoader |
| `--lr` | No | 0.001 | Learning rate for Adam optimizer |
| `--random-state` | No | 42 | Random seed for train/test split |
| `--test-size` | No | 0.2 | Fraction of data for testing |
| `--device` | No | auto | Device: `mps`, `cuda`, `cpu`, or auto-detect |
| `--num-workers` | No | 0 (Mac) / 2 (Kaggle) | DataLoader worker processes |
| `--weights` | No | auto | Path to saved `.pth` file for evaluation (default: auto-derived from FIPS + model name) |
| `--notes` | No | None | Optional notes saved in experiment log |

### `get_new_data.py`

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--shapefile` | Yes | — | Path to census tract shapefile (.shp) |
| `--fips` | Yes | — | State FIPS code (e.g. `09` for Connecticut) |
| `--mode` | No | `single` | `single` (centroid) or `multi` (grid-sampled) |
| `--census-api-link` | No | 2023 ACS 5-year | Census API base URL |

---

## Device Support

The pipeline auto-detects the best available device: MPS (Apple Silicon GPU) on Mac, CUDA (NVIDIA GPU) on Kaggle, or CPU as fallback.

Override with `--device`:
```
python3 new_pipeline/run_experiment.py --device cpu --cache ...
```

**Note on MPS workers:** On macOS, DataLoader workers are set to 0 by default because MPS (Metal Performance Shaders) has fork-safety issues that can cause hangs with `num_workers > 0`. Override with `--num-workers` or `ECON_NUM_WORKERS` if you need parallel loading.

---

## Environment Switching (Mac vs Kaggle)

### On Mac (default)

ECON_ENV defaults to "mac" — no setup needed:
```
python3 new_pipeline/run_experiment.py --cache data/cache/09_tracts.pt --model resnet_frozen
```

### On Kaggle

The Kaggle notebook clones the repo from GitHub into `/kaggle/working/economic-satellite-project/`,
so the entry point actually lives at:

```
/kaggle/working/economic-satellite-project/new_pipeline/run_experiment.py
```

Note that `new_pipeline/` is nested one folder deep inside the cloned repo.

Step 1: In your Kaggle notebook, set the environment variables:
```python
import os
os.environ["ECON_ENV"] = "kaggle"
os.environ["ECON_KAGGLE_DATASET"] = "test1data"   # name of the Dataset you uploaded
```

Step 2: Clone the repo (copies the code into `/kaggle/working/economic-satellite-project/`):
```
!rm -rf economic-satellite-project
!cd /kaggle/working && git clone -b main https://github.com/braydenwinnicki/economic-satellite-project.git
```

Step 3: Run the experiment using the **full cloned path**:
```
!python3 /kaggle/working/economic-satellite-project/new_pipeline/run_experiment.py \
    --cache /kaggle/input/datasets/braydenwinnicki/test1data/09_tracts_multi.pt \
    --csv   /kaggle/input/datasets/braydenwinnicki/test1data/09_tracts_multi.csv \
    --model resnet_frozen \
    --mode train \
    --epochs 10 \
    --batch-size 8 \
    --lr 0.001 \
    --random-state 42 \
    --num-workers 2
```

> **Common mistake:** running `python3 new_pipeline/run_experiment.py` (or
> `/kaggle/working/new_pipeline/run_experiment.py`) fails with
> `can't open file ...: No such file or directory` because the code is kept under
> the cloned `economic-satellite-project/` folder. Always use the full path above.

What happens on Kaggle:
- Inputs (cache, CSVs) are read from your uploaded Dataset.
- Outputs (models, results, figures) are written to `/kaggle/working/data/`
- Device auto-detects CUDA GPU
- Workers default to 2

You can download outputs from the Kaggle "Output" tab after the run.

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ECON_ENV` | `mac` | Set to `mac` or `kaggle` to switch file paths |
| `ECON_KAGGLE_DATASET` | `economic-satellite-data` | Name of your Kaggle Dataset |
| `ECON_NUM_WORKERS` | 0 (Mac) / 2 (Kaggle) | DataLoader worker count |
| `GOOGLE_MAPS_API_KEY` | — | **Required** for satellite image downloads |
| `CENSUS_API_KEY` | — | **Required** for Census income data |

---

## Model Options

| Model | Description |
|-------|-------------|
| `cnn` | Small CNN from scratch, no pretrained weights. Good for baselines and quick tests. |
| `resnet_frozen` | ResNet18 with all convolutional layers frozen. Fast training with transfer learning. |
| `resnet_l3` | ResNet18 with layers 3, 4, and FC unfrozen. More adaptation to your data. |
| `resnet_l4` | ResNet18 with layer 4 and FC unfrozen. Middle ground. |

When using multi-tile mode, the pipeline automatically selects the multi-tile variant of each model (e.g., `MultiTileResNetRegressor` instead of `ResNetRegressor`). Multi-tile models process each tile independently through the same backbone, then average tile-level predictions (masked to ignore padding) into a single tract-level prediction.

---

## Data Splitting

### Single-tile
Standard `train_test_split` from scikit-learn on the flat image tensor.

### Multi-tile
`split_by_tract()` splits by unique GEOID so all tiles from the same tract stay together in either train or test (never split across sets). This prevents data leakage where tiles from the same tract appear in both train and test.

Z-score normalization statistics (mean, std) are computed from training tracts only to avoid data leakage.

---

## Output Files

After an experiment, you'll find:
- Model weights at `data/models/<fips>_<model>[_multi].pth`
- Results CSV at `data/results/<fips>_<model>[_multi]_results.csv` (per-tract predictions)
- Performance chart at `data/figures/<fips>_<model>[_multi]_performance.png`
- Predicted vs actual scatter plot at `data/figures/<fips>_<model>[_multi]_predicted_vs_actual.png`
- Residual distribution at `data/figures/<fips>_<model>[_multi]_residuals.png`
- Experiment log at `experiment_logs/<timestamp>_<model>[_multi].json` (full experiment record)

---

## Cross-State Evaluation

The pipeline supports **cross-state evaluation**: train a model on one state's data, then apply it to make predictions on satellite imagery from a different state. This tests how well the model generalizes beyond the region it was trained on.

### Overview

The key idea is that the model is trained on a source state, saved as `.pth` weights, and then loaded during evaluation on a target state's cache/CSV. Both states need to have been downloaded and cached using `get_new_data.py` first.

You use `--mode eval --weights <path-to-source-weights>` to skip training on the target state and instead load the source-trained model:

```
python3 new_pipeline/run_experiment.py \
    --cache new_pipeline/data/cache/25_tracts_multi.pt \
    --csv new_pipeline/data/25_tracts_multi.csv \
    --model resnet_l4 \
    --mode eval \
    --weights new_pipeline/data/models/09_resnet_l4_multi.pth \
    --batch-size 8
```

### Step-by-Step: Train on CT → Predict on MA

1. **Get data for both states**
   ```
   # CT (source)
   python3 new_pipeline/get_new_data.py --shapefile /path/to/ct_tracts.shp --fips 09 --mode multi

   # MA (target)
   python3 new_pipeline/get_new_data.py --shapefile /path/to/ma_tracts.shp --fips 25 --mode multi
   ```
   This creates CT caches at `data/cache/09_tracts_multi.pt` / `data/09_tracts_multi.csv` and MA caches at `data/cache/25_tracts_multi.pt` / `data/25_tracts_multi.csv`.

2. **Train the model on CT**
   ```
   python3 new_pipeline/run_experiment.py \
       --cache new_pipeline/data/cache/09_tracts_multi.pt \
       --csv new_pipeline/data/09_tracts_multi.csv \
       --model resnet_l4 \
       --mode train \
       --epochs 10 \
       --batch-size 8 \
       --lr 0.001
   ```
   The trained weights save to `data/models/09_resnet_l4_multi.pth`.

3. **Evaluate the CT model on MA**
   ```
   python3 new_pipeline/run_experiment.py \
       --cache new_pipeline/data/cache/25_tracts_multi.pt \
       --csv new_pipeline/data/25_tracts_multi.csv \
       --model resnet_l4 \
       --mode eval \
       --weights new_pipeline/data/models/09_resnet_l4_multi.pth \
       --batch-size 8
   ```
   This loads the CT-trained weights, runs inference on all MA tracts, and saves results to `data/results/25_resnet_l4_multi_results.csv` with charts in `data/figures/`.

### Important Details

- **Model architecture must match.** The `--model` flag must be the same architecture used during training (e.g., both `resnet_l4`). If they don't match, weight loading will fail.
- **Single-tile also works.** Replace `_multi` paths with single-tile paths and omit `--csv`:
  ```
  python3 new_pipeline/run_experiment.py \
      --cache new_pipeline/data/cache/25_tracts.pt \
      --model resnet_l4 \
      --mode eval \
      --weights new_pipeline/data/models/09_resnet_l4.pth
  ```
- **Normalization is automatic.** The target state's incomes are z-score normalized using the target's own train/test split statistics (computed internally by `run_experiment.py`). The predictions are then denormalized back to dollar amounts. This is fine for evaluation metrics — you are measuring how well the source-trained model predicts the target's normalized incomes.
- **Output files use the target state's FIPS.** Results, model paths, and figures are all named after the state you pass in via `--cache`/`--csv`, not the source state.
- **No data leakage.** The cross-state workflow keeps the source and target data completely separate by design — the model never sees the target state during training.

### Batch Evaluation Across States

To run the same source-trained model across multiple target states, use a shell loop or script:

```bash
# Train once on CT
python3 new_pipeline/run_experiment.py \
    --cache data/cache/09_tracts_multi.pt \
    --csv data/09_tracts_multi.csv \
    --model resnet_l4 --mode train --epochs 10 --batch-size 8

# Evaluate on multiple states
for fips in 25 36 42; do
    python3 new_pipeline/run_experiment.py \
        --cache "data/cache/${fips}_tracts_multi.pt" \
        --csv "data/${fips}_tracts_multi.csv" \
        --model resnet_l4 \
        --mode eval \
        --weights data/models/09_resnet_l4_multi.pth \
        --batch-size 8
done
```

### Comparing Cross-State vs. In-State Performance

After running both in-state (train+eval on same state) and cross-state (train on source, eval on target) experiments, you can compare the results CSVs:

| Experiment | Results File | MAE | RMSE |
|---|---|---|---|
| In-state (CT) | `09_resnet_l4_multi_results.csv` | $X | $Y |
| Cross-state CT→MA | `25_resnet_l4_multi_results.csv` | $A | $B |

Typically you'd expect cross-state MAE/RMSE to be higher (worse) than in-state, indicating how much the model's performance degrades when applied to a new region.

---

## Experiment Logs

Every run automatically saves a detailed JSON log to `experiment_logs/`. Each log contains:
- Timestamp and duration
- All parameters (model, cache, epochs, batch size, learning rate, etc.)
- Environment info (mac/kaggle, device, workers)
- Data info (single vs multi-tile, train/test sizes, income stats)
- Training history (loss per epoch)
- Evaluation metrics (MAE, RMSE, R2, test loss)
- Output paths (model weights, results CSV, figures directory)
- Optional user notes (from `--notes`)

This makes it easy to compare experiments and track what you have run.

---

## Cache Formats

### Single-tile cache (produced by `build_cache`)
```
cache["images"]  — torch.Tensor of shape (N, 3, 224, 224)
cache["incomes"] — torch.Tensor of shape (N,)  (float32)
```

### Multi-tile cache (produced by `build_tract_cache`)
```
cache["images"]  — dict mapping GEOID -> torch.Tensor of shape (n_tiles, 3, 224, 224)
cache["income"]  — dict mapping GEOID -> float (median income)
```

The pipeline auto-detects which format is loaded and selects the appropriate dataset class and model variant.