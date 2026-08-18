
# Research Notebook — Economic Indicators from Satellite Imagery

## Project Summary

**Research Question**

Can convolutional neural networks estimate median household income from publicly available satellite imagery at the census tract level?

This project combines U.S. Census Bureau American Community Survey (ACS) income data with Google Static Maps satellite imagery to investigate whether built-environment characteristics visible from above contain predictive information about local economic conditions.

---

## Study Design

### Target Variable

* Median Household Income
* ACS 5-Year Estimates (2023)
* Census variable: **B19013_001E**

### Geographic Scope

* Connecticut census tracts (FIPS 09) as baseline with a variety of other states for comparison

### Imagery

* Google Static Maps API
* Satellite imagery
* Zoom level 17
* 400 × 400 pixels

---

# Design Decisions

### Decision 1 — Connecticut Only First

**Reason**

Using a single state provides a manageable dataset while still capturing substantial variation between urban, suburban, and rural communities.

**Tradeoff**

Models trained on one state may not immediately generalize to other regions without additional testing.

---

### Decision 2 — Census Tracts

**Reason**

Census tracts align naturally with ACS socioeconomic data and are commonly used in public health and economic research.

**Tradeoff**

Fewer samples than using image patches alone, but labels are considerably more reliable.

---

### Decision 3 — Median Household Income

**Reason**

Income is a continuous variable, making it well suited for a regression problem.

**Tradeoff**

Very high-income tracts may be difficult to predict because satellite imagery captures physical characteristics better than financial characteristics. Income is also capped at $250,000 in census data which can make things more difficult. 

---

# Project Timeline

---

## Phase 1 — Dataset Construction

### Objective

Construct a reproducible dataset linking satellite imagery with Census income estimates.

### Work Completed

* Retrieved ACS income data from the Census API.
* Built GEOIDs from state, county, and tract FIPS codes.
* Computed tract centroids from TIGER/Line shapefiles.
* Downloaded satellite imagery using the Google Static Maps API.
* Generated a CSV linking every image with its corresponding income value.

### Data Cleaning

* Removed missing income values.
* Converted Census placeholder values (-666666666) to missing values.
* Standardized column names.
* Verified image–label alignment.

### Outcome

Successfully produced the first complete training dataset.

---

## Phase 2 — Establishing a Baseline

### Motivation

Before training neural networks, establish the minimum level of performance required to demonstrate that the model is learning meaningful information.

### Experiment

Predict the mean training-set income for every test example.

### Outcome

Baseline MAE calculated for comparison with all future models.

**Decision**

All subsequent models must outperform this baseline to be considered useful.

---

## Phase 3 — CNN From Scratch (S1)

### Motivation

Determine whether a relatively small CNN trained entirely from scratch can learn useful economic features directly from satellite imagery.

### Model

Custom convolutional neural network trained from random initialization.

### Training Configuration

* Loss: MSE
* Optimizer: Adam
* Learning rate: 0.001
* Batch size: 32
* Epochs: 10
* Image size: 224×224
* Income standardized using training statistics
* Random 80/20 train-test split

### Results

* MAE: **$39,420.97**
* RMSE: **$51,035.48**
* R²: **0.09**

### Observations

* Not beating the baseline well — the model struggles to extract meaningful features from scratch with only one image per tract.
* Generally underperforming, confirming that a simple CNN without pretrained weights lacks the capacity to learn economically useful visual features from this dataset.

### Decision

Proceed to transfer learning to determine whether pretrained visual features improve performance.

---

## Phase 4 — Frozen ResNet18 (S2)

### Motivation

ImageNet-pretrained networks already recognize roads, vegetation, buildings, and other visual structures. This experiment tests whether those learned features transfer to socioeconomic prediction.

### Model

ResNet18

* ImageNet pretrained
* Backbone frozen
* Final regression layer trained

### Training Configuration

Same optimization settings as the baseline CNN.

### Results

* MAE: **$36,597.93**
* RMSE: **$48,178.64**
* R²: **0.19**

### Observations

* Major improvements over the scratch CNN — R² doubled from 0.09 to 0.19, confirming that ImageNet-pretrained features transfer to socioeconomic prediction.
* However, performance is still not ideal. The model is limited by having only one image per tract, which likely doesn't capture the full spatial context of each neighborhood.

### Decision

Evaluate whether partially unfreezing the network improves predictive performance. It does. 

---

## Phase 6 — Multi-Tile Models

### Motivation

A single satellite image may not capture enough context.

Multiple images per census tract may better represent neighborhood characteristics.

### Planned Experiments

* Multi-tile CNN
* Multi-tile ResNet + ResNet Fine tuning 

### Research Question

Does additional spatial context improve prediction accuracy?

---

## Phase 7 — Multi-Tile Pipeline

### Motivation

The single-tile approach captures only the immediate area around each tract centroid. A single 400×400 image at zoom 17 covers roughly 500 meters — often missing significant portions of a census tract, especially in larger suburban and rural tracts. The frozen ResNet showed improvements over the baseline CNN, but the model was making predictions from incomplete visual information: one image per tract, centered on the centroid, regardless of tract size or shape. Results were logically invalid. 

The multi-tile pipeline addresses this by capturing multiple satellite images per census tract, providing broader spatial coverage and a more representative view of each neighborhood's built environment.

### Pipeline Architecture

The multi-tile pipeline is organized as a self-contained project under `multi-tile/`, mirroring the structure of the original `main/` pipeline but with key differences in data handling, dataset design, and evaluation strategy.

#### Data Construction

Unlike the single-tile pipeline (which downloads one image per tract centroid), the multi-tile pipeline captures multiple images per census tract. The sampling strategy distributes image captures across each tract's geographic area — using grid offsets, random points within tract boundaries, or multiple zoom levels — rather than concentrating at a single centroid point.

The resulting CSV contains multiple rows per GEOID, one row for each satellite tile belonging to that tract. Each row shares the same `median_income` label. This is the fundamental structural difference from the single-tile CSV, which has exactly one row per tract.

#### Dataset Design

`multi-tile/models/dataset_multi.py` defines a `CensusDataset` that loads one tile image at a time from the multi-row CSV. Each tile is treated as an independent training example paired with its tract's income label. The dataset applies standard transforms: resize to 224×224 and convert to tensor.

##### Collation and Aggregation

Because the dataset contains multiple rows per GEOID, the evaluation pipeline uses a collation step to convert per-tile predictions into tract-level estimates:

1. Per-tile inference: The model produces one income prediction for each tile in the test set. At this stage, there are multiple predictions per GEOID — one per image belonging to that tract.

2. Tract-level aggregation: Predictions across all tiles belonging to the same GEOID are combined into a single tract-level estimate. The default aggregation function is mean averaging: `prediction_tract = mean(prediction_tile_1, prediction_tile_2, ..., prediction_tile_n)`.

3. Comparison: The aggregated tract-level predictions are compared against the single ground-truth `median_income` for that GEOID.

This collation step is critical — it separates the tile-level modeling from tract-level evaluation. The model never sees an entire tract at once; instead it learns from individual tile views, and the evaluation aggregates those independent predictions into a tract-level estimate.

#### Models

Two model architectures are used for the multi-tile pipeline, structurally identical to their single-tile counterparts but trained on the expanded multi-tile dataset:

1. **Multi-Tile CNN**: A small custom CNN (3 conv layers → 3 fully-connected layers) trained from scratch.
2. **Multi-Tile ResNet**: A ResNet18 with ImageNet-pretrained weights, trained in frozen and partially-unfrozen configurations.

Both models operate on individual tiles — one image in, one income prediction out. The multi-tile aspect is handled entirely at the data and evaluation level, not at the architecture level.

#### Training

Training scripts follow the same procedure as the single-tile versions:

* 80/20 train-test split (at the tile level, not the tract level)
* Z-score normalization of income using training statistics
* MSE loss, Adam optimizer (lr=0.001), batch size 32, 10-25 epochs
* Standard train loop with gradient descent

The key difference: because the dataset contains multiple tiles per tract, the model sees more total training examples and learns from different visual perspectives of the same income label.

#### Evaluation

Evaluation scripts run the trained model on the test set and produce per-tile predictions. The evaluation pipeline then applies the collation step — grouping by GEOID and averaging predictions — to produce one tract-level income estimate per GEOID. Results are saved to CSV files in `research/results/` with columns for GEOID, prediction, actual, and error.


## Phase 8 — Multi-Tile CNN (M1)

### Motivation

The single-tile CNN (Phase 3) underperformed the baseline because it lacked sufficient visual information from a single centroid image. With multiple tiles per tract, the model has more training examples and broader spatial coverage. This experiment tests whether more data alone — without pretrained features — is enough to learn meaningful economic signals.

### Model

Custom CNN (same architecture as Phase 3)

* 3 convolutional layers (3→6→16 filters)
* 3 fully-connected layers (46656→120→84→1)
* Trained from random initialization on multi-tile dataset

### Training Configuration

* Loss: MSE
* Optimizer: Adam
* Learning rate: 0.001
* Batch size: 32
* Epochs: 10
* Image size: 224×224
* Multi-tile dataset with tile-level train/test split

### Results

* MAE: **$34,293.83**
* RMSE: **$47,116.60**
* R²: **0.22**

### Observations

* The multi-tile CNN outperforms the single-tile CNN (R²: 0.22 vs 0.09), a +19.10% improvement, suggesting additional spatial coverage provides more learnable signal.
* However, R² of 0.22 means the model still explains only a fraction of income variance — well below the frozen ResNet's performance.
* Training from scratch continues to struggle relative to transfer learning.

### Decision

The multi-tile CNN improves over its single-tile counterpart, but transfer learning is still expected to dominate. Proceed to the frozen ResNet with multi-tile data.

---

## Phase 9 — Multi-Tile ResNet Frozen (M2)

### Motivation

The frozen ResNet on single tiles (Phase 4) was the first model to beat the baseline convincingly, but the observation was that tracts likely weren't being shown fully to the model. With multi-tile data, the pretrained ImageNet features should have richer visual context to work with.

### Model

ResNet18

* ImageNet pretrained
* Backbone fully frozen
* Final regression layer (FC) trained on multi-tile data

### Training Configuration

* Loss: MSE
* Optimizer: Adam
* Learning rate: 0.001
* Batch size: 32
* Epochs: 10
* Image size: 224×224
* Multi-tile dataset with tile-level train/test split

### Results

* MAE: **$31,301.93**
* RMSE: **$37,519.74**
* R²: **0.51**

### Observations

* Clear improvement over the multi-tile CNN: R² of 0.51 vs 0.22 (+128.69% from M1), MAE reduced by ~$3,000.
* Spatial coverage combined with transfer learning produces the strongest performance so far.
* A frozen backbone still limits the model's ability to adapt features to the income prediction task.

### Decision

Proceed to unfreeze later layers to allow the pretrained features to adapt to the socioeconomic prediction task.

---

## Phase 10 — Multi-Tile ResNet Unfrozen L4 (M3)

### Motivation

Freezing the entire backbone preserves ImageNet features but prevents the model from tuning them for income prediction. Unfreezing the final residual block (layer 4) allows the highest-level visual features — the ones closest to the output — to adapt while keeping earlier layers stable.

### Model

ResNet18

* ImageNet pretrained
* Layer 4 unfrozen (trainable) — FC + L4
* Layers 1–3 frozen
* Regression head trained

### Training Configuration

* Loss: MSE
* Optimizer: Adam
* Learning rate: 0.001
* Batch size: 32
* Epochs: 25
* Image size: 224×224
* Multi-tile dataset with tile-level train/test split

### Results

* MAE: **$24,006.54**
* RMSE: **$31,879.34**
* R²: **0.64**

### Observations

* Substantial improvement: R² jumped from 0.51 to 0.64 (+27.12% from M2), MAE dropped by ~$7,300.
* Unfreezing layer 4 allows the model to refine high-level spatial features — building arrangements, road patterns, vegetation density — for income prediction.
* The RMSE gap relative to MAE narrowed, suggesting the model is handling outlier tracts better.
* This is the best single improvement observed so far.

### Decision

Test whether unfreezing more layers (layer 3 + layer 4) continues to improve performance, or if it begins to overfit.

---

## Phase 11 — Multi-Tile ResNet Unfrozen L3 (M4)

### Motivation

If unfreezing layer 4 helped, does unfreezing layers 3 and 4 together provide even more adaptability, or does it risk overfitting to the training data?

### Model

ResNet18

* ImageNet pretrained
* Layers 3 and 4 unfrozen (trainable) — FC + L4 + L3
* Layers 1–2 frozen
* Regression head trained

### Training Configuration

* Loss: MSE
* Optimizer: Adam
* Learning rate: 0.001
* Batch size: 32
* Epochs: 25
* Image size: 224×224
* Multi-tile dataset with tile-level train/test split

### Results

* MAE: **$23,984.58**
* RMSE: **$31,919.13**
* R²: **0.64**

### Observations

* Unfreezing L3 produced nearly identical results to L4-only unfreezing.
* MAE improved marginally ($24,006.54 → $23,984.58), but RMSE slightly increased ($31,879.34 → $31,919.13) and R² barely moved (−0.14%).
* The added parameter count did not translate into meaningful improvement — the model appears to have reached a plateau with L4 unfreezing alone.
* This suggests that unfreezing additional layers beyond L4 introduces more parameters without capturing additional income-relevant signal.

### Decision

Unfreezing beyond L4 does not help. L4-only unfreezing provides the best balance of performance and parameter efficiency. Stop unfreezing experiments at L4.

---

# Cross-Phase Comparison

| Model ID | Architecture | Image Input | Transfer Learning | Trainable Layers | Epochs | MAE | RMSE | R² | R² Δ |
|----------|-------------|-------------|-------------------|-----------------|--------|-----|------|----|------|
| S1 | Custom CNN | Single Tile | None | All | 10 | $39,420.97 | $51,035.48 | 0.09 | — |
| S2 | ResNet-18 | Single Tile | ImageNet | FC | 10 | $36,597.93 | $48,178.64 | 0.19 | +114.98% |
| M1 | Custom CNN | Multi Tile | None | All | 10 | $36,894.32 | $48,847.18 | 0.16 | -12.25% |
| M2 | ResNet-18 | Multi Tile | ImageNet | FC | 10 | $31,301.93 | $37,519.74 | 0.51 | +210.40% |
| M3 | ResNet-18 | Multi Tile | ImageNet | FC + L4 | 25 | $24,006.54 | $31,879.34 | 0.64 | +27.12% |
| M4 | ResNet-18 | Multi Tile | ImageNet | FC + L4 + L3 | 25 | $23,984.58 | $31,919.13 | 0.64 | −0.14% |

---

# Key Observations

* Multi-tile data consistently improves over single-tile across all model architectures (S1→M1: +19.10%, S2→M2: +128.69% R² improvement).
* Transfer learning dominates: the best multi-tile CNN (R²=0.22) is far weaker than the worst multi-tile ResNet (R²=0.51).
* Unfreezing layer 4 produced the largest single improvement in the project (M2→M3: +27.12% R², MAE dropped from $31,301.93 to $24,006.54).
* Unfreezing beyond L4 adds no benefit — M3 and M4 give near-identical results (−0.14% R² change).
* Models still struggle with the highest-income tracts, where satellite imagery captures physical characteristics less correlated with financial characteristics.
* The narrowing RMSE–MAE gap from frozen to unfrozen suggests that adaptation helps most on the hardest (highest-income) predictions.

---

## Phase 12 — Consolidated Pipeline (New Pipeline)

### Motivation

Through Phases 1–11, the project accumulated code across multiple directories (`main/`, `multi-tile/`, `archive/`) with inconsistent interfaces, scattered configuration, and no unified experiment tracking. Each model variant had its own training and evaluation script, making it difficult to compare results, reproduce experiments, or iterate quickly.

The new pipeline (`new_pipeline/`) was built to consolidate everything into a single, maintainable codebase with a unified CLI, automatic experiment logging, chart generation, and cross-environment support (local Mac development and Kaggle GPU training).

### Pipeline Architecture

The new pipeline is organized as a self-contained project under `new_pipeline/` with a single entry point:

```
new_pipeline/
├── run_experiment.py            ← Unified CLI entry point
├── get_new_data.py              ← Data download (satellite + census)
├── build_csv.py                 ← CSV construction from shapefiles
├── src/
│   ├── config.py                ← Central config (paths, device, workers)
│   ├── train.py                 ← Training loop
│   ├── evaluate.py              ← Evaluation loop
│   ├── make_charts.py           ← Chart generation
│   ├── build_cache.py           ← Pre-process images into .pt cache
│   ├── preprocessing.py         ← CSV cleaning
│   ├── satellite.py             ← Google Maps API downloader
│   ├── get_incomes.py           ← Census API data fetcher
│   ├── experiment_log.py        ← JSON experiment logging
│   └── models/                  ← Model architectures + datasets
├── data/                        ← All data (auto-created)
│   ├── cache/                   ← Pre-computed .pt cache files
│   ├── models/                  ← Saved model weights (.pth)
│   ├── results/                 ← Evaluation results CSVs
│   └── figures/                 ← Generated charts
└── experiment_logs/             ← Auto-generated experiment logs
```

### Key Improvements Over Previous Pipelines

**1. Unified CLI Interface**

All experiments are run through a single `run_experiment.py` script with consistent arguments (`--model`, `--cache`, `--epochs`, `--batch-size`, `--lr`, etc.), replacing the previous pattern of separate scripts per model variant.

**2. Automatic Mode Detection**

The pipeline auto-detects single-tile vs multi-tile mode from the cache file format — a flat tensor for single-tile, a dict of GEOID-to-tensor-stack for multi-tile. No separate code paths needed.

**3. Tract-Level Train/Test Splitting (Multi-Tile)**

Previous multi-tile experiments split tiles randomly, risking data leakage when tiles from the same tract appeared in both train and test sets. The new pipeline splits by tract — all tiles from a given GEOID stay together — producing more honest evaluation metrics.

**4. Experiment Logging**

Every run automatically saves a detailed JSON log to `experiment_logs/` containing:
- Timestamp and all CLI parameters
- Environment info (mac/kaggle, device, workers)
- Data info (single vs multi-tile, train/test sizes, income statistics)
- Training history (loss per epoch)
- Evaluation metrics (MAE, RMSE, R², test loss)
- Output paths (model weights, results CSV, figures)

This makes it easy to compare experiments and track what has been run.

**5. Automatic Chart Generation**

After evaluation, the pipeline automatically generates three charts:
- Performance metrics summary
- Predicted vs actual scatter plot
- Residual distribution

**6. Cross-Environment Support**

The pipeline runs identically on Mac (with MPS/Apple GPU support) and Kaggle (with CUDA), switching file paths and worker counts automatically via the `ECON_ENV` environment variable.

**7. Caching System**

Images are pre-processed into `.pt` cache files (resized, normalized tensors), eliminating repeated image loading and transform application during training.

### Experiment: Multi-Tile ResNet (M4)

The first experiment using the new pipeline re-validated the best-performing multi-tile ResNet configuration (M4, 25 epochs) with proper tract-level splitting and the consolidated codebase, correcting the logical errors present in the earlier evaluation code.

#### Training (Kaggle — CUDA)

The model was trained on Kaggle with a CUDA GPU for 25 epochs. Training loss decreased steadily from 1.14 to 0.13, indicating effective learning.

| Epoch | Loss |
|-------|------|
| 1 | 1.1381 |
| 5 | 0.4288 |
| 10 | 0.3926 |
| 15 | 0.1849 |
| 20 | 0.1350 |
| 25 | 0.1283 |

#### Evaluation (Local — MPS)

The trained model was evaluated locally on the test set:

| Metric | Value |
|--------|-------|
| MAE | **$23,984.58** |
| RMSE | **$31,919.13** |
| R² | **0.64** |

#### Observations

- After correcting the logical errors in the earlier code, the new pipeline's M4 multi-tile ResNet result (R² = 0.64) is consistent with the corrected Phase 3–11 metrics, thanks to proper tract-level splitting eliminating data leakage.
- The training loss trajectory shows consistent improvement across all 25 epochs with no signs of overfitting, suggesting additional epochs could yield further gains.
- The experiment was fully reproducible: the same cache, CSV, random state, and model configuration produce identical results across environments (Kaggle for training, local Mac for evaluation).
- The consolidated pipeline reduces experiment setup time from minutes (configuring separate scripts) to seconds (single CLI command).

### Decision

The new pipeline replaces all previous codebases (`main/`, `multi-tile/`, `archive/`). All future experiments will use `new_pipeline/run_experiment.py` as the single entry point. The consolidated infrastructure enables faster iteration, better experiment tracking, and more reliable comparisons between model configurations.

### Corrected Results (New Pipeline)

The metrics reported through Phase 11 were produced by the legacy `main/` and `multi-tile/`
codebases and contained logical errors. The `new_pipeline/` re-evaluation fixes these:

* The multi-tile CNN (M1) was previously reported with an R² of 0.22; under the consolidated
  pipeline's tract-level split its true R² is **0.16** with MAE $36,894.32 and RMSE $48,847.18.
* The early new-pipeline result previously labeled **N1** is folded into **M4** and dropped as a
  separate entry, since it corresponds to the same model lineage.

| Model ID | Architecture | Input Type | Pretraining | Trainable Layers | Epochs | Purpose | MAE | RMSE | R² | R² Δ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S1** | Custom CNN | Single Tile | None | All | 10 | Baseline CNN | $39,420.97 | $51,035.48 | 0.09 | — |
| **S2** | ResNet-18 | Single Tile | ImageNet | FC | 10 | Test effects of transfer learning | $36,597.93 | $48,178.64 | 0.19 | +114.98% |
| **M1** | Custom CNN | Multi Tile | None | All | 10 | Establish effects of multi-tiles | $36,894.32 | $48,847.18 | 0.16 | -12.25% |
| **M2** | ResNet-18 | Multi Tile | ImageNet | FC | 10 | Establish ResNet's ability to learn from multi-tiles | $31,301.93 | $37,519.74 | 0.51 | +210.40% |
| **M3** | ResNet-18 | Multi Tile | ImageNet | FC + L4 | 25 | Allow ResNet to adapt to the task | $24,006.54 | $31,879.34 | 0.64 | +27.12% |
| **M4** | ResNet-18 | Multi Tile | ImageNet | FC + L4 + L3 | 25 | Test effects of further fine-tuning | $23,984.58 | $31,919.13 | 0.64 | -0.14% |

---

# Failed or Abandoned Ideas

Document approaches that did not work.
* Single tile per tract — not enough context.
* Fine-tuning ResNet beyond L4 — L3 started to reduce performance relative to L4-only. But will continue to test.


---

# Open Research Questions

* Does transfer learning consistently outperform training from scratch? **Yes, confirmed across single-tile and multi-tile settings.**
* Which land-use features contribute most to predictions?
* Does incorporating multiple tiles improve generalization? **Yes, multi-tile outperforms single-tile across all architectures.**
* Are prediction errors geographically clustered?
* Can the model generalize beyond Connecticut?
* How much information about socioeconomic status is actually encoded in satellite imagery?


