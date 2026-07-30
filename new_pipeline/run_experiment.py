"""
run_experiment.py — User-facing entry point for ML experiments.

Usage
-----
    # Single-tile (one image per tract):
    python3 new_pipeline/run_experiment.py \\
        --cache new_pipeline/data/cache/09_tracts.pt \\
        --model resnet_frozen \\
        --mode both \\
        --epochs 10 \\
        --batch-size 32 \\
        --lr 0.001 \\
        --random-state 42

    # Multi-tile (multiple images per tract, auto-detected from cache format):
    python3 new_pipeline/run_experiment.py \\
        --cache new_pipeline/data/cache/09_tracts_multi.pt \\
        --csv new_pipeline/data/09_tracts.csv \\
        --model resnet_frozen \\
        --mode both \\
        --epochs 10 \\
        --batch-size 8 \\
        --lr 0.001 \\
        --random-state 42

    # With MPS (Apple GPU) or CPU override:
    python3 new_pipeline/run_experiment.py \\
        --cache data/cache/09_tracts.pt \\
        --model resnet_frozen \\
        --device mps \\
        --num-workers 4

    # On Kaggle:
        import os
        os.environ["ECON_ENV"] = "kaggle"
        os.environ["ECON_KAGGLE_DATASET"] = "test1data"  # optional, has a default

        !rm -rf economic-satellite-project
        !cd /kaggle/working && git clone -b main https://github.com/braydenwinnicki/economic-satellite-project.git

        !python3 /kaggle/working/economic-satellite-project/new_pipeline/run_experiment.py --cache /kaggle/input/datasets/braydenwinnicki/test1data/09_tracts_multi.pt --model resnet_frozen --mode train --epochs 10 --batch-size 8 --lr 0.001 --random-state 42 --num-workers 2 --csv /kaggle/input/datasets/braydenwinnicki/test1data/09_tracts_multi.csv 

The pipeline:
  1. Loads the pre-computed .pt cache
  2. Auto-detects single-tile vs multi-tile from cache format
  3. Splits into train/test sets (by tract for multi-tile)
  4. Creates the specified model (multi-tile variant if needed)
  5. Trains (if mode is train or both)
  6. Evaluates (if mode is eval or both)
  7. Generates charts from evaluation results
"""

import argparse
import sys
from pathlib import Path

# Path(__file__) = this script's path; .resolve() = absolute path; .parents[1] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from new_pipeline.src.config import (
    DEVICE,
    FIGURES_DIR,
    MODELS_DIR,
    NUM_WORKERS,
    RESULTS_DIR,
)
from new_pipeline.src.evaluate import evaluate_model
from new_pipeline.src.experiment_log import ExperimentLog
from new_pipeline.src.make_charts import make_charts
from new_pipeline.src.models.cnn import ConvNN
from new_pipeline.src.models.cnn_multi import MultiTileConvNN
from new_pipeline.src.models.collate import collate_fn
from new_pipeline.src.models.dataset import CacheDataset
from new_pipeline.src.models.dataset_multi import MultiTileDataset
from new_pipeline.src.models.resnet import ResNetRegressor
from new_pipeline.src.models.resnet_multi import MultiTileResNetRegressor
from new_pipeline.src.train import train_model


def get_model(model_name):
    """Create a model instance based on the model name string."""
    if model_name == "cnn":
        return ConvNN()
    elif model_name == "resnet_frozen":
        return ResNetRegressor(freeze_mode="frozen")
    elif model_name == "resnet_l3":
        return ResNetRegressor(freeze_mode="l3")
    elif model_name == "resnet_l4":
        return ResNetRegressor(freeze_mode="l4")
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Options: cnn, resnet_frozen, resnet_l3, resnet_l4"
        )


def get_multi_model(model_name):
    """Create a multi-tile model instance based on the model name string."""
    if model_name == "cnn":
        return MultiTileConvNN()
    elif model_name == "resnet_frozen":
        return MultiTileResNetRegressor(freeze_mode="frozen")
    elif model_name == "resnet_l3":
        return MultiTileResNetRegressor(freeze_mode="l3")
    elif model_name == "resnet_l4":
        return MultiTileResNetRegressor(freeze_mode="l4")
    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Options: cnn, resnet_frozen, resnet_l3, resnet_l4"
        )


def get_transform(model):
    """Get the appropriate transform for a model.
    For ResNet models, use the pretrained weights' expected transforms.
    For CNN, use None (images are already tensors from cache)."""
    if hasattr(model, "weights"):
        return model.weights.transforms()
    return None


def split_by_tract(df, test_size=0.2, random_state=42):
    """
    Split a tile-level dataframe by tract so all tiles from one tract stay together.
    """
    if "GEOID" not in df.columns:
        raise KeyError("Input dataframe must include a GEOID column.")

    # grab all unique tract IDs so we split by tract, not by tile
    tract_ids = df["GEOID"].drop_duplicates().to_numpy()
    # train_test_split splits the array of tract IDs, not the rows directly
    train_tracts, test_tracts = train_test_split(
        tract_ids,
        test_size=test_size,
        random_state=random_state,
    )

    # .isin() returns a boolean mask — True for rows whose GEOID is in the train set
    train_df = df[df["GEOID"].isin(train_tracts)].copy()
    test_df = df[df["GEOID"].isin(test_tracts)].copy()

    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(
        description="Run an ML experiment on satellite imagery data."
    )
    parser.add_argument(
        "--cache",
        required=True,
        help="Path to the .pt cache file (produced by build_cache.py or build_tract_cache.py)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to the tile-level CSV (required for multi-tile mode)",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["cnn", "resnet_frozen", "resnet_l3", "resnet_l4"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["train", "eval", "both"],
        help="Whether to train, evaluate, or do both (default: both)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for DataLoader (default: 32)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate for Adam optimizer (default: 0.001)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split (default: 42)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data to use for testing (default: 0.2)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use: 'mps', 'cuda', 'cpu', or None for auto-detect (default: auto)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of DataLoader worker processes (default: from config, based on env)",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to a saved .pth model file to load for evaluation "
        "(default: auto-derived from FIPS and model name)",
    )
    parser.add_argument(
        "--notes",
        default=None,
        type=str,
        help="Optional notes to include in the experiment log (e.g., 'ran with lr=0.01')",
    )

    args = parser.parse_args()

    # Resolve device: CLI arg overrides auto-detect
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = DEVICE  # from config.py (auto-detected)

    # Resolve num_workers: CLI arg overrides config default
    num_workers = args.num_workers if args.num_workers is not None else NUM_WORKERS

    print(f"[run_experiment] Device: {device}  Workers: {num_workers}")

    # Step 1: Load cache and detect mode (single-tile vs multi-tile)
    print("=" * 60)
    print("Step 1: Loading cache and detecting mode...")
    print("=" * 60)

    cache = torch.load(args.cache, weights_only=False)

    # Auto-detect: if cache["images"] is a dict, it's multi-tile format
    # (dict mapping GEOID -> tensor of shape (n_tiles, 3, 224, 224))
    # Otherwise it's single-tile format (flat tensor of shape (N, 3, 224, 224))
    is_multi_tile = isinstance(cache["images"], dict)

    if is_multi_tile:
        print("  Detected multi-tile cache format (images grouped by GEOID)")
        if args.csv is None:
            raise ValueError(
                "Multi-tile mode requires a --csv argument pointing to the "
                "tile-level CSV file."
            )
    else:
        print("  Detected single-tile cache format (flat image tensor)")

    # Initialize experiment log (after is_multi_tile is known)
    log = ExperimentLog(args, device, num_workers, is_multi_tile)

    # Step 2: Split data into train/test
    print("\n" + "=" * 60)
    print("Step 2: Splitting data...")
    print("=" * 60)

    if is_multi_tile:
        # For multi-tile, we load the CSV and split by tract (not by row)
        # so all tiles from the same tract stay together
        df = pd.read_csv(args.csv, dtype={"GEOID": str})
        df.columns = df.columns.str.strip()

        df_train, df_test = split_by_tract(
            df, test_size=args.test_size, random_state=args.random_state
        )

        # Compute normalization stats from training tracts only (avoid data leakage)
        train_tract_labels = df_train.drop_duplicates(subset="GEOID")["median_income"]
        mean_income = train_tract_labels.mean()
        std_income = train_tract_labels.std()
        if std_income == 0:
            std_income = 1.0

        # Z-score normalize training labels
        df_train["median_income"] = (
            df_train["median_income"] - mean_income
        ) / std_income

        # Apply the same transformation to test data using train's stats
        df_test["median_income"] = (df_test["median_income"] - mean_income) / std_income

        log.add_data_info(
            n_train=len(df_train),
            n_test=len(df_test),
            mean_income=mean_income,
            std_income=std_income,
        )

        print(
            f"  Train tracts: {df_train['GEOID'].nunique()}  Test tracts: {df_test['GEOID'].nunique()}"
        )
        print(f"  Train tiles: {len(df_train)}  Test tiles: {len(df_test)}")
        print(f"  Income stats (train): mean={mean_income:.2f}, std={std_income:.2f}")

    else:
        # Single-tile: flat tensors, standard train_test_split
        images = cache["images"]
        incomes = cache["incomes"]

        print(f"  Loaded {len(images)} samples from {args.cache}")

        train_images, test_images, train_incomes, test_incomes = train_test_split(
            images, incomes, test_size=args.test_size, random_state=args.random_state
        )

        print(f"  Train: {len(train_images)}  Test: {len(test_images)}")

        # Z-score normalization
        mean_income = train_incomes.mean().item()
        std_income = train_incomes.std().item()

        train_incomes = (train_incomes - mean_income) / std_income
        test_incomes = (test_incomes - mean_income) / std_income

        log.add_data_info(
            n_train=len(train_images),
            n_test=len(test_images),
            mean_income=mean_income,
            std_income=std_income,
        )

    # Step 3: Create model and datasets
    print("\n" + "=" * 60)
    print(f"Step 3: Creating model ({args.model})...")
    print("=" * 60)

    if is_multi_tile:
        model = get_multi_model(args.model)
    else:
        model = get_model(args.model)

    transform = get_transform(model)

    if is_multi_tile:
        # Multi-tile dataset: groups by GEOID, looks up image stacks from cache
        train_dataset = MultiTileDataset(df_train, args.cache)
        test_dataset = MultiTileDataset(df_test, args.cache)

        # DataLoader with custom collate_fn for variable tile counts
        # persistent_workers=True avoids re-creating workers each epoch
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            collate_fn=collate_fn,
        )
    else:
        # Single-tile dataset: flat tensors
        train_dataset = CacheDataset(train_images, train_incomes, transform=transform)
        test_dataset = CacheDataset(test_images, test_incomes, transform=transform)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )

    # Extract FIPS code from cache filename (e.g., "09" from "09_tracts_multi.pt")
    cache_path = Path(args.cache)
    fips_code = cache_path.stem.split("_")[0]  # "09"
    safe_name = f"{fips_code}_{args.model}"
    if is_multi_tile:
        safe_name = f"{safe_name}_multi"
    model_save_path = MODELS_DIR / f"{safe_name}.pth"
    results_save_path = RESULTS_DIR / f"{safe_name}_results.csv"

    # Step 4: Train (if requested)
    if args.mode in ("train", "both"):
        print("\n" + "=" * 60)
        print("Step 4: Training model...")
        print("=" * 60)

        model, epoch_losses = train_model(
            model=model,
            train_loader=train_loader,
            epochs=args.epochs,
            lr=args.lr,
            model_save_path=model_save_path,
            device=device,
        )

        # Record training history in log
        for epoch, loss in enumerate(epoch_losses, 1):
            log.add_training_epoch(epoch, loss)

    # Step 5: Evaluate (if requested)
    if args.mode in ("eval", "both"):
        print("\n" + "=" * 60)
        print("Step 5: Evaluating model...")
        print("=" * 60)

        # If we didn't just train, load saved weights
        if args.mode == "eval":
            # Load weights to CPU first, then move to device
            state_dict = torch.load(model_save_path, map_location="cpu")
            model.load_state_dict(state_dict)
            if device is not None:
                model = model.to(device)

        _results_df, metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            mean_income=mean_income,
            std_income=std_income,
            results_save_path=results_save_path,
            device=device,
        )

        # Record evaluation metrics and output paths in log
        log.add_evaluation(metrics)
        log.add_outputs(
            model_path=model_save_path,
            results_path=results_save_path,
            figures_dir=FIGURES_DIR,
        )

        # Step 6: Generate charts
        print("\n" + "=" * 60)
        print("Step 6: Generating charts...")
        print("=" * 60)

        make_charts(
            results_csv=results_save_path,
            figures_dir=FIGURES_DIR,
            model_name=safe_name,
        )

    # Save experiment log
    log.save()

    print("\n" + "=" * 60)
    print("Experiment complete!")
    print(f"  Model weights: {model_save_path}")
    print(f"  Results CSV:   {results_save_path}")
    print(f"  Figures:       {FIGURES_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
