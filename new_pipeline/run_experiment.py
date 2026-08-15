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
import re
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


def infer_state_fips(path_value):
    """Return the 2-digit state/FIPS code embedded in a file path or filename.

    Looks for a two-digit number (not part of a longer digit sequence)
    in the filename stem — e.g. '09' from '09_resnet_l4_multi.pth'.
    Returns None if no match is found or the input is None/empty.
    """
    if path_value is None:
        return None
    text = str(path_value)
    if not text:
        return None
    # Prefer the stem (filename without extension) to avoid matching digits
    # in directory paths.
    stem = Path(text).stem
    match = re.search(r"(?<!\d)(\d{2})(?!\d)", stem)
    if match:
        return match.group(1)
    return None


def build_experiment_safe_name(
    model_name, is_multi_tile, target_fips, source_fips=None
):
    """Build a state-aware name that reflects cross-state evaluation.

    In-state (source == target):
        <target_fips>_<model>[_multi]

    Cross-state (source != target):
        <source_fips>_to_<target_fips>_<model>[_multi]
    """
    if source_fips and target_fips and source_fips != target_fips:
        safe_name = f"{source_fips}_to_{target_fips}_{model_name}"
    else:
        safe_name = f"{target_fips}_{model_name}"
    if is_multi_tile:
        safe_name = f"{safe_name}_multi"
    return safe_name


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

    # ── Resolve state codes and derived output paths ──────────────────
    cache_path = Path(args.cache)
    target_fips = infer_state_fips(cache_path) or cache_path.stem.split("_")[0]
    source_fips = infer_state_fips(args.weights) if args.weights else None

    # For training runs, the source state is the target state being trained on.
    if args.mode == "train" and source_fips is None:
        source_fips = target_fips
    if args.mode in ("eval", "both") and source_fips is None and args.weights is None:
        source_fips = target_fips

    state_mismatch = bool(
        source_fips and target_fips and source_fips != target_fips
    )
    if state_mismatch:
        print(
            f"  Cross-state evaluation detected: trained on {source_fips}, "
            f"evaluating on {target_fips}."
        )
    else:
        print(f"  In-state evaluation: {target_fips} (no source/target mismatch)")

    safe_name = build_experiment_safe_name(
        model_name=args.model,
        is_multi_tile=is_multi_tile,
        target_fips=target_fips,
        source_fips=source_fips,
    )
    log.add_state_info(source_state=source_fips, target_state=target_fips)
    model_save_path = MODELS_DIR / f"{safe_name}.pth"
    results_save_path = RESULTS_DIR / f"{safe_name}_results.csv"

    # ── Resolve the normalization frame ───────────────────────────────
    # Training runs always normalize labels with stats computed from the
    # target train split (saved with the model). For pure evaluation we reuse
    # the stats that were used to TRAIN the loaded model, so cross-state
    # evaluation denormalizes predictions back into the correct dollar frame
    # instead of the target state's own (incorrect) frame.
    loaded_ckpt = None
    norm_mean = None
    norm_std = None
    if args.mode == "eval":
        eval_weights_path = args.weights if args.weights else model_save_path
        loaded_ckpt = torch.load(eval_weights_path, map_location="cpu", weights_only=False)
        print(f"  Loading weights from: {eval_weights_path}")
        if isinstance(loaded_ckpt, dict) and "mean_income" in loaded_ckpt:
            norm_mean = loaded_ckpt["mean_income"]
            norm_std = loaded_ckpt["std_income"]
            print(
                f"  Using normalization stats saved with the source model "
                f"(mean={norm_mean:.2f}, std={norm_std:.2f})."
            )
        else:
            print(
                "  ⚠ Checkpoint has no saved normalization stats (legacy "
                "format); falling back to the target dataset's own stats. "
                "Cross-state numbers will NOT be reliable."
            )

    # Step 2: Split data into train/test
    print("\n" + "=" * 60)
    print("Step 2: Splitting data...")
    print("=" * 60)

    if is_multi_tile:
        # For multi-tile, we load the CSV and split by tract (not by row)
        # so all tiles from the same tract stay together
        df = pd.read_csv(args.csv, dtype={"GEOID": str})
        df.columns = df.columns.str.strip()

        # Canonicalize GEOIDs to zero-padded 11-digit form. Caches built before
        # the fix (or GEOIDs read as int) can drop the leading zero for states
        # whose FIPS starts with 0 (e.g. 09 = Connecticut), which would otherwise
        # make every tract look "missing" when matched against the cache.
        df["GEOID"] = df["GEOID"].astype(str).str.zfill(11)

        # Only keep tracts that have cached images.  build_tract_cache skips
        # tracts whose tile images all failed to load, so the CSV may contain
        # GEOIDs that are absent from cache["images"] — leaving them in would
        # cause a KeyError in MultiTileDataset.__getitem__ during iteration.
        cache_geoids = {str(g).zfill(11) for g in cache["images"]}
        before_tracts = df["GEOID"].nunique()
        df = df[df["GEOID"].isin(cache_geoids)].copy()
        dropped = before_tracts - df["GEOID"].nunique()
        if dropped:
            print(f"  Filtered CSV: {dropped} tracts not in cache were dropped.")

        df_train, df_test = split_by_tract(
            df, test_size=args.test_size, random_state=args.random_state
        )

        # Determine the normalization frame.
        #  - Training runs: compute from the training tracts only (avoids data
        #    leakage) and save those stats with the model when it is saved.
        #  - Pure eval with weights: reuse the stats saved with the source
        #    model, so cross-state predictions denormalize into the correct
        #    dollar frame rather than the target state's own frame.
        if norm_mean is not None and norm_std is not None:
            mean_income = norm_mean
            std_income = norm_std
        else:
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

        # Z-score normalization. Training runs compute stats from the training
        # split; pure eval with weights reuses the stats saved with the source
        # model so cross-state predictions denormalize into the correct frame.
        if norm_mean is not None and norm_std is not None:
            mean_income = norm_mean
            std_income = norm_std
        else:
            mean_income = train_incomes.mean().item()
            std_income = train_incomes.std().item()
        if std_income == 0:
            std_income = 1.0

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
        # Multi-tile dataset: groups by GEOID, looks up image stacks from cache.
        # Pass the model's expected transform (ImageNet normalization for ResNet)
        # so pretrained backbones receive properly normalized input.
        train_dataset = MultiTileDataset(df_train, args.cache, transform=transform)
        test_dataset = MultiTileDataset(df_test, args.cache, transform=transform)

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
            mean_income=mean_income,
            std_income=std_income,
        )

        # Record training history in log
        for epoch, loss in enumerate(epoch_losses, 1):
            log.add_training_epoch(epoch, loss)

    # Step 5: Evaluate (if requested)
    if args.mode in ("eval", "both"):
        print("\n" + "=" * 60)
        print("Step 5: Evaluating model...")
        print("=" * 60)

        # If we didn't just train, load saved weights from the checkpoint that
        # was already loaded up front (it also carries the normalization stats).
        if args.mode == "eval":
            # The checkpoint may be the new {state_dict, mean_income, std_income}
            # format or a legacy raw state_dict.
            if isinstance(loaded_ckpt, dict) and "state_dict" in loaded_ckpt:
                model.load_state_dict(loaded_ckpt["state_dict"])
            else:
                model.load_state_dict(loaded_ckpt)
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
