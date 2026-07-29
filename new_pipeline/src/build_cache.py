"""
build_cache.py

Pre-loads all images from a processed CSV, resizes them to 224×224,
converts them to PyTorch tensors, and saves everything as a single .pt
file. This avoids re-loading and re-transforming images from disk every
time you train or evaluate a model.
"""

import sys
from pathlib import Path

import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms

# Default transform: resize to 224×224 and convert pixel values
# from 0–255 (uint8) to 0.0–1.0 (float32) in (C, H, W) order.
DEFAULT_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def build_cache(csv_path, cache_path, transform=DEFAULT_TRANSFORM):
    """
    Load images listed in *csv_path*, apply *transform*, and save a
    single .pt file at *cache_path* containing:

        cache["images"]  — torch.Tensor of shape (N, 3, 224, 224)
        cache["incomes"] — torch.Tensor of shape (N,)  (float32)

    Parameters
    ----------
    csv_path : str or Path
        Path to the processed CSV (must have columns ``image_path``
        and ``median_income``).
    cache_path : str or Path
        Where to write the output ``.pt`` file.
    transform : callable, optional
        Torchvision transform pipeline.  Defaults to Resize + ToTensor.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    n = len(df)
    print(f"Building cache from {n} samples ...")

    images = []
    incomes = []

    for idx in range(n):
        row = df.iloc[idx]

        # Load image
        img_path = str(row["image_path"]).strip()
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"  WARNING: could not load {img_path} — skipping ({e})")
            continue

        # Transform to tensor
        img_tensor = transform(img)  # shape (3, 224, 224)

        images.append(img_tensor)
        incomes.append(float(row["median_income"]))

    # Guard: if every image failed to load, we can't stack an empty list
    if len(images) == 0:
        raise RuntimeError(
            f"All {n} images failed to load from {csv_path}. "
            "Check that image paths are correct and files exist."
        )

    # Stack into single tensors
    # torch.stack takes a list of (3, 224, 224) tensors → (N, 3, 224, 224)
    images_tensor = torch.stack(images)
    incomes_tensor = torch.tensor(incomes, dtype=torch.float32)

    # Save
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"images": images_tensor, "incomes": incomes_tensor}, cache_path)

    print(f"Cache saved → {cache_path}")
    print(f"  images shape:  {tuple(images_tensor.shape)}")
    print(f"  incomes shape: {tuple(incomes_tensor.shape)}")
    print(f"  samples kept:  {len(images_tensor)} / {n}")

    return cache_path


def build_tract_cache(csv_path, cache_path, transform=DEFAULT_TRANSFORM):
    """
    Load images listed in *csv_path*, group them by GEOID, and save a
    single .pt file at *cache_path* containing:

        cache["images"]  — dict mapping GEOID -> torch.Tensor of shape (n_tiles, 3, 224, 224)
        cache["income"]  — dict mapping GEOID -> float (median income)

    Each GEOID (tract) has a variable number of tiles. The cache preserves
    this grouping so the multi-tile dataset can look up all tiles for a tract
    at once.

    Parameters
    ----------
    csv_path : str or Path
        Path to the processed CSV (must have columns ``image_path``,
        ``GEOID``, and ``median_income``).
    cache_path : str or Path
        Where to write the output ``.pt`` file.
    transform : callable, optional
        Torchvision transform pipeline.  Defaults to Resize + ToTensor.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # cache dict will map GEOID (string) -> images tensor and income scalar
    cache = {"images": {}, "income": {}}

    # group dataframe rows by GEOID so we can collect tiles per tract
    groups = df.groupby("GEOID")

    print(f"Building tract cache from {len(df)} samples ({len(groups)} tracts) ...")

    for geoid, group in groups:

        images = []  # list to collect transformed image tensors for this tract

        for _, row in group.iterrows():

            # open the image file, strip whitespace, and convert to RGB
            try:
                img = Image.open(row["image_path"].strip()).convert("RGB")
            except Exception as e:
                print(
                    f"  WARNING: could not load image for GEOID {geoid} — skipping ({e})"
                )
                continue

            # apply the model's transform pipeline (resize, to-tensor, normalize)
            img = transform(img)

            # collect the transformed tensor
            images.append(img)

        # Guard: skip tracts where every tile failed to load
        if len(images) == 0:
            print(f"  WARNING: no valid images for GEOID {geoid} — skipping tract")
            continue

        # stack per-tract images into a single float32 tensor
        images = torch.stack(images).float()

        # store images in cache under the stringified GEOID
        cache["images"][str(geoid)] = images

        # store the median income label once for this GEOID (first row)
        cache["income"][str(geoid)] = group["median_income"].iloc[0]

    # Guard: if no tracts had valid images, we can't save a useful cache
    if len(cache["images"]) == 0:
        raise RuntimeError(
            f"No valid images found in {csv_path}. "
            "Check that image paths are correct and files exist."
        )

    # persist the cache to disk for later fast loading
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(cache, cache_path)

    print(f"Tract cache saved → {cache_path}")
    print(f"  tracts: {len(cache['images'])}")
    tile_counts = [v.shape[0] for v in cache["images"].values()]
    print(
        f"  tiles per tract: min={min(tile_counts)}, max={max(tile_counts)}, avg={sum(tile_counts)/len(tile_counts):.1f}"
    )

    return cache_path
