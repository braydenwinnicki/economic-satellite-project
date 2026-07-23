import os

# Prevent PyTorch memory fragmentation on Kaggle GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from pathlib import Path
import sys

# Ensure project imports work when run as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import gc
import pandas as pd
from sklearn.metrics import mean_absolute_error
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.collate import collate_fn
from models.dataset import CensusDataset
from models.resnet_frozen import ResNetRegressor
from src.splitting import split_by_tract


def main():
    # Clear leftover Kaggle CUDA memory
    gc.collect()
    torch.cuda.empty_cache()

    # Instantiate frozen resnet
    model = ResNetRegressor()

    # Device autodetection
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    model.to(device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Load CSV of tiles and GEOIDs
    df = pd.read_csv(
        "/kaggle/input/datasets/braydenwinnicki/processed-csv-tracts-tiles/processed_ct_tracts_tiles.csv"
    )
    df.columns = df.columns.str.strip()

    # Split train/test by tract
    df_train, df_test = split_by_tract(df, test_size=0.20, random_state=42)

    # Compute normalization stats
    train_tract_labels = df_train.drop_duplicates(subset="GEOID")[
        "median_income"
    ]
    mean_income = train_tract_labels.mean()
    std_income = train_tract_labels.std()
    if std_income == 0:
        std_income = 1.0

    df_train["median_income"] = (
        df_train["median_income"] - mean_income
    ) / std_income

    # Create dataset
    train_dataset = CensusDataset(
        df_train,
        "/kaggle/input/datasets/braydenwinnicki/census-images-cache-pt/census_images_cache.pt",
    )

    # Safety batch sizing: mini-batch 16 + accumulation steps = effective batch 32
    mini_batch = 16 if device.type == "cuda" else 8
    accumulation_steps = 2 if device.type == "cuda" else 4

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=mini_batch,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        persistent_workers=True,
        pin_memory=pin_memory,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.model.fc.parameters(), lr=0.001)

    # Gradient Scaler for Mixed Precision
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    epochs = 10
    model.train()

    print(
        f"Starting training on {device} (Effective Batch Size: {mini_batch * accumulation_steps})..."
    )

    for epoch in range(epochs):
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, (images, mask, incomes, geoids) in enumerate(
            train_loader
        ):
            # Async tensor transfer
            images = images.float().to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            incomes = incomes.to(device, non_blocking=True)

            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                predictions = model(images, mask)
                loss = (
                    criterion(predictions.squeeze(), incomes.float())
                    / accumulation_steps
                )

            # Scaled backward pass
            scaler.scale(loss).backward()

            # Step optimizer every N mini-batches
            if (batch_idx + 1) % accumulation_steps == 0 or (
                batch_idx + 1
            ) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * accumulation_steps

        avg_loss = total_loss / len(train_loader)
        print(f"Train Epoch {epoch+1}/{epochs} - MSE Loss: {avg_loss:.4f}")

    # Save the trained model weights
    save_path = "/kaggle/working/resnet-frozen.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Successfully saved frozen model checkpoint to {save_path}")


if __name__ == "__main__":
    main()