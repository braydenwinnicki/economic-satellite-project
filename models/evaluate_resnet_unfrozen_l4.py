import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from models.resnet_frozen import ResNetRegressor
import pandas as pd
from models.dataset import CensusDataset
from torch.utils.data import DataLoader
import torch
from torch.utils.data import DataLoader
from models.collate import collate_fn
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from src.splitting import split_by_tract


def main():
    model = ResNetRegressor()
    transform = model.weights.transforms()

    # device autodetection: prefer CUDA, then MPS, else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # load trained weights onto the selected device
    model.load_state_dict(
        torch.load(PROJECT_ROOT / "models" / "resnet_unfrozen_l4.pth", map_location=device)
    )
    model.to(device)
    model.eval()

    # split data using the tile-level processed CSV that exists in this repo
    # (same dataset form used by train_resnet_frozen.py)
    input_csv = PROJECT_ROOT / "data" / "processed" / "processed_ct_tracts_tiles.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Evaluation CSV missing: {input_csv}")

    df = pd.read_csv(input_csv, dtype={"GEOID": str})

    df.columns = df.columns.str.strip()

    df_train, df_test = split_by_tract(df, test_size=0.20, random_state=42)

    # compute normalizing stats from the training tracts only
    train_tract_labels = df_train.drop_duplicates(subset="GEOID")["median_income"]
    mean_income = train_tract_labels.mean()
    std_income = train_tract_labels.std()
    if std_income == 0:
        std_income = 1.0

    df_train["median_income"] = (df_train["median_income"] - mean_income) / std_income
    df_test["median_income"] = (df_test["median_income"] - mean_income) / std_income

    print(f"Device: {device}, test rows: {len(df_test)}")

    cache_path = PROJECT_ROOT / "data" / "processed" / "census_images_cache.pt"
    if not cache_path.exists():
        raise FileNotFoundError(f"Image cache missing: {cache_path}")

    print(f"Loading dataset from cache: {cache_path}")

    test_dataset = CensusDataset(df_test, str(cache_path))

    pin_memory = device.type == "cuda"
    num_workers = 0 if sys.platform == "darwin" else 2
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=4,
    )
    print(
        f"test loader batch size: {test_loader.batch_size}, num_workers={test_loader.num_workers}"
    )

    criterion = nn.MSELoss()  # mean squared loss
    optimizer = torch.optim.Adam(  # an optimizer (not needed for eval)
        model.parameters(), lr=0.001
    )

    # testing
    all_predictions = []
    all_targets = []
    all_geoids = []

    with torch.no_grad():
        total_loss = 0

        for images, mask, incomes, geoids in test_loader:
            # move tensors to device and ensure float dtype for images
            images = images.float().to(device)
            mask = mask.to(device)
            incomes = incomes.to(device)

            # forward pass
            predictions = model(images, mask)

            all_predictions.extend(predictions.squeeze().tolist())
            all_targets.extend(incomes.tolist())
            all_geoids.extend(geoids)

            # calculate error
            loss = criterion(predictions.squeeze(), incomes.float())

            total_loss += loss.item()

        avg_test_loss = total_loss / len(test_loader)

    print("prediction sample:", all_predictions[:5])
    print("target sample:", all_targets[:5])
    print("mean/std:", mean_income, std_income)

    # convert back to dollars
    predictions_dollars = [p * std_income + mean_income for p in all_predictions]
    targets_dollars = [t * std_income + mean_income for t in all_targets]

    mae = mean_absolute_error(targets_dollars, predictions_dollars)
    rmse = np.sqrt(mean_squared_error(targets_dollars, predictions_dollars))
    r2 = r2_score(targets_dollars, predictions_dollars)

    print(f"AVG TEST LOSS: {avg_test_loss}")
    print(f"TESTING MAE: {mae}")
    print(f"TESTING RMSE: {rmse}")
    print(f"TESTING Rsquared: {r2}")

    # put results in a dataframe for inspection
    results = pd.DataFrame(
        {
            "GEOID": all_geoids,
            "prediction": predictions_dollars,
            "actual": targets_dollars,
        }
    )

    results["error"] = (results["prediction"] - results["actual"]).abs()

    worst = results.sort_values(by="error", ascending=False)

    print(worst.head(10))


if __name__ == "__main__":
    main()
