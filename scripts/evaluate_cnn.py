import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from models.dataset import CensusDataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from models.cnn import ConvNN
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from models.collate import collate_fn
from src.splitting import split_by_tract


def main():
    model = ConvNN()  # instantiate the CNN model

    # device autodetection: prefer CUDA, then MPS, else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    # load saved weights onto the selected device
    model.load_state_dict(
        torch.load(PROJECT_ROOT / "models" / "cnn_v1_tiles.pth", map_location=device)
    )
    model.to(device)
    model.eval()  # set to evaluation mode (disables training-only behavior)

    # load and prepare dataset
    df = pd.read_csv(
        PROJECT_ROOT / "data" / "processed" / "processed_ct_tracts_tiles.csv",
        dtype={"GEOID": str},
    )
    df.columns = df.columns.str.strip()

    df_train, df_test = split_by_tract(df, test_size=0.20, random_state=42)

    # compute normalization stats from training tracts only
    train_tract_labels = df_train.drop_duplicates(subset="GEOID")["median_income"]
    mean_income = train_tract_labels.mean()
    std_income = train_tract_labels.std()
    if std_income == 0:
        std_income = 1.0

    # normalize labels
    df_train["median_income"] = (df_train["median_income"] - mean_income) / std_income
    df_test["median_income"] = (df_test["median_income"] - mean_income) / std_income

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )

    test_dataset = CensusDataset(
        df_test,
        str(PROJECT_ROOT / "data" / "processed" / "census_images_cache.pt"),
    )

    pin_memory = device.type == "cuda"
    num_workers = 0 if sys.platform == "darwin" else 2
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )

    criterion = nn.MSELoss()  # mean squared loss
    optimizer = torch.optim.Adam(  # optimizer (not used in eval but kept for parity)
        model.parameters(), lr=0.001
    )

    # testing / evaluation loop
    all_predictions = []
    all_targets = []
    all_geoids = []

    with torch.no_grad():
        total_loss = 0

        model.eval()  # ensure model is in eval mode

        for images, mask, incomes, geoids in test_loader:
            # move tensors to device and ensure float dtype for images
            images = images.float().to(device)
            mask = mask.to(device)
            incomes = incomes.to(device)

            # forward pass to get tract-level predictions
            predictions = model(images, mask)

            # collect results into python lists for later metrics/plots
            all_predictions.extend(predictions.squeeze().tolist())
            all_targets.extend(incomes.tolist())
            all_geoids.extend(geoids)

            # compute batch loss for reporting
            loss = criterion(predictions.squeeze(), incomes.float())

            total_loss += loss.item()

        avg_test_loss = total_loss / len(test_loader)

    # convert normalized predictions back to dollar values for human-readable metrics
    predictions_dollars = [p * std_income + mean_income for p in all_predictions]
    targets_dollars = [t * std_income + mean_income for t in all_targets]

    mae = mean_absolute_error(targets_dollars, predictions_dollars)

    rmse = np.sqrt(mean_squared_error(targets_dollars, predictions_dollars))

    r2 = r2_score(targets_dollars, predictions_dollars)

    print(f"AVG TEST LOSS: {avg_test_loss}")
    print(f"TESTING MAE: {mae}")
    print(f"TESTING RMSE: {rmse}")
    print(f"TESTING Rsquared: {r2}")

    # quick histogram of tract median incomes (original, not predictions)
    import matplotlib.pyplot as plt

    plt.xlabel("Median income")
    plt.ylabel("Number of tracts")
    tract_level = df.drop_duplicates(subset="GEOID")
    plt.hist(tract_level["median_income"], bins=40)
    plt.show()

    # assemble results DataFrame and show worst errors
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
