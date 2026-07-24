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
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

model = ResNetRegressor()
transform = model.weights.transforms()


# load the saved weights from training. .state_dict() was saved as a dict,
# .load_state_dict() loads it back into the model's parameters.
model.load_state_dict(torch.load(PROJECT_ROOT / "models" / "resnet18_frozen.pth"))

model.eval()  # switch to evaluation mode (disables dropout, fixes batchnorm)


# split data

df = pd.read_csv(
    "/Users/braydenwinnicki/Desktop/econ_project/data/processed/processed_ct_tracts.csv"
)
# fix file paths in the CSV if they point to the old CODE directory
df["image_path"] = df["image_path"].str.replace(
    "/Users/braydenwinnicki/CODE/econ_project",
    "/Users/braydenwinnicki/Desktop/econ_project"
)

df_train, df_test = train_test_split(df, test_size=0.20, random_state=42)

# Z-score normalization — must use the same mean/std from training
# so the model sees test data in the same scale it was trained on
mean_income = df_train["median_income"].mean()
std_income = df_train["median_income"].std()

df_train["median_income"] = (df_train["median_income"] - mean_income) / std_income
df_test["median_income"] = (df_test["median_income"] - mean_income) / std_income

train_dataset = CensusDataset(df_train, transform=transform)
test_dataset = CensusDataset(df_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# testing loop — no gradients needed, so wrap in torch.no_grad()
# to save memory and speed things up

all_predictions = []
all_targets = []


with torch.no_grad():

    total_loss = 0

    for images, incomes in test_loader:

        predictions = model(images)

        # extend appends each element of the list individually
        # so we build up a flat list of all predictions/targets
        all_predictions.extend(predictions.squeeze().tolist())
        all_targets.extend(incomes.tolist())

        loss = criterion(predictions.squeeze(), incomes.float())

        total_loss += loss.item()

    avg_test_loss = total_loss / len(test_loader)


# Convert normalized predictions back to dollar amounts
# undo the z-score: prediction * std + mean = original dollar value
predictions_dollars = [p * std_income + mean_income for p in all_predictions]
targets_dollars = [t * std_income + mean_income for t in all_targets]

mae = mean_absolute_error(targets_dollars, predictions_dollars)

avg_test_loss = total_loss / len(test_loader)

rmse = np.sqrt(mean_squared_error(targets_dollars, predictions_dollars))

r2 = r2_score(targets_dollars, predictions_dollars)

print(f"AVG TEST LOSS: {avg_test_loss}")
print(f"TESTING MAE: {mae}")
print(f"TESTING RMSE: {rmse}")
print(f"TESTING Rsquared: {r2}")


# graph results
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))


"""
# Scatter plot
plt.scatter(targets_dollars, predictions_dollars, alpha=0.6)

# Perfect prediction line
min_income = min(targets_dollars)
max_income = max(targets_dollars)

plt.plot(
    [min_income, max_income],
    [min_income, max_income],
    "r--",
    linewidth=2,
    label="Perfect Prediction",
)

plt.xlabel("Actual Median Income ($)")
plt.ylabel("Predicted Median Income ($)")
plt.title("Predicted vs Actual Median Income")

plt.legend()

plt.tight_layout()
plt.show()
"""

# put results in a dataframe

results = df_test.copy().reset_index(drop=True)

results["prediction"] = predictions_dollars
results["actual"] = targets_dollars

results["error"] = (results["prediction"] - results["actual"]).abs()

worst = results.sort_values(by="error", ascending=False)
# print 10 worst for eval
print(worst[["GEOID", "median_income", "prediction", "error", "image_path"]].head(10))