import sys
from pathlib import Path
import pandas as pd

# Path(__file__) = this script's path; .resolve() = absolute path; .parents[1] = 2 levels up → project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from models.dataset import CensusDataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from models.CNN import ConvNN
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

model = ConvNN()

model.load_state_dict(torch.load(PROJECT_ROOT / "models" / "cnn_v1.pth"))

model.eval()


# split data

df = pd.read_csv(
    "/Users/braydenwinnicki/Desktop/econ_project/data/processed/processed_ct_tracts.csv"
)

# .str.replace() does a find-and-replace on every string in the "image_path" column
df["image_path"] = df["image_path"].str.replace(
    "/Users/braydenwinnicki/CODE/econ_project",
    "/Users/braydenwinnicki/Desktop/econ_project"
)

df_train, df_test = train_test_split(df, test_size=0.20, random_state=42)

# use z-scale normalizing to shrink numbers and help the dataset. dont use test.mean() becuase it would leak

# .mean() computes the average of the "median_income" column for z-score normalization
mean_income = df_train["median_income"].mean()
# .std() computes the standard deviation (spread) of the "median_income" column
std_income = df_train["median_income"].std()

# Z-score normalization: (value - mean) / std  -> shifts to mean=0, std=1
# Only uses train stats to avoid leaking test data into training
df_train["median_income"] = (df_train["median_income"] - mean_income) / std_income

# Apply the same train-derived mean/std to test data so it's on the same scale
df_test["median_income"] = (df_test["median_income"] - mean_income) / std_income


transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


test_dataset = CensusDataset(df_test, transform=transform)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


criterion = nn.MSELoss()  # mean squared loss

# testing

all_predictions = []
all_targets = []


with torch.no_grad():

    total_loss = 0

    for images, incomes in test_loader:

        # forward pass
        predictions = model(images)

        # .squeeze() removes the extra dimension: shape (batch, 1) - (batch,)
        # .tolist() converts the PyTorch tensor to a Python list
        # .extend() appends each element individually - builds a flat list
        all_predictions.extend(predictions.squeeze().tolist())
        all_targets.extend(incomes.tolist())

        # calculate error
        # .squeeze() matches shapes: model outputs (batch, 1), incomes are (batch,)
        # .float() ensures incomes are float32 for the MSE loss computation
        loss = criterion(predictions.squeeze(), incomes.float())

        # .item() pulls the single scalar value out of a 1-element tensor, converting to a Python float
        total_loss += loss.item()

    avg_test_loss = total_loss / len(test_loader)


# calculate error via MAE

# Undo z-score normalization: prediction * std + mean = original dollar amount
# (This is a list comprehension — it applies the formula to every prediction in the list)
predictions_dollars = [p * std_income + mean_income for p in all_predictions]

# Same denormalization for the actual income values
targets_dollars = [t * std_income + mean_income for t in all_targets]

mae = mean_absolute_error(targets_dollars, predictions_dollars)

rmse = np.sqrt(mean_squared_error(targets_dollars, predictions_dollars))

r2 = r2_score(targets_dollars, predictions_dollars)

print(f"AVG TEST LOSS: {avg_test_loss}")
print(f"TESTING MAE: {mae}")
print(f"TESTING RMSE: {rmse}")
print(f"TESTING Rsquared: {r2}")


# put results in a dataframe

# .copy() avoids a SettingWithCopyWarning when modifying the DataFrame
# .reset_index(drop=True) resets row indices to 0, 1, 2... and drops the old index column
results = df_test.copy().reset_index(drop=True)

results["prediction"] = predictions_dollars
results["actual"] = targets_dollars

# (prediction - actual)  for each row
# .abs() takes the absolute value so errors are always positive
results["error"] = (results["prediction"] - results["actual"]).abs()

# .sort_values(by="error", ascending=False) sorts rows from highest error to lowest
worst = results.sort_values(by="error", ascending=False)
# print 10 worst for eval
# [["GEOID", "median_income", "prediction", "error", "image_path"]] selects specific columns
# .head(10) takes only the first 10 rows (the 10 worst predictions after the sort above)
print(worst[["GEOID", "median_income", "prediction", "error", "image_path"]].head(10))
