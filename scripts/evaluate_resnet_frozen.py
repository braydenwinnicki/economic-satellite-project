import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from models.resnet_frozen import ResNetRegressor
import pandas as pd
from models.dataset import CensusDataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

model = ResNetRegressor()

# .transforms() returns the exact image preprocessing this pretrained model expects
# (resize + ImageNet mean/std normalization). Using basic transforms would hurt accuracy.
transform = model.weights.transforms()


# load the saved weights from training. .state_dict() was saved as a dict,
# .load_state_dict() loads it back into the model's parameters.
model.load_state_dict(torch.load(PROJECT_ROOT / "models" / "resnet18_frozen.pth"))

# model.eval() switches to evaluation mode (disables dropout, fixes batchnorm stats)
model.eval()


# split data

df = pd.read_csv(
    "/Users/braydenwinnicki/Desktop/econ_project/data/processed/processed_ct_tracts.csv"
)
# fix file paths in the CSV if they point to the old CODE directory
# .str.replace() does a find-and-replace on every string in the "image_path" column
df["image_path"] = df["image_path"].str.replace(
    "/Users/braydenwinnicki/CODE/econ_project",
    "/Users/braydenwinnicki/Desktop/econ_project"
)

df_train, df_test = train_test_split(df, test_size=0.20, random_state=42)

# Z-score normalization — must use the same mean/std from training
# so the model sees test data in the same scale it was trained on
# .mean() = average; .std() = standard deviation (how spread out the values are)
mean_income = df_train["median_income"].mean()
std_income = df_train["median_income"].std()

# (value - mean) / std shifts so the average becomes 0 and spread becomes 1
df_train["median_income"] = (df_train["median_income"] - mean_income) / std_income

# Apply the same transformation using train's stats (not test's own mean/std)
df_test["median_income"] = (df_test["median_income"] - mean_income) / std_income

test_dataset = CensusDataset(df_test, transform=transform)

# DataLoader batches the data; shuffle=False means we process in a fixed order
# (shuffling isn't needed for evaluation)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


criterion = nn.MSELoss()


# testing loop — no gradients needed, so wrap in torch.no_grad()
# to save memory and speed things up

all_predictions = []
all_targets = []


with torch.no_grad():

    total_loss = 0

    for images, incomes in test_loader:

        # Forward pass — generate predictions without tracking gradients
        predictions = model(images)

        # .squeeze() removes extra dimension: (batch, 1) -> (batch,)
        # .tolist() converts tensor to a Python list
        # .extend() appends each element individually, building one flat list
        all_predictions.extend(predictions.squeeze().tolist())
        all_targets.extend(incomes.tolist())

        # .float() ensures incomes are float32 for the loss calculation
        loss = criterion(predictions.squeeze(), incomes.float())

        # .item() extracts the scalar value from a 1-element tensor as a Python float
        total_loss += loss.item()

    avg_test_loss = total_loss / len(test_loader)


# Convert normalized predictions back to dollar amounts
# undo the z-score: prediction * std + mean = original dollar value
# (This is a list comprehension — applies the formula to every prediction)
predictions_dollars = [p * std_income + mean_income for p in all_predictions]

# Same denormalization for the actual income values
targets_dollars = [t * std_income + mean_income for t in all_targets]

# mean_absolute_error = average of |actual - predicted| (in dollars)
mae = mean_absolute_error(targets_dollars, predictions_dollars)

avg_test_loss = total_loss / len(test_loader)

# RMSE = Root Mean Squared Error — sqrt of average squared error
# np.sqrt() computes the square root; mean_squared_error computes the average squared difference
rmse = np.sqrt(mean_squared_error(targets_dollars, predictions_dollars))

# R-squared = how much of the variance in actuals is explained by predictions
# 1.0 = perfect, 0.0 = no better than guessing the mean, negative = worse than guessing the mean
r2 = r2_score(targets_dollars, predictions_dollars)

print(f"AVG TEST LOSS: {avg_test_loss}")
print(f"TESTING MAE: {mae}")
print(f"TESTING RMSE: {rmse}")
print(f"TESTING Rsquared: {r2}")


# put results in a dataframe

# .copy() avoids a SettingWithCopyWarning when modifying the DataFrame
# .reset_index(drop=True) resets row indices to 0,1,2... and discards the old index column
results = df_test.copy().reset_index(drop=True)

results["prediction"] = predictions_dollars
results["actual"] = targets_dollars

# .abs() takes the absolute value so errors are always positive
results["error"] = (results["prediction"] - results["actual"]).abs()

# .sort_values(by="error", ascending=False) sorts from highest error to lowest
worst = results.sort_values(by="error", ascending=False)
# print 10 worst for eval
# [["GEOID", "median_income", ...]] picks specific columns
# .head(10) takes only the first 10 rows (the worst offenders after sorting)
print(worst[["GEOID", "median_income", "prediction", "error", "image_path"]].head(10))