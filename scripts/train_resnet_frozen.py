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
from sklearn.metrics import mean_absolute_error


model = ResNetRegressor()
transform = model.weights.transforms()


# split data

df = pd.read_csv(
    "/Users/braydenwinnicki/CODE/econ_project/data/processed/processed_ct_tracts.csv"
)

df_train, df_test = train_test_split(df, test_size=0.20, random_state=42)

# Z-score normalization: shift incomes so they have mean=0 and std=1.
# This helps the model converge faster. We only use train's mean/std
# to avoid leaking information from test into training.
mean_income = df_train["median_income"].mean()
std_income = df_train["median_income"].std()

df_train["median_income"] = (df_train["median_income"] - mean_income) / std_income

df_test["median_income"] = (df_test["median_income"] - mean_income) / std_income

# Create datasets — CensusDataset loads images on-the-fly from file paths
train_dataset = CensusDataset(df_train, transform=transform)
test_dataset = CensusDataset(df_test, transform=transform)

# DataLoader batches the data and shuffles it each epoch.
# batch_size=32 means 32 images get processed together in one forward pass.
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


criterion = nn.MSELoss()  # mean squared error — penalizes large errors more
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training loop

epochs = 10

model.train()  # enables training-specific behavior (affects things like dropout/batchnorm)

for epoch in range(epochs):

    total_loss = 0

    for images, incomes in train_loader:

        # images shape: (batch_size, 3, 224, 224)
        predictions = model(images)

        # squeeze() removes the extra dimension: (batch_size, 1) → (batch_size,)
        # so it matches the shape of incomes for the loss calculation
        loss = criterion(predictions.squeeze(), incomes.float())

        # zero_grad clears old gradient values so they don't accumulate
        optimizer.zero_grad()

        # backward() computes the gradient of the loss with respect to
        # every parameter that has requires_grad=True
        loss.backward()

        # step() uses the gradients to update the model weights
        optimizer.step()

        total_loss += loss.item()  # .item() extracts the scalar from a 1-element tensor

    avg_loss = total_loss / len(train_loader)

    print(f"train Epoch {epoch+1}: {avg_loss:.4f}")


# Save the trained weights so we can load them later for evaluation.
# .state_dict() returns all the learnable parameters as a dictionary of tensors.
torch.save(model.state_dict(), PROJECT_ROOT / "models" / "resnet18_frozen.pth")

print("Saved model to models/resnet18_frozen.pth")