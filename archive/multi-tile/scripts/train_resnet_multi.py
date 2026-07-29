import sys
from pathlib import Path

# Path(__file__) = this script's path; .resolve() makes it absolute; .parents[1] = 2 directories up = project root
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

# .transforms() returns the exact image preprocessing that this pretrained model expects
# This includes resizing to 224x224 AND normalizing with ImageNet's mean/std values
# If you use basic transforms instead, the pretrained weights won't work as well
transform = model.weights.transforms()


# split data

df = pd.read_csv(
    "/Users/braydenwinnicki/CODE/econ_project/data/processed/processed_ct_tracts.csv"
)

# train_test_split splits rows into two groups: train (80%) and test (20%)
# random_state=42 ensures you get the same split every time
df_train, df_test = train_test_split(df, test_size=0.20, random_state=42)

# Z-score normalization: shift incomes so they have mean=0 and std=1.
# This helps the model converge faster. We only use train's mean/std
# to avoid leaking information from test into training.
# .mean() = average of the column; .std() = standard deviation (spread)
mean_income = df_train["median_income"].mean()
std_income = df_train["median_income"].std()

# (value - mean) / std shifts each value so the new average is 0 and spread is 1
df_train["median_income"] = (df_train["median_income"] - mean_income) / std_income

# Apply the same transformation to test data using train's derived stats
df_test["median_income"] = (df_test["median_income"] - mean_income) / std_income

# Create datasets — CensusDataset loads images on-the-fly from file paths
train_dataset = CensusDataset(df_train, transform=transform)
test_dataset = CensusDataset(df_test, transform=transform)

# DataLoader batches the data and shuffles it each epoch.
# batch_size=32 means 32 images get processed together in one forward pass.
# shuffle=True randomizes sample order each epoch (prevents overfitting to order)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# nn.MSELoss() = mean squared error — penalizes large errors more than small ones
criterion = nn.MSELoss()
# torch.optim.Adam adjusts model weights using gradient descent
# model.parameters() gives the optimizer access to all trainable weights
# lr=0.001 is the learning rate — controls how big each weight update is
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# training loop

epochs = 10

# model.train() enables training-specific behavior (affects batchnorm, dropout, etc.)
model.train()

for epoch in range(epochs):

    total_loss = 0

    # Each inner loop iteration processes one batch of 32 images
    for images, incomes in train_loader:

        # images shape: (batch_size=32, 3 color channels, 224 height, 224 width)
        # model(images) runs the forward pass through all ResNet layers
        predictions = model(images)

        # squeeze() removes the extra dimension: (batch_size, 1) → (batch_size,)
        # so it matches the shape of incomes for the loss calculation
        # .float() ensures incomes are float32 (might be int64 from CSV)
        loss = criterion(predictions.squeeze(), incomes.float())

        # zero_grad clears old gradient values so they don't accumulate across batches
        optimizer.zero_grad()

        # backward() computes the gradient of the loss with respect to
        # every parameter that has requires_grad=True
        loss.backward()

        # step() uses the gradients to update the model weights
        optimizer.step()

        # .item() extracts the scalar value from a 1-element tensor as a Python float
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"train Epoch {epoch+1}: {avg_loss:.4f}")


# Save the trained weights so we can load them later for evaluation.
# .state_dict() returns all the learnable parameters as a dictionary of tensors.
torch.save(model.state_dict(), PROJECT_ROOT / "models" / "resnet18_frozen.pth")

print("Saved model to models/resnet18_frozen.pth")
