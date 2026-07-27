import sys
from pathlib import Path
import pandas as pd

# Path(__file__) = this script's path; .resolve() makes it absolute; .parents[1] = 2 directories up = project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from models.dataset import CensusDataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from models.cnn import ConvNN
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


model = ConvNN()

# split data

df = pd.read_csv(
    "/Users/braydenwinnicki/CODE/econ_project/data/processed/processed_ct_tracts.csv"
)

# train_test_split splits rows into two groups: train (80%) and test (20%)
# random_state=42 ensures you get the same split every time
df_train, df_test = train_test_split(df, test_size=0.20, random_state=42)

# Z-score normalization: shift incomes to mean=0, std=1.
# Only use train's stats to avoid leaking test info into training.
# .mean() = average of the column; .std() = standard deviation (spread)
mean_income = df_train["median_income"].mean()
std_income = df_train["median_income"].std()

# (value - mean) / std shifts so the average becomes 0 and spread becomes 1
df_train["median_income"] = (df_train["median_income"] - mean_income) / std_income

# Apply the same transformation to test data using train's stats
df_test["median_income"] = (df_test["median_income"] - mean_income) / std_income


# transforms.Compose chains multiple image processing steps together
# Resize((224, 224)) makes all images 224x224 pixels (what ResNet expects)
# ToTensor() converts pixel values from 0-255 to 0.0-1.0 and rearranges to (C, H, W) format
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


# CensusDataset loads images on-the-fly from file paths listed in the DataFrame
train_dataset = CensusDataset(df_train, transform=transform)
test_dataset = CensusDataset(df_test, transform=transform)

# DataLoader batches the data and shuffles it each epoch
# batch_size=32 means 32 images get processed together in one forward pass
# shuffle=True randomizes the order of samples each epoch (helps training)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


# nn.MSELoss() = mean squared error loss — penalizes big errors more than small ones
criterion = nn.MSELoss()
# torch.optim.Adam adjusts model weights using gradient descent
# model.parameters() gives the optimizer access to all trainable weights
# lr=0.001 is the learning rate — how big each weight update step is
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# training loop

epochs = 10

# model.train() enables training-specific behaviors
model.train()

for epoch in range(epochs):

    total_loss = 0

    # Each inner loop iteration processes one batch of 32 images
    for images, incomes in train_loader:

        # images shape: (batch_size=32, 3 color channels, 224 height, 224 width)
        # model(images) runs the forward pass — generates predictions from pixel values
        predictions = model(images)

        # squeeze() removes the extra dimension: (batch_size, 1) → (batch_size,)
        # This matches the shape of incomes so the loss calculation works
        # .float() ensures incomes are float32 (might be int64 from CSV)
        loss = criterion(predictions.squeeze(), incomes.float())

        # zero_grad clears old gradient values so they don't accumulate across batches
        optimizer.zero_grad()
        # backward() computes gradients of the loss with respect to every parameter
        loss.backward()
        # step() updates the model weights using the computed gradients
        optimizer.step()

        # .item() extracts the single scalar value from a 1-element tensor as a Python float
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"train Epoch {epoch+1}: {avg_loss:.4f}")


# save the trained weights
# .state_dict() returns all learnable parameters as a dictionary of tensors
# torch.save() writes that dictionary to disk
torch.save(model.state_dict(), PROJECT_ROOT / "models" / "cnn_v1.pth")

print("Saved model to models/cnn_v1.pth")
