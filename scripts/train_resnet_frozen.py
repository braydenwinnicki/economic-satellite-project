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

# use z-scale normalizing to shrink numbers and help the dataset. dont use test.mean() becuase it would leak

mean_income = df_train["median_income"].mean()
std_income = df_train["median_income"].std()

df_train["median_income"] = (df_train["median_income"] - mean_income) / std_income

df_test["median_income"] = (df_test["median_income"] - mean_income) / std_income

train_dataset = CensusDataset(df_train, transform=transform)

test_dataset = CensusDataset(df_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


criterion = nn.MSELoss()  # mean squared loss
optimizer = torch.optim.Adam(  # an optimizer adjusts weights via gradient
    model.parameters(), lr=0.001
)

# training

epochs = 10

model.train()  # turn on train mode

for epoch in range(epochs):

    total_loss = 0

    for images, incomes in train_loader:

        # forward pass
        predictions = model(images)

        # calculate error
        loss = criterion(predictions.squeeze(), incomes.float())

        # clear old gradients
        optimizer.zero_grad()

        # calculate gradients
        loss.backward()

        # update weights
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"train Epoch {epoch+1}: {avg_loss:.4f}")


# save model
torch.save(model.state_dict(), PROJECT_ROOT / "models" / "resnet18_frozen.pth")

print("Saved model to models/resnet18_frozen.pth")
