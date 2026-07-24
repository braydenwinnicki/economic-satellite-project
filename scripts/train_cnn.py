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
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


model = ConvNN()

# split data

df = pd.read_csv(
    "/Users/braydenwinnicki/CODE/econ_project/data/processed/processed_ct_tracts.csv"
)

df_train, df_test = train_test_split(df, test_size=0.20, random_state=42)

# Z-score normalization: shift incomes to mean=0, std=1.
# Only use train's stats to avoid leaking test info into training.
mean_income = df_train["median_income"].mean()
std_income = df_train["median_income"].std()

df_train["median_income"] = (df_train["median_income"] - mean_income) / std_income
df_test["median_income"] = (df_test["median_income"] - mean_income) / std_income


transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


train_dataset = CensusDataset(df_train, transform=transform)
test_dataset = CensusDataset(df_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# training loop

epochs = 10

model.train()

for epoch in range(epochs):

    total_loss = 0

    for images, incomes in train_loader:

        # images shape: (batch_size, 3, 224, 224)
        predictions = model(images)

        # squeeze() removes the extra dim: (batch_size, 1) → (batch_size,)
        loss = criterion(predictions.squeeze(), incomes.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    print(f"train Epoch {epoch+1}: {avg_loss:.4f}")


# save the trained weights
torch.save(model.state_dict(), PROJECT_ROOT / "models" / "cnn_v1.pth")

print("Saved model to models/cnn_v1.pth")