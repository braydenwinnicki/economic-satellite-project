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

model.load_state_dict(
    torch.load(PROJECT_ROOT / "models" / "cnn_v1.pth")
)

model.eval()


#split data

df = pd.read_csv("/Users/braydenwinnicki/CODE/econ_project/data/processed/processed_ct_tracts.csv")

df_train, df_test = train_test_split(df, test_size=.20, random_state=42)

#use z-scale normalizing to shrink numbers and help the dataset. dont use test.mean() becuase it would leak

mean_income = df_train["median_income"].mean()
std_income = df_train["median_income"].std()

df_train["median_income"] = (
    df_train["median_income"] - mean_income
) / std_income

df_test["median_income"] = (
    df_test["median_income"] - mean_income
) / std_income



transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])


train_dataset = CensusDataset(
    df_train,
    transform=transform
)

test_dataset = CensusDataset(
    df_test,
    transform=transform
    )

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=True
)


criterion = nn.MSELoss() #mean squared loss
optimizer = torch.optim.Adam(  #an optimizer adjusts weights via gradient
    model.parameters(),
    lr=0.001
)

#testing 

all_predictions = []
all_targets = []


with torch.no_grad():


    total_loss = 0
    
    model.eval() # put on test mode


    for images, incomes in test_loader:

        # forward pass
        predictions = model(images)


        all_predictions.extend(predictions.squeeze().tolist())
        all_targets.extend(incomes.tolist())

        # calculate error
        loss = criterion(
            predictions.squeeze(),
            incomes.float()
        )

        total_loss += loss.item()

    avg_test_loss = total_loss / len(test_loader)


        


#calculate error via MAE 

predictions_dollars = [
    p * std_income + mean_income
    for p in all_predictions]

targets_dollars = [
    t * std_income + mean_income
    for t in all_targets]

mae = mean_absolute_error(targets_dollars, predictions_dollars)

avg_test_loss = total_loss / len(test_loader)

print(f"AVG TEST LOSS: {avg_test_loss}")
print(f"TESTING MAE: {mae}")


#graph results 
import matplotlib.pyplot as plt


# Scatter plot
plt.scatter(
    targets_dollars,
    predictions_dollars,
    alpha=0.6
)

# Perfect prediction line
min_income = min(targets_dollars)
max_income = max(targets_dollars)

plt.plot(
    [min_income, max_income],
    [min_income, max_income],
    "r--",
    linewidth=2,
    label="Perfect Prediction"
)

plt.xlabel("Actual Median Income ($)")
plt.ylabel("Predicted Median Income ($)")
plt.title("Predicted vs Actual Median Income")

plt.legend()

plt.tight_layout()
plt.show()

#print 10 worst errors

errors = [
    abs(p - t)
    for p, t in zip(predictions_dollars, targets_dollars)
]

results = pd.DataFrame({
    "Actual": targets_dollars,
    "Predicted": predictions_dollars,
    "Error": errors
})

print(
    results.sort_values(
        by="Error",
        ascending=False
    ).head(10)
)
