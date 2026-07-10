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

model = ConvNN()

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


#training 

epochs = 10

model.train() #turn on train mode

for epoch in range(epochs):

    total_loss = 0

    for images, incomes in train_loader:

        # forward pass
        predictions = model(images)

        # calculate error
        loss = criterion(
            predictions.squeeze(),
            incomes.float()
        )

        # clear old gradients
        optimizer.zero_grad()

        # calculate gradients
        loss.backward()

        # update weights
        optimizer.step()

        total_loss += loss.item()


    avg_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}: {avg_loss:.4f}")



#testing 


with torch.no_grad():

    total_loss = 0
    
    model.eval() # put on test mode

    for images, incomes in test_loader:

        # forward pass
        predictions = model(images)

        # calculate error
        loss = criterion(
            predictions.squeeze(),
            incomes.float()
        )

        total_loss += loss.item()


    avg_test_loss = total_loss / len(test_loader)

    print(f"AVG TEST LOSS: {avg_test_loss}")
