import sys  # allow running as a script with project imports
from pathlib import Path
import pandas as pd  # data handling

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))  # make project importable
from models.dataset import CensusDataset  # dataset class that loads cached images
from torch.utils.data import DataLoader  # loader to batch/pad data
import torch  # main PyTorch package
from torchvision import transforms  # image transforms (not heavily used here)
from torch.utils.data import DataLoader
from models.cnn import ConvNN  # the small ConvNet defined in models/cnn.py
import torch.nn as nn  # neural network losses and layers
from sklearn.metrics import mean_absolute_error  # evaluation metric (MAE)
from models.collate import collate_fn  # custom collate to pad variable-length bags
from src.splitting import split_by_tract


model = ConvNN()  # instantiate the convolutional model

# device autodetection: prefer CUDA, then MPS, else CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

model.to(device)
try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass

# load dataset CSV that lists tiles and GEOIDs
df = pd.read_csv(
    "/Users/braydenwinnicki/CODE/econ_project/data/processed/processed_ct_tracts_tiles.csv"
)

# trim column whitespace just in case
df.columns = df.columns.str.strip()


# split dataset into train/test sets by tract (20% test)
df_train, df_test = split_by_tract(df, test_size=0.20, random_state=42)

# compute z-score normalization stats from training tracts only (avoid leakage)
train_tract_labels = df_train.drop_duplicates(subset="GEOID")["median_income"]
mean_income = train_tract_labels.mean()
std_income = train_tract_labels.std()
if std_income == 0:
    std_income = 1.0

# z-score normalize the training labels
df_train["median_income"] = (df_train["median_income"] - mean_income) / std_income

# create the train dataset using the prebuilt image cache
train_dataset = CensusDataset(
    df_train,
    "/Users/braydenwinnicki/CODE/econ_project/data/processed/census_images_cache.pt",
)


# DataLoader handles batching and uses a custom collate_fn to pad variable bag sizes
pin_memory = device.type == "cuda"
# lower batch size on MPS to reduce memory usage
batch_size = 32 if device.type != "mps" else 8
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    pin_memory=pin_memory,
)


criterion = nn.MSELoss()  # mean squared error loss for regression
optimizer = torch.optim.Adam(  # Adam optimizer adjusts weights using gradients
    model.parameters(), lr=0.001
)


# training loop parameters
epochs = 10

model.train()  # set model to training mode (enables dropout, batchnorm updates if present)

for epoch in range(epochs):

    total_loss = 0

    for images, mask, incomes, geoids in train_loader:

        # move tensors to device; cached images are float32
        images = images.float().to(device)
        mask = mask.to(device)
        incomes = incomes.to(device)

        # forward pass: model returns per-tract predictions (batched)
        predictions = model(images, mask)

        # compute loss between predictions and targets (squeeze removes extra dims)
        loss = criterion(predictions.squeeze(), incomes.float())

        # zero existing gradients before backward pass
        optimizer.zero_grad()

        # backward pass computes gradients
        loss.backward()

        # optimizer step applies gradient updates to model parameters
        optimizer.step()

        total_loss += loss.item()  # accumulate scalar loss

    avg_loss = total_loss / len(train_loader)  # average loss per batch

    print(f"train Epoch {epoch+1}: {avg_loss:.4f}")


# save model weights to project models directory
torch.save(model.state_dict(), PROJECT_ROOT / "models" / "cnn_v1_tiles.pth")

print("Saved model to models/cnn_v1_tiles.pth")
