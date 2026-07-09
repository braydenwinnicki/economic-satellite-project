import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

class CensusDataset(Dataset):
    def __init__(self, file, transform=transform):
        self.data = pd.read_csv(file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        row = self.data.iloc[idx]

        image = Image.open(row["image_path"].strip()).convert("RGB")

        income = row["median_income"]

        if self.transform:
            image = self.transform(image)

        return image, income



i = CensusDataset("/Users/braydenwinnicki/CODE/econ_project/data/processed/processed_ct_tracts.csv")


loader = DataLoader(i, batch_size=32, shuffle=True)


