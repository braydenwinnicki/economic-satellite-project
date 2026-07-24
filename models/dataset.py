import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# transforms.Compose chains transformations together. Resize makes images
# 224x224 so they match what ResNet expects. ToTensor converts pixel values
# from 0-255 to 0.0-1.0 and rearranges to (C, H, W) format.
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


class CensusDataset(Dataset):
    """
    Each row in the CSV is one satellite tile. This loads one image at a time,
    transforms it, and returns (image_tensor, income_label).
    """

    def __init__(self, data, transform=transform):
        # `data` is a DataFrame — each row = one tile + its income
        self.data = data
        self.transform = transform

    def __len__(self):
        # __len__ tells PyTorch how many total samples exist
        return len(self.data)

    def __getitem__(self, idx):
        # iloc[idx] grabs the row at position idx from the DataFrame
        row = self.data.iloc[idx]

        # .strip() trims whitespace from the file path, .convert("RGB")
        # ensures we get 3 color channels (some images might be grayscale)
        image = Image.open(row["image_path"].strip()).convert("RGB")

        income = row["median_income"]

        if self.transform:
            # apply the resize + to-tensor pipeline
            image = self.transform(image)

        return image, income