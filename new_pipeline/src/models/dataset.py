import torch
from torch.utils.data import Dataset, DataLoader


class CacheDataset(Dataset):
    """
    Loads pre-computed image tensors and incomes from a .pt cache file
    (produced by build_cache.py). This avoids re-loading and re-transforming
    images from disk every epoch.
    """

    def __init__(self, images, incomes, transform=None):
        # images: torch.Tensor of shape (N, 3, 224, 224)
        # incomes: torch.Tensor of shape (N,)
        self.images = images
        self.incomes = incomes
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        income = self.incomes[idx]

        # Normalize any uint8 [0,255] cache to float32 [0,1] so model input is
        # consistent with the float32 [0,1] caches (see dataset_multi.py note).
        if image.dtype == torch.uint8:
            image = image.float().div(255.0)

        if self.transform:
            image = self.transform(image)

        return image, income
