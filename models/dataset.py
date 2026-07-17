import pandas as pd  # DataFrame handling for CSVs and tabular data
from PIL import Image  # Image IO utilities (used elsewhere in project)
import torch  # main PyTorch package for tensors and I/O

from torch.utils.data import Dataset  # base class for PyTorch datasets
import torchvision.transforms as transforms  # image transforms (unused here but common)
from torch.utils.data import DataLoader  # data loader (used by callers)


class CensusDataset(Dataset):

    def __init__(self, data, cache_file):

        # `data` is a DataFrame listing tiles and GEOIDs; group rows by tract GEOID
        self.groups = list(data.groupby("GEOID"))

        # load the prebuilt cache (maps GEOID -> tensor of images, and income)
        # Some PyTorch versions default to `weights_only=True` which prevents
        # loading arbitrary python objects. The cache was saved as a full
        # dict of tensors, so explicitly allow full loading.
        self.cache = torch.load(cache_file, weights_only=False)

    def __len__(self):
        # dataset length = number of unique GEOIDs (tracts)
        return len(self.groups)

    def __getitem__(self, idx):

        # unpack the (GEOID, group_df) tuple for this index
        geoid, _ = self.groups[idx]

        # ensure GEOID is a string for cache lookup
        geoid = str(geoid)

        # get stacked image tensors for this tract from the cache
        images = self.cache["images"][geoid]

        # get the median income label for this tract from the cache
        income = self.groups[idx][1]["median_income"].iloc[0]
        # return (images tensor, income scalar, geoid string)
        return images, income, geoid
