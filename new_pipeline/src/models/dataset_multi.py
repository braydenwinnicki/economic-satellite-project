"""
dataset_multi.py — PyTorch Dataset for multi-tile satellite imagery.

Each census tract has a variable number of satellite tiles. This dataset
wraps the pre-built .pt cache (produced by ``build_tract_cache``) and
looks up image stacks by GEOID, pairing them with their income labels.

The companion ``collate_fn`` in ``collate.py`` handles padding the
variable-length tile sets into fixed-size batches.
"""

import pandas as pd  # DataFrame handling for CSVs and tabular data
from PIL import Image  # Image IO utilities (used elsewhere in project)
import torch  # main PyTorch package for tensors and I/O

from torch.utils.data import Dataset  # base class for PyTorch datasets
import torchvision.transforms as transforms  # image transforms (unused here but common)
from torch.utils.data import DataLoader  # data loader (used by callers)


class MultiTileDataset(Dataset):
    """
    Wraps the prebuilt image cache so we can feed it into a PyTorch DataLoader.

    Each tract has a variable number of satellite tiles. The cache already has
    all the images stacked into tensors, so this class just looks them up by
    GEOID and pairs them with the income label.

    The DataLoader (used in the training/eval scripts) will call __getitem__
    repeatedly and batch the results using collate_fn.

    Parameters
    ----------
    data : pd.DataFrame
        Tile-level DataFrame with a ``GEOID`` column. Rows are grouped by
        GEOID to define the dataset samples (one per tract).
    cache_file : str or Path
        Path to the .pt cache file produced by ``build_tract_cache``.
        Must contain ``cache["images"]`` (dict mapping GEOID -> tensor)
        and ``cache["income"]`` (dict mapping GEOID -> float).

    Yields (via __getitem__)
    ------------------------
    images : torch.Tensor
        Shape ``(n_tiles, 3, 224, 224)`` — all tiles for this tract.
    income : float
        The median household income for this tract (z-score normalized).
    geoid : str
        The GEOID string for this tract.
    """

    def __init__(self, data, cache_file):

        # load the prebuilt cache (maps GEOID -> tensor of images, and income)
        # Some PyTorch versions default to `weights_only=True` which prevents
        # loading arbitrary python objects. The cache was saved as a full
        # dict of tensors, so explicitly allow full loading.
        self.cache = torch.load(cache_file, weights_only=False)

        # Only keep tracts whose GEOIDs exist in the cache.  build_tract_cache
        # skips tracts whose tile images all failed to load, so the input
        # `data` DataFrame may contain GEOIDs that are absent from the cache.
        cache_geoids = set(str(g) for g in self.cache["images"].keys())
        data = data[data["GEOID"].astype(str).isin(cache_geoids)].copy()

        # `data` is a DataFrame listing tiles and GEOIDs; group rows by tract GEOID
        self.groups = list(data.groupby("GEOID"))

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
