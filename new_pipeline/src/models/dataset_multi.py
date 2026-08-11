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
    transform : callable, optional
        Transform applied to every tile before it is returned. The cache
        stores raw ``[0, 1]`` tensors, so pretrained models (e.g. ResNet)
        need their expected ImageNet mean/std normalization applied here.
        CNN models pass ``None``.

    Yields (via __getitem__)
    ------------------------
    images : torch.Tensor
        Shape ``(n_tiles, 3, 224, 224)`` — all tiles for this tract.
    income : float
        The median household income for this tract (z-score normalized).
    geoid : str
        The GEOID string for this tract.
    """

    def __init__(self, data, cache_file, transform=None):

        # load the prebuilt cache (maps GEOID -> tensor of images, and income)
        # Some PyTorch versions default to `weights_only=True` which prevents
        # loading arbitrary python objects. The cache was saved as a full
        # dict of tensors, so explicitly allow full loading.
        self.cache = torch.load(cache_file, weights_only=False)

        # Only keep tracts whose GEOIDs exist in the cache.  build_tract_cache
        # skips tracts whose tile images all failed to load, so the input
        # `data` DataFrame may contain GEOIDs that are absent from the cache.
        # GEOIDs are canonicalized to zero-padded 11-digit form so that caches
        # whose keys lost a leading zero (states with a 0-leading FIPS) still
        # match the string GEOIDs read from a CSV with dtype=str.
        self.cache["images"] = {
            str(k).zfill(11): v for k, v in self.cache["images"].items()
        }
        cache_geoids = set(self.cache["images"].keys())
        data = data[data["GEOID"].astype(str).str.zfill(11).isin(cache_geoids)].copy()

        # `data` is a DataFrame listing tiles and GEOIDs; group rows by tract GEOID
        self.groups = list(data.groupby("GEOID"))

        # Optional per-tile transform (e.g. ImageNet normalization for ResNet)
        self.transform = transform

    def __len__(self):
        # dataset length = number of unique GEOIDs (tracts)
        return len(self.groups)

    def __getitem__(self, idx):

        # unpack the (GEOID, group_df) tuple for this index
        geoid, _ = self.groups[idx]

        # ensure GEOID is a string for cache lookup
        geoid = str(geoid).zfill(11)  # canonical zero-padded 11-digit GEOID

        # get stacked image tensors for this tract from the cache
        images = self.cache["images"][geoid]

        # Cache images are raw [0, 1] tensors.  Pretrained models (ResNet)
        # expect their ImageNet mean/std normalization, which we apply here
        # per tile.  CNN models pass transform=None and are used as-is.
        if self.transform is not None:
            images = torch.stack([self.transform(img) for img in images])

        # get the median income label for this tract from the cache
        income = self.groups[idx][1]["median_income"].iloc[0]
        # return (images tensor, income scalar, geoid string)
        return images, income, geoid
