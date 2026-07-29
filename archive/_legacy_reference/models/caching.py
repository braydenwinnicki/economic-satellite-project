from tqdm import tqdm  # progress bar for long-running loops
import pandas as pd  # dataframe utilities (data argument is a DataFrame)
from PIL import Image  # image reading
import torch  # tensor creation and saving

from torch.utils.data import Dataset  # dataset base class (not used here directly)
import torchvision.transforms as transforms  # image transforms passed in
from torch.utils.data import DataLoader  # data loader (not used here)


def create_image_cache(data, transform, save_path):

    # cache dict will map GEOID (string) -> images tensor and income scalar
    cache = {"images": {}, "income": {}}

    # group dataframe rows by GEOID so we can collect tiles per tract
    groups = data.groupby("GEOID")

    # iterate over each tract (GEOID) and its rows
    for geoid, group in tqdm(groups):

        images = []  # list to collect transformed image tensors for this tract

        for _, row in group.iterrows():

            # open the image file, strip whitespace, and convert to RGB
            img = Image.open(row["image_path"].strip()).convert("RGB")

            # apply the model's transform pipeline (resize, to-tensor, normalize)
            img = transform(img)

            # collect the transformed tensor
            images.append(img)

        # stack per-tract images into a single tensor and convert to half precision
        # stack per-tract images into a single float32 tensor (safer for CPU/GPU)
        images = torch.stack(images).float()

        # store images in cache under the stringified GEOID
        cache["images"][str(geoid)] = images

        # store the median income label once for this GEOID (first row)
        cache["income"][str(geoid)] = group["median_income"].iloc[0]

    # persist the cache to disk for later fast loading
    torch.save(cache, save_path)
