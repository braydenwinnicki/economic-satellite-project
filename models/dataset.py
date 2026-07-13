import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])


class CensusDataset(Dataset):
    """
    Each item is one TRACT, not one image. A tract's item is a variable-length
    bag of tile images (shape: [n_tiles, 3, 224, 224]) plus the tract's income label.

    n_tiles differs per tract (anywhere from 4 to 50, depending on tract area),
    so batching requires a custom collate function (see get_collate_fn below)
    that pads bags to the max size WITHIN each batch and tracks a mask so the
    model knows which tiles are real vs padding.
    """

    def __init__(self, data, transform=transform):
        self.groups = list(
            data.groupby("GEOID")
        )  # now its a list of touples (GEOID, DF/ correlated images)
        self.transform = transform

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        geoid, group = self.groups[idx]  # seperates geoid and the correlated rows

        images = []
        for _, row in group.iterrows():  # grabs each row
            img = Image.open(row["image_path"].strip()).convert(
                "RGB"
            )  # runs coversioms
            if self.transform:
                img = self.transform(img)  # makes it a tensor
            images.append(img)  # adds it to list of images

        images = torch.stack(images)  # shape: (n_tiles_for_this_tract, 3, 224, 224)
        income = group["median_income"].iloc[
            0
        ]  # same value repeated for every row in the group

        return images, income

    def collate_fn(batch):
        """
        Custom collate function - required because each tract has a DIFFERENT
        number of tiles, so torch can't just stack them into one tensor like
        it would with fixed-size items.

        Pads every bag in this batch up to the size of the LARGEST bag in this
        specific batch (not the dataset-wide max), and returns a boolean mask
        marking which entries are real tiles (True) vs padding (False), so the
        model's pooling step can ignore the padded entries.
        """
        images_list, incomes = zip(
            *batch
        )  # unzip the list of (images, income) tuples into seperate lists

        batch_size = len(images_list)  # get lenth of batch
        max_n = max(
            img.shape[0] for img in images_list
        )  # largest bag in THIS batch, shape = (n_value, 3, 224, 224)

        # allocate padded tensors, all zeros to start
        padded_images = torch.zeros(
            batch_size, max_n, 3, 224, 224
        )  # create a tensor of zeros, one for each image
        """
        if n = 5
        padded images:
        image# 0 1 2 3 4 
        tracta 0 0 0 0 0
        tractb 0 0 0 0 0
        tractc 0 0 0 0 0
        """

        mask = torch.zeros(
            batch_size, max_n, dtype=torch.bool
        )  # turns all zeros into falses

        """
        if n = 5
        mask: 
        image# 0 1 2 3 4 
        tracta f f f f f
        tractb f f f f f
        tractc f f f f f
        """

        for i, imgs in enumerate(images_list):
            n = imgs.shape[0]
            padded_images[i, :n] = (
                imgs  # fill in the real tiles, tract 1, tile slots 0 through n-1, all channels, all rows, all columns(ommited in code)
            )
            """
                takes this
                if n = 5
                padded images:
                image# 0 1 2 3 4 
                tracta 0 0 0 0 0
                tractb 0 0 0 0 0
                tractc 0 0 0 0 0

                and fills in image data where needed.
                image# 0 1 2 3 4 
                tracta image_data 0 0 image_data 0
                tractb image_data 0 0 0 0
                tractc 0 0 0 0 0
                """

            mask[i, :n] = True  # mark those positions as real (not padding)

        """
        takes mask grid from earlier-
        image# 0 1 2 3 4 
        tracta f f f f f
        tractb f f f f f
        tractc f f f f f

        and adds true where there are images:

        image# 0 1 2 3 4 
        tracta t f f t f
        tractb t f f f f
        tractc f f f f f

        this grid intentionally matches padded_images. the model will use it to figure out which images are real, and which are padding
        
        """

        incomes = torch.tensor(incomes, dtype=torch.float32)

        return padded_images, mask, incomes
