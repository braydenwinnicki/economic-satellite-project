"""
collate.py — Custom collation for variable-length multi-tile batches.

PyTorch's default collation cannot handle datasets where each sample has a
different number of tiles. This module provides ``collate_fn``, which pads
all tile sets in a batch to the same length and produces a boolean mask so
downstream models can ignore the padded entries.

All tensors are created on CPU. The training loop moves them to the target
device after the DataLoader returns them, avoiding CUDA re-initialization
errors in forked worker processes.
"""

import torch


def collate_fn(batch):
    """
    Custom collate function for variable-length multi-tile batches.

    Because each tract has a different number of tiles, PyTorch's default
    ``default_collate`` cannot stack them. This function pads every tract's
    tile set to the size of the largest set *in this batch* and returns a
    boolean mask indicating which tiles are real vs. padding.

    Parameters
    ----------
    batch : list of tuples
        Each element is ``(images, income, geoid)`` where:
          - images: ``torch.Tensor`` of shape ``(n_tiles, 3, 224, 224)``
          - income: float (the z-score normalized median income)
          - geoid: str (the GEOID for the tract)

    Returns
    -------
    padded_images : torch.Tensor
        Shape ``(B, max_n, 3, 224, 224)`` — zero-padded image tensor.
    mask : torch.BoolTensor
        Shape ``(B, max_n)`` — True where a real tile exists, False for padding.
    incomes : torch.Tensor
        Shape ``(B,)`` — float32 tensor of z-score normalized incomes.
    geoids : tuple of str
        The GEOIDs for each sample in the batch (not padded).

    Notes
    -----
    Padding occurs at the batch level only, so ``max_n`` can vary between
    batches. This is more memory-efficient than padding to the dataset-wide
    maximum tile count.
    """
    images_list, incomes, geoids = zip(
        *batch
    )  # unzip the list of (images, income) tuples into seperate lists

    batch_size = len(images_list)  # get lenth of batch
    max_n = max(
        img.shape[0] for img in images_list
    )  # largest bag in THIS batch, shape = (n_value, 3, 224, 224)

    # allocate padded tensors, all zeros to start
    padded_images = torch.zeros(
        batch_size, max_n, 3, 224, 224
    )  # create a tensor of zeros, one for each image (on CPU)
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
    )  # turns all zeros into falses (on CPU)

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

    return padded_images, mask, incomes, geoids