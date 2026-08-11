import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(
    model, train_loader, epochs, lr, model_save_path, device=None,
    mean_income=None, std_income=None,
):
    """
    Train a regression model on satellite image data.

    Parameters
    ----------
    model : nn.Module
        The model to train.
    train_loader : DataLoader
        DataLoader yielding (images, incomes) batches for single-tile,
        or (images, mask, incomes, geoids) for multi-tile.
    epochs : int
        Number of training epochs.
    lr : float
        Learning rate for the Adam optimizer.
    model_save_path : str or Path
        Where to save the trained model weights (.pth).
    device : torch.device, optional
        Device to run training on (e.g. mps, cuda, cpu).
        If None, stays on whatever device the model is already on.
    mean_income : float, optional
        Mean of the income labels used to z-score normalize them during
        training. If provided, it is saved alongside the weights so later
        (including cross-state) evaluation can denormalize predictions back
        into the correct dollar frame.
    std_income : float, optional
        Std of the income labels used for z-score normalization. Saved with
        the weights alongside ``mean_income``.

    Returns
    -------
    model : nn.Module
        The trained model (with best weights loaded).
    epoch_losses : list
        List of average loss per epoch.
    """

    # Move model to the target device
    if device is not None:
        model = model.to(device)

    # nn.MSELoss() = mean squared error loss — penalizes big errors more than small ones
    criterion = nn.MSELoss()
    # torch.optim.Adam adjusts model weights using gradient descent
    # model.parameters() gives the optimizer access to all trainable weights
    # lr=0.001 is the learning rate — how big each weight update step is
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # model.train() enables training-specific behaviors (affects dropout, batchnorm, etc.)
    model.train()

    epoch_losses = []

    for epoch in range(epochs):

        total_loss = 0

        # Each iteration of this inner loop processes one batch
        for batch in train_loader:

            # Detect multi-tile mode: if the batch has 4 elements (images, mask, incomes, geoids)
            # it's a multi-tile batch. Otherwise it's single-tile (images, incomes).
            if len(batch) == 4:
                images, mask, incomes, geoids = batch
                # Move tensors to device
                if device is not None:
                    images = images.to(device)
                    mask = mask.to(device)
                    incomes = incomes.to(device)
                # multi-tile model takes both images and mask
                # images shape: (batch_size, max_n_tiles, 3, 224, 224)
                predictions = model(images, mask)
            else:
                images, incomes = batch
                # Move tensors to device
                if device is not None:
                    images = images.to(device)
                    incomes = incomes.to(device)
                # single-tile model takes just images
                # images shape: (batch_size, 3, 224, 224)
                predictions = model(images)

            # .view(-1) flattens (batch_size, 1) → (batch_size,) safely, even when batch=1
            # This matches the shape of incomes so the loss calculation works
            # .float() ensures incomes are float32 (might be int64 from CSV)
            loss = criterion(predictions.view(-1), incomes.float())

            # zero_grad clears old gradient values so they don't accumulate across batches
            optimizer.zero_grad()
            # backward() computes gradients of the loss with respect to every parameter
            loss.backward()
            # step() updates the model weights using the computed gradients
            optimizer.step()

            # .item() extracts the single scalar value from a 1-element tensor as a Python float
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        print(f"train Epoch {epoch+1}: {avg_loss:.4f}")

    # save the trained weights
    # Move model to CPU before saving so weights are portable (can load on CPU later)
    model_cpu = model.cpu()
    # .state_dict() returns all learnable parameters as a dictionary of tensors.
    # We save it inside a checkpoint dict together with the z-score stats used
    # during training, so later (potentially cross-state) evaluation can map
    # predictions back into the correct dollar frame.
    checkpoint = {"state_dict": model_cpu.state_dict()}
    if mean_income is not None:
        checkpoint["mean_income"] = float(mean_income)
    if std_income is not None:
        checkpoint["std_income"] = float(std_income)
    torch.save(checkpoint, model_save_path)

    print(f"Saved model to {model_save_path}")

    return model, epoch_losses
