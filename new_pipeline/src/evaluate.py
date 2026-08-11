import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd


def evaluate_model(
    model, test_loader, mean_income, std_income, results_save_path, device=None
):
    """
    Evaluate a trained regression model and save results.

    Parameters
    ----------
    model : nn.Module
        The trained model (in eval mode).
    test_loader : DataLoader
        DataLoader yielding (images, incomes) batches for single-tile,
        or (images, mask, incomes, geoids) for multi-tile.
    mean_income : float
        Mean income from training set (for denormalization).
    std_income : float
        Std income from training set (for denormalization).
    results_save_path : str or Path
        Where to save the results CSV.
    device : torch.device, optional
        Device to run evaluation on (e.g. mps, cuda, cpu).
        If None, stays on whatever device the model is already on.

    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with GEOID, actual, prediction, error columns.
    metrics : dict
        Dictionary with mae, rmse, r2, avg_test_loss.
    """

    # Move model to the target device
    if device is not None:
        model = model.to(device)

    # model.eval() switches to evaluation mode (disables dropout, fixes batchnorm stats)
    model.eval()

    criterion = nn.MSELoss()

    all_predictions = []
    all_targets = []
    all_geoids = []

    with torch.no_grad():

        total_loss = 0

        for batch in test_loader:

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
                predictions = model(images, mask)
                all_geoids.extend(geoids)
            else:
                images, incomes = batch
                # Move tensors to device
                if device is not None:
                    images = images.to(device)
                    incomes = incomes.to(device)
                # single-tile model takes just images
                predictions = model(images)

            # .view(-1) flattens (batch, 1) → (batch,) safely, even when batch=1
            # .tolist() converts the PyTorch tensor to a Python list
            # .extend() appends each element individually - builds a flat list
            all_predictions.extend(predictions.view(-1).tolist())
            all_targets.extend(incomes.tolist())

            # .view(-1) matches shapes: model outputs (batch, 1), incomes are (batch,)
            # .float() ensures incomes are float32 for the MSE loss computation
            loss = criterion(predictions.view(-1), incomes.float())

            # .item() pulls the single scalar value out of a 1-element tensor, converting to a Python float
            total_loss += loss.item()

        avg_test_loss = total_loss / len(test_loader)

    # Undo z-score normalization: prediction * std + mean = original dollar amount
    # (This is a list comprehension — it applies the formula to every prediction in the list)
    predictions_dollars = [p * std_income + mean_income for p in all_predictions]

    # Same denormalization for the actual income values
    targets_dollars = [t * std_income + mean_income for t in all_targets]

    # ── Sanity guard ──────────────────────────────────────────────────
    # Cross-state evaluation with a normalization frame that doesn't match the
    # model's training frame (or a degenerate model) often produces wildly
    # implausible dollar predictions (e.g. negative millions). Warn loudly so a
    # broken evaluation isn't silently accepted.
    if predictions_dollars and targets_dollars:
        pred_mean = float(np.mean(predictions_dollars))
        target_mean = float(np.mean(targets_dollars))
        if pred_mean < 0 or abs(pred_mean - target_mean) > 5 * abs(mean_income):
            print(
                "  ⚠ SANITY CHECK FAILED: mean prediction "
                f"${pred_mean:,.0f} vs mean actual ${target_mean:,.0f}. "
                "Predictions are implausible — check that the normalization "
                "frame matches the model's training frame (e.g. cross-state "
                "evaluation without saved stats)."
            )

    mae = mean_absolute_error(targets_dollars, predictions_dollars)

    rmse = np.sqrt(mean_squared_error(targets_dollars, predictions_dollars))

    r2 = r2_score(targets_dollars, predictions_dollars)

    print(f"AVG TEST LOSS: {avg_test_loss}")
    print(f"TESTING MAE: {mae}")
    print(f"TESTING RMSE: {rmse}")
    print(f"TESTING Rsquared: {r2}")

    # put results in a dataframe
    results_df = pd.DataFrame()
    if all_geoids:
        results_df["GEOID"] = all_geoids
    results_df["prediction"] = predictions_dollars
    results_df["actual"] = targets_dollars

    # (prediction - actual) for each row
    # .abs() takes the absolute value so errors are always positive
    results_df["error"] = (results_df["prediction"] - results_df["actual"]).abs()

    # .sort_values(by="error", ascending=False) sorts rows from highest error to lowest
    worst = results_df.sort_values(by="error", ascending=False)
    # print 10 worst for eval
    # .head(10) takes only the first 10 rows (the 10 worst predictions after the sort above)
    print(worst.head(10))

    # Save results
    results_df.to_csv(results_save_path, index=False)
    print(f"Results saved to {results_save_path}")

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "avg_test_loss": avg_test_loss,
    }

    return results_df, metrics
