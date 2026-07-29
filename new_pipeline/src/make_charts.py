"""
make_charts.py

Generates evaluation charts from a results CSV:
  1. model_performance_comparison.png  - Bar chart of MAE, RMSE, R2
  2. predicted_vs_actual.png            - Scatter plot
  3. residual_distributions.png         - KDE plot of residuals

Usage (as a module):
    from new_pipeline.src.make_charts import make_charts
    make_charts(results_csv, figures_dir, model_name)
"""

import os
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def make_charts(results_csv, figures_dir, model_name="Model"):
    """
    Generate three evaluation charts from a single results CSV.

    Parameters
    ----------
    results_csv : str or Path
        Path to the results CSV (must have 'prediction' and 'actual' columns).
    figures_dir : str or Path
        Directory where the charts will be saved.
    model_name : str
        Name of the model (used in titles and filenames).
    """

    # Create figures directory if it doesn't exist
    figures_dir = str(figures_dir)
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    # Load results
    df = pd.read_csv(results_csv, skipinitialspace=True)
    df.columns = [col.strip() for col in df.columns]
    df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
    df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
    df = df.dropna(subset=["prediction", "actual"]).reset_index(drop=True)

    actual = df["actual"].values
    predicted = df["prediction"].values

    # Calculate metrics
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(
        f"  {model_name}  MAE = {round(mae, 2)}  RMSE = {round(rmse, 2)}  R2 = {round(r2, 4)}"
    )

    # Safe filename prefix
    safe_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")

    sns.set_theme(style="whitegrid")

    # Chart 1: Performance bar chart (single model)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics_data = {"mae": mae, "rmse": rmse, "r2": r2}
    colors = {"mae": "steelblue", "rmse": "coral", "r2": "seagreen"}

    for idx, (metric, value) in enumerate(metrics_data.items()):
        ax = axes[idx]
        ax.bar([model_name], [value], color=colors[metric], width=0.6)
        ax.set_xticks([0])
        ax.set_xticklabels([model_name], rotation=30, ha="right", fontsize=9)
        ax.set_title(metric.upper(), fontsize=13, fontweight="bold")
        ax.set_ylabel(metric.upper())
        ax.text(0, value + abs(value) * 0.02, f"{value:.4f}", ha="center", fontsize=11)

    fig.suptitle(
        f"Model Performance — {model_name}", fontsize=15, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(figures_dir, f"{safe_name}_performance.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved {safe_name}_performance.png")

    # Chart 2: Predicted vs Actual scatter plot

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(actual, predicted, alpha=0.5, s=25, color="steelblue")

    all_values = list(actual) + list(predicted)
    min_val = min(all_values)
    max_val = max(all_values)
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=1.5,
        label="Ideal (y=x)",
    )

    ax.text(
        0.05,
        0.95,
        f"R2 = {r2:.4f}",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"),
    )

    ax.set_xlabel("Actual Median Income ($)")
    ax.set_ylabel("Predicted Median Income ($)")
    ax.set_title(f"Predicted vs Actual — {model_name}", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(figures_dir, f"{safe_name}_predicted_vs_actual.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved {safe_name}_predicted_vs_actual.png")

    # Chart 3: Residual distribution
    fig, ax = plt.subplots(figsize=(10, 6))

    residuals = actual - predicted

    sns.kdeplot(
        residuals,
        ax=ax,
        label=model_name,
        color="steelblue",
        fill=True,
        alpha=0.3,
        linewidth=2,
    )

    mean_res = np.mean(residuals)
    ax.axvline(
        mean_res,
        color="steelblue",
        linestyle=":",
        linewidth=1.2,
        label=f"Mean residual = {mean_res:.0f}",
    )

    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Zero (no bias)")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Residual Distribution — {model_name}", fontsize=14, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(figures_dir, f"{safe_name}_residuals.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"  Saved {safe_name}_residuals.png")

    print(f"  All charts saved to {figures_dir}/")
