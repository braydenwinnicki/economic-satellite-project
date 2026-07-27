"""
make_comparison_charts.py

Loads evaluation CSVs from results/ and generates three charts in figures/:
  1. model_performance_comparison.png  - Bar chart of MAE, RMSE, R2
  2. predicted_vs_actual.png            - Scatter plots for each model
  3. residual_distributions.png         - KDE plots of residuals

Usage: python3 scripts/make_comparison_charts.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

if not os.path.exists(FIGURES_DIR):
    os.makedirs(FIGURES_DIR)

MODELS = [
    ("Baseline (Mean)", None),
    ("Multi-Tile CNN", "cnn_multi_results.csv"),
    ("Multi-Tile ResNet Frozen", "resnet_frozen_results.csv"),
    ("Multi-Tile ResNet Unfrozen L3", "resnet_unfrozen_l3_results.csv"),
    ("Multi-Tile ResNet Unfrozen L4", "resnet_unfrozen_l4_results.csv"),
]

COLORS = {
    "Baseline (Mean)": "gray",
    "Multi-Tile CNN": "blue",
    "Multi-Tile ResNet Frozen": "green",
    "Multi-Tile ResNet Unfrozen L3": "orange",
    "Multi-Tile ResNet Unfrozen L4": "red",
}


def load_data():
    """Load all CSV files and return a dict of model_name -> DataFrame."""
    all_models = {}

    for name, csv_file in MODELS:
        if csv_file is None:
            continue

        df = pd.read_csv(os.path.join(RESULTS_DIR, csv_file), skipinitialspace=True)
        df.columns = [col.strip() for col in df.columns]
        df["prediction"] = pd.to_numeric(df["prediction"], errors="coerce")
        df["actual"] = pd.to_numeric(df["actual"], errors="coerce")
        df = df.dropna(subset=["prediction", "actual"]).reset_index(drop=True)

        all_models[name] = df
        print("  Loaded", name, "--", len(df), "rows")

    # Baseline: predict the mean of all actual values
    if all_models:
        first_model = list(all_models.values())[0]
        mean_actual = first_model["actual"].mean()

        baseline = pd.DataFrame()
        baseline["GEOID"] = first_model["GEOID"]
        baseline["prediction"] = mean_actual
        baseline["actual"] = first_model["actual"]
        all_models["Baseline (Mean)"] = baseline
        print("  Computed Baseline (Mean) =", round(mean_actual, 2))

    ordered = {}
    for name, _ in MODELS:
        if name in all_models:
            ordered[name] = all_models[name]

    return ordered


def calculate_metrics(models):
    """Calculate MAE, RMSE, and R2 for each model."""
    results = []

    for name, df in models.items():
        actual = df["actual"].values
        predicted = df["prediction"].values

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)

        results.append({
            "model": name,
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        })
        print("  ", name, "  MAE =", round(mae, 2), "  RMSE =", round(rmse, 2), "  R2 =", round(r2, 4))

    return pd.DataFrame(results)


def make_bar_chart(metrics_df):
    """Bar chart comparing MAE, RMSE, and R2 across all models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    model_names = list(metrics_df["model"])
    x_positions = np.arange(len(model_names))

    for idx, metric in enumerate(["mae", "rmse", "r2"]):
        ax = axes[idx]
        values = metrics_df[metric].values
        colors = [COLORS.get(name, "gray") for name in model_names]

        ax.bar(x_positions, values, color=colors, width=0.6)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(model_names, rotation=30, ha="right", fontsize=9)
        ax.set_title(metric.upper(), fontsize=13, fontweight="bold")
        ax.set_ylabel(metric.upper())

        for i, v in enumerate(values):
            ax.text(i, v + max(values) * 0.01, f"{v:.4f}", ha="center", fontsize=9)

    fig.suptitle("Model Performance Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_performance_comparison.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved model_performance_comparison.png")


def make_scatter_plots(models):
    """Grid of predicted-vs-actual scatter plots, one per model."""
    model_names = list(models.keys())
    n = len(model_names)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    for i, name in enumerate(model_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row][col]

        df = models[name]
        actual = df["actual"].values
        predicted = df["prediction"].values

        ax.scatter(actual, predicted, alpha=0.5, s=25, color=COLORS.get(name, "gray"))

        all_values = list(actual) + list(predicted)
        min_val = min(all_values)
        max_val = max(all_values)
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal (y=x)")

        r2 = r2_score(actual, predicted)
        ax.text(0.05, 0.95, f"R2 = {r2:.4f}",
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

        ax.set_xlabel("Actual Median Income ($)")
        ax.set_ylabel("Predicted Median Income ($)")
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    for i in range(n, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row][col].set_visible(False)

    fig.suptitle("Predicted vs Actual Median Income", fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "predicted_vs_actual.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved predicted_vs_actual.png")


def make_residual_plot(models):
    """Overlaid KDE plots of residuals (Actual - Predicted)."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, df in models.items():
        residuals = df["actual"].values - df["prediction"].values
        color = COLORS.get(name, "gray")

        sns.kdeplot(residuals, ax=ax, label=name, color=color, fill=True, alpha=0.15, linewidth=2)

        mean_res = np.mean(residuals)
        ax.axvline(mean_res, color=color, linestyle=":", linewidth=1.2)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.5, label="Zero (no bias)")
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_ylabel("Density")
    ax.set_title("Residual Distributions Across Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "residual_distributions.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("  Saved residual_distributions.png")


def main():
    sns.set_theme(style="whitegrid")

    print("\nLoading CSV files from results/ ...")
    models = load_data()

    if not models:
        print("ERROR: No data found. Exiting.")
        return

    print("\nModels loaded:", list(models.keys()))

    print("\nCalculating metrics ...")
    metrics_df = calculate_metrics(models)

    print("\nCreating charts ...")
    make_bar_chart(metrics_df)
    make_scatter_plots(models)
    make_residual_plot(models)

    print("\nAll charts saved to", FIGURES_DIR)
    print("\nMetrics summary:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
