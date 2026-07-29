"""
experiment_log.py — Auto-logging for ML experiments.

Automatically records all parameters, environment settings, training history,
and evaluation metrics for every experiment run. Logs are saved as JSON files
in experiment_logs/ for easy comparison and tracking.
"""

import json
import time
from datetime import datetime
from pathlib import Path


class ExperimentLog:
    """
    Records and saves experiment metadata, parameters, and results.

    Usage:
        log = ExperimentLog(args, device, num_workers, is_multi_tile)
        log.add_training_epoch(epoch, loss)
        log.add_evaluation(metrics)
        log.save()
    """

    def __init__(self, args, device, num_workers, is_multi_tile):
        self.start_time = time.time()
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Environment
        from new_pipeline.src.config import ENV, NUM_WORKERS

        self.data = {
            "timestamp": self.timestamp,
            "environment": {
                "env": ENV,
                "device": str(device),
                "num_workers": num_workers,
            },
            "parameters": {
                "cache": str(args.cache),
                "csv": str(args.csv) if args.csv else None,
                "model": args.model,
                "mode": args.mode,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "random_state": args.random_state,
                "test_size": args.test_size,
            },
            "data_info": {
                "is_multi_tile": is_multi_tile,
            },
            "training": {
                "epochs": [],
            },
            "evaluation": {},
            "outputs": {},
        }

    def add_data_info(
        self, n_train=None, n_test=None, mean_income=None, std_income=None
    ):
        """Record data split information."""
        if n_train is not None:
            self.data["data_info"]["n_train"] = n_train
        if n_test is not None:
            self.data["data_info"]["n_test"] = n_test
        if mean_income is not None:
            self.data["data_info"]["mean_income"] = round(mean_income, 2)
        if std_income is not None:
            self.data["data_info"]["std_income"] = round(std_income, 2)

    def add_training_epoch(self, epoch, loss):
        """Record loss for a single training epoch."""
        self.data["training"]["epochs"].append(
            {
                "epoch": epoch,
                "loss": round(loss, 4),
            }
        )

    def add_evaluation(self, metrics):
        """Record evaluation metrics (MAE, RMSE, R2, test loss)."""
        self.data["evaluation"] = {
            k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()
        }

    def add_outputs(self, model_path=None, results_path=None, figures_dir=None):
        """Record output file paths."""
        outputs = {}
        if model_path:
            outputs["model_weights"] = str(model_path)
        if results_path:
            outputs["results_csv"] = str(results_path)
        if figures_dir:
            outputs["figures_dir"] = str(figures_dir)
        self.data["outputs"] = outputs

    def save(self):
        """Save the log to experiment_logs/<timestamp>_<model>.json."""
        # Record duration
        duration = time.time() - self.start_time
        self.data["duration_seconds"] = round(duration, 1)
        self.data["duration_minutes"] = round(duration / 60, 1)

        # Build filename
        model_name = self.data["parameters"]["model"]
        if self.data["data_info"].get("is_multi_tile"):
            model_name = f"{model_name}_multi"
        log_dir = Path(__file__).resolve().parents[1] / "experiment_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{self.timestamp}_{model_name}.json"

        # Write
        with open(log_path, "w") as f:
            json.dump(self.data, f, indent=2)

        print(f"Experiment log saved → {log_path}")
        return log_path
