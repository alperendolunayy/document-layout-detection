"""Training script with hydra config and mlflow logging."""

import gc
import json
import subprocess
from pathlib import Path

import hydra
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from docuvision.data.dataset import DocLayNetDataModule
from docuvision.training.lightning_module import DETRLightningModule

matplotlib.use("Agg")

METRICS_FILE = "plots/metrics_history.json"


def get_git_commit():
    """Get current git commit hash for logging."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_metrics_history():
    """Load previous metrics from json file."""
    path = Path(METRICS_FILE)
    if path.exists():
        with open(path) as file:
            data = json.load(file)
        # make sure new keys exist for older json files
        if "val_map_50" not in data:
            data["val_map_50"] = []
        if "val_recall" not in data:
            data["val_recall"] = []
        return data
    return {
        "train_loss": [],
        "val_loss": [],
        "val_map": [],
        "val_map_50": [],
        "val_recall": [],
    }


def save_metrics_history(history):
    """Save metrics to json file."""
    path = Path(METRICS_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(history, file, indent=2)


def save_plots(history):
    """Save training plots from metrics history."""
    plots_dir = Path("plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = list(range(len(history["train_loss"])))

    if history["train_loss"]:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history["train_loss"], marker="o", label="train loss")
        if history["val_loss"]:
            plt.plot(epochs, history["val_loss"], marker="s", label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / "training_loss.png", dpi=150, bbox_inches="tight")
        plt.close()

    if history["val_loss"]:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, history["val_loss"], marker="s", color="orange")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Validation Loss")
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / "validation_loss.png", dpi=150, bbox_inches="tight")
        plt.close()

    # plot all validation metrics together
    has_metrics = history.get("val_map") or history.get("val_map_50")
    if has_metrics:
        plt.figure(figsize=(10, 6))
        if history.get("val_map_50"):
            map50_epochs = list(range(len(history["val_map_50"])))
            plt.plot(
                map50_epochs, history["val_map_50"], marker="^", color="green", label="mAP@0.5"
            )
        if history.get("val_map"):
            map_epochs = list(range(len(history["val_map"])))
            plt.plot(map_epochs, history["val_map"], marker="s", color="blue", label="mAP@0.5:0.95")
        if history.get("val_recall"):
            recall_epochs = list(range(len(history["val_recall"])))
            plt.plot(
                recall_epochs,
                history["val_recall"],
                marker="o",
                color="red",
                label="Recall (mAR@100)",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Validation Metrics")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / "metrics_summary.png", dpi=150, bbox_inches="tight")
        plt.close()


class PlotCallback(pl.Callback):
    """Saves plots after every validation epoch."""

    def __init__(self):
        self.history = load_metrics_history()

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = trainer.logged_metrics

        if pl_module.training_losses:
            avg_train = sum(pl_module.training_losses) / len(pl_module.training_losses)
            self.history["train_loss"].append(round(avg_train, 4))
            pl_module.training_losses.clear()

        if "val/loss" in metrics:
            val = metrics["val/loss"]
            if hasattr(val, "item"):
                val = val.item()
            self.history["val_loss"].append(round(val, 4))

        if "val/mAP" in metrics:
            val = metrics["val/mAP"]
            if hasattr(val, "item"):
                val = val.item()
            self.history["val_map"].append(round(val, 4))

        if "val/mAP_50" in metrics:
            val = metrics["val/mAP_50"]
            if hasattr(val, "item"):
                val = val.item()
            self.history["val_map_50"].append(round(val, 4))

        if "val/recall" in metrics:
            val = metrics["val/recall"]
            if hasattr(val, "item"):
                val = val.item()
            self.history["val_recall"].append(round(val, 4))

        save_metrics_history(self.history)
        save_plots(self.history)
        gc.collect()
        torch.cuda.empty_cache()


def find_last_checkpoint(checkpoint_dir):
    """Find the last saved checkpoint to resume from."""
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        return str(last_ckpt)
    return None


def run_training(cfg: DictConfig):
    """Main training function."""
    try:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.search_experiments()
        tracking_uri = cfg.mlflow_tracking_uri
    except Exception:
        tracking_uri = "file:./mlruns"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow server not available, logging to {tracking_uri}")

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg.project_name,
        tracking_uri=tracking_uri,
    )

    mlflow_logger.log_hyperparams(
        {
            "model": cfg.model.name,
            "num_classes": cfg.model.num_classes,
            "batch_size": cfg.data.batch_size,
            "learning_rate": cfg.training.learning_rate,
            "lr_backbone": cfg.training.lr_backbone,
            "max_epochs": cfg.training.max_epochs,
            "subset_ratio": cfg.data.subset_ratio,
            "git_commit": get_git_commit(),
        }
    )

    best_checkpoint = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_loss:.4f}",
        dirpath="checkpoints",
    )

    last_checkpoint = ModelCheckpoint(
        every_n_epochs=1,
        save_top_k=1,
        filename="last",
        dirpath="checkpoints",
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=cfg.training.patience,
        mode="min",
    )

    plot_callback = PlotCallback()

    datamodule = DocLayNetDataModule(
        data_dir=cfg.data.data_dir,
        model_name=cfg.model.name,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        subset_ratio=cfg.data.subset_ratio,
    )

    model = DETRLightningModule(
        model_name=cfg.model.name,
        num_classes=cfg.model.num_classes,
        learning_rate=cfg.training.learning_rate,
        lr_backbone=cfg.training.lr_backbone,
        weight_decay=cfg.training.weight_decay,
        num_queries=cfg.model.num_queries,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=mlflow_logger,
        callbacks=[best_checkpoint, last_checkpoint, early_stop_callback, plot_callback],
        accelerator="auto",
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=10,
    )

    resume_ckpt = find_last_checkpoint("checkpoints")
    if resume_ckpt:
        print(f"Resuming from checkpoint: {resume_ckpt}")

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)

    print("Training done!")
    print(f"Best model saved at: {best_checkpoint.best_model_path}")


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig):
    """Hydra entry point."""
    run_training(cfg)


if __name__ == "__main__":
    train()
