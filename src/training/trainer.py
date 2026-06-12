"""Trainer class for the Crop Disease Detection project."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed. Trainer will not function.")
    _TORCH_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


class Trainer:
    """Manages the full training and validation loop.

    Parameters
    ----------
    model:
        PyTorch model to train.
    train_loader:
        DataLoader for training data.
    val_loader:
        DataLoader for validation data.
    config:
        ``TrainingConfig`` dataclass with hyper-parameters.
    device:
        Device string (``'cpu'``, ``'cuda'``, or ``'mps'``).
    logger:
        Optional external logger; falls back to module logger.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        config,
        device: str = "cpu",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for the Trainer.")

        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device(device)
        self.log = logger or logging.getLogger(__name__)

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.lr_factor,
            patience=config.patience // 2,
            min_lr=config.lr_min,
            verbose=True,
        )

        self.best_val_loss: float = float("inf")
        self.best_val_acc: float = 0.0
        self.epochs_without_improvement: int = 0

    def train_epoch(self) -> Dict[str, float]:
        """Run one full training epoch.

        Returns
        -------
        Dict[str, float]
            ``{loss, accuracy}`` averaged over all mini-batches.
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

            if batch_idx % 50 == 0:
                self.log.debug(
                    "  Batch %d/%d — loss: %.4f",
                    batch_idx, len(self.train_loader), loss.item(),
                )

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}

    def validate_epoch(self) -> Dict[str, float]:
        """Run one full validation pass.

        Returns
        -------
        Dict[str, float]
            ``{loss, accuracy}`` averaged over the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

        avg_loss = total_loss / total if total > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        return {"loss": avg_loss, "accuracy": accuracy}

    def train(self, num_epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """Execute the full training loop with early stopping and checkpointing.

        Parameters
        ----------
        num_epochs:
            Number of epochs. Defaults to ``config.epochs``.

        Returns
        -------
        Dict[str, List[float]]
            History with keys ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``.
        """
        epochs = num_epochs or self.config.epochs
        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []
        }

        self.log.info("Starting training for %d epochs on device '%s'.", epochs, self.device)

        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate_epoch()

            history["train_loss"].append(train_metrics["loss"])
            history["val_loss"].append(val_metrics["loss"])
            history["train_acc"].append(train_metrics["accuracy"])
            history["val_acc"].append(val_metrics["accuracy"])

            self.scheduler.step(val_metrics["loss"])

            self.log.info(
                "Epoch %d/%d — train_loss: %.4f, train_acc: %.4f | val_loss: %.4f, val_acc: %.4f",
                epoch, epochs,
                train_metrics["loss"], train_metrics["accuracy"],
                val_metrics["loss"], val_metrics["accuracy"],
            )

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_val_acc = val_metrics["accuracy"]
                self.epochs_without_improvement = 0
                self.save_checkpoint(
                    filepath="models/best_model.pt",
                    epoch=epoch,
                    metric=val_metrics["loss"],
                )
                self.log.info("  ✓ Best model saved (val_loss=%.4f).", self.best_val_loss)
            else:
                self.epochs_without_improvement += 1

            if self.epochs_without_improvement >= self.config.patience:
                self.log.info(
                    "Early stopping after %d epochs without improvement.",
                    self.config.patience,
                )
                break

        self.log.info(
            "Training complete. Best val_loss=%.4f, val_acc=%.4f",
            self.best_val_loss, self.best_val_acc,
        )
        return history

    def save_checkpoint(self, filepath: str, epoch: int, metric: float) -> None:
        """Serialise model, optimiser, and training state to disk.

        Parameters
        ----------
        filepath:
            Destination path (``*.pt``).
        epoch:
            Current epoch number.
        metric:
            Validation metric at this checkpoint.
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metric": metric,
            "config": self.config,
        }
        torch.save(checkpoint, filepath)
        self.log.debug("Checkpoint saved to %s.", filepath)

    def load_checkpoint(self, filepath: str) -> Dict:
        """Restore model and optimiser from a checkpoint file.

        Parameters
        ----------
        filepath:
            Path to the checkpoint ``*.pt`` file.

        Returns
        -------
        Dict
            The full checkpoint dictionary.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.log.info(
            "Checkpoint loaded from %s (epoch %d, metric=%.4f).",
            filepath,
            checkpoint.get("epoch", -1),
            checkpoint.get("metric", float("nan")),
        )
        return checkpoint

    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        save_path: Optional[str] = None,
    ) -> None:
        """Plot training and validation loss/accuracy curves.

        Parameters
        ----------
        history:
            Dictionary returned by :meth:`train`.
        save_path:
            File path to save the figure. Displays if None.
        """
        if not _MPL_AVAILABLE:
            self.log.warning("Matplotlib not installed; cannot plot training history.")
            return

        epochs_range = range(1, len(history["train_loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="royalblue")
        ax1.plot(epochs_range, history["val_loss"], label="Val Loss", color="tomato")
        ax1.set_title("Training & Validation Loss", fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(epochs_range, [a * 100 for a in history["train_acc"]], label="Train Acc", color="royalblue")
        ax2.plot(epochs_range, [a * 100 for a in history["val_acc"]], label="Val Acc", color="tomato")
        ax2.set_title("Training & Validation Accuracy", fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.suptitle("Crop Disease CNN — Training History", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            self.log.info("Training history plot saved to %s.", save_path)
        else:
            plt.show()
        plt.close(fig)
