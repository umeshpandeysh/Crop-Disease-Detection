"""Model evaluation utilities for Crop Disease Detection."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed.")
    _TORCH_AVAILABLE = False

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False

try:
    from sklearn.metrics import (
        accuracy_score, classification_report, confusion_matrix,
        f1_score, precision_score, recall_score,
    )
    _SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not installed. Some metrics unavailable.")
    _SKLEARN_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _VIZ_AVAILABLE = True
except ImportError:
    logger.warning("matplotlib/seaborn not installed. Plotting unavailable.")
    _VIZ_AVAILABLE = False


class Evaluator:
    """Runs inference and computes evaluation metrics for a trained model.

    Parameters
    ----------
    model:
        Trained PyTorch model.
    test_loader:
        DataLoader for the test set.
    class_names:
        Ordered list of class labels.
    device:
        Computation device string.
    """

    def __init__(
        self,
        model,
        test_loader,
        class_names: List[str],
        device: str = "cpu",
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for the Evaluator.")
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        self._all_preds: List[int] = []
        self._all_labels: List[int] = []
        self._all_probs: Optional["np.ndarray"] = None
        self._evaluated: bool = False

    def _run_inference(self) -> None:
        """Collect predictions and labels over the entire test set."""
        import torch.nn.functional as F
        all_preds: List[int] = []
        all_labels: List[int] = []
        all_probs_list = []

        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                probs = F.softmax(outputs, dim=1).cpu().numpy()
                preds = outputs.argmax(dim=1).cpu().numpy().tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy().tolist())
                all_probs_list.append(probs)

        self._all_preds = all_preds
        self._all_labels = all_labels
        if _NP_AVAILABLE:
            import numpy as np
            self._all_probs = np.concatenate(all_probs_list, axis=0)
        self._evaluated = True
        logger.info("Inference complete on %d samples.", len(all_preds))

    def evaluate(self) -> Dict:
        """Run inference and compute comprehensive evaluation metrics.

        Returns
        -------
        Dict
            ``{accuracy, precision, recall, f1, per_class_metrics}``.
        """
        if not _SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for evaluate().")
        if not self._evaluated:
            self._run_inference()

        acc = accuracy_score(self._all_labels, self._all_preds)
        precision = precision_score(self._all_labels, self._all_preds, average="weighted", zero_division=0)
        recall = recall_score(self._all_labels, self._all_preds, average="weighted", zero_division=0)
        f1 = f1_score(self._all_labels, self._all_preds, average="weighted", zero_division=0)

        report = classification_report(
            self._all_labels, self._all_preds,
            target_names=self.class_names, output_dict=True, zero_division=0,
        )

        per_class: Dict[str, Dict[str, float]] = {}
        for cls in self.class_names:
            if cls in report:
                per_class[cls] = {
                    "precision": report[cls]["precision"],
                    "recall": report[cls]["recall"],
                    "f1": report[cls]["f1-score"],
                    "support": report[cls]["support"],
                }

        metrics = {
            "accuracy": acc, "precision": precision,
            "recall": recall, "f1": f1,
            "per_class_metrics": per_class,
        }
        logger.info("Evaluation — acc=%.4f, precision=%.4f, recall=%.4f, f1=%.4f", acc, precision, recall, f1)
        return metrics

    def get_confusion_matrix(self) -> "np.ndarray":
        """Return the confusion matrix as a NumPy array."""
        if not _SKLEARN_AVAILABLE or not _NP_AVAILABLE:
            raise ImportError("scikit-learn and NumPy are required.")
        if not self._evaluated:
            self._run_inference()
        return confusion_matrix(self._all_labels, self._all_preds)

    def plot_confusion_matrix(
        self,
        cm: Optional["np.ndarray"] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a labelled confusion matrix heatmap.

        Parameters
        ----------
        cm:
            Confusion matrix. Computed automatically if None.
        save_path:
            Destination file path for the figure.
        """
        if not _VIZ_AVAILABLE or not _NP_AVAILABLE:
            logger.warning("matplotlib/seaborn/numpy not available for plotting.")
            return
        if cm is None:
            cm = self.get_confusion_matrix()

        import numpy as np
        fig, ax = plt.subplots(figsize=(12, 10))
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=self.class_names, yticklabels=self.class_names, ax=ax,
        )
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        ax.set_title("Normalised Confusion Matrix", fontsize=14, fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Confusion matrix saved to %s.", save_path)
        else:
            plt.show()
        plt.close(fig)

    def plot_class_performance(
        self,
        metrics: Optional[Dict] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Bar chart of per-class precision, recall, and F1.

        Parameters
        ----------
        metrics:
            Dict from :meth:`evaluate`. Computed automatically if None.
        save_path:
            Destination file path for the figure.
        """
        if not _VIZ_AVAILABLE:
            logger.warning("matplotlib not available for plotting.")
            return
        if metrics is None:
            metrics = self.evaluate()

        per_class = metrics["per_class_metrics"]
        classes = list(per_class.keys())
        precision = [per_class[c]["precision"] for c in classes]
        recall = [per_class[c]["recall"] for c in classes]
        f1 = [per_class[c]["f1"] for c in classes]

        x = range(len(classes))
        width = 0.28
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.bar([i - width for i in x], precision, width, label="Precision", color="steelblue")
        ax.bar(list(x), recall, width, label="Recall", color="seagreen")
        ax.bar([i + width for i in x], f1, width, label="F1", color="tomato")
        ax.set_xticks(list(x))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Score")
        ax.set_title("Per-Class Performance", fontsize=14, fontweight="bold")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info("Class performance plot saved to %s.", save_path)
        else:
            plt.show()
        plt.close(fig)

    def get_misclassified_samples(self, n: int = 16) -> List[Dict]:
        """Return a list of misclassified sample descriptions.

        Parameters
        ----------
        n:
            Maximum number of samples to return.

        Returns
        -------
        List[Dict]
            Dicts with ``index``, ``true_label``, ``pred_label``,
            ``true_class``, ``pred_class``.
        """
        if not self._evaluated:
            self._run_inference()

        misclassified = []
        for idx, (true, pred) in enumerate(zip(self._all_labels, self._all_preds)):
            if true != pred:
                entry: Dict = {
                    "index": idx,
                    "true_label": true,
                    "pred_label": pred,
                    "true_class": self.class_names[true],
                    "pred_class": self.class_names[pred],
                }
                if self._all_probs is not None:
                    entry["confidence"] = float(self._all_probs[idx][pred])
                misclassified.append(entry)
                if len(misclassified) >= n:
                    break
        logger.info("Found %d misclassified samples (capped at %d).", len(misclassified), n)
        return misclassified

    def generate_report(self, output_dir: str) -> None:
        """Save all metrics and plots to *output_dir*.

        Parameters
        ----------
        output_dir:
            Directory where evaluation artefacts will be written.
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        metrics = self.evaluate()
        metrics_path = out / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        logger.info("Metrics saved to %s.", metrics_path)

        cm = self.get_confusion_matrix()
        self.plot_confusion_matrix(cm, save_path=str(out / "confusion_matrix.png"))
        self.plot_class_performance(metrics, save_path=str(out / "class_performance.png"))

        misclassified = self.get_misclassified_samples()
        misclassified_path = out / "misclassified.json"
        with open(misclassified_path, "w", encoding="utf-8") as fh:
            json.dump(misclassified, fh, indent=2)
        logger.info("Evaluation report written to %s.", output_dir)
