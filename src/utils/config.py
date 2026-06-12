"""Configuration dataclasses for Crop Disease Detection project."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)

DISEASE_CLASSES: List[str] = [
    "Healthy",
    "Early_Blight",
    "Late_Blight",
    "Leaf_Mold",
    "Septoria_Leaf_Spot",
    "Spider_Mites",
    "Target_Spot",
    "Mosaic_Virus",
    "Yellow_Leaf_Curl",
    "Bacterial_Spot",
]


@dataclass
class DataConfig:
    """Configuration for dataset paths and loading parameters."""

    data_dir: str = "dataset"
    train_dir: str = "dataset/train"
    val_dir: str = "dataset/val"
    test_dir: str = "dataset/test"
    img_size: int = 224
    num_workers: int = 4
    class_names: List[str] = field(default_factory=lambda: list(DISEASE_CLASSES))

    def __post_init__(self) -> None:
        if self.img_size <= 0:
            raise ValueError(f"img_size must be positive, got {self.img_size}")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be >= 0, got {self.num_workers}")
        logger.debug("DataConfig initialised: img_size=%d, num_workers=%d", self.img_size, self.num_workers)


@dataclass
class TrainingConfig:
    """Hyper-parameter and training loop configuration."""

    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    dropout_rate: float = 0.3
    weight_decay: float = 1e-4
    patience: int = 7
    lr_factor: float = 0.5
    lr_min: float = 1e-6
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 < self.learning_rate < 1.0:
            raise ValueError(f"learning_rate must be in (0,1), got {self.learning_rate}")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError(f"dropout_rate must be in [0,1), got {self.dropout_rate}")
        logger.debug(
            "TrainingConfig: epochs=%d, batch_size=%d, lr=%.5f",
            self.epochs, self.batch_size, self.learning_rate,
        )


@dataclass
class PathConfig:
    """File-system path configuration."""

    models_dir: str = "models"
    logs_dir: str = "logs"
    outputs_dir: str = "outputs"
    checkpoint_name: str = "best_model.pt"

    @property
    def best_checkpoint_path(self) -> str:
        """Return the full path of the best model checkpoint."""
        import os
        return os.path.join(self.models_dir, self.checkpoint_name)


def get_config() -> Tuple[DataConfig, TrainingConfig, PathConfig]:
    """Return default configuration objects for the project.

    Returns
    -------
    Tuple[DataConfig, TrainingConfig, PathConfig]
        A tuple of (data_config, training_config, path_config).
    """
    data_cfg = DataConfig()
    train_cfg = TrainingConfig()
    path_cfg = PathConfig()
    logger.info("Loaded default project configuration.")
    return data_cfg, train_cfg, path_cfg
