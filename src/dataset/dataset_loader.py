"""PyTorch Dataset and DataLoader utilities for Crop Disease Detection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    _TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed. Dataset utilities will be limited.")
    _TORCH_AVAILABLE = False
    Dataset = object  # type: ignore[misc,assignment]

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    logger.warning("Pillow not installed. Image loading will fail.")
    _PIL_AVAILABLE = False

_VALID_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


class CropDiseaseDataset(Dataset):  # type: ignore[misc]
    """PyTorch Dataset for crop disease classification.

    Expects directory structure::

        data_dir/
            train/
                Healthy/
                    img001.jpg
                Early_Blight/
                    ...
            val/
                ...
            test/
                ...

    Parameters
    ----------
    data_dir:
        Root directory of the dataset.
    split:
        One of ``'train'``, ``'val'``, or ``'test'``.
    transform:
        Optional callable applied to each PIL image.
    class_names:
        Optional explicit list of class folder names. Auto-discovered if None.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.split_dir = self.data_dir / split

        if not self.split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {self.split_dir}. "
                "Please download and prepare the dataset first."
            )

        if class_names is not None:
            self.class_names: List[str] = class_names
        else:
            self.class_names = sorted(
                [d.name for d in self.split_dir.iterdir() if d.is_dir()]
            )

        self.class_to_idx: Dict[str, int] = {
            cls: idx for idx, cls in enumerate(self.class_names)
        }

        self.samples: List[Tuple[Path, int]] = []
        self._build_index()
        logger.info(
            "CropDiseaseDataset [%s]: %d samples, %d classes.",
            split, len(self.samples), len(self.class_names),
        )

    def _build_index(self) -> None:
        """Scan the split directory and populate ``self.samples``."""
        self.samples = []
        for class_name in self.class_names:
            class_dir = self.split_dir / class_name
            if not class_dir.is_dir():
                logger.warning("Class directory not found, skipping: %s", class_dir)
                continue
            label_idx = self.class_to_idx[class_name]
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in _VALID_EXTENSIONS:
                    self.samples.append((img_path, label_idx))
        if not self.samples:
            logger.warning("No images found in split '%s'.", self.split)

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int):
        """Return (image_tensor, label) for sample at *idx*."""
        if not _PIL_AVAILABLE:
            raise RuntimeError("Pillow is required for image loading.")
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for dataset usage.")

        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            logger.error("Failed to load image %s: %s", img_path, exc)
            raise

        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_class_weights(self):
        """Compute inverse-frequency class weights.

        Returns
        -------
        torch.Tensor
            1-D tensor of shape ``(num_classes,)``.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required.")
        counts = [0] * len(self.class_names)
        for _, label in self.samples:
            counts[label] += 1
        total = sum(counts)
        weights = [total / (c + 1e-8) for c in counts]
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        weight_tensor = weight_tensor / weight_tensor.sum()
        return weight_tensor

    def get_class_distribution(self) -> Dict[str, int]:
        """Return mapping of class name to sample count."""
        distribution: Dict[str, int] = {name: 0 for name in self.class_names}
        for _, label in self.samples:
            distribution[self.class_names[label]] += 1
        return distribution


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: int = 224,
    num_workers: int = 4,
    class_names: Optional[List[str]] = None,
) -> Tuple:
    """Create train, validation, and test DataLoaders.

    Parameters
    ----------
    data_dir:
        Root directory containing ``train/``, ``val/``, ``test/``.
    batch_size:
        Number of samples per mini-batch.
    img_size:
        Height and width to resize images to.
    num_workers:
        Number of parallel data-loading workers.
    class_names:
        Optional explicit class name list.

    Returns
    -------
    Tuple
        ``(train_loader, val_loader, test_loader, class_names)``.
    """
    if not _TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for create_data_loaders.")

    from src.preprocessing.transforms import get_train_transforms, get_val_transforms

    train_transform = get_train_transforms(img_size)
    val_transform = get_val_transforms(img_size)

    train_dataset = CropDiseaseDataset(
        data_dir, split="train", transform=train_transform, class_names=class_names
    )
    val_dataset = CropDiseaseDataset(
        data_dir, split="val", transform=val_transform, class_names=class_names
    )
    test_dataset = CropDiseaseDataset(
        data_dir, split="test", transform=val_transform, class_names=class_names
    )

    class_weights = train_dataset.get_class_weights()
    sample_weights = torch.tensor(
        [class_weights[label].item() for _, label in train_dataset.samples],
        dtype=torch.float32,
    )
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_dataset),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    logger.info(
        "DataLoaders ready — train: %d, val: %d, test: %d batches.",
        len(train_loader), len(val_loader), len(test_loader),
    )
    return train_loader, val_loader, test_loader, train_dataset.class_names
