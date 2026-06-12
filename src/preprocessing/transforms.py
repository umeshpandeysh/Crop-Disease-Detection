"""Image augmentation and preprocessing transforms for Crop Disease Detection."""

from __future__ import annotations

import logging
import warnings

logger = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

try:
    from torchvision import transforms
    _TV_AVAILABLE = True
except ImportError:
    warnings.warn(
        "torchvision is not installed. Transform utilities will not function.",
        ImportWarning,
        stacklevel=2,
    )
    _TV_AVAILABLE = False


def _check_torchvision() -> None:
    """Raise ``ImportError`` if torchvision is not available."""
    if not _TV_AVAILABLE:
        raise ImportError(
            "torchvision is required. Install with: pip install torchvision"
        )


def get_train_transforms(img_size: int = 224):
    """Return the augmentation pipeline for training images.

    Applies random flips, rotations, colour jitter, and random crops to
    improve model generalisation. Normalised with ImageNet statistics.

    Parameters
    ----------
    img_size:
        Target height and width in pixels.

    Returns
    -------
    transforms.Compose
        Composed torchvision transform.
    """
    _check_torchvision()
    padding = img_size // 8
    return transforms.Compose([
        transforms.Resize((img_size + padding, img_size + padding)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(img_size: int = 224):
    """Return deterministic transforms for validation/testing images.

    Parameters
    ----------
    img_size:
        Target height and width in pixels.

    Returns
    -------
    transforms.Compose
        Composed torchvision transform.
    """
    _check_torchvision()
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(img_size: int = 224):
    """Return transforms used during single-image inference.

    Identical to :func:`get_val_transforms`.

    Parameters
    ----------
    img_size:
        Target height and width in pixels.

    Returns
    -------
    transforms.Compose
        Composed torchvision transform.
    """
    return get_val_transforms(img_size)
