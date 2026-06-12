"""Image utility functions for dataset validation and analysis."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from PIL import Image, UnidentifiedImageError
    _PIL_AVAILABLE = True
except ImportError:
    logger.warning("Pillow not installed. Image utilities will be limited.")
    _PIL_AVAILABLE = False

try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    logger.warning("NumPy not installed.")
    _NP_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not installed.")
    _MPL_AVAILABLE = False

_VALID_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp")


def is_valid_image(filepath: str) -> bool:
    """Check whether *filepath* is a valid, non-corrupt RGB image.

    Parameters
    ----------
    filepath:
        Path to the image file.

    Returns
    -------
    bool
        ``True`` if the image can be opened and has non-zero dimensions.
    """
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required for is_valid_image.")
    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            img = img.convert("RGB")
            width, height = img.size
            if width == 0 or height == 0:
                return False
        return True
    except (OSError, UnidentifiedImageError, SyntaxError):
        return False


def resize_image(image: "Image.Image", size: Tuple[int, int]) -> "Image.Image":
    """Resize a PIL image using Lanczos resampling.

    Parameters
    ----------
    image:
        Source PIL image.
    size:
        Target ``(width, height)`` in pixels.

    Returns
    -------
    PIL.Image.Image
        Resized image.
    """
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required for resize_image.")
    return image.resize(size, Image.LANCZOS)


def normalize_image(image: "np.ndarray") -> "np.ndarray":
    """Per-channel normalisation of an image array to [0, 1].

    Parameters
    ----------
    image:
        NumPy array of shape ``(H, W, C)``.

    Returns
    -------
    np.ndarray
        Float32 array with values in ``[0, 1]``.
    """
    if not _NP_AVAILABLE:
        raise ImportError("NumPy is required for normalize_image.")
    image = image.astype(np.float32)
    for c in range(image.shape[2]):
        ch_min = image[:, :, c].min()
        ch_max = image[:, :, c].max()
        denom = ch_max - ch_min
        if denom > 0:
            image[:, :, c] = (image[:, :, c] - ch_min) / denom
        else:
            image[:, :, c] = 0.0
    return image


def validate_dataset_directory(data_dir: str) -> Dict:
    """Scan *data_dir* and report image counts and invalid files.

    Parameters
    ----------
    data_dir:
        Root directory of the dataset.

    Returns
    -------
    Dict
        ``{total_images, by_class, invalid_files}``.
    """
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow is required for validate_dataset_directory.")

    root = Path(data_dir)
    by_class: Dict[str, int] = {}
    invalid_files: List[str] = []
    total = 0

    for img_path in root.rglob("*"):
        if img_path.suffix.lower() not in _VALID_EXTENSIONS:
            continue
        total += 1
        class_name = img_path.parent.name
        by_class[class_name] = by_class.get(class_name, 0) + 1
        if not is_valid_image(str(img_path)):
            invalid_files.append(str(img_path))

    logger.info("Dataset validation — total: %d, invalid: %d", total, len(invalid_files))
    return {"total_images": total, "by_class": by_class, "invalid_files": invalid_files}


def compute_dataset_stats(data_dir: str, n_samples: int = 1000) -> Dict:
    """Estimate per-channel mean and std deviation over the dataset.

    Parameters
    ----------
    data_dir:
        Root directory of the dataset.
    n_samples:
        Maximum number of images to sample.

    Returns
    -------
    Dict
        ``{mean: [R, G, B], std: [R, G, B], n_sampled: int}``.
    """
    if not _PIL_AVAILABLE or not _NP_AVAILABLE:
        raise ImportError("Pillow and NumPy are required for compute_dataset_stats.")

    root = Path(data_dir)
    all_paths = [p for p in root.rglob("*") if p.suffix.lower() in _VALID_EXTENSIONS]
    sampled = random.sample(all_paths, min(n_samples, len(all_paths)))

    channel_sums = np.zeros(3, dtype=np.float64)
    channel_sq_sums = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for img_path in sampled:
        try:
            with Image.open(img_path) as img:
                arr = np.array(img.convert("RGB"), dtype=np.float64) / 255.0
            h, w, _ = arr.shape
            channel_sums += arr.sum(axis=(0, 1))
            channel_sq_sums += (arr ** 2).sum(axis=(0, 1))
            pixel_count += h * w
        except Exception as exc:
            logger.warning("Skipping %s: %s", img_path, exc)

    mean = (channel_sums / pixel_count).tolist() if pixel_count > 0 else [0.0, 0.0, 0.0]
    variance = channel_sq_sums / pixel_count - np.array(mean) ** 2 if pixel_count > 0 else np.zeros(3)
    std = np.sqrt(np.maximum(variance, 0)).tolist()

    logger.info("Dataset stats — mean: %s, std: %s (n=%d)", mean, std, len(sampled))
    return {"mean": mean, "std": std, "n_sampled": len(sampled)}


def visualize_samples(
    data_dir: str,
    n_samples: int = 4,
    save_path: Optional[str] = None,
) -> None:
    """Render a grid of sample images, one row per class.

    Parameters
    ----------
    data_dir:
        Root directory with class sub-directories.
    n_samples:
        Images to display per class.
    save_path:
        Save figure here instead of displaying if given.
    """
    if not _PIL_AVAILABLE or not _MPL_AVAILABLE:
        raise ImportError("Pillow and Matplotlib are required for visualize_samples.")

    root = Path(data_dir)
    class_dirs = sorted([
        d for d in root.rglob("*")
        if d.is_dir() and any(f.suffix.lower() in _VALID_EXTENSIONS for f in d.iterdir() if f.is_file())
    ])

    if not class_dirs:
        logger.warning("No class directories found in %s.", data_dir)
        return

    n_classes = len(class_dirs)
    fig, axes = plt.subplots(n_classes, n_samples, figsize=(n_samples * 3, n_classes * 3))
    if n_classes == 1:
        axes = [axes]

    for row_idx, class_dir in enumerate(class_dirs):
        images = [
            f for f in class_dir.iterdir()
            if f.is_file() and f.suffix.lower() in _VALID_EXTENSIONS
        ][:n_samples]
        for col_idx in range(n_samples):
            ax = axes[row_idx][col_idx] if n_samples > 1 else axes[row_idx]
            if col_idx < len(images):
                with Image.open(images[col_idx]) as img:
                    ax.imshow(img.convert("RGB"))
            else:
                ax.axis("off")
            if col_idx == 0:
                ax.set_ylabel(class_dir.name, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Sample Images per Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Sample grid saved to %s.", save_path)
    else:
        plt.show()
    plt.close(fig)
