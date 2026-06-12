"""Unit tests for the CropDiseaseDataset and image utilities."""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest
from pathlib import Path


def _create_temp_dataset(root: str, classes: list, n_per_class: int = 3) -> None:
    """Create a minimal directory tree with blank image files."""
    for split in ("train", "val", "test"):
        for cls in classes:
            cls_dir = Path(root) / split / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                (cls_dir / f"img_{i:03d}.jpg").touch()


class TestCropDiseaseDataset(unittest.TestCase):
    """Tests for CropDiseaseDataset."""

    CLASSES = ["Healthy", "Early_Blight", "Late_Blight"]

    def _make_dataset_dir(self, n_per_class: int = 5) -> str:
        tmp = tempfile.mkdtemp()
        _create_temp_dataset(tmp, self.CLASSES, n_per_class=n_per_class)
        return tmp

    def test_init_raises_on_missing_split_dir(self):
        """FileNotFoundError when split directory is absent."""
        from src.dataset.dataset_loader import CropDiseaseDataset
        with self.assertRaises(FileNotFoundError):
            CropDiseaseDataset("/nonexistent/path", split="train")

    def test_init_discovers_classes_automatically(self):
        """Auto-discovers class folders when class_names is None."""
        from src.dataset.dataset_loader import CropDiseaseDataset
        tmp = self._make_dataset_dir()
        try:
            ds = CropDiseaseDataset(tmp, split="train", class_names=None)
            self.assertEqual(sorted(ds.class_names), sorted(self.CLASSES))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_len_returns_correct_count(self):
        """__len__ equals number of image files scanned."""
        from src.dataset.dataset_loader import CropDiseaseDataset
        n_per_class = 5
        tmp = self._make_dataset_dir(n_per_class=n_per_class)
        try:
            ds = CropDiseaseDataset(tmp, split="train", class_names=self.CLASSES)
            self.assertEqual(len(ds), n_per_class * len(self.CLASSES))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_class_to_idx_mapping(self):
        """class_to_idx maps every class to a unique integer."""
        from src.dataset.dataset_loader import CropDiseaseDataset
        tmp = self._make_dataset_dir()
        try:
            ds = CropDiseaseDataset(tmp, split="train", class_names=self.CLASSES)
            self.assertEqual(len(ds.class_to_idx), len(self.CLASSES))
            self.assertEqual(set(ds.class_to_idx.values()), set(range(len(self.CLASSES))))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_get_class_distribution(self):
        """get_class_distribution returns dict with correct counts."""
        from src.dataset.dataset_loader import CropDiseaseDataset
        n_per_class = 5
        tmp = self._make_dataset_dir(n_per_class=n_per_class)
        try:
            ds = CropDiseaseDataset(tmp, split="train", class_names=self.CLASSES)
            dist = ds.get_class_distribution()
            for cls in self.CLASSES:
                self.assertEqual(dist[cls], n_per_class)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_samples_list_contains_tuples(self):
        """Each element of samples is a (Path, int) tuple."""
        from src.dataset.dataset_loader import CropDiseaseDataset
        tmp = self._make_dataset_dir()
        try:
            ds = CropDiseaseDataset(tmp, split="train", class_names=self.CLASSES)
            for item in ds.samples:
                self.assertIsInstance(item, tuple)
                self.assertEqual(len(item), 2)
                path, label = item
                self.assertIsInstance(path, Path)
                self.assertIsInstance(label, int)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestImageUtils(unittest.TestCase):
    """Tests for image utility functions."""

    def _create_temp_png(self, size: tuple = (10, 10)) -> str:
        """Write a small valid PNG to a temp file and return its path."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed.")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fh:
            path = fh.name
        img = Image.new("RGB", size, color=(0, 0, 0))
        img.save(path)
        return path

    def test_is_valid_image_returns_true_for_valid_png(self):
        """Valid 10x10 PNG should be accepted."""
        from src.preprocessing.image_utils import is_valid_image
        path = self._create_temp_png()
        try:
            self.assertTrue(is_valid_image(path))
        finally:
            os.unlink(path)

    def test_is_valid_image_returns_false_for_nonexistent(self):
        """Non-existent path returns False."""
        from src.preprocessing.image_utils import is_valid_image
        self.assertFalse(is_valid_image("/nonexistent/image.png"))

    def test_is_valid_image_returns_false_for_corrupt_file(self):
        """Garbage bytes should be identified as invalid."""
        from src.preprocessing.image_utils import is_valid_image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as fh:
            fh.write(b"not-an-image-xyz123")
            path = fh.name
        try:
            self.assertFalse(is_valid_image(path))
        finally:
            os.unlink(path)

    def test_resize_image(self):
        """resize_image returns image with requested dimensions."""
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed.")
        from src.preprocessing.image_utils import resize_image
        img = Image.new("RGB", (100, 100))
        resized = resize_image(img, (50, 60))
        self.assertEqual(resized.size, (50, 60))

    def test_normalize_image_output_range(self):
        """normalize_image produces values in [0, 1]."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy not installed.")
        from src.preprocessing.image_utils import normalize_image
        arr = np.random.randint(0, 256, (64, 64, 3)).astype(np.uint8)
        result = normalize_image(arr)
        self.assertAlmostEqual(float(result.min()), 0.0, places=5)
        self.assertAlmostEqual(float(result.max()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
