"""Unit tests for preprocessing transforms."""

from __future__ import annotations

import unittest


class TestTrainTransforms(unittest.TestCase):
    """Tests for get_train_transforms."""

    def _get_compose_type(self):
        try:
            from torchvision import transforms as tv_transforms
            return tv_transforms.Compose
        except ImportError:
            return None

    def test_returns_compose_object(self):
        """get_train_transforms returns a torchvision Compose object."""
        compose_type = self._get_compose_type()
        if compose_type is None:
            self.skipTest("torchvision not installed.")
        from src.preprocessing.transforms import get_train_transforms
        t = get_train_transforms(224)
        self.assertIsInstance(t, compose_type)

    def test_compose_has_more_than_five_transforms(self):
        """Training pipeline includes at least 6 steps."""
        compose_type = self._get_compose_type()
        if compose_type is None:
            self.skipTest("torchvision not installed.")
        from src.preprocessing.transforms import get_train_transforms
        t = get_train_transforms(224)
        self.assertGreater(len(t.transforms), 5)

    def test_different_sizes_produce_valid_compose(self):
        """Compose created successfully for non-default img_size."""
        compose_type = self._get_compose_type()
        if compose_type is None:
            self.skipTest("torchvision not installed.")
        from src.preprocessing.transforms import get_train_transforms
        for size in (64, 128, 256, 320):
            t = get_train_transforms(size)
            self.assertIsInstance(t, compose_type)


class TestValTransforms(unittest.TestCase):
    """Tests for get_val_transforms."""

    def _get_compose_type(self):
        try:
            from torchvision import transforms as tv_transforms
            return tv_transforms.Compose
        except ImportError:
            return None

    def test_returns_compose_object(self):
        """get_val_transforms returns a torchvision Compose object."""
        compose_type = self._get_compose_type()
        if compose_type is None:
            self.skipTest("torchvision not installed.")
        from src.preprocessing.transforms import get_val_transforms
        t = get_val_transforms(224)
        self.assertIsInstance(t, compose_type)

    def test_val_has_fewer_transforms_than_train(self):
        """Validation pipeline is shorter than training pipeline."""
        compose_type = self._get_compose_type()
        if compose_type is None:
            self.skipTest("torchvision not installed.")
        from src.preprocessing.transforms import get_train_transforms, get_val_transforms
        train_t = get_train_transforms(224)
        val_t = get_val_transforms(224)
        self.assertLess(len(val_t.transforms), len(train_t.transforms))

    def test_inference_matches_val(self):
        """get_inference_transforms returns same number of steps as get_val_transforms."""
        compose_type = self._get_compose_type()
        if compose_type is None:
            self.skipTest("torchvision not installed.")
        from src.preprocessing.transforms import get_inference_transforms, get_val_transforms
        val_t = get_val_transforms(224)
        inf_t = get_inference_transforms(224)
        self.assertEqual(len(val_t.transforms), len(inf_t.transforms))


class TestImageNormalize(unittest.TestCase):
    """Tests for normalize_image."""

    def test_normalize_all_same_channel_returns_zeros(self):
        """Uniform channel values normalise to 0."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy not installed.")
        from src.preprocessing.image_utils import normalize_image
        arr = np.full((32, 32, 3), 128, dtype=np.uint8)
        result = normalize_image(arr)
        self.assertTrue((result == 0.0).all())

    def test_normalize_output_dtype_float32(self):
        """Output dtype is float32."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy not installed.")
        from src.preprocessing.image_utils import normalize_image
        arr = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        result = normalize_image(arr)
        self.assertEqual(result.dtype, np.float32)

    def test_normalize_output_shape_unchanged(self):
        """Output shape is identical to input shape."""
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy not installed.")
        from src.preprocessing.image_utils import normalize_image
        arr = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
        result = normalize_image(arr)
        self.assertEqual(result.shape, arr.shape)


if __name__ == "__main__":
    unittest.main()
