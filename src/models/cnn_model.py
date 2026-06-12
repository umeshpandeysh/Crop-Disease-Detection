"""CNN and Transfer Learning model definitions for Crop Disease Detection."""

from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed.")
    _TORCH_AVAILABLE = False
    nn = None  # type: ignore[assignment]


def _require_torch() -> None:
    """Raise ImportError if PyTorch is not installed."""
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required. Install with: pip install torch torchvision"
        )


class CropDiseaseCNN(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """Lightweight custom CNN for multi-class plant disease classification.

    Architecture::

        Input (3, 224, 224)
          Conv(3->32) + BN + ReLU + MaxPool(2x2)    -> 112x112
          Conv(32->64) + BN + ReLU + MaxPool(2x2)   ->  56x56
          Conv(64->128) + BN + ReLU + MaxPool(2x2)  ->  28x28
          Conv(128->256) + BN + ReLU + AdaptiveAvgPool(1x1)
          Flatten -> Linear(256->256) -> ReLU -> Dropout
          Linear(256->num_classes)

    Parameters
    ----------
    num_classes:
        Number of output classes.
    img_size:
        Input image resolution (stored for reference; not used in computation).
    dropout_rate:
        Dropout probability in the classifier head.
    """

    def __init__(
        self,
        num_classes: int,
        img_size: int = 224,
        dropout_rate: float = 0.3,
    ) -> None:
        _require_torch()
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.dropout_rate = dropout_rate

        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 4: 28x28 -> 1x1 (via adaptive pool)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

        self._init_weights()
        logger.info(
            "CropDiseaseCNN: %d classes, dropout=%.2f, params=%d",
            num_classes, dropout_rate, self.get_num_params(),
        )

    def _init_weights(self) -> None:
        """Kaiming/Xavier weight initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        x = self.features(x)
        return self.classifier(x)

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class EfficientNetTransfer(nn.Module if _TORCH_AVAILABLE else object):  # type: ignore[misc]
    """EfficientNet-B0 fine-tuned for crop disease classification.

    The pre-trained backbone can be optionally frozen during initial training
    and unfrozen for fine-tuning via :meth:`unfreeze_base`.

    Parameters
    ----------
    num_classes:
        Number of output classes.
    dropout_rate:
        Dropout probability before the final linear layer.
    freeze_base:
        If ``True``, backbone weights are frozen on construction.
    """

    def __init__(
        self,
        num_classes: int,
        dropout_rate: float = 0.3,
        freeze_base: bool = True,
    ) -> None:
        _require_torch()
        super().__init__()
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        try:
            from torchvision import models
            weights = models.EfficientNet_B0_Weights.DEFAULT
            self.base = models.efficientnet_b0(weights=weights)
        except AttributeError:
            from torchvision import models
            self.base = models.efficientnet_b0(pretrained=True)  # type: ignore[call-arg]

        in_features: int = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, num_classes),
        )

        if freeze_base:
            self.freeze_base()

        logger.info(
            "EfficientNetTransfer: %d classes, frozen=%s, params=%d",
            num_classes, freeze_base, self.get_num_params(),
        )

    def freeze_base(self) -> None:
        """Freeze all backbone parameters (train classifier head only)."""
        for name, param in self.base.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        logger.info("EfficientNet backbone frozen.")

    def unfreeze_base(self) -> None:
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self.base.parameters():
            param.requires_grad = True
        logger.info("EfficientNet backbone unfrozen for fine-tuning.")

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        """Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(B, 3, H, W)``.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, num_classes)``.
        """
        return self.base(x)

    def get_num_params(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_model(model_type: str, num_classes: int, **kwargs) -> "nn.Module":
    """Instantiate a model by name.

    Parameters
    ----------
    model_type:
        One of ``'cnn'`` or ``'efficientnet'``.
    num_classes:
        Number of output classes.
    **kwargs:
        Additional arguments forwarded to the model constructor.

    Returns
    -------
    nn.Module
        Instantiated model.

    Raises
    ------
    ValueError
        If *model_type* is not recognised.
    """
    _require_torch()
    registry: Dict[str, type] = {
        "cnn": CropDiseaseCNN,
        "efficientnet": EfficientNetTransfer,
    }
    if model_type not in registry:
        raise ValueError(
            f"Unknown model type '{model_type}'. Choose from: {list(registry.keys())}"
        )
    model = registry[model_type](num_classes=num_classes, **kwargs)
    logger.info("Model '%s' instantiated with %d classes.", model_type, num_classes)
    return model
