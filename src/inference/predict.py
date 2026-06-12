"""Inference module for single-image and batch disease prediction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not installed. Inference will not function.")
    _TORCH_AVAILABLE = False

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False


class DiseasePredictor:
    """Wraps a trained model for convenient single and batch inference.

    Parameters
    ----------
    model_path:
        Path to the saved PyTorch checkpoint (``*.pt``).
    class_names:
        Ordered list of class label strings.
    device:
        Computation device string.
    """

    def __init__(
        self,
        model_path: str,
        class_names: List[str],
        device: str = "cpu",
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DiseasePredictor.")
        self.model_path = model_path
        self.class_names = class_names
        self.device = torch.device(device)
        self.model: Optional["torch.nn.Module"] = None
        self._transform = None
        self.load_model()

    def load_model(self) -> None:
        """Load model from :attr:`model_path` and set to eval mode."""
        from src.preprocessing.transforms import get_inference_transforms
        checkpoint = torch.load(self.model_path, map_location=self.device)

        if "model_state_dict" in checkpoint:
            from src.models.cnn_model import CropDiseaseCNN
            self.model = CropDiseaseCNN(num_classes=len(self.class_names))
            self.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model = checkpoint

        self.model.to(self.device)
        self.model.eval()
        self._transform = get_inference_transforms(img_size=224)
        logger.info("Model loaded from %s.", self.model_path)

    def _preprocess(self, image_path: str) -> "torch.Tensor":
        """Load and preprocess a single image.

        Parameters
        ----------
        image_path:
            Path to the image file.

        Returns
        -------
        torch.Tensor
            Preprocessed tensor of shape ``(1, 3, H, W)``.
        """
        if not _PIL_AVAILABLE:
            raise ImportError("Pillow is required for image loading.")
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        tensor = self._transform(image)
        return tensor.unsqueeze(0).to(self.device)

    def predict_image(self, image_path: str) -> Dict:
        """Predict disease class for a single image.

        Parameters
        ----------
        image_path:
            Path to the input image.

        Returns
        -------
        Dict
            ``{label, confidence, probabilities: {class: prob}}``.
        """
        tensor = self._preprocess(image_path)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0)

        prob_values = probs.cpu().numpy().tolist()
        pred_idx = int(probs.argmax().item())
        confidence = prob_values[pred_idx]

        probabilities = {
            cls: round(prob, 6)
            for cls, prob in zip(self.class_names, prob_values)
        }
        result = {
            "label": self.class_names[pred_idx],
            "confidence": round(confidence, 6),
            "probabilities": probabilities,
        }
        logger.debug("Prediction: %s (%.4f)", result["label"], confidence)
        return result

    def predict_batch(self, image_paths: List[str]) -> List[Dict]:
        """Predict disease classes for a list of images.

        Parameters
        ----------
        image_paths:
            List of file paths to process.

        Returns
        -------
        List[Dict]
            One prediction dict per input image.
        """
        results = []
        for path in image_paths:
            try:
                results.append(self.predict_image(path))
            except Exception as exc:
                logger.error("Failed to predict %s: %s", path, exc)
                results.append({"label": "ERROR", "confidence": 0.0, "error": str(exc)})
        return results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crop Disease Predictor — classify plant leaf images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, type=str, help="Path to model checkpoint (.pt).")
    parser.add_argument("--image", required=True, type=str, help="Path to the input image.")
    parser.add_argument("--classes_file", type=str, default=None, help="JSON file with class names list.")
    parser.add_argument("--device", type=str, default="cpu", help="Computation device.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top predictions to display.")
    return parser


def _load_class_names(classes_file: Optional[str]) -> List[str]:
    """Load class names from JSON or return project defaults."""
    if classes_file and Path(classes_file).exists():
        with open(classes_file, encoding="utf-8") as fh:
            return json.load(fh)
    return [
        "Healthy", "Early_Blight", "Late_Blight", "Leaf_Mold",
        "Septoria_Leaf_Spot", "Spider_Mites", "Target_Spot",
        "Mosaic_Virus", "Yellow_Leaf_Curl", "Bacterial_Spot",
    ]


def main() -> None:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    class_names = _load_class_names(args.classes_file)
    predictor = DiseasePredictor(
        model_path=args.model, class_names=class_names, device=args.device,
    )

    result = predictor.predict_image(args.image)
    sorted_probs = sorted(result["probabilities"].items(), key=lambda kv: kv[1], reverse=True)

    print(f"\n{'=' * 50}")
    print(" Crop Disease Prediction")
    print(f"{'=' * 50}")
    print(f" Image   : {args.image}")
    print(f" Result  : {result['label']}")
    print(f" Confidence: {result['confidence'] * 100:.2f}%")
    print(f"\n Top-{args.top_k} Predictions:")
    for rank, (cls, prob) in enumerate(sorted_probs[: args.top_k], 1):
        bar = "█" * int(prob * 30)
        print(f"   {rank}. {cls:<25} {prob * 100:6.2f}%  {bar}")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
