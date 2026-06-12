#!/usr/bin/env python3
"""Crop Disease Detection — Training Entry Point.

Usage
-----
    python training/train.py --data dataset/ --model efficientnet --epochs 30

This script:
    1. Loads the dataset using CropDiseaseDataset.
    2. Applies train/val transforms.
    3. Trains the selected CNN model (custom or EfficientNet-B0).
    4. Saves the best checkpoint to models/.
    5. Prints final metrics.

Requires PyTorch. If PyTorch is not installed, prints a clear error.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
)
logger = logging.getLogger('train')

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Crop Disease CNN Training')
    p.add_argument('--data',    default='dataset',
                   help='Root dataset directory (expects train/val/test subdirs)')
    p.add_argument('--model',   default='efficientnet', choices=['cnn', 'efficientnet'],
                   help='Model architecture to train')
    p.add_argument('--epochs',  type=int, default=30)
    p.add_argument('--batch',   type=int, default=32)
    p.add_argument('--lr',      type=float, default=1e-3)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--device',  default='cpu',
                   help='Torch device: cpu, cuda, or mps')
    p.add_argument('--output',  default='models/',
                   help='Directory to save checkpoints')
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not TORCH_AVAILABLE:
        logger.error('PyTorch is not installed. Run: pip install torch torchvision')
        sys.exit(1)

    from src.utils.config import DataConfig, TrainingConfig, PathConfig
    from src.models.cnn_model import get_model
    from src.dataset.dataset_loader import create_data_loaders
    from src.training.trainer import Trainer

    data_cfg = DataConfig(
        data_dir=args.data,
        train_dir=str(Path(args.data) / 'train'),
        val_dir=str(Path(args.data) / 'val'),
        test_dir=str(Path(args.data) / 'test'),
        num_workers=args.workers,
    )
    train_cfg = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch,
        learning_rate=args.lr,
    )
    path_cfg = PathConfig(models_dir=args.output)

    logger.info('Loading dataset from %s', args.data)
    try:
        train_loader, val_loader, _, class_names = create_data_loaders(
            data_dir=args.data,
            batch_size=train_cfg.batch_size,
            img_size=data_cfg.img_size,
            num_workers=data_cfg.num_workers,
        )
    except Exception as exc:
        logger.error(
            'Failed to load dataset: %s\n'
            'Ensure the dataset is in the expected structure:\n'
            '  %s/train/<class_name>/<image.jpg>\n'
            '  %s/val/<class_name>/<image.jpg>\n'
            'See dataset/README.md for download instructions.',
            exc, args.data, args.data
        )
        sys.exit(1)

    num_classes = len(class_names)
    logger.info('Classes (%d): %s', num_classes, class_names)

    model = get_model(
        model_type=args.model,
        num_classes=num_classes,
        dropout_rate=train_cfg.dropout_rate,
    )
    logger.info('Model: %s  Parameters: %s',
                args.model, f'{model.get_num_params():,}' if hasattr(model, 'get_num_params') else 'N/A')

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_cfg,
        device=args.device,
    )

    logger.info('Starting training for %d epochs on %s', args.epochs, args.device)
    history = trainer.train(num_epochs=args.epochs)

    best_val_acc = max(history['val_acc']) * 100 if history['val_acc'] else 0.0
    logger.info('Training complete. Best val_acc: %.2f%%', best_val_acc)
    logger.info('Best checkpoint saved to: %s', path_cfg.best_checkpoint_path)


if __name__ == '__main__':
    main()
