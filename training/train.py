"""
train.py
Crop Disease Detection — CNN Training Script

Trains a CNN for multi-class plant disease classification.
Supports both PyTorch and Keras backends.

Usage:
    python training/train.py --data_dir dataset/ --epochs 30 --batch_size 32
"""

import argparse
import os
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Crop Disease CNN Training")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img_size", type=int, default=224, help="Input image size")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save trained model")
    return parser.parse_args()


def check_dataset(data_dir: str):
    """
    Verify dataset directory exists and has expected structure.

    Expected:
        dataset/
            train/
                class_name_1/   (*.jpg, *.png)
                class_name_2/
            val/
                class_name_1/
                class_name_2/
    """
    data_path = Path(data_dir)
    train_path = data_path / "train"
    val_path = data_path / "val"

    if not train_path.exists():
        print(f"[ERROR] Training data not found at: {train_path}")
        print("[INFO] Please download a plant disease dataset (e.g., PlantVillage) and")
        print("       organize it into dataset/train/ and dataset/val/ with class subdirectories.")
        return False

    classes = [d.name for d in train_path.iterdir() if d.is_dir()]
    print(f"[INFO] Found {len(classes)} disease classes: {classes}")
    return True


def build_model_keras(num_classes: int, img_size: int, dropout_rate: float):
    """
    Build a CNN model using Keras.

    Args:
        num_classes: Number of disease categories.
        img_size: Input image dimension (square).
        dropout_rate: Dropout probability.

    Returns:
        Compiled Keras model.
    """
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError:
        print("[ERROR] TensorFlow not installed. Run: pip install tensorflow")
        return None

    model = keras.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        # Classifier head
        layers.Dense(256, activation='relu'),
        layers.Dropout(dropout_rate),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model


def train_with_keras(data_dir: str, epochs: int, batch_size: int, img_size: int,
                     dropout: float, output_dir: str):
    """
    Train CNN using Keras ImageDataGenerator pipeline.

    Args:
        data_dir: Root dataset directory.
        epochs: Training epochs.
        batch_size: Batch size.
        img_size: Image dimension.
        dropout: Dropout rate.
        output_dir: Directory to save model.
    """
    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError:
        print("[ERROR] TensorFlow not installed.")
        return

    # Data generators with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2]
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )
    val_gen = val_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical'
    )

    num_classes = train_gen.num_classes
    print(f"[INFO] Classes: {num_classes}")

    model = build_model_keras(num_classes, img_size, dropout)
    if model is None:
        return

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, "best_model.keras"),
            save_best_only=True
        )
    ]

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)

    print("[INFO] Training complete. Model saved to:", output_dir)
    return model, history


if __name__ == "__main__":
    args = parse_args()
    print("[INFO] Crop Disease Detection — CNN Training")
    print(f"[INFO] Data: {args.data_dir} | Epochs: {args.epochs} | Batch: {args.batch_size}")

    if check_dataset(args.data_dir):
        train_with_keras(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            img_size=args.img_size,
            dropout=args.dropout,
            output_dir=args.output_dir
        )
    else:
        print("[INFO] Training skipped — please add dataset first.")
