# Crop Disease Detection — System Architecture

## Pipeline Overview

1. **Data Ingestion**: Raw leaf images organized by disease class
2. **Preprocessing**: Resize (224×224), normalize, type conversion
3. **Augmentation**: Random flips, rotations, color jitter (training only)
4. **CNN Training**: Sequential Conv-Pool-Conv-Pool-FC architecture
5. **Evaluation**: Precision, Recall, F1-Score, Confusion Matrix on test split
6. **Model Persistence**: Best checkpoint saved during training

## Regularisation Strategy

- Dropout (0.3–0.5) on fully connected layers
- L2 weight decay on Dense layers
- Early stopping (patience=5 epochs)
- Learning rate reduction on plateau
- Data augmentation as implicit regulariser

## Why CNN for this task?

CNNs are the established state-of-the-art for image classification tasks because:
- Convolutional filters learn spatially-invariant features
- Pooling layers provide translation invariance
- Hierarchical feature learning matches the hierarchical nature of visual disease patterns
- Much more parameter-efficient than fully-connected networks for image data
