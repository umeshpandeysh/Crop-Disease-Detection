# Architecture Documentation — Crop Disease Detection

---

## 1. CNN Architecture Diagram

```
+------------------------------------------------------+
|            INPUT  (B x 3 x 224 x 224)                |
+------------------------+-----------------------------+
                         |
         +---------------v-----------------+
         |  Conv2d(3->32, 3x3, pad=1)      |
         |  BatchNorm2d(32) + ReLU         |
         |  MaxPool2d(2x2)  -> 112x112     |
         +---------------+-----------------+
                         |
         +---------------v-----------------+
         |  Conv2d(32->64, 3x3, pad=1)     |
         |  BatchNorm2d(64) + ReLU         |
         |  MaxPool2d(2x2)  ->  56x56      |
         +---------------+-----------------+
                         |
         +---------------v-----------------+
         |  Conv2d(64->128, 3x3, pad=1)    |
         |  BatchNorm2d(128) + ReLU        |
         |  MaxPool2d(2x2)  ->  28x28      |
         +---------------+-----------------+
                         |
         +---------------v-----------------+
         |  Conv2d(128->256, 3x3, pad=1)   |
         |  BatchNorm2d(256) + ReLU        |
         |  AdaptiveAvgPool2d(1x1)         |
         +---------------+-----------------+
                         |
         +---------------v-----------------+
         |  Flatten  ->  256               |
         |  Linear(256->256) + ReLU        |
         |  Dropout(p=0.3)                 |
         +---------------+-----------------+
                         |
         +---------------v-----------------+
         |  Linear(256->10)                |
         |  OUTPUT: logits (B x 10)        |
         +---------------------------------+
```

**Total parameters (CropDiseaseCNN):** ~1.2 million trainable parameters

---

## 2. Data Pipeline Flow

```
Raw Images (JPEG/PNG)
        |
        v
+-------------------+      +-------------------------------+
|  Dataset Scan     |----->|  CropDiseaseDataset           |
|  (pathlib)        |      |  samples: [(Path, label_int)] |
+-------------------+      +----------+--------------------+
                                      |
                    +-----------------v-----------------+
                    |         get_train_transforms()    |
                    |  Resize -> Flip -> Rotate ->      |
                    |  ColorJitter -> RandomCrop ->     |
                    |  ToTensor -> Normalize            |
                    +-----------------+-----------------+
                                      |
                    +-----------------v-----------------+
                    |   WeightedRandomSampler           |
                    |   (handles class imbalance)       |
                    +-----------------+-----------------+
                                      |
                    +-----------------v-----------------+
                    |         DataLoader                |
                    |   batch_size=32, num_workers=4    |
                    +-----------------+-----------------+
                                      |
                    +-----------------v-----------------+
                    |         Model Forward Pass        |
                    |         Loss (CrossEntropy)       |
                    |         Backward + Optimiser Step |
                    +-----------------------------------+
```

---

## 3. Augmentation Strategy Rationale

| Augmentation | Probability | Rationale |
|-------------|:-----------:|-----------|
| `RandomHorizontalFlip` | 0.50 | Leaves have no preferred L/R orientation |
| `RandomVerticalFlip` | 0.30 | Occasional upside-down captures |
| `RandomRotation(30°)` | always | Camera tilt variation |
| `ColorJitter` | always | Lighting and sensor variation |
| `RandomCrop` (with padding) | always | Scale and framing variation |
| `Normalize` (ImageNet μ/σ) | always | Transfer-learning compatibility |

Augmentations are applied **only during training**.

---

## 4. Training Hyper-parameters

| Parameter | Value | Justification |
|-----------|------:|---------------|
| `epochs` | 30 | Sufficient for convergence with early stopping |
| `batch_size` | 32 | Stable gradient estimates |
| `learning_rate` | 1e-3 | Standard Adam starting point |
| `weight_decay` | 1e-4 | L2 regularisation |
| `dropout_rate` | 0.30 | Moderate regularisation |
| `patience` | 7 | ~25% of max epochs before early-stopping |
| `lr_factor` | 0.50 | Halve LR on plateau |
| `grad_clip` | 1.00 | Prevent gradient explosion |
| Optimiser | Adam | Adaptive LR; robust to sparse gradients |
| Scheduler | `ReduceLROnPlateau` | Data-driven LR decay |
| Loss | `CrossEntropyLoss` | Standard multi-class loss |

---

## 5. Transfer Learning Strategy (EfficientNet-B0)

```
Phase 1 — Classifier Warm-up (epochs 1-5)
  Backbone  : FROZEN  (ImageNet weights preserved)
  Head      : TRAINABLE
  LR        : 1e-3
  Goal      : Quickly train the new classifier head

Phase 2 — Fine-tuning (epochs 6+)
  Backbone  : UNFROZEN  (call model.unfreeze_base())
  Head      : TRAINABLE
  LR        : 1e-4  (reduce by 10x to protect backbone)
  Goal      : Adapt low-level features to crop domain
```

---

## 6. Future Improvements

| Priority | Improvement | Expected Impact |
|:--------:|-------------|----------------|
| High | **Grad-CAM visualisation** for explainability | Trust + debugging |
| High | **Test-Time Augmentation (TTA)** | +1-2% accuracy |
| Medium | **EfficientNet-B3 / ViT-B/16** backbone | Higher accuracy |
| Medium | **Mixup / CutMix** augmentation | Better minority class recall |
| Medium | **ONNX export** for mobile deployment | Production viability |
| Low | **Semi-supervised learning** on unlabelled field images | Domain adaptation |
| Low | **Multi-crop support** (pepper, potato) | Generalisation |
