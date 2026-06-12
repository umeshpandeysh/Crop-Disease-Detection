# Model Card — Crop Disease Detection CNN

> **Version:** 1.0.0  
> **Author:** Umesh Pandey  
> **Date:** 2025  
> **Framework:** PyTorch 2.x

---

## Model Description

This repository provides two trained models for **multi-class plant leaf disease classification**:

| Model | Architecture | Parameters | Notes |
|-------|-------------|------------|-------|
| `CropDiseaseCNN` | Custom 4-block CNN | ~1.2 M | Lightweight, fast |
| `EfficientNetTransfer` | EfficientNet-B0 (fine-tuned) | ~4.0 M trainable | Higher accuracy |

Both models output a probability distribution over **10 disease classes** from a single RGB leaf image.

---

## Intended Use

### Primary use case
- Automated triage of plant disease severity from smartphone photographs
- Decision support for farmers and agricultural extension workers
- Research baseline for comparing CNN architectures on PlantVillage

### Out-of-scope use
- Clinical or regulatory diagnosis of any plant pathogen
- Deployment on plant species not present in PlantVillage
- Images taken under severely different lighting without re-calibration

---

## Training Data

**Dataset:** PlantVillage (Kaggle mirror)  
**Total images used:** ~16,160 (10 classes)  
**Image format:** RGB JPEG, resized to 224 × 224 px  
**Split:** 70 % train / 20 % validation / 10 % test (stratified)

### Class distribution

| Class | Train | Val | Test |
|-------|------:|----:|-----:|
| Healthy | 1,114 | 318 | 159 |
| Early_Blight | 700 | 200 | 100 |
| Late_Blight | 1,336 | 382 | 191 |
| Leaf_Mold | 666 | 191 | 95 |
| Septoria_Leaf_Spot | 1,240 | 354 | 177 |
| Spider_Mites | 1,173 | 335 | 168 |
| Target_Spot | 983 | 281 | 140 |
| Mosaic_Virus | 261 | 75 | 37 |
| Yellow_Leaf_Curl | 3,750 | 1,071 | 536 |
| Bacterial_Spot | 1,489 | 425 | 213 |

---

## Architecture

### CropDiseaseCNN

```
Input  (B, 3, 224, 224)
  Conv2d(3->32, 3x3, pad=1)  -> BN -> ReLU -> MaxPool(2x2)
  Conv2d(32->64, 3x3, pad=1) -> BN -> ReLU -> MaxPool(2x2)
  Conv2d(64->128, 3x3,pad=1) -> BN -> ReLU -> MaxPool(2x2)
  Conv2d(128->256,3x3,pad=1) -> BN -> ReLU -> AdaptiveAvgPool(1x1)
  Flatten -> Linear(256->256) -> ReLU -> Dropout(0.3)
  Linear(256->10)
Output (B, 10)  — logits
```

### EfficientNetTransfer

- **Backbone:** EfficientNet-B0 pre-trained on ImageNet
- **Modified head:** `Dropout(0.3) -> Linear(1280 -> 10)`
- **Training strategy:** Backbone frozen for epoch 1–5, then unfrozen

---

## Evaluation Metrics

Results on the held-out **test set** (stratified 10 %):

| Model | Accuracy | Precision | Recall | F1-score | Notes |
|-------|:--------:|:---------:|:------:|:--------:|-------|
| CropDiseaseCNN | 91.3 % | 0.912 | 0.913 | 0.911 | Trained 30 epochs |
| EfficientNet-B0 | 96.7 % | 0.968 | 0.967 | 0.967 | Fine-tuned 15 epochs |

*Metrics are macro-averaged. Per-class breakdowns in `outputs/metrics.json`.*

---

## Limitations and Biases

1. **Dataset bias:** PlantVillage images were taken under controlled greenhouse conditions.
2. **Species scope:** Trained exclusively on *Solanum lycopersicum* (tomato) leaves.
3. **Class imbalance:** `Yellow_Leaf_Curl` (~5,357) vastly outnumbers `Mosaic_Virus` (~373).
4. **Resolution sensitivity:** Images should be at least 224 × 224 px.

---

## How to Use

```python
from src.inference.predict import DiseasePredictor

predictor = DiseasePredictor(
    model_path="models/best_model.pt",
    class_names=[
        "Healthy", "Early_Blight", "Late_Blight", "Leaf_Mold",
        "Septoria_Leaf_Spot", "Spider_Mites", "Target_Spot",
        "Mosaic_Virus", "Yellow_Leaf_Curl", "Bacterial_Spot",
    ],
    device="cpu",
)

result = predictor.predict_image("path/to/leaf.jpg")
print(result)
# {'label': 'Late_Blight', 'confidence': 0.9312, 'probabilities': {...}}
```

Or via CLI:

```bash
python -m src.inference.predict \
  --model models/best_model.pt \
  --image path/to/leaf.jpg \
  --top_k 3
```
