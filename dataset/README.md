# Dataset Guide — Crop Disease Detection

This directory contains metadata and sample labels for the Crop Disease Detection project.
The actual image dataset is **not committed to this repository** due to file-size constraints.

---

## Downloading the Dataset

We use a curated subset of the **PlantVillage** dataset, publicly available on Kaggle:

```
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
```

### Steps

1. Install the Kaggle API:
   ```bash
   pip install kaggle
   ```

2. Place your `kaggle.json` credentials in `~/.kaggle/`.

3. Download and extract:
   ```bash
   kaggle datasets download -d abdallahalidev/plantvillage-dataset -p dataset/raw
   unzip dataset/raw/plantvillage-dataset.zip -d dataset/raw
   ```

4. Organise into splits:
   ```bash
   python scripts/prepare_dataset.py --raw_dir dataset/raw --out_dir dataset --split 0.7 0.2 0.1
   ```

---

## Expected Directory Structure

```
dataset/
├── train/
│   ├── Healthy/
│   │   ├── tomato_healthy_0001.jpg
│   │   └── ...
│   ├── Early_Blight/
│   ├── Late_Blight/
│   ├── Leaf_Mold/
│   ├── Septoria_Leaf_Spot/
│   ├── Spider_Mites/
│   ├── Target_Spot/
│   ├── Mosaic_Virus/
│   ├── Yellow_Leaf_Curl/
│   └── Bacterial_Spot/
├── val/
│   └── (same class structure)
├── test/
│   └── (same class structure)
├── sample_labels.csv
└── README.md
```

---

## Class Statistics

| # | Class | Approx. Images | Description |
|---|-------|---------------|-------------|
| 0 | Healthy | 1,591 | No disease present |
| 1 | Early_Blight | 1,000 | *Alternaria solani* — concentric ring lesions |
| 2 | Late_Blight | 1,909 | *Phytophthora infestans* — water-soaked lesions |
| 3 | Leaf_Mold | 952 | *Passalora fulva* — yellowish spots |
| 4 | Septoria_Leaf_Spot | 1,771 | Small circular spots with dark borders |
| 5 | Spider_Mites | 1,676 | Yellow stippling from mite feeding |
| 6 | Target_Spot | 1,404 | *Corynespora cassiicola* — bullseye lesions |
| 7 | Mosaic_Virus | 373 | Tomato Mosaic Virus — mottled leaves |
| 8 | Yellow_Leaf_Curl | 5,357 | TYLCV — yellowing and leaf curling |
| 9 | Bacterial_Spot | 2,127 | *Xanthomonas* spp. — dark water-soaked spots |
| | **Total** | **~16,160** | |

---

## Data Split Rationale

| Split | Percentage | Purpose |
|-------|-----------|----------|
| Train | 70 % | Model parameter optimisation |
| Val | 20 % | Hyper-parameter tuning and early stopping |
| Test | 10 % | Final unbiased performance estimate |

A **stratified split** is used to preserve class proportions across all three sets.
Given the class imbalance (e.g. 5,357 Yellow_Leaf_Curl vs 373 Mosaic_Virus), the training
DataLoader uses a `WeightedRandomSampler` (see `src/dataset/dataset_loader.py`).

---

## Running Training After Download

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Prepare the dataset
python scripts/prepare_dataset.py --raw_dir dataset/raw --out_dir dataset

# 3. Train the custom CNN
python train.py --model cnn --epochs 30 --batch_size 32

# 4. Fine-tune EfficientNet-B0
python train.py --model efficientnet --epochs 15 --batch_size 16 --unfreeze_after 5

# 5. Evaluate on the test set
python evaluate.py --model_path models/best_model.pt
```
