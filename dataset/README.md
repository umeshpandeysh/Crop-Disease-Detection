# Dataset Setup

## Recommended Dataset

This project works with plant disease image datasets organized in a directory-per-class format.

### Recommended: PlantVillage Dataset
- Source: [Kaggle — PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Classes: 38 plant/disease combinations
- Images: ~54,000 leaf images

### Directory Structure

After downloading, organize as:

```
dataset/
├── train/
│   ├── Apple___Apple_scab/
│   │   ├── image001.jpg
│   │   └── ...
│   ├── Apple___healthy/
│   ├── Tomato___Early_blight/
│   └── ...
└── val/
    ├── Apple___Apple_scab/
    ├── Apple___healthy/
    └── ...
```

### Train/Val Split

Split ratio: **80% train / 20% validation**

Use the following command to create the split automatically:

```bash
python -c "
import os, shutil, random
from pathlib import Path

# Set your paths
SOURCE = 'dataset/raw_plantvillage'
TRAIN = 'dataset/train'
VAL = 'dataset/val'
SPLIT = 0.8

for cls in os.listdir(SOURCE):
    images = list(Path(f'{SOURCE}/{cls}').glob('*.jpg'))
    random.shuffle(images)
    split_idx = int(len(images) * SPLIT)
    for img in images[:split_idx]:
        dest = Path(TRAIN) / cls
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(img, dest)
    for img in images[split_idx:]:
        dest = Path(VAL) / cls
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copy(img, dest)
print('Split complete.')
"
```
