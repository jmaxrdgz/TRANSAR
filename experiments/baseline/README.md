# YOLO Baseline Experiments

This directory contains baseline YOLO training scripts using Ultralytics YOLO (v11/v8) for comparison with custom backbone approaches. Default model: YOLOv11l

## Setup

1. Install Ultralytics:
```bash
pip install ultralytics
```

2. Convert SIVED dataset to YOLO format:

The SIVED dataset uses oriented bounding boxes (8 coordinates), but YOLO requires axis-aligned boxes in normalized format. Run the conversion script:

```bash
python experiments/baseline/convert_to_yolo_format.py
```

This will create `dataset/supervised/SIVED/ImageSets/labels/{train,valid,test}/` directories with YOLO-formatted labels.

The dataset structure should be:
```
dataset/supervised/SIVED/ImageSets/
├── images/           # Images
│   ├── train/
│   ├── valid/
│   └── test/
├── labelTxt/         # Original oriented bounding boxes
│   ├── train/
│   ├── valid/
│   └── test/
└── labels/           # YOLO format labels (created by conversion script)
    ├── train/
    ├── valid/
    └── test/
```

## Training

### Recommended: Train with Config File (Uses Detection Experiment Hyperparameters)

This script uses the same hyperparameters from `config_experiment.yaml` for fair comparison:

```bash
python experiments/baseline/train_yolo_with_config.py
```

This will train YOLOv11l using:
- Hyperparameters from `experiments/detection/configs/config_experiment.yaml`
- Image size: 256
- Batch size: 8
- Learning rate: 0.001 (HEAD_LR from config)
- Epochs: 50
- Minimal augmentation for SAR images
- Results saved to: `logs/detection_experiments/yolo_baseline`

### Alternative: Basic Training with Custom Parameters

Train a YOLOv11 large model with custom settings:

```bash
python experiments/baseline/train_yolo_ultralytics.py \
    --data experiments/baseline/data_config_sived.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 256
```

### Available Model Variants

**YOLOv11 (Recommended - Latest):**
- `yolo11n.pt` - Nano (fastest, smallest)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large (default)
- `yolo11x.pt` - Extra Large (best accuracy, slowest)

**YOLOv8 (Also supported):**
- `yolov8n.pt` - Nano
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large

### Example Configurations

**Quick test run:**
```bash
python experiments/baseline/train_yolo_ultralytics.py \
    --data experiments/baseline/data_config_template.yaml \
    --model yolo11n.pt \
    --epochs 10 \
    --batch 8 \
    --imgsz 256 \
    --name quick_test
```

**Full training with default YOLOv11l:**
```bash
python experiments/baseline/train_yolo_ultralytics.py \
    --data experiments/baseline/data_config_template.yaml \
    --epochs 100 \
    --batch 16 \
    --imgsz 256 \
    --lr0 0.01 \
    --patience 50 \
    --name yolo11l_baseline
```

**Training with SAR-specific settings (minimal augmentation):**
```bash
python experiments/baseline/train_yolo_ultralytics.py \
    --data experiments/baseline/data_config_template.yaml \
    --model yolo11l.pt \
    --epochs 100 \
    --batch 16 \
    --imgsz 256 \
    --hsv_h 0.0 \
    --hsv_s 0.0 \
    --hsv_v 0.2 \
    --degrees 0.0 \
    --translate 0.1 \
    --scale 0.3 \
    --flipud 0.0 \
    --fliplr 0.5 \
    --mosaic 0.5 \
    --name yolo11l_sar
```

### Resume Training

Resume from a checkpoint:
```bash
python experiments/baseline/train_yolo_ultralytics.py \
    --resume logs/baseline_yolo/train/weights/last.pt
```

## Key Parameters

### Model
- `--model`: YOLOv8 variant (n/s/m/l/x)
- `--pretrained`: Path to custom pretrained weights

### Training
- `--epochs`: Number of training epochs (default: 100)
- `--batch`: Batch size (default: 16)
- `--imgsz`: Image size (default: 256)
- `--lr0`: Initial learning rate (default: 0.01)
- `--patience`: Early stopping patience (default: 50)

### Augmentation
- `--hsv_h/s/v`: HSV augmentation ranges
- `--degrees`: Rotation range
- `--translate`: Translation range
- `--scale`: Scale range
- `--fliplr/flipud`: Flip probabilities
- `--mosaic`: Mosaic augmentation probability
- `--mixup`: Mixup augmentation probability

### System
- `--device`: GPU device(s) or 'cpu'
- `--workers`: Number of data loading workers
- `--project`: Directory for saving results
- `--name`: Experiment name
- `--seed`: Random seed for reproducibility

## Validation

After training, validate the model:
```bash
yolo detect val model=logs/baseline_yolo/train/weights/best.pt data=experiments/baseline/data_config_template.yaml
```

## Inference

Run inference on test images:
```bash
yolo detect predict model=logs/baseline_yolo/train/weights/best.pt source=path/to/images imgsz=256
```

## Results

Training results are saved to `logs/baseline_yolo/{name}/`:
- `weights/best.pt` - Best model weights
- `weights/last.pt` - Last epoch weights
- `results.png` - Training curves
- `confusion_matrix.png` - Confusion matrix
- `val_batch*.jpg` - Validation predictions

## Notes

- SAR images are typically single-channel (grayscale), but YOLOv8 expects 3-channel input. The images will be automatically converted during loading.
- For SAR images, consider reducing color-based augmentations (hsv_h, hsv_s) or setting them to 0.
- The default COCO pretrained weights work reasonably well as a starting point even for SAR imagery.
- Adjust batch size based on your GPU memory (reduce if you get OOM errors).
