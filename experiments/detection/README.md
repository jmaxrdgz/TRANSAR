# Object Detection Finetuning Experiments

This directory contains the object detection finetuning pipeline for testing detection capabilities of different backbone architectures using Faster R-CNN.

## Overview

The detection pipeline uses:
- **Faster R-CNN** with Feature Pyramid Network (FPN) for multi-scale detection
- **Custom backbones** from timm (Swin Transformer, ResNet, ConvNeXt, etc.)
- **Progressive unfreezing** to fine-tune pretrained backbones
- **F1 score and mAP metrics** for comprehensive evaluation

## Quick Start

### Basic Training

```bash
python experiments/detection/finetune_detection.py \
    --backbone swinv2_tiny_window8_256 \
    --epochs 50
```

### Training with Pretrained Weights

```bash
python experiments/detection/finetune_detection.py \
    --backbone swinv2_tiny_window8_256 \
    --pretrained_weights path/to/pretrained/backbone.pth \
    --num_blocks_to_unfreeze 2 \
    --epochs 50
```

### Testing Different Backbones

```bash
# Swin Transformer V2
python experiments/detection/finetune_detection.py --backbone swinv2_tiny_window8_256

# ResNet-50
python experiments/detection/finetune_detection.py --backbone resnet50

# ConvNeXt
python experiments/detection/finetune_detection.py --backbone convnext_tiny

# EfficientNet
python experiments/detection/finetune_detection.py --backbone efficientnet_b3
```

## Command-Line Arguments

### Model Configuration
- `--backbone`: Backbone architecture (default: from config)
  - Examples: `swinv2_tiny_window8_256`, `resnet50`, `convnext_tiny`
- `--pretrained_weights`: Path to custom pretrained backbone weights
- `--num_blocks_to_unfreeze`: Number of backbone blocks to unfreeze (0=freeze all)

### Data Configuration
- `--data_path`: Path to dataset directory
- `--batch_size`: Training batch size (default: 4)
- `--num_workers`: Number of data loading workers

### Training Configuration
- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 0.0001)
- `--gpus`: Number of GPUs to use
- `--seed`: Random seed for reproducibility

### Experiment Tracking
- `--experiment_name`: Name for the experiment
- `--version`: Version identifier
- `--save_top_k`: Number of best checkpoints to save (default: 3)

### Utilities
- `--resume`: Path to checkpoint to resume training
- `--fast_dev_run`: Quick test run (1 batch per epoch)

## Configuration File

The pipeline uses `experiments/detection/configs/config_experiment.yaml`. Key sections:

```yaml
MODEL:
  BACKBONE:
    NAME: swinv2_tiny_window8_256
    PRETRAINED: true
    WEIGHTS: null  # Path to custom weights

  NUM_BLOCKS_TO_UNFREEZE: 0  # Progressive unfreezing

  RPN:  # Region Proposal Network
    ANCHOR_SIZES: [32, 64, 128, 256, 512]
    ASPECT_RATIOS: [0.5, 1.0, 2.0]

DATA:
  DATA_PATH: dataset/supervised/synthetic_yolo
  NUM_CLASSES: 3  # Including background class

TRAIN:
  BATCH_SIZE: 4
  LR: 0.0001
  EPOCHS: 50
```

## Dataset Format

The pipeline expects YOLO format datasets:

```
dataset/
  images/
    train/
      image_001.png
      image_002.png
    val/
      image_003.png
  labels/
    train/
      image_001.txt  # Format: class_id x_center y_center width height (normalized)
      image_002.txt
    val/
      image_003.txt
```

## Evaluation Metrics

The pipeline computes:
- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **mAP@0.75**: Mean Average Precision at IoU=0.75
- **mAP@0.5:0.95**: COCO-style mAP (averaged over IoU 0.5 to 0.95)
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)

## Progressive Unfreezing Strategy

Control which parts of the backbone are trainable:

```bash
# Freeze entire backbone (train only FPN + detection heads)
python experiments/finetune_detection.py --num_blocks_to_unfreeze 0

# Unfreeze last 1 block
python experiments/finetune_detection.py --num_blocks_to_unfreeze 1

# Unfreeze last 2 blocks
python experiments/finetune_detection.py --num_blocks_to_unfreeze 2

# Fully fine-tune (all blocks)
python experiments/finetune_detection.py --num_blocks_to_unfreeze 4
```

## Output Structure

```
logs/detection_experiments/
  faster_rcnn_detection/
    version_0/
      checkpoints/
        epoch=XX-mAP=0.XXXX.ckpt  # Best checkpoints
        last.ckpt                  # Last checkpoint
      events.out.tfevents.*        # TensorBoard logs
      hparams.yaml                 # Hyperparameters
```

## Monitoring Training

### TensorBoard

```bash
tensorboard --logdir logs/detection_experiments
```

View:
- Training losses (classifier, box regression, RPN)
- Validation metrics (mAP, F1, precision, recall)
- Learning rate schedule

## Code Structure

```
experiments/detection/
  ├── models/
  │   ├── backbone_adapter.py       # Adapts timm backbones to torchvision
  │   ├── faster_rcnn_wrapper.py    # Lightning module for Faster R-CNN
  │   └── __init__.py
  ├── data/
  │   ├── detection_transforms.py   # YOLO → torchvision format conversion
  │   └── __init__.py
  ├── configs/
  │   └── config_experiment.yaml    # Experiment configuration
  ├── finetune_detection.py         # Main training script
  └── README.md                     # This file
```

All detection experiment code is self-contained in the `experiments/detection/` directory, separate from the main TRANSAR codebase.

## Example Workflows

### 1. Baseline with Frozen Backbone

```bash
python experiments/detection/finetune_detection.py \
    --experiment_name baseline_frozen \
    --num_blocks_to_unfreeze 0 \
    --epochs 50
```

### 2. Fine-tune with Pretrained Weights

```bash
python experiments/detection/finetune_detection.py \
    --experiment_name pretrained_swin \
    --pretrained_weights checkpoints/pretrain/backbone.pth \
    --num_blocks_to_unfreeze 2 \
    --epochs 100
```

### 3. Compare Multiple Backbones

```bash
# Swin V2
python experiments/detection/finetune_detection.py \
    --experiment_name compare_swin \
    --backbone swinv2_tiny_window8_256

# ResNet-50
python experiments/detection/finetune_detection.py \
    --experiment_name compare_resnet \
    --backbone resnet50

# ConvNeXt
python experiments/detection/finetune_detection.py \
    --experiment_name compare_convnext \
    --backbone convnext_tiny
```

### 4. Resume Training

```bash
python experiments/detection/finetune_detection.py \
    --resume logs/detection_experiments/faster_rcnn_detection/version_0/checkpoints/last.ckpt
```

## Tips and Best Practices

1. **Batch Size**: Start with batch_size=4 for Faster R-CNN (memory intensive)
2. **Learning Rate**: Use 0.0001 for frozen backbone, lower (0.00001) when fine-tuning
3. **Progressive Unfreezing**: Start with frozen backbone, then gradually unfreeze blocks
4. **Gradient Clipping**: Enabled by default (clip_val=1.0) for stability
5. **Mixed Precision**: Automatically enabled on CUDA for faster training

## Troubleshooting

### Out of Memory
- Reduce `--batch_size` (try 2 or 1)
- Reduce image size in config
- Use gradient accumulation

### Poor Performance
- Check if labels match expected format
- Verify NUM_CLASSES includes background
- Increase training epochs
- Try unfreezing more backbone blocks

### No Detections
- Lower BOX_SCORE_THRESH in config
- Check anchor sizes match your object scales
- Verify ground truth boxes are correct
