# Linear Probing Finetuning

This module provides scripts for finetuning pretrained backbones on classification tasks using linear probing.

## Overview

Linear probing is a technique for evaluating and finetuning pretrained models by:
1. Loading a pretrained backbone (frozen or partially frozen)
2. Adding a single linear layer for classification
3. Training only the linear layer (or the linear layer + last N blocks)

This approach is useful for:
- Quick evaluation of pretrained representations
- Transfer learning with limited data
- Efficient finetuning on downstream classification tasks

## Files

- `linear_probing.py`: Lightning module for linear probing classifier
- `finetune_linear_probing.py`: Main training script with CLI
- `classification_dataset.py`: Example dataset implementations
- `README.md`: This file

## Quick Start

### 1. Prepare Your Dataset

The script supports ImageFolder structure by default:

```
data/classification/
├── class_0/
│   ├── img1.png
│   └── img2.png
├── class_1/
│   ├── img3.png
│   └── img4.png
└── class_2/
    ├── img5.png
    └── img6.png
```

For custom datasets, see the `classification_dataset.py` examples.

### 2. Run Linear Probing (Frozen Backbone)

Train only the linear classification layer:

```bash
python experiments/finetune_linear_probing.py \
    --backbone_path checkpoints/pretrain/backbone_final.pth \
    --dataset_path data/classification \
    --num_classes 10 \
    --num_blocks_unfreeze 0 \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 1e-3
```

### 3. Run Finetuning (Unfreeze Last N Blocks)

Finetune the last 2 blocks + linear layer:

```bash
python experiments/finetune_linear_probing.py \
    --backbone_path checkpoints/pretrain/backbone_final.pth \
    --dataset_path data/classification \
    --num_classes 10 \
    --num_blocks_unfreeze 2 \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 5e-4
```

## CLI Arguments

### Model Arguments

- `--backbone_path` (required): Path to pretrained backbone weights (.pth file)
- `--backbone_name`: Backbone architecture from timm (default: `swinv2_tiny_window8_256`)
- `--num_classes` (required): Number of output classes
- `--num_blocks_unfreeze`: Number of blocks to unfreeze from the end (default: 0)
  - `0`: Only linear layer trainable (linear probing)
  - `1`: Last block + linear layer trainable
  - `2`: Last 2 blocks + linear layer trainable
  - etc.
- `--in_chans`: Number of input channels (1 for SAR, 3 for RGB)

### Dataset Arguments

- `--dataset_path` (required): Path to dataset directory
- `--dataset_type`: Type of dataset (default: `folder`)
  - `folder`: ImageFolder structure
  - `custom`: Custom implementation (requires modification)
- `--val_split`: Validation split ratio (default: 0.2)

### Training Arguments

- `--batch_size`: Batch size (default: 32)
- `--epochs`: Maximum number of epochs (default: 100)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--weight_decay`: Weight decay for AdamW (default: 0.05)
- `--warmup_epochs`: Number of warmup epochs (default: 5)

### Data Augmentation

- `--img_size`: Input image size (default: 256)
- `--no_augmentation`: Disable training augmentations

### Training Configuration

- `--num_workers`: Data loading workers (default: 4)
- `--precision`: Training precision (default: `16-mixed`)
- `--accelerator`: Accelerator type (default: `auto`)
- `--devices`: Number of devices (default: 1)
- `--gradient_clip_val`: Gradient clipping (default: 1.0)

### Checkpointing and Logging

- `--output_dir`: Output directory (default: `checkpoints/linear_probing`)
- `--experiment_name`: Experiment name (default: auto-generated)
- `--save_top_k`: Save top k checkpoints (default: 3)
- `--early_stopping_patience`: Early stopping patience (default: 15, 0 to disable)
- `--resume_from`: Resume from checkpoint

## Examples

### Example 1: Linear Probing on 10-class SAR Dataset

```bash
python experiments/finetune_linear_probing.py \
    --backbone_path checkpoints/pretrain/swinv2_tiny_w8/backbone_final.pth \
    --dataset_path data/sar_classification_10class \
    --num_classes 10 \
    --num_blocks_unfreeze 0 \
    --batch_size 64 \
    --epochs 50 \
    --learning_rate 1e-3 \
    --img_size 256 \
    --in_chans 1
```

### Example 2: Full Finetuning (All Blocks Unfrozen)

```bash
python experiments/finetune_linear_probing.py \
    --backbone_path checkpoints/pretrain/swinv2_tiny_w8/backbone_final.pth \
    --dataset_path data/sar_classification_10class \
    --num_classes 10 \
    --num_blocks_unfreeze 4 \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 5e-5 \
    --weight_decay 0.01
```

### Example 3: Resume Training

```bash
python experiments/finetune_linear_probing.py \
    --backbone_path checkpoints/pretrain/swinv2_tiny_w8/backbone_final.pth \
    --dataset_path data/sar_classification_10class \
    --num_classes 10 \
    --num_blocks_unfreeze 2 \
    --resume_from checkpoints/linear_probing/my_experiment/last.ckpt
```

## Custom Dataset Implementation

To use a custom dataset, modify `finetune_linear_probing.py` in the `create_dataloaders` function:

```python
from classification_dataset import SARClassificationDataset, get_classification_transforms

def create_dataloaders(args):
    train_transform, val_transform = get_classification_transforms(
        img_size=args.img_size,
        augment=not args.no_augmentation,
        in_chans=args.in_chans
    )

    train_dataset = SARClassificationDataset(
        root=os.path.join(args.dataset_path, 'train'),
        transform=train_transform,
        file_format='npy'  # or 'png', 'tif', etc.
    )

    val_dataset = SARClassificationDataset(
        root=os.path.join(args.dataset_path, 'val'),
        transform=val_transform,
        file_format='npy'
    )

    # ... create dataloaders
```

## Monitoring Training

### TensorBoard

Logs are saved to `{output_dir}/{experiment_name}/`. View them with:

```bash
tensorboard --logdir checkpoints/linear_probing
```

### Metrics

The script logs the following metrics:
- `train/loss`: Training cross-entropy loss
- `train/acc`: Training accuracy
- `val/loss`: Validation cross-entropy loss
- `val/acc`: Validation accuracy
- `val/f1`: Validation F1 score (macro-averaged)

## Tips

### Learning Rate Selection

- **Linear probing** (frozen backbone): Use higher LR (1e-3 to 1e-2)
- **Partial finetuning** (few blocks unfrozen): Use medium LR (5e-4 to 1e-3)
- **Full finetuning** (all blocks unfrozen): Use lower LR (1e-5 to 1e-4)

### Number of Blocks to Unfreeze

For Swin Transformer variants:
- Swin Tiny: 4 stages total
- Swin Small: 4 stages total

Start with `--num_blocks_unfreeze 0` (linear probing), then gradually increase if needed.

### Batch Size

- Larger batches (64-128): Better for linear probing
- Smaller batches (16-32): Better for full finetuning
- Adjust based on GPU memory

### Data Augmentation

For SAR images, useful augmentations include:
- Random horizontal/vertical flips
- Random rotation
- Color jitter (brightness, contrast)

Disable augmentation with `--no_augmentation` if needed.

## Output Structure

```
checkpoints/linear_probing/
└── {experiment_name}/
    ├── version_0/
    │   ├── events.out.tfevents.xxx  # TensorBoard logs
    │   └── hparams.yaml              # Hyperparameters
    ├── epoch000-val_acc0.8500.ckpt   # Top-k checkpoints
    ├── epoch010-val_acc0.9200.ckpt
    ├── epoch025-val_acc0.9500.ckpt
    └── last.ckpt                      # Latest checkpoint
```

## Troubleshooting

### Out of Memory

- Reduce `--batch_size`
- Reduce `--img_size`
- Use `--precision 16-mixed` or `--precision bf16-mixed`

### Poor Performance

- Try unfreezing more blocks (`--num_blocks_unfreeze`)
- Increase training epochs (`--epochs`)
- Adjust learning rate (`--learning_rate`)
- Enable/disable data augmentation

### Backbone Loading Errors

- Ensure `--backbone_path` points to a valid .pth file
- Check that the backbone architecture matches (`--backbone_name`)
- Verify input channels match (`--in_chans`)

## Citation

If you use this code, please cite the TRANSAR paper:

```bibtex
@article{transar2024,
  title={TRANSAR: Transformer for SAR Ship Detection},
  author={...},
  journal={...},
  year={2024}
}
```
