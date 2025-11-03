# SimMIM Pretraining Documentation

This directory contains documentation for SimMIM pretraining with block-wise masking as implemented for the TRANSAR project.

## Quick Start

```bash
# Run pretraining with default configuration
python pretrain.py

# Run with custom settings
python pretrain.py --override MODEL.MASK_SIZE=16 TRAIN.BATCH_SIZE=8
```

## Documentation

### [PRETRAINING_GUIDE.md](./PRETRAINING_GUIDE.md)
Complete guide to using the SimMIM pretraining pipeline:
- Architecture overview
- Configuration options
- Usage examples
- Experimentation guide
- Troubleshooting

### [BLOCKWISE_MASKING.md](./BLOCKWISE_MASKING.md)
Detailed documentation on block-wise masking:
- TRANSAR paper specifications
- Implementation details
- Comparison with random patch masking
- Visualization tools
- Valid configurations

## Key Files

### Model Implementation
- `models/unsupervised/simmim.py` - SimMIM model with block-wise masking
- `models/unsupervised/backbones.py` - Swin Transformer backbone factory
- `models/unsupervised/__init__.py` - Module exports

### Training Scripts
- `pretrain.py` - Main pretraining script
- `configs/config_pretrain.yaml` - Configuration file

### Data Pipeline
- `data/data_pretrain.py` - Data loading and augmentation
- `scripts/compute_global_std.py` - Compute SAR normalization statistics
- `scripts/chip_capella.py` - Preprocess Capella SAR data

### Utilities
- `scripts/visualize_masking.py` - Visualize block-wise vs patch-wise masking
- `scripts/pretrain.sh` - Simple training script

## TRANSAR Paper Configuration

The implementation uses the following configuration from the TRANSAR paper:

```yaml
MODEL:
  BACKBONE: swinv2_tiny_w8    # Swin v2 Tiny, window_size=8
  MASK_SIZE: 8                # 8×8 block-wise masking
  MASK_RATIO: 0.6             # 60% of blocks masked

DATA:
  IMG_SIZE: 512               # 512×512 images (128×128 patches)

TRAIN:
  BATCH_SIZE: 16              # Batch size
  EPOCHS: 100                 # Training epochs
```

## Architecture

```
Input: 512×512 SAR images
  ↓
Patch Embedding (patch_size=4)
  ↓
128×128 patches
  ↓
Block-wise Masking (8×8 blocks, 60% masked)
  ↓
Swin Transformer Encoder (with mask tokens)
  ↓
Linear Decoder
  ↓
Reconstruct masked patches
  ↓
L1 Loss (on masked patches only)
```

## Directory Structure

```
.
├── docs/pretraining/           # Documentation (you are here)
│   ├── README.md
│   ├── PRETRAINING_GUIDE.md
│   └── BLOCKWISE_MASKING.md
├── models/unsupervised/        # Model implementations
│   ├── __init__.py
│   ├── simmim.py
│   └── backbones.py
├── data/
│   └── data_pretrain.py        # Data loading
├── configs/
│   └── config_pretrain.yaml    # Configuration
├── scripts/
│   ├── pretrain.sh
│   ├── compute_global_std.py
│   ├── chip_capella.py
│   └── visualize_masking.py
└── pretrain.py                 # Main training script
```

## Key Features

### ✅ Block-wise Masking
- Mask contiguous 8×8 blocks instead of random patches
- TRANSAR found this produces best detection scores
- Configurable mask_size (4, 8, 16)

### ✅ SAR-Specific Normalization
- Per-chip mean centering
- Global standard deviation normalization
- Optional Capella log2 normalization

### ✅ Flexible Architecture
- Multiple Swin v1/v2 backbones
- Variable input sizes (256, 384, 512, etc.)
- Easy experimentation via config overrides

### ✅ Production Ready
- Lightning-based training
- Multi-GPU support
- Automatic checkpointing
- Progress monitoring

## Getting Started

1. **Prepare your data**:
   ```bash
   # For Capella SAR data
   python scripts/chip_capella.py /path/to/capella/data --chip_size 512

   # Compute global std for normalization
   python scripts/compute_global_std.py --data_path dataset/pretrain/unlabeled
   ```

2. **Update configuration**:
   ```yaml
   # configs/config_pretrain.yaml
   DATA:
     TRAIN_DATA: dataset/pretrain/unlabeled
     GLOBAL_STD: 1.234  # Value from compute_global_std.py
   ```

3. **Start training**:
   ```bash
   python pretrain.py
   ```

4. **Use pretrained weights in finetuning**:
   ```yaml
   # configs/config_finetune.yaml
   MODEL:
     BACKBONE:
       WEIGHTS: checkpoints/pretrain/swinv2_tiny_w8/backbone_final.pth
   ```

## References

- **SimMIM Paper**: "SimMIM: A Simple Framework for Masked Image Modeling" (Xie et al., 2021)
- **TRANSAR Paper**: Uses SimMIM with block-wise masking (mask_size=8)
- **Swin Transformer v2**: "Swin Transformer V2: Scaling Up Capacity and Resolution" (Liu et al., 2022)

## Support

For issues or questions:
1. Check the documentation files in this directory
2. Review the code comments in `models/unsupervised/`
3. Visualize masking patterns with `scripts/visualize_masking.py`
4. See configuration examples in `configs/config_pretrain.yaml`
