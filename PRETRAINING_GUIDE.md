# SimMIM Pretraining Guide

## Overview

This implementation provides a complete SimMIM (Simple Framework for Masked Image Modeling) pretraining pipeline for Swin Transformer backbones with support for:

- **Multiple backbones**: Swin v1 and Swin v2 (Tiny/Small variants)
- **Flexible input sizes**: Easy experimentation with different resolutions
- **SAR image support**: 1-channel input handling
- **Config-based experiments**: CLI overrides for quick testing

## Architecture

### Key Components

1. **Learnable Mask Token**: Replaces masked patches (critical difference from MAE)
2. **Full Sequence Encoding**: All patches pass through encoder (masked ones use mask token)
3. **Simple Linear Decoder**: Minimal decoder (just linear projection)
4. **L1 Loss**: Better than MSE for pixel reconstruction
5. **Masked-only Loss**: Only compute loss on masked patches

### Backbone Options

```
Swin v1 (Fixed resolution):
- swin_tiny         (224x224, window=7)
- swin_small        (224x224, window=7)

Swin v2 (Variable resolution, continuous position bias):
- swinv2_tiny_w8    (256x256, window=8)  ← Recommended
- swinv2_tiny_w16   (256x256, window=16)
- swinv2_small_w8   (256x256, window=8)
- swinv2_small_w16  (256x256, window=16)
```

**Recommendation**: Use `swinv2_tiny_w8` for best compatibility with your finetuning pipeline.

## Usage

### Basic Usage

```bash
# Run with default config
python pretrain.py

# Or use the script
bash scripts/pretrain.sh
```

### Configuration

Edit `configs/config_pretrain.yaml`:

```yaml
MODEL:
  BACKBONE: swinv2_tiny_w8  # Choose backbone
  IN_CHANS: 1               # 1 for SAR, 3 for RGB
  MASK_RATIO: 0.6           # 40-75% typical

DATA:
  IMG_SIZE: 256             # Must be compatible with window size
  TRAIN_DATA: dataset/pretrain/unlabeled

TRAIN:
  BATCH_SIZE: 128           # Adjust for GPU memory
  EPOCHS: 300
  LR: 0.001
  N_GPU: 2
```

### CLI Overrides

```bash
# Test different backbone
python pretrain.py --override MODEL.BACKBONE=swinv2_tiny_w16

# Test different input size
python pretrain.py --override DATA.IMG_SIZE=192 TRAIN.BATCH_SIZE=256

# Test different mask ratio
python pretrain.py --override MODEL.MASK_RATIO=0.75

# Multiple overrides
python pretrain.py --override \
    MODEL.BACKBONE=swinv2_small_w8 \
    DATA.IMG_SIZE=384 \
    TRAIN.BATCH_SIZE=64 \
    TRAIN.EPOCHS=100
```

## Input Size Constraints

Input size must satisfy:
1. Divisible by `patch_size` (always 4)
2. `(img_size / patch_size)` divisible by `window_size`

**Valid sizes by window:**
- `window=7` (Swin v1): 224
- `window=8` (Swin v2): 128, 192, 256, 320, 384, 512
- `window=16` (Swin v2): 256, 320, 384, 512

## Checkpoints

Checkpoints are saved to:
```
checkpoints/pretrain/{backbone}/
├── simmim-epoch=XXX-train_loss=Y.YYYY.ckpt  # Full checkpoints
├── last.ckpt                                  # Last checkpoint
└── backbone/
    ├── backbone-epoch=XXX.ckpt               # Backbone only
    └── backbone_final.pth                    # Final backbone weights
```

Use `backbone_final.pth` for finetuning.

## Transfer to Finetuning

Load pretrained backbone in `finetune.py`:

```python
# In finetune.py, around line 30-35
if config.MODEL.BACKBONE.WEIGHTS is not None:
    backbone = timm.create_model(
        "swinv2_tiny_window8_256",
        pretrained=False,
        features_only=True
    )
    # Load pretrained weights
    backbone.load_state_dict(
        torch.load(config.MODEL.BACKBONE.WEIGHTS, map_location="cpu")
    )
else:
    # Use ImageNet pretrained
    backbone = timm.create_model(
        "swinv2_tiny_window8_256",
        pretrained=True,
        features_only=True
    )
```

Then in `config_finetune.yaml`:
```yaml
MODEL:
  BACKBONE:
    WEIGHTS: checkpoints/pretrain/swinv2_tiny_w8/backbone_final.pth
```

## Experimentation Guide

### 1. Backbone Comparison (Swin v1 vs v2)

```bash
# Swin v1
python pretrain.py --override MODEL.BACKBONE=swin_tiny DATA.IMG_SIZE=224

# Swin v2 with window=8
python pretrain.py --override MODEL.BACKBONE=swinv2_tiny_w8 DATA.IMG_SIZE=256

# Swin v2 with window=16
python pretrain.py --override MODEL.BACKBONE=swinv2_tiny_w16 DATA.IMG_SIZE=256
```

**Hypothesis**: Swin v2 should generalize better to different input sizes during finetuning.

### 2. Input Size Ablation

```bash
# Small (faster, less memory)
python pretrain.py --override DATA.IMG_SIZE=192 TRAIN.BATCH_SIZE=256

# Medium (balanced)
python pretrain.py --override DATA.IMG_SIZE=256 TRAIN.BATCH_SIZE=128

# Large (more detail)
python pretrain.py --override DATA.IMG_SIZE=384 TRAIN.BATCH_SIZE=64
```

**Hypothesis**: Larger sizes learn better representations but cost more compute.

### 3. Mask Ratio Ablation

```bash
# Low masking (easier task)
python pretrain.py --override MODEL.MASK_RATIO=0.4

# Medium masking (SimMIM default)
python pretrain.py --override MODEL.MASK_RATIO=0.6

# High masking (harder task, forces better representations)
python pretrain.py --override MODEL.MASK_RATIO=0.75
```

**Hypothesis**: Higher mask ratios force the model to learn better semantic representations.

## Monitoring

Training metrics are logged to:
- Console progress bar
- TensorBoard: `logs/pretrain/{backbone}/`

View with:
```bash
tensorboard --logdir logs/pretrain/
```

Key metrics:
- `train_loss`: Reconstruction loss (L1) on masked patches
- `mask_ratio`: Actual proportion of masked patches (should match config)

## Troubleshooting

### OOM (Out of Memory)
- Reduce `TRAIN.BATCH_SIZE` (128 → 64 → 32)
- Reduce `DATA.IMG_SIZE` (256 → 192)
- Use smaller backbone (`swin_tiny` instead of `swin_small`)

### Invalid Input Size
Error: "not divisible by patch size" or dimension mismatch

**Solution**: Use valid sizes for your window size (see "Input Size Constraints")

### Slow Training
- Increase `DATA.NUM_WORKERS` (default: 8)
- Ensure `persistent_workers=True` in DataLoader
- Use larger batch size if memory allows
- Check data loading isn't bottleneck (should be <10% of step time)

## Expected Results

**Training time** (approximate, 2x V100):
- 100 epochs, 256x256, batch=128: ~12 hours (depends on dataset size)
- 300 epochs: ~36 hours

**Loss curves**:
- Initial loss: ~0.15-0.25 (L1 on normalized pixels)
- Converged loss: ~0.05-0.10 (depends on mask ratio)
- Should decrease smoothly with warmup, then cosine decay

## Implementation Details

### SimMIM vs MAE Differences

| Aspect | SimMIM | MAE |
|--------|--------|-----|
| Masked patches | Replaced with learnable token | Removed from sequence |
| Encoder input | Full sequence (N patches) | Visible patches only (~25% of N) |
| Decoder | Simple linear layer | Deep transformer decoder |
| Loss function | L1 loss | MSE loss |
| Mask ratio | 60% (moderate) | 75% (aggressive) |

### Why SimMIM for SAR?

1. **Simpler architecture**: Easier to debug and adapt
2. **Less aggressive masking**: Better for textures/patterns in SAR
3. **Faster training**: Minimal decoder overhead
4. **Proven for detection**: Good transfer to downstream tasks

## Next Steps

1. **Run pretraining**: Start with default config to verify setup
2. **Monitor convergence**: Check loss decreases smoothly
3. **Test on finetuning**: Compare vs ImageNet pretrained
4. **Ablation studies**: Experiment with different configs
5. **Scale up**: Once validated, train for full 300+ epochs

## References

- SimMIM paper: "SimMIM: A Simple Framework for Masked Image Modeling" (Xie et al., 2021)
- Swin Transformer v2: "Swin Transformer V2: Scaling Up Capacity and Resolution" (Liu et al., 2022)
