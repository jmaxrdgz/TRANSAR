# Block-wise Masking Implementation

## Overview

Implemented block-wise masking for SimMIM pretraining as specified in the TRANSAR paper (Section C.1). This creates structured, contiguous masked regions instead of randomly scattered masked patches.

## TRANSAR Paper Configuration

```
img_size: 512 pixels
patch_size: 4 pixels
window_size: 8 patches
mask_size: 8 patches (blocks)
mask_ratio: 0.6 (60%)
```

### Masking Hierarchy

```
512×512 image
├─ 128×128 patches (after patch_embed with patch_size=4)
└─ 16×16 blocks (grouped into mask_size=8 blocks)
   └─ Each block contains 8×8 = 64 patches
```

### Masking Statistics

- **Total patches**: 128 × 128 = 16,384 patches
- **Total blocks**: 16 × 16 = 256 blocks
- **Patches per block**: 8 × 8 = 64 patches
- **Blocks to mask** (60%): ~154 blocks
- **Patches masked** (60%): ~9,830 patches

## Implementation Details

### Block-wise Masking Algorithm

```python
def random_mask(self, B, N, device):
    """Generate block-wise mask."""
    H = W = int(N ** 0.5)  # 128 for 512×512 images

    # 1. Calculate block grid
    num_blocks_per_dim = H // mask_size  # 16
    num_blocks = num_blocks_per_dim ** 2  # 256

    # 2. Randomly select blocks to mask
    num_blocks_to_mask = int(num_blocks * mask_ratio)  # 154
    mask_indices = torch.randperm(num_blocks)[:num_blocks_to_mask]

    # 3. Create block mask [B, 16, 16]
    block_mask[mask_indices] = 1

    # 4. Expand blocks to patches [B, 128, 128]
    mask = block_mask.repeat_interleave(mask_size, dim=1)
    mask = mask.repeat_interleave(mask_size, dim=2)

    return mask.flatten()
```

### Key Differences from Random Patch Masking

| Aspect | Patch-wise (Original SimMIM) | Block-wise (TRANSAR) |
|--------|------------------------------|----------------------|
| **Masking unit** | Individual patches | 8×8 blocks of patches |
| **Spatial structure** | Scattered, random | Contiguous regions |
| **Reconstruction task** | Fill scattered holes | Reconstruct large regions |
| **Difficulty** | Easier (local context) | Harder (requires understanding) |
| **Performance** | Good baseline | Better for detection tasks |

## Why Block-wise Masking Works Better

From TRANSAR paper ablation study (Section C.1):

### Mask Size = 4 (Too Small)
- **Problem**: Patches too close together
- **Result**: Model uses local context to blur reconstruction
- **Impact**: Doesn't learn semantic understanding

### Mask Size = 8 (Optimal) ✅
- **Benefit**: Balanced difficulty
- **Result**: Forces model to understand structure
- **Impact**: Best fine-tuning detection scores

### Mask Size > 8 (Too Large)
- **Problem**: Masks too large regions
- **Result**: Ignores background intensity variations
- **Impact**: Poor reconstruction, loses detail

## Visualization

Run the visualization script to see the difference:

```bash
python scripts/visualize_masking.py
```

This generates:
- `results/masking_visualization/blockwise_masking.png` - Shows contiguous 8×8 blocks
- `results/masking_visualization/patchwise_masking.png` - Shows scattered patches

Blue grid lines show the 8×8 block boundaries.

## Configuration

### TRANSAR Paper Settings

```yaml
# config_pretrain.yaml
MODEL:
  BACKBONE: swinv2_tiny_w8    # window_size=8, patch_size=4
  MASK_SIZE: 8                # 8×8 blocks
  MASK_RATIO: 0.6             # 60% of blocks masked

DATA:
  IMG_SIZE: 512               # 128×128 patches -> 16×16 blocks
```

### Constraints

For block-wise masking to work:

1. **Window compatibility**: `(img_size / patch_size) % window_size == 0`
   - ✅ `(512 / 4) % 8 = 128 % 8 = 0`

2. **Block compatibility**: `(img_size / patch_size) % mask_size == 0`
   - ✅ `(512 / 4) % 8 = 128 % 8 = 0`

3. **Square images**: Currently assumes `H == W` (can be extended)

### Valid Configurations

For `window_size=8`, `patch_size=4`, `mask_size=8`:

| IMG_SIZE | Patches | Blocks | Valid? |
|----------|---------|--------|--------|
| 256 | 64×64 | 8×8 | ✅ |
| 384 | 96×96 | 12×12 | ✅ |
| 512 | 128×128 | 16×16 | ✅ (TRANSAR) |
| 640 | 160×160 | 20×20 | ✅ |

## Experimentation

Test different mask sizes:

```bash
# Mask size 4 (too small - blurry)
python pretrain.py --override MODEL.MASK_SIZE=4

# Mask size 8 (optimal - TRANSAR)
python pretrain.py --override MODEL.MASK_SIZE=8

# Mask size 16 (too large - loses details)
python pretrain.py --override MODEL.MASK_SIZE=16
```

Expected results (from TRANSAR paper):
- **mask_size=4**: Lower fine-tuning F1 scores
- **mask_size=8**: Best fine-tuning F1 scores ⭐
- **mask_size=16**: Lower fine-tuning F1 scores

## Files Modified

### Core Implementation
- `pretrain.py`: Added `MASK_SIZE` parameter and block-wise masking logic
- `configs/config_pretrain.yaml`: Set `IMG_SIZE=512`, `MASK_SIZE=8`

### Documentation
- `PRETRAINING_GUIDE.md`: Updated with block-wise masking details
- `BLOCKWISE_MASKING.md`: This file

### Utilities
- `scripts/visualize_masking.py`: Visualization tool for comparing masking strategies

## References

- TRANSAR paper, Section C.1: "Pretraining Configuration"
  - Quote: "the model pretrained with the mask size of 8 gives the best detection scores in the fine-tuning stage"
- SimMIM paper: Original masked image modeling approach
- MAE paper: Related work using encoder-only masking

## Summary

✅ **Implemented**: Block-wise masking with `mask_size=8`
✅ **Configuration**: 512×512 images, window_size=8, compatible with Swin v2
✅ **Performance**: Expected to match TRANSAR paper results
✅ **Flexibility**: Can easily test different mask sizes via config
✅ **Visualization**: Tool to verify masking patterns

The implementation follows TRANSAR paper specifications exactly and should produce the same pretraining quality as reported in the paper.
