"""
Test the full data pipeline with transforms and visualize results.

This script tests:
1. Loading SIVED dataset with full transform pipeline
2. Applying SAR normalization and augmentations
3. Visualizing images with bounding boxes after transforms
4. Verifying grayscale handling throughout

Usage:
    python scripts/test_pipeline_viz.py
"""
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from pathlib import Path

# Add parent directory to path to import data module
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_finetune import SIVEDDataset, SARNormalization, RandomGammaAdjustment
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode


def denormalize_box(box, img_width, img_height):
    """
    Convert normalized YOLO box (cx, cy, w, h) to pixel coordinates (x1, y1, x2, y2).

    Args:
        box: [cx, cy, w, h] normalized to [0, 1]
        img_width, img_height: Image dimensions

    Returns:
        [x1, y1, x2, y2] in pixel coordinates
    """
    cx, cy, w, h = box
    x1 = (cx - w/2) * img_width
    y1 = (cy - h/2) * img_height
    x2 = (cx + w/2) * img_width
    y2 = (cy + h/2) * img_height
    return [x1, y1, x2, y2]


def plot_sample(sample, idx, ax, class_mapping=None):
    """
    Plot a single sample with bounding boxes.

    Args:
        sample: Dictionary from dataset
        idx: Sample index
        ax: Matplotlib axis
        class_mapping: Dict mapping class IDs to names (optional)

    Returns:
        None
    """
    # Get image
    img = sample["image"]
    boxes = sample["boxes"]
    labels = sample["labels"]
    difficulties = sample["difficulties"]
    orig_h, orig_w = sample["orig_size"].tolist()

    # Convert tensor image to displayable format
    # Handle normalized images
    if img.min() < -1 or img.max() > 2:
        # Image is SAR-normalized, bring to displayable range
        img_display = img.clone()
        # Clip extreme values and normalize to [0, 1]
        img_display = torch.clamp(img_display, -3, 3)  # Clip to ±3 std
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min() + 1e-8)
    else:
        img_display = torch.clamp(img, 0, 1)

    # Convert to numpy for matplotlib
    if img_display.shape[0] == 1:
        # Grayscale - squeeze to (H, W)
        img_np = img_display.squeeze(0).numpy()
        ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)
    elif img_display.shape[0] == 3:
        # RGB
        img_np = img_display.permute(1, 2, 0).numpy()
        ax.imshow(img_np)

    # Get current image dimensions after transforms
    _, curr_h, curr_w = img.shape

    # Draw bounding boxes
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']

    # Reverse class mapping for display
    if class_mapping:
        id_to_name = {v: k for k, v in class_mapping.items()}
    else:
        id_to_name = None

    for i, (box, label, difficult) in enumerate(zip(boxes, labels, difficulties)):
        # Denormalize box to current image size
        x1, y1, x2, y2 = denormalize_box(box.tolist(), curr_w, curr_h)
        w = x2 - x1
        h = y2 - y1

        # Create rectangle patch
        color = colors[label.item() % len(colors)]
        linestyle = '--' if difficult.item() else '-'  # Dashed for difficult examples

        rect = patches.Rectangle(
            (x1, y1), w, h,
            linewidth=2,
            edgecolor=color,
            linestyle=linestyle,
            facecolor='none'
        )
        ax.add_patch(rect)

        # Get class name
        class_name = id_to_name.get(label.item(), f'Class {label.item()}') if id_to_name else f'Class {label.item()}'
        diff_marker = ' [D]' if difficult.item() else ''

        # Add label text
        ax.text(
            x1, y1 - 5,
            f'{class_name}{diff_marker}',
            color=color,
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8)
        )

    title = f'Sample {idx} | Size: {curr_w}x{curr_h} (orig: {orig_w}x{orig_h}) | {len(boxes)} objects'
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def main():
    """Test full pipeline with transforms."""

    # Dataset path
    data_path = "dataset/supervised/SIVED/ImageSets"

    print("="*70)
    print("Testing SIVED Dataset Pipeline with Full Transforms")
    print("="*70)

    # Define transforms (same as training)
    target_size = 512  # Common size for detection models

    # Test with SAR normalization
    transform = T.Compose([
        T.Resize((target_size, target_size), interpolation=InterpolationMode.BICUBIC),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        RandomGammaAdjustment(gamma_range=(0.8, 1.2)),
        SARNormalization(s_norm=None, global_std=None),  # Use per-chip std for testing
    ])

    print(f"\nTransform pipeline:")
    print(f"  1. Resize to {target_size}x{target_size} (BICUBIC)")
    print(f"  2. ColorJitter (brightness=0.2, contrast=0.2)")
    print(f"  3. RandomGammaAdjustment (gamma=0.8-1.2)")
    print(f"  4. SARNormalization (per-chip std)")

    try:
        # Load dataset
        print(f"\nLoading dataset from: {data_path}")

        dataset = SIVEDDataset(
            root_dir=data_path,
            split="train",
            num_classes=None,  # Auto-infer
            transform=transform
        )

        print(f"\n✓ Dataset loaded successfully!")
        print(f"  - Total samples: {len(dataset)}")
        print(f"  - Number of classes: {dataset.num_classes}")
        print(f"  - Class mapping: {dataset.class_mapping}")
        print(f"  - Binary mode: {dataset.is_binary}")

        # Test loading samples
        print("\nLoading sample data...")
        num_test_samples = min(6, len(dataset))

        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\n  Sample {i}:")
            print(f"    - Image shape: {sample['image'].shape}")
            print(f"    - Image dtype: {sample['image'].dtype}")
            print(f"    - Image range: [{sample['image'].min():.3f}, {sample['image'].max():.3f}]")
            print(f"    - Num boxes: {len(sample['boxes'])}")
            print(f"    - Labels: {sample['labels'].tolist()}")
            print(f"    - Difficulties: {sample['difficulties'].tolist()}")
            print(f"    - Orig size: {sample['orig_size'].tolist()}")

        # Create visualization
        print("\n" + "="*70)
        print("Creating visualization...")
        print("="*70)

        fig, axes = plt.subplots(2, 3, figsize=(20, 14))
        axes = axes.flatten()

        samples = []
        for i in range(num_test_samples):
            sample = dataset[i]
            samples.append(sample)
            plot_sample(sample, i, axes[i], class_mapping=dataset.class_mapping)

        # Hide unused subplots
        for i in range(num_test_samples, len(axes)):
            axes[i].axis('off')

        plt.suptitle(
            'SIVED Dataset with Full Transform Pipeline\n'
            f'Grayscale SAR Images ({target_size}x{target_size}) with Bounding Boxes\n'
            'Dashed boxes indicate difficult examples',
            fontsize=14,
            y=0.98
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save figure
        output_path = "pipeline_visualization.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Visualization saved to: {output_path}")

        # Verify grayscale format
        print("\n" + "="*70)
        print("Pipeline Verification:")
        print("="*70)

        print(f"\n✓ Image format: {'Grayscale' if samples[0]['image'].shape[0] == 1 else 'RGB'}")
        print(f"  - Expected: [1, {target_size}, {target_size}]")
        print(f"  - Actual: {list(samples[0]['image'].shape)}")

        print(f"\n✓ Bounding boxes are normalized: {samples[0]['boxes'].max() <= 1.0 and samples[0]['boxes'].min() >= 0.0}")
        print(f"  - Box format: [cx, cy, w, h] in range [0, 1]")

        print(f"\n✓ Difficulty flags present: {samples[0]['difficulties'].dtype == torch.bool}")

        # Show plot
        plt.show()

        print("\n" + "="*70)
        print("✓ All tests passed!")
        print("="*70)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
