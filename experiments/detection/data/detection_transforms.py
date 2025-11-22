"""
Data transforms for object detection.
Converts YOLO format to torchvision format for Faster R-CNN.
"""

import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import torch
import torchvision.transforms.functional as TF
from typing import Dict, List, Tuple
from PIL import Image
import numpy as np


def letterbox_resize(
    image: Image.Image,
    target_size: int,
    fill: int = 128
) -> Tuple[Image.Image, float, Tuple[int, int]]:
    """
    Resize image with aspect ratio preservation using letterbox (padding).
    This is the standard YOLO preprocessing approach.

    Args:
        image: PIL Image to resize
        target_size: Target size for both width and height (square output)
        fill: Fill value for padding (default: 128 for gray)

    Returns:
        Tuple of:
            - Resized and padded image (square)
            - Scale factor applied
            - Padding (pad_left, pad_top) in pixels
    """
    # Get original dimensions
    orig_width, orig_height = image.size

    # Calculate scale factor to fit image in target_size while preserving aspect ratio
    scale = min(target_size / orig_width, target_size / orig_height)

    # Calculate new dimensions after scaling
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    # Resize image with aspect ratio preserved
    resized_image = image.resize((new_width, new_height), Image.BILINEAR)

    # Calculate padding to center the image
    pad_left = (target_size - new_width) // 2
    pad_top = (target_size - new_height) // 2
    pad_right = target_size - new_width - pad_left
    pad_bottom = target_size - new_height - pad_top

    # Create new image with padding
    padded_image = Image.new(image.mode, (target_size, target_size), color=(fill, fill, fill))
    padded_image.paste(resized_image, (pad_left, pad_top))

    return padded_image, scale, (pad_left, pad_top)


def adjust_boxes_for_letterbox(
    boxes: torch.Tensor,
    scale: float,
    pad_x: int,
    pad_y: int,
    orig_size: Tuple[int, int],
    target_size: int
) -> torch.Tensor:
    """
    Adjust YOLO normalized boxes after letterbox resizing.

    YOLO boxes are in normalized format (x_center, y_center, width, height) in [0, 1].
    After letterbox:
    1. Convert to absolute coords on original image
    2. Apply scale and padding
    3. Convert back to normalized coords on letterboxed image

    Args:
        boxes: YOLO format boxes [N, 4] normalized to [0, 1]
        scale: Scale factor from letterbox resize
        pad_x: Left padding in pixels
        pad_y: Top padding in pixels
        orig_size: Original image (height, width)
        target_size: Target size after letterbox

    Returns:
        Adjusted boxes [N, 4] in YOLO format normalized to letterboxed image
    """
    if len(boxes) == 0:
        return boxes

    orig_h, orig_w = orig_size

    # Convert from normalized [0,1] to absolute pixels on original image
    boxes_abs = boxes.clone()
    boxes_abs[:, 0] = boxes[:, 0] * orig_w  # x_center
    boxes_abs[:, 1] = boxes[:, 1] * orig_h  # y_center
    boxes_abs[:, 2] = boxes[:, 2] * orig_w  # width
    boxes_abs[:, 3] = boxes[:, 3] * orig_h  # height

    # Apply scale and padding
    boxes_abs[:, 0] = boxes_abs[:, 0] * scale + pad_x  # x_center
    boxes_abs[:, 1] = boxes_abs[:, 1] * scale + pad_y  # y_center
    boxes_abs[:, 2] = boxes_abs[:, 2] * scale  # width
    boxes_abs[:, 3] = boxes_abs[:, 3] * scale  # height

    # Convert back to normalized coordinates [0, 1] on letterboxed image
    boxes_normalized = boxes_abs.clone()
    boxes_normalized[:, 0] = boxes_abs[:, 0] / target_size  # x_center
    boxes_normalized[:, 1] = boxes_abs[:, 1] / target_size  # y_center
    boxes_normalized[:, 2] = boxes_abs[:, 2] / target_size  # width
    boxes_normalized[:, 3] = boxes_abs[:, 3] / target_size  # height

    # Clamp to valid range [0, 1]
    boxes_normalized = boxes_normalized.clamp(min=0.0, max=1.0)

    return boxes_normalized


def yolo_to_torchvision_format(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    image_size: Tuple[int, int]
) -> Dict[str, torch.Tensor]:
    """
    Convert YOLO format boxes to torchvision format.

    Args:
        boxes: YOLO format [N, 4] with (x_center, y_center, width, height) normalized to [0, 1]
        labels: Class labels [N]
        image_size: (height, width) of the image

    Returns:
        Dict with 'boxes' in (x1, y1, x2, y2) absolute coordinates and 'labels'
    """
    if len(boxes) == 0:
        return {
            'boxes': torch.zeros((0, 4), dtype=torch.float32),
            'labels': torch.zeros((0,), dtype=torch.int64)
        }

    h, w = image_size

    # Convert YOLO (x_center, y_center, width, height) to (x1, y1, x2, y2)
    x_center = boxes[:, 0] * w
    y_center = boxes[:, 1] * h
    width = boxes[:, 2] * w
    height = boxes[:, 3] * h

    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2

    # Stack into [N, 4] tensor
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=1)

    # Clip to image boundaries
    boxes_xyxy[:, 0].clamp_(min=0, max=w)
    boxes_xyxy[:, 1].clamp_(min=0, max=h)
    boxes_xyxy[:, 2].clamp_(min=0, max=w)
    boxes_xyxy[:, 3].clamp_(min=0, max=h)

    # Convert labels to int64 (keep 0-indexed: no background class)
    labels = labels.long()

    return {
        'boxes': boxes_xyxy,
        'labels': labels
    }


def detection_collate_fn(batch: List[Dict], target_size: int = 256) -> Dict:
    """
    Custom collate function for object detection with letterbox resizing.
    Preserves aspect ratio using letterbox (resize + padding).

    Args:
        batch: List of dicts from SARDetYoloDataset with keys:
            - 'image': PIL Image or Tensor [C, H, W]
            - 'boxes': Tensor [N, 4] in YOLO format (normalized)
            - 'labels': Tensor [N]
            - 'image_id': int
            - 'orig_size': Tensor [H, W]
        target_size: Target image size for resizing (from config)

    Returns:
        Dict with:
            - 'images': List of image tensors [3, target_size, target_size]
            - 'targets': List of target dicts with 'boxes', 'labels', 'image_id'
    """
    images = []
    targets = []

    for sample in batch:
        image = sample['image']
        boxes = sample['boxes']
        labels = sample['labels']
        image_id = sample['image_id']
        orig_size = sample['orig_size']  # [H, W]

        # Convert tensor to PIL Image if needed for letterbox resize
        if isinstance(image, torch.Tensor):
            # Convert tensor [C, H, W] to PIL Image
            if image.shape[0] == 1:
                # Single channel - convert to grayscale PIL
                image = TF.to_pil_image(image, mode='L')
                # Convert to RGB for consistency
                image = image.convert('RGB')
            else:
                # Multi-channel
                image = TF.to_pil_image(image)
        elif isinstance(image, Image.Image):
            # Ensure RGB mode
            if image.mode != 'RGB':
                image = image.convert('RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # Apply letterbox resize (preserves aspect ratio)
        letterboxed_image, scale, (pad_x, pad_y) = letterbox_resize(
            image, target_size, fill=128
        )

        # Adjust bounding boxes for letterbox transformation
        if len(boxes) > 0:
            boxes = adjust_boxes_for_letterbox(
                boxes=boxes,
                scale=scale,
                pad_x=pad_x,
                pad_y=pad_y,
                orig_size=(orig_size[0].item(), orig_size[1].item()),  # (H, W)
                target_size=target_size
            )

        # Convert to tensor
        image = TF.to_tensor(letterboxed_image)

        # Replicate single channel to 3 channels if needed
        # (letterbox_resize already ensures RGB, so this is redundant but safe)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        # Get image size (should be target_size x target_size)
        _, h, w = image.shape
        image_size = (h, w)

        # Convert to torchvision format (YOLO normalized -> absolute xyxy)
        target = yolo_to_torchvision_format(boxes, labels, image_size)
        target['image_id'] = torch.tensor([image_id])

        images.append(image)
        targets.append(target)

    return {
        'images': images,
        'targets': targets
    }


def create_detection_dataloaders(config):
    """
    Create detection dataloaders using existing SARDetYoloDataset.

    Args:
        config: Configuration object

    Returns:
        train_dataloader, val_dataloader
    """
    from torch.utils.data import DataLoader
    from data.data_finetune import SARDetYoloDataset, SARNormalization
    from torchvision import transforms as T
    from functools import partial

    # Build transforms
    train_transforms = []
    val_transforms = []

    # SAR normalization
    if config.DATA.SAR_NORM_SCALE is not None:
        sar_norm = SARNormalization(
            s_norm=config.DATA.SAR_NORM_SCALE,
            global_std=config.DATA.GLOBAL_STD
        )
        train_transforms.append(sar_norm)
        val_transforms.append(sar_norm)

    # Radiometric augmentations (train only)
    if hasattr(config.DATA, 'AUGMENTATION') and config.DATA.AUGMENTATION.ENABLED:
        if config.DATA.AUGMENTATION.COLOR_JITTER:
            train_transforms.append(
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.0
                )
            )

        if config.DATA.AUGMENTATION.RANDOM_GAMMA:
            from data.data_finetune import RandomGammaAdjustment
            train_transforms.append(
                RandomGammaAdjustment(gamma_range=(0.8, 1.2))
            )

    # Compose transforms
    train_transform = T.Compose(train_transforms) if train_transforms else None
    val_transform = T.Compose(val_transforms) if val_transforms else None

    # Create datasets
    train_dataset = SARDetYoloDataset(
        root_dir=config.DATA.DATA_PATH,
        split='train',
        num_classes=config.DATA.NUM_CLASSES,
        transform=train_transform
    )

    val_dataset = SARDetYoloDataset(
        root_dir=config.DATA.DATA_PATH,
        split='val',
        num_classes=config.DATA.NUM_CLASSES,
        transform=val_transform
    )

    # Infer num_classes from dataset if not specified in config
    if config.DATA.NUM_CLASSES is None:
        config.DATA.NUM_CLASSES = train_dataset.num_classes
        print(f"Inferred num_classes from dataset: {config.DATA.NUM_CLASSES}")

    # Create collate function with target size from config
    collate_fn_with_size = partial(detection_collate_fn, target_size=config.MODEL.IN_SIZE)

    # Create dataloaders with detection collate function
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=config.TRAIN.NUM_WORKERS,
        collate_fn=collate_fn_with_size,
        pin_memory=True,
        persistent_workers=False  # Disabled to avoid caching issues during development
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=config.TRAIN.NUM_WORKERS,
        collate_fn=collate_fn_with_size,
        pin_memory=True,
        persistent_workers=False  # Disabled to avoid caching issues during development
    )

    return train_dataloader, val_dataloader


if __name__ == "__main__":
    # Test the transforms
    print("Testing YOLO to torchvision format conversion...")

    # Create sample YOLO boxes
    yolo_boxes = torch.tensor([
        [0.5, 0.5, 0.2, 0.3],  # Center box
        [0.25, 0.25, 0.1, 0.1],  # Top-left box
        [0.75, 0.75, 0.15, 0.2],  # Bottom-right box
    ])
    labels = torch.tensor([0, 1, 2])
    image_size = (256, 256)

    result = yolo_to_torchvision_format(yolo_boxes, labels, image_size)

    print(f"\nInput YOLO boxes (normalized):\n{yolo_boxes}")
    print(f"\nOutput torchvision boxes (absolute):\n{result['boxes']}")
    print(f"\nLabels (shifted +1 for background):\n{result['labels']}")

    # Verify conversion
    print("\nVerification:")
    for i, (yolo_box, tv_box) in enumerate(zip(yolo_boxes, result['boxes'])):
        x_c, y_c, w, h = yolo_box
        x1, y1, x2, y2 = tv_box

        # Convert back to check
        x_c_calc = ((x1 + x2) / 2) / image_size[1]
        y_c_calc = ((y1 + y2) / 2) / image_size[0]
        w_calc = (x2 - x1) / image_size[1]
        h_calc = (y2 - y1) / image_size[0]

        print(f"Box {i}: Original ({x_c:.3f}, {y_c:.3f}, {w:.3f}, {h:.3f}) "
              f"-> Recovered ({x_c_calc:.3f}, {y_c_calc:.3f}, {w_calc:.3f}, {h_calc:.3f})")
