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

    # Convert labels to int64 and add 1 to shift from [0, num_classes-1] to [1, num_classes]
    # (torchvision uses 0 for background)
    labels = labels.long() + 1

    return {
        'boxes': boxes_xyxy,
        'labels': labels
    }


def detection_collate_fn(batch: List[Dict], target_size: int = 256) -> Dict:
    """
    Custom collate function for object detection.
    Converts batches from YOLO dataset format to torchvision Faster R-CNN format.

    Args:
        batch: List of dicts from SARDetYoloDataset with keys:
            - 'image': Tensor [C, H, W]
            - 'boxes': Tensor [N, 4] in YOLO format
            - 'labels': Tensor [N]
            - 'image_id': int
            - 'orig_size': Tensor [H, W]
        target_size: Target image size for resizing (from config)

    Returns:
        Dict with:
            - 'images': List of image tensors
            - 'targets': List of target dicts with 'boxes', 'labels', 'image_id'
    """
    images = []
    targets = []

    for sample in batch:
        image = sample['image']
        boxes = sample['boxes']
        labels = sample['labels']
        image_id = sample['image_id']

        # Convert PIL Image to tensor if needed
        if isinstance(image, Image.Image):
            # Resize to target size
            image = TF.resize(image, [target_size, target_size])
            image = TF.to_tensor(image)
        else:
            # Already a tensor, resize it
            image = TF.resize(image, [target_size, target_size])

        # Replicate single channel to 3 channels for Faster R-CNN (expects RGB)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        # Get image size (H, W) after resizing
        _, h, w = image.shape
        image_size = (h, w)

        # Convert to torchvision format
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
        num_classes=config.DATA.NUM_CLASSES - 1,  # SARDetYoloDataset expects num foreground classes
        transform=train_transform
    )

    val_dataset = SARDetYoloDataset(
        root_dir=config.DATA.DATA_PATH,
        split='val',
        num_classes=config.DATA.NUM_CLASSES - 1,
        transform=val_transform
    )

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
