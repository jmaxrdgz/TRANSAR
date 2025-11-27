import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from PIL import Image
import torch
from typing import Dict, List
import numpy as np


#------------------------------
#   DataLoaders & Transforms
#------------------------------
class SARNormalization:
    """
    SAR-specific logarithmic normalization as described in TRANSAR paper.

    From paper Section D.1 (Equation 3-4):
    x_hat = log2(x) / s_norm  (optional, for Capella data)
    x_hat_norm = (x_hat - mu_c) / sigma_g

    where mu_c is calculated per chip and sigma_g is the global standard deviation
    computed from the entire training dataset.
    """
    def __init__(self, s_norm=None, global_std=None):
        """
        Args:
            s_norm: Normalization scale constant for log normalization.
                    If None, skip log normalization (default for non-Capella data).
                    Set to 16 for Capella SAR data as used in the paper.
            global_std: Global standard deviation computed from training dataset.
                       Should be pre-computed using a script and stored in config.
                       If None, will use per-chip std (not recommended for production).
        """
        self.s_norm = s_norm
        self.global_std = global_std
        self._logged = False  # Flag to log normalization config only once

    def _log_normalization_config(self):
        """Log the normalization configuration once for debugging purposes."""
        if not self._logged:
            norm_type = []
            if self.s_norm is not None:
                norm_type.append(f"log2 scaling (s_norm={self.s_norm})")
            else:
                norm_type.append("no log scaling")

            if self.global_std is not None:
                norm_type.append(f"global std (σ={self.global_std:.4f})")
            else:
                norm_type.append("per-chip std")

            print(f"[SARNormalization] Using: {' + '.join(norm_type)}")
            print(f"[SARNormalization] Formula: x_norm = (x - μ_chip) / σ")
            self._logged = True

    def __call__(self, img):
        """
        Apply normalization to SAR image.

        Args:
            img: PIL Image (grayscale) or torch Tensor [1, H, W] in range [0, 1] or [0, 255]

        Returns:
            Normalized tensor [1, H, W]
        """
        # Log normalization config on first call
        self._log_normalization_config()

        if isinstance(img, Image.Image):
            # Convert PIL grayscale image to tensor [1, H, W]
            img = T.ToTensor()(img)

        # Convert to float if needed
        if img.dtype == torch.uint8:
            img = img.float() / 255.0

        # Optional logarithmic normalization (for Capella data)
        if self.s_norm is not None:
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            img = torch.log2(img + epsilon) / self.s_norm

        # Per-chip mean normalization
        mu_c = img.mean()

        # Use global std if available, otherwise fall back to per-chip std
        if self.global_std is not None:
            sigma = self.global_std
        else:
            epsilon = 1e-10
            sigma = img.std() + epsilon

        img_norm = (img - mu_c) / sigma

        return img_norm


class RandomGammaAdjustment:
    """Radiometric augmentation: random gamma correction for grayscale images."""
    def __init__(self, gamma_range=(0.8, 1.2)):
        self.gamma_range = gamma_range

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # Convert tensor to PIL for gamma adjustment
            img = T.ToPILImage()(img)

        gamma = np.random.uniform(*self.gamma_range)
        # Apply gamma correction to grayscale image
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.power(img_array, gamma)
        img_array = (img_array * 255.0).clip(0, 255).astype(np.uint8)
        # Return grayscale PIL image (mode 'L')
        return Image.fromarray(img_array, mode='L')


class RandomHorizontalFlipWithBoxes:
    """
    Random horizontal flip for images with YOLO-format bounding boxes.

    Applies horizontal flip to both image and boxes with probability p.
    Boxes are in normalized YOLO format: [x_center, y_center, width, height]

    For horizontal flip:
        x_center_new = 1.0 - x_center_old
        y_center, width, height remain unchanged

    Args:
        p: Probability of applying flip (default: 0.5)
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Apply flip to image and boxes.

        Args:
            sample: Dict with keys 'image' (PIL Image), 'boxes' (Tensor [N, 4])

        Returns:
            sample: Dict with flipped image and transformed boxes
        """
        if np.random.random() < self.p:
            # Flip the image
            sample['image'] = T.functional.hflip(sample['image'])

            # Transform boxes if there are any
            if len(sample['boxes']) > 0:
                boxes = sample['boxes'].clone()
                # Flip x_center: x_new = 1.0 - x_old
                boxes[:, 0] = 1.0 - boxes[:, 0]
                sample['boxes'] = boxes

        return sample


class ImageOnlyTransform:
    """
    Wrapper for transforms that only operate on images.
    Passes through boxes and other metadata unchanged.

    This allows existing image-only transforms (ColorJitter, Gamma, Normalization)
    to work in a dict-based transform pipeline.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        sample['image'] = self.transform(sample['image'])
        return sample


class ComposeWithBoxes:
    """
    Compose transforms that operate on dict samples with images and boxes.

    Compatible with both dict-based transforms (that modify boxes)
    and image-only transforms (wrapped in ImageOnlyTransform).
    """
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


def build_dataloaders(config):
    """
    Build train and validation dataloaders with augmentations from TRANSAR paper.

    From paper Section D.1:
    - Radiometric augmentations: brightness, contrast, gamma
    - SAR-specific normalization: optional logarithmic normalization for Capella data

    Note: Geometric augmentations (flips, affine transforms) from the paper are NOT included
    because they require proper bounding box transformation, which is not implemented yet.
    These would need albumentations library or custom implementation to work correctly.

    Config parameters:
    - DATA.SAR_NORM_SCALE: s_norm parameter (None by default, 16 for Capella data)
    - DATA.GLOBAL_STD: Global std computed from training dataset (pre-computed via script)
    """

    s_norm = getattr(config.DATA, 'SAR_NORM_SCALE', None)  # None by default, 16 for Capella
    global_std = getattr(config.DATA, 'GLOBAL_STD', None)  # Pre-computed from training data

    train_transform = ComposeWithBoxes([
        # Spatial transforms
        ImageOnlyTransform(T.Resize((config.MODEL.IN_SIZE, config.MODEL.IN_SIZE), interpolation=InterpolationMode.BICUBIC)),

        # Geometric augmentations (NEW! Now with proper bbox transformation)
        RandomHorizontalFlipWithBoxes(p=0.5),

        # Radiometric augmentations (safe - don't affect box coordinates)
        ImageOnlyTransform(T.ColorJitter(
            brightness=0.2,  # Random brightness adjustment
            contrast=0.2,    # Random contrast adjustment
        )),
        ImageOnlyTransform(RandomGammaAdjustment(gamma_range=(0.8, 1.2))),

        # SAR-specific normalization (applied last)
        # NOTE: remove if using MGF
        ImageOnlyTransform(SARNormalization(s_norm=s_norm, global_std=global_std)),
    ])

    val_transform = ComposeWithBoxes([
        ImageOnlyTransform(T.Resize((config.MODEL.IN_SIZE, config.MODEL.IN_SIZE), interpolation=InterpolationMode.BICUBIC)),
        ImageOnlyTransform(SARNormalization(s_norm=s_norm, global_std=global_std)),
    ])

    # Create datasets
    train_dataset = SIVEDDataset(
        root_dir=config.DATA.DATA_PATH,
        split="train",
        num_classes=config.DATA.NUM_CLASS if hasattr(config.DATA, 'NUM_CLASS') else None,
        transform=train_transform
    )
    # Share class mapping from train to val for consistency
    val_dataset = SIVEDDataset(
        root_dir=config.DATA.DATA_PATH,
        split="valid",
        num_classes=config.DATA.NUM_CLASS if hasattr(config.DATA, 'NUM_CLASS') else None,
        transform=val_transform,
        class_mapping=train_dataset.class_mapping
    )

    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=config.TRAIN.NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=yolo_collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=config.TRAIN.NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=yolo_collate_fn
    )

    return train_dataloader, val_dataloader


#-------------
#   Dataset  
#-------------
# NOTE: Fix binary vs multi-class handling in SIVEDDataset
class SIVEDDataset(Dataset):
    """
    SIVED dataset for object detection in PascalVOC format.

    Each image file is 512x512 grayscale JPEG (loaded as single-channel).

    Each label file contains:
        x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
    where:
        - x1 y1 x2 y2 x3 y3 x4 y4: absolute rotated bbox coordinates
        - class_name: string name of the class
        - difficulty: boolean flag (0 or 1) indicating if the example is difficult to predict

    Returns:
        Dictionary with:
            - image: Tensor [1, H, W] - single-channel SAR image
            - boxes: Tensor [N, 4] - normalized YOLO boxes (x_center, y_center, width, height)
            - labels: Tensor [N] - class IDs
            - difficulties: Tensor [N] - difficulty flags
            - image_id: int - sample index
            - orig_size: Tensor [2] - original image size (H, W)
    """

    def __init__(self, root_dir, split="train", num_classes=None, transform=None, class_mapping=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_classes = num_classes

        self.class_mapping = {} if class_mapping is None else dict(class_mapping)

        self.img_dir = os.path.join(root_dir, "images", split)
        self.lbl_dir = os.path.join(root_dir, "labelTxt", split)

        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.exists(self.lbl_dir):
            raise FileNotFoundError(f"Label directory not found: {self.lbl_dir}")

        self.img_files = sorted([
            f for f in os.listdir(self.img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg", ".bmp"))
        ])

        if len(self.img_files) == 0:
            raise ValueError(f"No images found in {self.img_dir}")

        self.samples = []
        next_class_id = len(self.class_mapping)

        # Build class mapping and samples list simultaneously
        for f in self.img_files:
            img_path = os.path.join(self.img_dir, f)
            label_path = os.path.join(self.lbl_dir, os.path.splitext(f)[0] + ".txt")

            labels = []

            if os.path.exists(label_path):
                with open(label_path, "r") as lf:
                    for line in lf:
                        line = line.strip()
                        if not line:
                            continue

                        parts = line.split()
                        if len(parts) < 10:
                            print(f"Warning: Invalid label format in {label_path}: {line}")
                            continue

                        class_name = parts[8]

                        # Dynamic mapping creation
                        if class_name not in self.class_mapping:
                            self.class_mapping[class_name] = next_class_id
                            next_class_id += 1

                        labels.append(self.class_mapping[class_name])

            self.samples.append({
                "image": img_path,
                "labels": labels
            })

        if class_mapping is None:
            print(f"[Dataset] Built class mapping with {len(self.class_mapping)} classes: {self.class_mapping}")

        # Infer num_classes if not provided
        if self.num_classes is None:
            self.num_classes = len(self.class_mapping) if self.class_mapping else 1

        self.is_binary = (self.num_classes == 1)

        if self.is_binary:
            print(f"[Dataset] Binary classification mode: all objects mapped to foreground (class 1)")

        else:
            print(f"[Dataset] Multi-class mode: {self.num_classes} classes")

    def __len__(self):
        return len(self.samples)

    def _rotated_to_xywhn(self, x1, y1, x2, y2, x3, y3, x4, y4, img_width, img_height):
        """
        Convert rotated bbox (4 corner points) to normalized axis-aligned bbox.

        Args:
            x1, y1, x2, y2, x3, y3, x4, y4: Absolute pixel coordinates of 4 corners
            img_width, img_height: Image dimensions for normalization

        Returns:
            Normalized [x_center, y_center, width, height] in range [0, 1]
        """
        # Get axis-aligned bounding box from rotated box
        x_coords = [x1, x2, x3, x4]
        y_coords = [y1, y2, y3, y4]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # Convert to normalized YOLO format (center_x, center_y, width, height)
        x_center = ((x_min + x_max) / 2) / img_width
        y_center = ((y_min + y_max) / 2) / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        return [x_center, y_center, width, height]

    def _map_labels_to_binary(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Map all object class IDs to foreground (class 1) for binary classification.

        Args:
            labels: Tensor of class IDs [N]

        Returns:
            Tensor with all non-zero values mapped to 1
        """
        if self.is_binary and len(labels) > 0:
            # All objects become foreground (class 1)
            return torch.ones_like(labels, dtype=torch.int64)
        return labels

    def get_sample_classes(self, idx: int) -> List[int]:
        """
        Get list of class IDs present in a sample.

        Used by AdaptiveSampler to determine sample class membership.

        Args:
            idx: Sample index

        Returns:
            List of class IDs present in this sample (empty list for background)
        """
        return self.samples[idx]["labels"]

    def get_class_distribution(self) -> Dict[int, int]:
        """
        Compute class distribution across the dataset.

        For binary classification (foreground/background):
        - Class 0: background (samples with no objects)
        - Class 1: foreground (samples with at least one object)

        Returns:
            Dictionary mapping class_id to count
        """
        class_counts = {0: 0, 1: 0}  # Background, Foreground

        for sample in self.samples:
            if len(sample["labels"]) == 0:
                class_counts[0] += 1  # Background
            else:
                class_counts[1] += 1  # Foreground

        return class_counts

    def __getitem__(self, idx):
        s = self.samples[idx]
        img_path = s["image"]

        # Load image as grayscale (SAR images are single-channel)
        img = Image.open(img_path).convert("L")
        W, H = img.size

        # Load YOLO boxes
        label_path = os.path.join(
            self.lbl_dir,
            os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )
        boxes = []
        labels = []
        difficulties = []

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    parts = line.split()
                    # Format: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
                    if len(parts) >= 10:
                        try:
                            # Parse rotated bbox coordinates (absolute pixels)
                            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
                            class_name = parts[8]
                            difficulty = int(parts[9])

                            # Map class name to ID
                            cls_id = self.class_mapping.get(class_name)
                            if cls_id is None:
                                print(f"Warning: Unknown class name '{class_name}' in {label_path}")
                                continue

                            # Convert to normalized axis-aligned bbox
                            box = self._rotated_to_xywhn(x1, y1, x2, y2, x3, y3, x4, y4, W, H)
                            boxes.append(box)
                            labels.append(cls_id)
                            difficulties.append(difficulty)
                        except (ValueError, IndexError):
                            print(f"Warning: Invalid box format in {label_path}: {line}")
                            continue

        # Convert to tensors for Lightning compatibility
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        difficulties = torch.tensor(difficulties, dtype=torch.bool) if difficulties else torch.zeros((0,), dtype=torch.bool)

        # Apply transforms to both image and boxes
        if self.transform is not None:
            sample = {
                'image': img,
                'boxes': boxes
            }
            sample = self.transform(sample)
            img = sample['image']
            boxes = sample['boxes']

        return {
            "image": img,           # Tensor [1, H, W] - single-channel SAR image
            "boxes": boxes,         # Tensor [N, 4] normalized YOLO boxes (x_center, y_center, width, height)
            "labels": labels,       # Tensor [N] class IDs
            "difficulties": difficulties,  # Tensor [N] difficulty flags (bool)
            "image_id": idx,        # Sample index
            "orig_size": torch.tensor([H, W], dtype=torch.int64)  # Original image size (H, W)
        }


def yolo_collate_fn(batch):
    """
    Custom collate function for batching YOLO-style data.

    Converts SIVED format to YOLODetector expected format:
    - Combines boxes and labels into target dicts
    - Keeps boxes in normalized [0,1] format

    Returns:
        Dictionary with:
            - images: Tensor [B, 1, H, W] - batch of single-channel SAR images
            - targets: List[Dict] - list of dicts with 'boxes' [N_i,4] and 'labels' [N_i]
            - image_id: Tensor [B] - sample indices
            - orig_size: Tensor [B, 2] - original image sizes (H, W)
    """
    images = torch.stack([item["image"] for item in batch])
    image_ids = torch.tensor([item["image_id"] for item in batch])
    orig_sizes = torch.stack([item["orig_size"] for item in batch])

    # Build targets in expected format: List[Dict]
    targets = []
    for item in batch:
        targets.append({
            'boxes': item["boxes"],    # [N, 4] normalized [x_center, y_center, w, h] in [0,1]
            'labels': item["labels"]   # [N]
        })

    return {
        "images": images,         # Tensor [B, 1, H, W]
        "targets": targets,       # List[Dict] with 'boxes' and 'labels'
        "image_id": image_ids,    # Tensor [B]
        "orig_size": orig_sizes   # Tensor [B, 2]
    }