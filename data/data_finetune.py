import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
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

    def __call__(self, img):
        """
        Apply normalization to SAR image.

        Args:
            img: PIL Image or torch Tensor [C, H, W] in range [0, 1] or [0, 255]

        Returns:
            Normalized tensor
        """
        if isinstance(img, Image.Image):
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
    """Radiometric augmentation: random gamma correction."""
    def __init__(self, gamma_range=(0.8, 1.2)):
        self.gamma_range = gamma_range

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            # Convert tensor to PIL for gamma adjustment
            img = T.ToPILImage()(img)

        gamma = np.random.uniform(*self.gamma_range)
        # Apply gamma correction
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.power(img_array, gamma)
        img_array = (img_array * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(img_array)


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

    # Training augmentations (radiometric only - safe without box transformation)
    train_transform = T.Compose([
        T.Resize((config.MODEL.IN_SIZE, config.MODEL.IN_SIZE)),
        # Radiometric augmentations (safe - don't affect box coordinates)
        T.ColorJitter(
            brightness=0.2,  # Random brightness adjustment
            contrast=0.2,    # Random contrast adjustment
        ),
        RandomGammaAdjustment(gamma_range=(0.8, 1.2)),
        # SAR-specific normalization
        SARNormalization(s_norm=s_norm, global_std=global_std),
    ])

    val_transform = T.Compose([
        T.Resize((config.MODEL.IN_SIZE, config.MODEL.IN_SIZE)),
        SARNormalization(s_norm=s_norm, global_std=global_std),
    ])

    # Create datasets
    train_dataset = SARDetYoloDataset(
        root_dir=config.DATA.DATA_PATH,
        split="train",
        num_classes=config.DATA.NUM_CLASS,
        transform=train_transform
    )
    val_dataset = SARDetYoloDataset(
        root_dir=config.DATA.DATA_PATH,
        split="val",
        num_classes=config.DATA.NUM_CLASS,
        transform=val_transform
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
class SARDetYoloDataset(Dataset):
    """
    Dataset for YOLO-style structure of SARDet-100k:
    dataset/
      images/{split}/image_name.jpg
      labels/{split}/image_name.txt

    Each label file contains:
        class_id x_center y_center width height
    (normalized coordinates in [0,1])
    """

    def __init__(self, root_dir, split="train", num_classes=None, transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.num_classes = num_classes  # Store before computing samples

        self.img_dir = os.path.join(root_dir, "images", split)
        self.lbl_dir = os.path.join(root_dir, "labels", split)

        # Verify directories exist
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

        self.num_classes = num_classes
        self.samples = []

        for f in self.img_files:
            img_path = os.path.join(self.img_dir, f)
            label_path = os.path.join(self.lbl_dir, os.path.splitext(f)[0] + ".txt")

            labels = []
            if os.path.exists(label_path):
                with open(label_path, "r") as lf:
                    for line in lf:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        parts = line.split()
                        if len(parts) >= 1:
                            try:
                                cls_id = int(parts[0])
                                labels.append(cls_id)
                            except ValueError:
                                print(f"Warning: Invalid class ID in {label_path}: {line}")
                                continue
            
            # Save sample representation
            self.samples.append({"image": img_path, "labels": labels})

        # Infer num_classes if not given
        if self.num_classes is None:
            all_labels = [l for s in self.samples for l in s["labels"]]
            self.num_classes = (max(all_labels) + 1) if all_labels else 1

        # Check if binary classification mode
        self.is_binary = (self.num_classes == 2)

        if self.is_binary:
            print(f"[Dataset] Binary classification mode: all objects mapped to foreground (class 1)")
        else:
            print(f"[Dataset] Multi-class mode: {self.num_classes} classes")

    def __len__(self):
        return len(self.samples)

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

        # Load image
        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        # Load YOLO boxes
        label_path = os.path.join(
            self.lbl_dir, 
            os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        )
        boxes = []
        labels = []
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    parts = line.split()
                    if len(parts) == 5:
                        try:
                            cls, x, y, w, h = map(float, parts)
                            boxes.append([x, y, w, h])  # normalized coords
                            labels.append(int(cls))
                        except ValueError:
                            print(f"Warning: Invalid box format in {label_path}: {line}")
                            continue

        # Convert to tensors for Lightning compatibility
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)

        # Apply transforms (should be applied BEFORE converting boxes if using albumentations)
        # If using torchvision transforms, apply to image only
        if self.transform is not None:
            img = self.transform(img)

        return {
            "image": img,           # Tensor ready for model
            "boxes": boxes,         # Tensor [N, 4] normalized YOLO boxes (x, y, w, h)
            "labels": labels,       # Tensor [N] class IDs
            "image_id": idx,        # Use image_id instead of idx for clarity
            "orig_size": torch.tensor([H, W], dtype=torch.int64)  # Original image size
        }


def yolo_collate_fn(batch):
    """
    Custom collate function for batching YOLO-style data.
    Since each image can have different number of boxes, we keep them as lists.
    """
    images = torch.stack([item["image"] for item in batch])
    boxes = [item["boxes"] for item in batch]
    labels = [item["labels"] for item in batch]
    image_ids = torch.tensor([item["image_id"] for item in batch])
    orig_sizes = torch.stack([item["orig_size"] for item in batch])
    
    return {
        "image": images,
        "boxes": boxes,           # List of tensors
        "labels": labels,         # List of tensors
        "image_id": image_ids,
        "orig_size": orig_sizes
    }