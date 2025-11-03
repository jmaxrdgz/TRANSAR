from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class NormalizeSAR:
    """
    SAR Normalization from TRANSAR paper (Section D.1, Equations 3-4).

    Implements the full normalization pipeline:
        1. (Optional) Logarithmic normalization: x_hat = log2(x) / s_norm
        2. Per-chip mean centering and global std normalization: x_norm = (x_hat - μ_c) / σ_g

    where:
        - s_norm is a normalization scale constant (optional, for Capella data)
        - μ_c is the mean calculated per chip (per image)
        - σ_g is the global standard deviation of the training dataset

    This normalization should be applied AFTER all spatial augmentations
    (crop, flip, etc.) so that statistics are computed on the augmented image.

    NOTE: If using Capella data, the log2 normalization should ONLY be applied here,
    not during chip preprocessing, to avoid applying it twice.

    Args:
        global_std: Global standard deviation σ_g computed from training set.
                    Use scripts/compute_global_std.py to calculate this value.
        s_norm: Optional normalization scale constant for logarithmic normalization.
                Set to 16 for Capella SAR data as used in the paper.
                Set to None to skip log normalization (default for most SAR data).
    """
    def __init__(self, global_std, s_norm=None):
        self.global_std = global_std
        self.s_norm = s_norm

    def __call__(self, img):
        """
        Apply SAR normalization pipeline.

        Args:
            img: Tensor of shape [C, H, W]

        Returns:
            normalized: Tensor of shape [C, H, W]
        """
        # Optional logarithmic normalization (for Capella data)
        if self.s_norm is not None:
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            img = torch.log2(img + epsilon) / self.s_norm

        # Subtract per-chip mean (μ_c)
        img = img - img.mean()

        # Divide by global standard deviation (σ_g)
        img = img / self.global_std

        return img


def build_loader(config):
    """
    Build DataLoader for pretraining.

    Args:
        config: Configuration object from build_config()

    Returns:
        train_loader: DataLoader for training
    """
    # Build transform pipeline
    transform_list = [
        # Random resized crop for scale invariance
        transforms.RandomResizedCrop(
            config.DATA.IMG_SIZE,
            scale=(0.2, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),

        # Ensure correct size (in case crop didn't produce exact size)
        transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)),

        # Augmentations (spatial + photometric)
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.5),
    ]

    # Add SAR normalization at the END of pipeline (after augmentations)
    # This ensures per-chip mean is computed on augmented image
    if hasattr(config.DATA, 'GLOBAL_STD') and config.DATA.GLOBAL_STD is not None:
        s_norm = config.DATA.S_NORM if hasattr(config.DATA, 'S_NORM') else None
        if s_norm is not None:
            print(f"[DataLoader] Using SAR normalization with global_std={config.DATA.GLOBAL_STD}, s_norm={s_norm} (Capella log2 normalization)")
        else:
            print(f"[DataLoader] Using SAR normalization with global_std={config.DATA.GLOBAL_STD}")
        transform_list.append(NormalizeSAR(global_std=config.DATA.GLOBAL_STD, s_norm=s_norm))
    else:
        print("[DataLoader] WARNING: GLOBAL_STD not set, skipping SAR normalization")
        print("[DataLoader] Run scripts/compute_global_std.py to compute it from your training data")

    transform = transforms.Compose(transform_list)

    # Create dataset
    train_set = NpyDataset(
        folder=config.DATA.TRAIN_DATA,
        transform=transform
    )

    print(f"[DataLoader] Loaded {len(train_set)} images from {config.DATA.TRAIN_DATA}")
    print(f"[DataLoader] Image size: {config.DATA.IMG_SIZE}")
    print(f"[DataLoader] Batch size: {config.TRAIN.BATCH_SIZE}")

    # Create DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=config.TRAIN.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        persistent_workers=True if config.DATA.NUM_WORKERS > 0 else False,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch for stable training
    )

    return train_loader


class NpyDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.files = sorted(Path(folder).glob("*.npy"))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        arr = np.load(self.files[idx])
        if arr.ndim == 2:
            arr = arr[np.newaxis, :, :]

        img = torch.from_numpy(arr).float()

        if self.transform:
            img = self.transform(img)

        assert img.shape[0] in [1, 3], "Image must have 1 or 3 channels"
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)

        return img, 0