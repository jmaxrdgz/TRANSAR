from typing import Optional
import numpy as np
import h5py

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

    # Create dataset based on format
    data_format = config.DATA.FORMAT.lower() if hasattr(config.DATA, 'FORMAT') else 'npy'

    if data_format == 'hdf5':
        print(f"[DataLoader] Using HDF5 dataset format")
        train_set = SARChipDataset(
            hdf5_path=config.DATA.TRAIN_DATA,
            transform=transform
        )
    elif data_format == 'npy':
        print(f"[DataLoader] Using NPY dataset format")
        train_set = SARChipDatasetNPY(
            npy_dir=config.DATA.TRAIN_DATA,
            transform=transform
        )
    else:
        raise ValueError(f"Unknown data format: {data_format}. Choose 'npy' or 'hdf5'")

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
    

class SARChipDataset(Dataset):
    """
    PyTorch Dataset for SAR chips stored in HDF5 format.
    
    Features:
    - Memory-efficient: doesn't load entire dataset into RAM
    - Fast random access via HDF5
    - Supports standard PyTorch DataLoader with multiple workers
    """
    
    def __init__(
        self,
        hdf5_path: str,
        transform: Optional[callable] = None,
    ):
        """
        Args:
            hdf5_path: Path to HDF5 file containing chips
            transform: Optional transform to apply to chips
        """
        self.hdf5_path = hdf5_path
        self.transform = transform
        
        # Open file to get metadata
        with h5py.File(hdf5_path, 'r') as f:
            self.num_chips = f.attrs['num_chips']
            self.chip_size = f.attrs['chip_size']
            
            # Load chip names if needed
            if 'chip_names' in f:
                self.chip_names = [
                    name.decode('utf-8') if isinstance(name, bytes) else name
                    for name in f['chip_names'][:]
                ]
            else:
                self.chip_names = [f"chip_{i}" for i in range(self.num_chips)]
        
        # Each worker will open its own file handle
        self._file = None
        self._dataset = None
    
    def __len__(self):
        return self.num_chips
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        # Lazy open file (important for multiprocessing)
        if self._file is None:
            self._file = h5py.File(self.hdf5_path, 'r')
            self._dataset = self._file['chips']
        
        # Load chip
        chip = self._dataset[idx].astype(np.float32)

        # Ensure 3D array: [C, H, W]
        if chip.ndim == 2:
            chip = chip[np.newaxis, :]  # Add channel dimension

        # Convert to torch tensor
        chip = torch.from_numpy(chip)

        # Apply transforms
        if self.transform:
            chip = self.transform(chip)

        return chip, 0  # Dummy label for unsupervised learning
    
    def __del__(self):
        # Clean up file handle
        if self._file is not None:
            self._file.close()
    
    def get_chip_name(self, idx: int) -> str:
        """Get the name/identifier of a chip."""
        return self.chip_names[idx]


class SARChipDatasetNPY(Dataset):
    """
    Efficient dataset for loading SAR chips from individual .npy files.

    Keeps single-channel SAR data without unnecessary conversions.
    Normalization is applied via transform pipeline, not built-in.
    """

    def __init__(
        self,
        npy_dir: str,
        transform: Optional[callable] = None
    ):
        """
        Args:
            npy_dir: Directory containing .npy chip files
            transform: Optional transform pipeline (augmentation + normalization)
            normalize: If True, apply built-in normalization (deprecated - use transform pipeline)
        """
        from pathlib import Path

        self.npy_dir = Path(npy_dir)
        self.transform = transform

        # Find all .npy files
        self.chip_files = sorted(list(self.npy_dir.glob("*.npy")))

        if len(self.chip_files) == 0:
            raise ValueError(f"No .npy files found in {npy_dir}")

    def __len__(self):
        return len(self.chip_files)

    def __getitem__(self, idx: int) -> tuple:
        # Load chip
        chip = np.load(self.chip_files[idx]).astype(np.float32)

        # Ensure 3D array: [C, H, W]
        if chip.ndim == 2:
            chip = chip[np.newaxis, :, :]  # Add channel dimension

        # Convert to torch tensor
        chip = torch.from_numpy(chip)

        # Apply transform pipeline (augmentation + normalization)
        if self.transform:
            chip = self.transform(chip)

        # Return single-channel image (efficient for SAR)
        # No conversion to 3 channels - happens in model if needed
        return chip, 0  # Dummy label for unsupervised learning

    def get_chip_name(self, idx: int) -> str:
        return self.chip_files[idx].stem
