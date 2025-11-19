"""
Example dataset implementations for classification tasks.

This module provides example dataset classes for loading SAR images
for classification finetuning. Users can extend or modify these classes
for their specific use cases.
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class SARClassificationDataset(Dataset):
    """
    SAR image classification dataset.

    Supports loading from:
    - ImageFolder structure: root/class1/img1.png, root/class2/img2.png, etc.
    - NPY files: root/class1/img1.npy, root/class2/img2.npy, etc.
    - CSV file with paths and labels

    Args:
        root: Root directory of dataset
        transform: Optional transform to apply to images
        file_format: Format of image files ('png', 'npy', 'tif', etc.)
        classes: Optional list of class names (inferred from folders if None)
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        file_format: str = 'png',
        classes: Optional[List[str]] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        self.file_format = file_format

        # Find all class folders
        if classes is None:
            class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
            self.classes = [d.name for d in class_dirs]
        else:
            self.classes = classes

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Build file list
        self.samples = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            if not class_dir.exists():
                continue

            class_idx = self.class_to_idx[class_name]

            # Find all files with specified format
            if file_format == 'npy':
                files = list(class_dir.glob('*.npy'))
            elif file_format in ['png', 'jpg', 'jpeg']:
                files = list(class_dir.glob(f'*.{file_format}'))
            elif file_format == 'tif':
                files = list(class_dir.glob('*.tif')) + list(class_dir.glob('*.tiff'))
            else:
                files = list(class_dir.glob(f'*.{file_format}'))

            for file_path in files:
                self.samples.append((str(file_path), class_idx))

        print(f"[SARClassificationDataset] Found {len(self.samples)} images")
        print(f"[SARClassificationDataset] Classes: {self.classes}")
        print(f"[SARClassificationDataset] Samples per class: {[sum(1 for _, idx in self.samples if idx == i) for i in range(len(self.classes))]}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Load and return a sample."""
        file_path, label = self.samples[idx]

        # Load image based on format
        if self.file_format == 'npy':
            # Load numpy array
            image = np.load(file_path)
            # Ensure it's a 2D array
            if image.ndim == 3:
                image = image.squeeze()
            # Convert to PIL Image for transforms
            # Normalize to 0-255 range if needed
            if image.max() > 1.0:
                image_pil = Image.fromarray(image.astype(np.uint8))
            else:
                image_pil = Image.fromarray((image * 255).astype(np.uint8))
        else:
            # Load image file
            image_pil = Image.open(file_path)
            # Convert to grayscale if needed
            if image_pil.mode != 'L':
                image_pil = image_pil.convert('L')

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image_pil)
        else:
            image = transforms.ToTensor()(image_pil)

        return {'image': image, 'label': label, 'path': file_path}


class SARClassificationDatasetCSV(Dataset):
    """
    SAR image classification dataset from CSV file.

    CSV format:
        image_path,label
        /path/to/img1.png,0
        /path/to/img2.npy,1
        ...

    Args:
        csv_file: Path to CSV file with image paths and labels
        root: Root directory for relative paths (optional)
        transform: Optional transform to apply to images
        num_classes: Number of classes (optional, inferred from data if None)
    """

    def __init__(
        self,
        csv_file: str,
        root: Optional[str] = None,
        transform: Optional[Callable] = None,
        num_classes: Optional[int] = None,
    ):
        import pandas as pd

        self.root = Path(root) if root is not None else None
        self.transform = transform

        # Load CSV
        df = pd.read_csv(csv_file)
        self.image_paths = df['image_path'].tolist()
        self.labels = df['label'].tolist()

        # Infer number of classes
        if num_classes is None:
            self.num_classes = max(self.labels) + 1
        else:
            self.num_classes = num_classes

        print(f"[SARClassificationDatasetCSV] Loaded {len(self.image_paths)} images")
        print(f"[SARClassificationDatasetCSV] Number of classes: {self.num_classes}")
        print(f"[SARClassificationDatasetCSV] Label distribution: {np.bincount(self.labels)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load and return a sample."""
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Handle relative paths
        if self.root is not None:
            image_path = self.root / image_path

        # Load image based on extension
        if image_path.endswith('.npy'):
            # Load numpy array
            image = np.load(image_path)
            if image.ndim == 3:
                image = image.squeeze()
            # Convert to PIL Image
            if image.max() > 1.0:
                image_pil = Image.fromarray(image.astype(np.uint8))
            else:
                image_pil = Image.fromarray((image * 255).astype(np.uint8))
        else:
            # Load image file
            image_pil = Image.open(image_path)
            if image_pil.mode != 'L':
                image_pil = image_pil.convert('L')

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image_pil)
        else:
            image = transforms.ToTensor()(image_pil)

        return {'image': image, 'label': label, 'path': str(image_path)}


def get_classification_transforms(
    img_size: int = 256,
    augment: bool = True,
    in_chans: int = 1,
    mean: Optional[List[float]] = None,
    std: Optional[List[float]] = None,
) -> Tuple[Callable, Callable]:
    """
    Get training and validation transforms for classification.

    Args:
        img_size: Target image size
        augment: Whether to use data augmentation for training
        in_chans: Number of input channels
        mean: Normalization mean (default: [0.5]*in_chans)
        std: Normalization std (default: [0.5]*in_chans)

    Returns:
        train_transform, val_transform
    """
    if mean is None:
        mean = [0.5] * in_chans
    if std is None:
        std = [0.5] * in_chans

    # Training transforms
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Grayscale(num_output_channels=in_chans),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(num_output_channels=in_chans),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=in_chans),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_transform


# Example usage
if __name__ == "__main__":
    # Example 1: ImageFolder structure
    print("Example 1: ImageFolder structure")
    print("-" * 60)

    # Assuming data structure:
    # data/classification/
    #   ├── class_0/
    #   │   ├── img1.png
    #   │   └── img2.png
    #   └── class_1/
    #       ├── img3.png
    #       └── img4.png

    train_transform, val_transform = get_classification_transforms(
        img_size=256,
        augment=True,
        in_chans=1
    )

    # dataset = SARClassificationDataset(
    #     root='data/classification',
    #     transform=train_transform,
    #     file_format='png'
    # )

    # print(f"Dataset size: {len(dataset)}")
    # print(f"Classes: {dataset.classes}")
    # print(f"First sample shape: {dataset[0]['image'].shape}")

    print("\nExample 2: CSV file")
    print("-" * 60)

    # Example CSV format:
    # image_path,label
    # class_0/img1.png,0
    # class_0/img2.png,0
    # class_1/img3.png,1

    # dataset_csv = SARClassificationDatasetCSV(
    #     csv_file='data/classification/train.csv',
    #     root='data/classification',
    #     transform=train_transform,
    #     num_classes=2
    # )

    # print(f"Dataset size: {len(dataset_csv)}")
    # print(f"Number of classes: {dataset_csv.num_classes}")
