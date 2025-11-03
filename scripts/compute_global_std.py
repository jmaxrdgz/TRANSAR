#!/usr/bin/env python3
"""
Compute global standard deviation of training dataset for SAR normalization.

The TRANSAR paper uses: x_norm = (x - μ_c) / σ_g
where:
    - μ_c is per-chip (per-image) mean
    - σ_g is global standard deviation of the training dataset

This script computes σ_g from all training images.

Usage:
    python scripts/compute_global_std.py --data_path dataset/pretrain/unlabeled
    python scripts/compute_global_std.py --data_path dataset/pretrain/unlabeled --sample_size 1000
"""

import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm


def compute_global_std(data_path, sample_size=None):
    """
    Compute global standard deviation of dataset using batched Welford's algorithm.

    This is memory efficient (processes one image at a time) and fast (vectorized).

    Args:
        data_path: Path to directory containing .npy files
        sample_size: If set, randomly sample this many images (for speed)

    Returns:
        global_std: Global standard deviation across all images
        global_mean: Global mean across all images
        n_pixels: Total number of pixels processed
    """
    data_path = Path(data_path)
    files = sorted(data_path.glob("*.npy"))

    if len(files) == 0:
        raise ValueError(f"No .npy files found in {data_path}")

    print(f"Found {len(files)} images in {data_path}")

    # Optionally sample for speed
    if sample_size is not None and sample_size < len(files):
        print(f"Sampling {sample_size} images for faster computation")
        np.random.seed(42)
        files = np.random.choice(files, size=sample_size, replace=False)

    # Batched Welford's algorithm (numerically stable + fast)
    # Process one image at a time, but vectorize operations within each image
    n_total = 0
    mean = 0.0
    M2 = 0.0

    print("Computing global statistics...")
    for file in tqdm(files, desc="Processing images"):
        img = np.load(file).astype(np.float32)

        # Flatten image to 1D array
        pixels = img.ravel()
        n_batch = len(pixels)

        # Update running statistics using batch update formulas
        # See: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        batch_mean = np.mean(pixels)
        batch_var = np.var(pixels)
        batch_M2 = batch_var * n_batch

        # Combine with running statistics
        n_new = n_total + n_batch

        if n_total == 0:
            # First batch
            mean = batch_mean
            M2 = batch_M2
        else:
            # Merge batches
            delta = batch_mean - mean
            mean = (n_total * mean + n_batch * batch_mean) / n_new
            M2 = M2 + batch_M2 + delta**2 * n_total * n_batch / n_new

        n_total = n_new

    variance = M2 / n_total if n_total > 1 else 0.0
    global_std = np.sqrt(variance)

    return global_std, mean, n_total


def compute_global_std_batch(data_path, sample_size=None):
    """
    Compute global std using batch method (faster but uses more memory).

    Args:
        data_path: Path to directory containing .npy files
        sample_size: If set, randomly sample this many images

    Returns:
        global_std: Global standard deviation
    """
    data_path = Path(data_path)
    files = sorted(data_path.glob("*.npy"))

    if len(files) == 0:
        raise ValueError(f"No .npy files found in {data_path}")

    print(f"Found {len(files)} images in {data_path}")

    # Optionally sample
    if sample_size is not None and sample_size < len(files):
        print(f"Sampling {sample_size} images for faster computation")
        np.random.seed(42)
        files = np.random.choice(files, size=sample_size, replace=False)

    # Collect all pixel values
    all_pixels = []
    print("Loading images...")
    for file in tqdm(files, desc="Processing images"):
        img = np.load(file).astype(np.float32)
        all_pixels.append(img.ravel())

    # Concatenate and compute std
    all_pixels = np.concatenate(all_pixels)
    global_std = np.std(all_pixels)
    global_mean = np.mean(all_pixels)

    return global_std, global_mean, len(all_pixels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute global standard deviation for SAR normalization")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to directory containing .npy training images"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Sample this many images for faster computation (default: use all)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["online", "batch"],
        default="online",
        help="Computation method: 'online' (memory efficient, fast) or 'batch' (fastest but loads all in RAM)"
    )

    args = parser.parse_args()

    print("="*60)
    print("Computing Global Standard Deviation for SAR Normalization")
    print("="*60)
    print()

    # Compute statistics
    if args.method == "online":
        global_std, global_mean, n_pixels = compute_global_std(
            args.data_path,
            sample_size=args.sample_size
        )
    else:
        global_std, global_mean, n_pixels = compute_global_std_batch(
            args.data_path,
            sample_size=args.sample_size
        )

    # Print results
    print()
    print("="*60)
    print("Results:")
    print("="*60)
    print(f"Number of pixels: {n_pixels:,}")
    print(f"Global mean: {global_mean:.6f}")
    print(f"Global std:  {global_std:.6f}")
    print()
    print("Add this to your config_pretrain.yaml:")
    print("-"*60)
    print(f"DATA:")
    print(f"  GLOBAL_STD: {global_std:.6f}")
    print("-"*60)
    print()
    print("This implements the normalization: x_norm = (x - μ_c) / σ_g")
    print("where μ_c is per-chip mean and σ_g is this global std")
    print("="*60)
