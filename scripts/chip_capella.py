#!/usr/bin/env python3
"""
Chip large Capella SAR images into smaller patches for training.

This script:
1. Reads Capella SAR TIFF images
2. Chips them into non-overlapping patches of specified size
3. Preprocesses each chip (absolute value, log normalization, float16)
4. Saves as .npy files for efficient loading

Usage:
    python scripts/chip_capella.py /path/to/images --chip_size 512
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm


def chip_sar_images(
    input_dir: str,
    chip_size: int = 512,
    output_dir: str = None
) -> None:
    """
    Chip SAR images into non-overlapping patches.

    Args:
        input_dir: Directory containing Capella SAR .tiff files
        chip_size: Size of square chips (default: 512)
        output_dir: Output directory (default: {input_dir}/chips_{chip_size})

    Raises:
        ValueError: If input_dir doesn't exist or contains no TIFF files
        ValueError: If chip_size is not positive
    """
    # Validation
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    if chip_size <= 0:
        raise ValueError(f"chip_size must be positive, got {chip_size}")

    # Create output directory
    if output_dir is None:
        output_path = input_path / f"chips_{chip_size}"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find TIFF files
    tiff_files = [
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in ['.tiff', '.tif']
    ]

    if len(tiff_files) == 0:
        raise ValueError(f"No TIFF files found in {input_dir}")

    print(f"Found {len(tiff_files)} TIFF files in {input_dir}")
    print(f"Chip size: {chip_size}x{chip_size}")
    print(f"Output directory: {output_path}")
    print()

    # Statistics
    total_chips = 0
    skipped_images = 0

    # Process each image
    for tiff_file in tqdm(tiff_files, desc="Processing images", unit="img"):
        try:
            chips_from_image = _process_image(
                tiff_file,
                output_path,
                chip_size
            )
            total_chips += chips_from_image

        except Exception as e:
            print(f"\nWarning: Failed to process {tiff_file.name}: {e}")
            skipped_images += 1
            continue

    # Summary
    print()
    print("="*60)
    print("Chipping complete!")
    print("="*60)
    print(f"Images processed: {len(tiff_files) - skipped_images}")
    print(f"Images skipped:   {skipped_images}")
    print(f"Total chips:      {total_chips}")
    print(f"Output location:  {output_path}")
    print("="*60)


def _process_image(
    tiff_file: Path,
    output_dir: Path,
    chip_size: int
) -> int:
    """
    Process a single TIFF image and extract chips.

    Args:
        tiff_file: Path to TIFF file
        output_dir: Directory to save chips
        chip_size: Size of chips

    Returns:
        Number of chips created from this image
    """
    with rasterio.open(tiff_file) as src:
        width, height = src.width, src.height

        # Validate image is large enough
        if width < chip_size or height < chip_size:
            raise ValueError(
                f"Image too small ({width}x{height}) for chip_size {chip_size}"
            )

        base_name = tiff_file.stem

        # Generate chip coordinates (non-overlapping)
        coords = [
            (x, y)
            for y in range(0, height - chip_size + 1, chip_size)
            for x in range(0, width - chip_size + 1, chip_size)
        ]

        # Extract and save each chip
        for x, y in tqdm(
            coords,
            desc=f"  Chips from {tiff_file.name}",
            leave=False,
            unit="chip"
        ):
            window = Window(x, y, chip_size, chip_size)
            chip = src.read(1, window=window)

            # Preprocess chip
            chip = _preprocess_chip(chip)

            # Save chip
            chip_name = f"{base_name}_{x}_{y}.npy"
            chip_path = output_dir / chip_name
            np.save(chip_path, chip)

    return len(coords)


def _preprocess_chip(chip: np.ndarray) -> np.ndarray:
    """
    Preprocess SAR chip with Capella-specific normalization.

    Steps:
    1. Take absolute value (for complex SAR data)
    2. Apply logarithmic normalization: log2(x) / 16 (as per TRANSAR paper)
    3. Convert to float16 for storage efficiency

    This implements the first step of Capella normalization from TRANSAR paper.
    The second step (per-chip mean and global std normalization) is applied
    later in the data loader pipeline.

    Args:
        chip: Raw SAR chip data

    Returns:
        Preprocessed chip as float16
    """
    # Take absolute value (handles complex SAR data)
    chip = np.abs(chip)

    # Logarithmic normalization with clipping to avoid log(0)
    # Uses log base 2 as specified in TRANSAR paper (Section D.1)
    chip = np.log2(np.maximum(chip, 1e-6)) / 16

    # Convert to float16 for storage efficiency
    chip = chip.astype(np.float16)

    return chip


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Chip Capella SAR images into smaller patches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing Capella SAR TIFF images"
    )
    parser.add_argument(
        "--chip_size",
        type=int,
        default=512,
        help="Size of square chips in pixels"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: {input_dir}/chips_{chip_size})"
    )

    args = parser.parse_args()

    try:
        chip_sar_images(
            args.input_dir,
            chip_size=args.chip_size,
            output_dir=args.output_dir
        )
    except Exception as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
