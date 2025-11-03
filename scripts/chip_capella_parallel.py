#!/usr/bin/env python3
"""
Fast parallel chipping of Capella SAR images with optimized I/O.

Key improvements over sequential version:
- Parallel processing across all images
- HDF5 output for efficient storage and loading
- Memory-efficient streaming
- Progress tracking across workers

Usage:
    python chip_capella_parallel.py /path/to/folder1 /path/to/folder2 ... \
        --chip_size 512 --output chips.h5 --workers 16
"""

import os
import argparse
from pathlib import Path
from typing import List, Tuple
import multiprocessing as mp
from functools import partial

import numpy as np
import rasterio
from rasterio.windows import Window
import h5py
from tqdm import tqdm


def process_single_image(
    tiff_path: Path,
    chip_size: int,
    temp_dir: Path
) -> Tuple[str, int, List[str]]:
    """
    Process a single image and save chips to temporary location.
    
    Returns:
        (image_name, num_chips, list of temp file paths)
    """
    try:
        with rasterio.open(tiff_path) as src:
            width, height = src.width, src.height
            
            if width < chip_size or height < chip_size:
                return (tiff_path.stem, 0, [])
            
            base_name = tiff_path.stem
            temp_files = []
            
            # Generate non-overlapping chip coordinates
            chip_count = 0
            for y in range(0, height - chip_size + 1, chip_size):
                for x in range(0, width - chip_size + 1, chip_size):
                    window = Window(x, y, chip_size, chip_size)
                    chip = src.read(1, window=window)
                    
                    # Preprocess
                    chip = _preprocess_chip(chip)
                    
                    # Save to temp file
                    temp_file = temp_dir / f"{base_name}_{x}_{y}.npy"
                    np.save(temp_file, chip)
                    temp_files.append(str(temp_file))
                    chip_count += 1
            
            return (base_name, chip_count, temp_files)
            
    except Exception as e:
        print(f"Error processing {tiff_path.name}: {e}")
        return (tiff_path.stem, 0, [])


def _preprocess_chip(chip: np.ndarray) -> np.ndarray:
    """
    Preprocess SAR chip with Capella-specific normalization.
    
    Steps:
    1. Take absolute value (for complex SAR data)
    2. Apply logarithmic normalization: log2(x) / 16
    3. Convert to float16 for storage efficiency
    """
    chip = np.abs(chip)
    chip = np.log2(np.maximum(chip, 1e-6)) / 16
    chip = chip.astype(np.float16)
    return chip


def consolidate_to_hdf5(
    temp_files: List[str],
    output_path: Path,
    chip_size: int,
    compression: str = 'gzip'
):
    """
    Consolidate all temporary .npy files into a single HDF5 file.
    """
    print(f"\nConsolidating {len(temp_files)} chips into HDF5...")
    
    with h5py.File(output_path, 'w') as f:
        # Create dataset with appropriate shape
        dset = f.create_dataset(
            'chips',
            shape=(len(temp_files), chip_size, chip_size),
            dtype='float16',
            compression=compression,
            chunks=(1, chip_size, chip_size)
        )
        
        # Store chip names for reference
        chip_names = []
        
        # Load and write each chip
        for idx, temp_file in enumerate(tqdm(temp_files, desc="Writing to HDF5")):
            chip = np.load(temp_file)
            dset[idx] = chip
            
            # Extract name from path
            chip_name = Path(temp_file).stem
            chip_names.append(chip_name)
            
            # Clean up temp file
            os.remove(temp_file)
        
        # Store metadata
        f.create_dataset(
            'chip_names',
            data=np.array(chip_names, dtype=h5py.string_dtype())
        )
        f.attrs['chip_size'] = chip_size
        f.attrs['num_chips'] = len(temp_files)


def chip_sar_images_parallel(
    input_dirs: List[str],
    chip_size: int = 512,
    output_file: str = "chips.h5",
    workers: int = None,
    use_hdf5: bool = True
) -> None:
    """
    Chip SAR images in parallel across multiple directories.
    
    Args:
        input_dirs: List of directories containing TIFF files
        chip_size: Size of square chips (default: 512)
        output_file: Output HDF5 file path
        workers: Number of parallel workers (default: CPU count)
        use_hdf5: If True, consolidate to HDF5; if False, keep as .npy files
    """
    if workers is None:
        workers = mp.cpu_count()
    
    print(f"Using {workers} parallel workers")
    print(f"Chip size: {chip_size}x{chip_size}")
    print()
    
    # Collect all TIFF files from all directories
    all_tiff_files = []
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"Warning: {input_dir} does not exist, skipping")
            continue
        
        tiff_files = [
            f for f in input_path.iterdir()
            if f.is_file() and f.suffix.lower() in ['.tiff', '.tif']
        ]
        all_tiff_files.extend(tiff_files)
    
    if len(all_tiff_files) == 0:
        raise ValueError("No TIFF files found in any input directory")
    
    print(f"Found {len(all_tiff_files)} TIFF files across {len(input_dirs)} directories")
    
    # Create temporary directory for chips
    temp_dir = Path(output_file).parent / "temp_chips"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images in parallel
    process_func = partial(
        process_single_image,
        chip_size=chip_size,
        temp_dir=temp_dir
    )
    
    print("\nProcessing images in parallel...")
    with mp.Pool(processes=workers) as pool:
        results = list(tqdm(
            pool.imap(process_func, all_tiff_files),
            total=len(all_tiff_files),
            desc="Processing images",
            unit="img"
        ))
    
    # Collect all temp files
    all_temp_files = []
    total_chips = 0
    for img_name, num_chips, temp_files in results:
        total_chips += num_chips
        all_temp_files.extend(temp_files)
    
    print(f"\nTotal chips created: {total_chips}")
    
    if use_hdf5:
        # Consolidate to HDF5
        output_path = Path(output_file)
        consolidate_to_hdf5(all_temp_files, output_path, chip_size)
        
        # Clean up temp directory
        try:
            temp_dir.rmdir()
        except:
            pass
        
        print(f"\nOutput saved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / (1024**3):.2f} GB")
    else:
        print(f"\nChips saved to: {temp_dir}")
        print(f"Total files: {len(all_temp_files)}")
    
    print("\n" + "="*60)
    print("Chipping complete!")
    print("="*60)


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Fast parallel chipping of Capella SAR images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "input_dirs",
        nargs='+',
        type=str,
        help="One or more directories containing Capella SAR TIFF images"
    )
    parser.add_argument(
        "--chip_size",
        type=int,
        default=512,
        help="Size of square chips in pixels"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="chips.h5",
        help="Output HDF5 file path"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)"
    )
    parser.add_argument(
        "--no-hdf5",
        action='store_true',
        help="Keep chips as individual .npy files instead of HDF5"
    )
    
    args = parser.parse_args()
    
    try:
        chip_sar_images_parallel(
            args.input_dirs,
            chip_size=args.chip_size,
            output_file=args.output,
            workers=args.workers,
            use_hdf5=not args.no_hdf5
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())