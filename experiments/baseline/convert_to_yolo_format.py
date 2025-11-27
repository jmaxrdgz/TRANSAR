"""
Convert SIVED dataset from oriented bounding box format to YOLO format.
SIVED format: x1 y1 x2 y2 x3 y3 x4 y4 class_name difficulty
YOLO format: class_id x_center y_center width height (normalized)
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np


def parse_sived_line(line):
    """Parse a line from SIVED label file."""
    parts = line.strip().split()
    if len(parts) < 10:
        return None

    # Extract oriented bounding box coordinates
    x1, y1 = float(parts[0]), float(parts[1])
    x2, y2 = float(parts[2]), float(parts[3])
    x3, y3 = float(parts[4]), float(parts[5])
    x4, y4 = float(parts[6]), float(parts[7])
    class_name = parts[8]

    # Convert to axis-aligned bounding box (take min/max of all coordinates)
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)

    return {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max,
        'class_name': class_name
    }


def convert_to_yolo_format(bbox, img_width, img_height, class_id=0):
    """Convert bbox to YOLO format."""
    x_center = (bbox['x_min'] + bbox['x_max']) / 2.0 / img_width
    y_center = (bbox['y_min'] + bbox['y_max']) / 2.0 / img_height
    width = (bbox['x_max'] - bbox['x_min']) / img_width
    height = (bbox['y_max'] - bbox['y_min']) / img_height

    # Clamp values to [0, 1]
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))

    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def convert_split(dataset_path, split_name='train'):
    """Convert a single split (train/valid/test) to YOLO format."""
    images_dir = dataset_path / 'ImageSets' / 'images' / split_name
    labels_input_dir = dataset_path / 'ImageSets' / 'labelTxt' / split_name
    labels_output_dir = dataset_path / 'ImageSets' / 'labels' / split_name

    # Create output directory
    labels_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting {split_name} split...")
    print(f"Images: {images_dir}")
    print(f"Input labels: {labels_input_dir}")
    print(f"Output labels: {labels_output_dir}")

    # Get all label files
    label_files = list(labels_input_dir.glob('*.txt'))
    print(f"Found {len(label_files)} label files")

    converted_count = 0
    error_count = 0

    for label_file in label_files:
        try:
            # Get corresponding image
            image_name = label_file.stem
            image_files = list(images_dir.glob(f"{image_name}.*"))

            if not image_files:
                print(f"Warning: No image found for {image_name}")
                error_count += 1
                continue

            image_file = image_files[0]

            # Get image dimensions
            with Image.open(image_file) as img:
                img_width, img_height = img.size

            # Read and convert labels
            yolo_lines = []
            with open(label_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    bbox = parse_sived_line(line)
                    if bbox is None:
                        continue

                    # Convert to YOLO format (class_id is always 0 for single class)
                    yolo_line = convert_to_yolo_format(bbox, img_width, img_height, class_id=0)
                    yolo_lines.append(yolo_line)

            # Write YOLO format labels
            output_file = labels_output_dir / label_file.name
            with open(output_file, 'w') as f:
                f.write('\n'.join(yolo_lines))
                if yolo_lines:  # Add final newline if there are labels
                    f.write('\n')

            converted_count += 1

            if converted_count % 100 == 0:
                print(f"Converted {converted_count}/{len(label_files)} files...")

        except Exception as e:
            print(f"Error processing {label_file.name}: {e}")
            error_count += 1

    print(f"Conversion complete: {converted_count} files converted, {error_count} errors")
    return converted_count, error_count


def main():
    """Main conversion function."""
    # Dataset path
    dataset_path = Path(__file__).parent.parent.parent / 'dataset' / 'supervised' / 'SIVED'

    print("="*80)
    print("SIVED to YOLO Format Conversion")
    print("="*80)
    print(f"Dataset path: {dataset_path}")

    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return

    # Convert all splits
    splits = ['train', 'valid', 'test']
    total_converted = 0
    total_errors = 0

    for split in splits:
        split_images_dir = dataset_path / 'ImageSets' / 'images' / split
        if split_images_dir.exists():
            converted, errors = convert_split(dataset_path, split)
            total_converted += converted
            total_errors += errors
        else:
            print(f"\nSkipping {split} split (not found)")

    print("\n" + "="*80)
    print("Conversion Summary")
    print("="*80)
    print(f"Total files converted: {total_converted}")
    print(f"Total errors: {total_errors}")
    print("="*80)

    # Update data config path
    print("\nNote: Update data_config_sived.yaml to use labels directory:")
    print("  Change 'labelTxt/{split}' to 'labels/{split}'")


if __name__ == '__main__':
    main()
