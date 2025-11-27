"""
Baseline YOLO training script using Ultralytics YOLO (v8/v11).
This script trains a standard YOLO model as a baseline for comparison
with custom backbone approaches. Default: YOLOv11l
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ultralytics import YOLO
import yaml
import torch


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 baseline model using Ultralytics'
    )

    # Model configuration
    parser.add_argument(
        '--model',
        type=str,
        default='yolo11l.pt',
        help='YOLO model variant (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x, or yolov8n/s/m/l/x)'
    )
    parser.add_argument(
        '--pretrained',
        type=str,
        default=None,
        help='Path to pretrained weights (if not using default COCO weights)'
    )

    # Data configuration
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data.yaml file defining dataset'
    )

    # Training configuration
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=16,
        help='Batch size'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=256,
        help='Image size for training'
    )
    parser.add_argument(
        '--lr0',
        type=float,
        default=0.01,
        help='Initial learning rate'
    )
    parser.add_argument(
        '--lrf',
        type=float,
        default=0.01,
        help='Final learning rate (lr0 * lrf)'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.937,
        help='SGD momentum'
    )
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0005,
        help='Weight decay'
    )
    parser.add_argument(
        '--warmup_epochs',
        type=float,
        default=3.0,
        help='Warmup epochs'
    )

    # Augmentation
    parser.add_argument(
        '--hsv_h',
        type=float,
        default=0.015,
        help='HSV-Hue augmentation (fraction)'
    )
    parser.add_argument(
        '--hsv_s',
        type=float,
        default=0.7,
        help='HSV-Saturation augmentation (fraction)'
    )
    parser.add_argument(
        '--hsv_v',
        type=float,
        default=0.4,
        help='HSV-Value augmentation (fraction)'
    )
    parser.add_argument(
        '--degrees',
        type=float,
        default=0.0,
        help='Image rotation (+/- deg)'
    )
    parser.add_argument(
        '--translate',
        type=float,
        default=0.1,
        help='Image translation (+/- fraction)'
    )
    parser.add_argument(
        '--scale',
        type=float,
        default=0.5,
        help='Image scale (+/- gain)'
    )
    parser.add_argument(
        '--flipud',
        type=float,
        default=0.0,
        help='Image flip up-down (probability)'
    )
    parser.add_argument(
        '--fliplr',
        type=float,
        default=0.5,
        help='Image flip left-right (probability)'
    )
    parser.add_argument(
        '--mosaic',
        type=float,
        default=1.0,
        help='Image mosaic (probability)'
    )
    parser.add_argument(
        '--mixup',
        type=float,
        default=0.0,
        help='Image mixup (probability)'
    )

    # System
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device to use (e.g., 0, 0,1,2,3 for GPU(s), or cpu)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='logs/baseline_yolo',
        help='Project directory for saving results'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='train',
        help='Experiment name'
    )
    parser.add_argument(
        '--exist_ok',
        action='store_true',
        help='Allow overwriting existing project/name'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    # Validation
    parser.add_argument(
        '--val',
        action='store_true',
        default=True,
        help='Validate after training'
    )
    parser.add_argument(
        '--save_period',
        type=int,
        default=-1,
        help='Save checkpoint every N epochs (-1 to disable)'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=50,
        help='Early stopping patience (epochs without improvement)'
    )

    # Additional options
    parser.add_argument(
        '--cache',
        type=str,
        default='ram',
        choices=['ram', 'disk', 'False'],
        help='Cache images (ram/disk/False)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--amp',
        action='store_true',
        default=True,
        help='Use Automatic Mixed Precision (AMP) training'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Print configuration
    print("\n" + "="*80)
    print("ULTRALYTICS YOLO BASELINE TRAINING (Default: YOLOv11l)")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Epochs: {args.epochs}")
    print(f"Initial LR: {args.lr0}")
    print(f"Device: {args.device if args.device else 'auto'}")
    print(f"Project: {args.project}")
    print(f"Name: {args.name}")
    print(f"Seed: {args.seed}")
    print("="*80 + "\n")

    # Load model
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        model = YOLO(args.resume)
    elif args.pretrained:
        print(f"Loading pretrained weights from: {args.pretrained}")
        model = YOLO(args.pretrained)
    else:
        print(f"Loading model: {args.model}")
        model = YOLO(args.model)

    # Prepare cache parameter
    cache_val = False if args.cache == 'False' else args.cache

    # Train model
    print("\nStarting training...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=args.degrees,
        translate=args.translate,
        scale=args.scale,
        flipud=args.flipud,
        fliplr=args.fliplr,
        mosaic=args.mosaic,
        mixup=args.mixup,
        device=args.device if args.device else None,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        val=args.val,
        save_period=args.save_period,
        patience=args.patience,
        cache=cache_val,
        verbose=args.verbose,
        seed=args.seed,
        amp=args.amp,
    )

    # Print results
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.project}/{args.name}")

    # Validate if requested
    if args.val:
        print("\nRunning final validation...")
        metrics = model.val()
        print(f"mAP@0.5: {metrics.box.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")

    print("="*80 + "\n")


if __name__ == '__main__':
    main()
