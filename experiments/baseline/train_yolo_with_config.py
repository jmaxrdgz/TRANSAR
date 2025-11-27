"""
Baseline YOLO training script using Ultralytics YOLOv11l.
This script uses hyperparameters from config_experiment.yaml to train a baseline
YOLO model for comparison with custom backbone approaches.
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

from utils.config import load_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train YOLOv11l baseline model using config from detection experiments'
    )

    # Config
    default_config = str(Path(__file__).parent.parent / 'detection' / 'configs' / 'config_experiment.yaml')
    parser.add_argument(
        '--config',
        type=str,
        default=default_config,
        help='Path to config file'
    )

    # Model override
    parser.add_argument(
        '--model',
        type=str,
        default='yolo11l.pt',
        help='YOLO model variant (yolo11n, yolo11s, yolo11m, yolo11l, yolo11x)'
    )

    # Data configuration
    parser.add_argument(
        '--data',
        type=str,
        default='experiments/baseline/data_config_sived.yaml',
        help='Path to data.yaml file defining dataset'
    )

    # Overrides
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--batch',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=None,
        help='Image size (overrides config)'
    )

    # System
    parser.add_argument(
        '--device',
        type=str,
        default='',
        help='Device to use (e.g., 0, 0,1,2,3 for GPU(s), or cpu)'
    )
    parser.add_argument(
        '--project',
        type=str,
        default='logs/detection_experiments',
        help='Project directory for saving results'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='yolo_baseline',
        help='Experiment name'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint'
    )
    parser.add_argument(
        '--exist_ok',
        action='store_true',
        help='Allow overwriting existing project/name'
    )

    return parser.parse_args()


def map_config_to_ultralytics(config, args):
    """Map config_experiment.yaml parameters to Ultralytics format."""

    # Get values from config or override with args
    epochs = args.epochs if args.epochs is not None else config.TRAIN.EPOCHS
    batch = args.batch if args.batch is not None else config.TRAIN.BATCH_SIZE
    imgsz = args.imgsz if args.imgsz is not None else config.MODEL.IN_SIZE

    # Learning rate - use HEAD_LR as the main learning rate for baseline
    lr0 = config.TRAIN.HEAD_LR

    # Final learning rate (use scheduler min_lr if available)
    if hasattr(config.TRAIN, 'SCHEDULER') and config.TRAIN.SCHEDULER.ENABLED:
        lrf = config.TRAIN.SCHEDULER.MIN_LR / lr0
    else:
        lrf = 0.01  # Default

    # Weight decay
    weight_decay = config.TRAIN.WEIGHT_DECAY

    # Workers
    workers = config.TRAIN.NUM_WORKERS

    # Seed
    seed = config.SEED

    # Augmentation settings - minimal for SAR images
    augmentation_config = {
        'hsv_h': 0.0,  # No hue augmentation for SAR
        'hsv_s': 0.0,  # No saturation augmentation for SAR
        'hsv_v': 0.2,  # Minimal value/brightness augmentation
        'degrees': 0.0,  # No rotation for SAR ships
        'translate': 0.1,  # Small translation
        'scale': 0.3,  # Scale augmentation
        'flipud': 0.0,  # No vertical flip for SAR ships
        'fliplr': 0.5,  # Horizontal flip
        'mosaic': 0.5,  # Reduced mosaic for SAR
        'mixup': 0.0,  # No mixup
    }

    # Check if augmentation is enabled in config
    if hasattr(config.DATA, 'AUGMENTATION') and not config.DATA.AUGMENTATION.ENABLED:
        # Disable most augmentations
        augmentation_config = {
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,
            'degrees': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'flipud': 0.0,
            'fliplr': 0.0,
            'mosaic': 0.0,
            'mixup': 0.0,
        }

    ultralytics_config = {
        'epochs': epochs,
        'batch': batch,
        'imgsz': imgsz,
        'lr0': lr0,
        'lrf': lrf,
        'momentum': 0.937,  # Standard SGD momentum
        'weight_decay': weight_decay,
        'warmup_epochs': 3.0,
        'workers': workers,
        'seed': seed,
        'patience': 50,  # Early stopping
        'save_period': -1,  # Don't save periodic checkpoints
        'amp': True,  # Automatic Mixed Precision
        'verbose': True,
        'val': True,
        'cache': False,  # Don't cache for large datasets
    }

    # Add augmentation settings
    ultralytics_config.update(augmentation_config)

    return ultralytics_config


def main():
    """Main training function."""
    args = parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config, args=[])

    # Map config to Ultralytics parameters
    train_config = map_config_to_ultralytics(config, args)

    # Print configuration
    print("\n" + "="*80)
    print("ULTRALYTICS YOLO BASELINE TRAINING (YOLOv11l)")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.data}")
    print(f"Config file: {args.config}")
    print("\nTraining Parameters:")
    print(f"  Image size: {train_config['imgsz']}")
    print(f"  Batch size: {train_config['batch']}")
    print(f"  Epochs: {train_config['epochs']}")
    print(f"  Initial LR: {train_config['lr0']}")
    print(f"  Final LR factor: {train_config['lrf']}")
    print(f"  Weight decay: {train_config['weight_decay']}")
    print(f"  Workers: {train_config['workers']}")
    print(f"  Seed: {train_config['seed']}")
    print("\nAugmentation:")
    print(f"  HSV (h/s/v): {train_config['hsv_h']}/{train_config['hsv_s']}/{train_config['hsv_v']}")
    print(f"  Rotation: {train_config['degrees']}°")
    print(f"  Translation: {train_config['translate']}")
    print(f"  Scale: {train_config['scale']}")
    print(f"  Flip LR/UD: {train_config['fliplr']}/{train_config['flipud']}")
    print(f"  Mosaic: {train_config['mosaic']}")
    print(f"  Mixup: {train_config['mixup']}")
    print(f"\nDevice: {args.device if args.device else 'auto'}")
    print(f"Project: {args.project}")
    print(f"Name: {args.name}")
    print("="*80 + "\n")

    # Load model
    if args.resume:
        print(f"Resuming training from: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"Loading model: {args.model}")
        model = YOLO(args.model)

    # Train model
    print("\nStarting training...")
    results = model.train(
        data=args.data,
        device=args.device if args.device else None,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        **train_config
    )

    # Print results
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.project}/{args.name}")

    # Final validation
    print("\nRunning final validation...")
    metrics = model.val()
    print(f"mAP@0.5: {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
