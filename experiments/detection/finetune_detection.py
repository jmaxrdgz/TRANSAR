"""
Finetuning pipeline for object detection using YOLO with custom backbones.
Tests detection capability of different backbone architectures.
"""

import os
import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

from utils.config import load_config
from experiments.detection.models import YOLODetector
from experiments.detection.data import create_detection_dataloaders
from experiments.detection.callbacks import ValidationVisualizationCallback


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Finetune YOLO for object detection on SAR images'
    )

    # Config and experiment
    default_config = str(Path(__file__).parent / 'configs' / 'config_experiment.yaml')
    parser.add_argument(
        '--config',
        type=str,
        default=default_config,
        help='Path to config file'
    )
    parser.add_argument(
        '--experiment_name',
        type=str,
        default=None,
        help='Experiment name (overrides config)'
    )
    parser.add_argument(
        '--version',
        type=str,
        default=None,
        help='Experiment version (overrides config)'
    )

    # Model
    parser.add_argument(
        '--backbone',
        type=str,
        default=None,
        help='Backbone architecture (e.g., swinv2_tiny_window8_256, resnet50, convnext_tiny)'
    )
    parser.add_argument(
        '--pretrained_weights',
        type=str,
        default=None,
        help='Path to pretrained backbone weights'
    )
    parser.add_argument(
        '--num_blocks_to_unfreeze',
        type=int,
        default=None,
        help='Number of backbone blocks to unfreeze (0=freeze all)'
    )

    # Data
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to dataset (overrides config)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=None,
        help='Number of data loading workers (overrides config)'
    )

    # Training
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of training epochs (overrides config)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--gpus',
        type=int,
        default=None,
        help='Number of GPUs to use (overrides config)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed (overrides config)'
    )

    # Checkpointing
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--save_top_k',
        type=int,
        default=None,
        help='Save top K checkpoints (overrides config)'
    )

    # Misc
    parser.add_argument(
        '--fast_dev_run',
        action='store_true',
        help='Run a quick test (1 batch per epoch)'
    )
    parser.add_argument(
        '--limit_train_batches',
        type=float,
        default=None,
        help='Limit number of training batches per epoch (for debugging)'
    )
    parser.add_argument(
        '--limit_val_batches',
        type=float,
        default=None,
        help='Limit number of validation batches per epoch (for debugging)'
    )

    return parser.parse_args()


def override_config(config, args):
    """Override config values with command-line arguments."""
    # Model overrides
    if args.backbone is not None:
        config.MODEL.BACKBONE.NAME = args.backbone
    if args.pretrained_weights is not None:
        config.MODEL.BACKBONE.WEIGHTS = args.pretrained_weights
    if args.num_blocks_to_unfreeze is not None:
        config.MODEL.NUM_BLOCKS_TO_UNFREEZE = args.num_blocks_to_unfreeze

    # Data overrides
    if args.data_path is not None:
        config.DATA.DATA_PATH = args.data_path
    if args.batch_size is not None:
        config.TRAIN.BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        config.TRAIN.NUM_WORKERS = args.num_workers

    # Training overrides
    if args.epochs is not None:
        config.TRAIN.EPOCHS = args.epochs
    if args.lr is not None:
        config.TRAIN.LR = args.lr
    if args.gpus is not None:
        config.TRAIN.GPUS = args.gpus
    if args.seed is not None:
        config.SEED = args.seed

    # Experiment overrides
    if args.experiment_name is not None:
        config.EXPERIMENT.NAME = args.experiment_name
    if args.version is not None:
        config.EXPERIMENT.VERSION = args.version
    if args.save_top_k is not None:
        config.LOGGING.SAVE_TOP_K = args.save_top_k

    return config


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load config (pass empty list to prevent load_config from parsing sys.argv)
    print(f"Loading config from: {args.config}")
    config = load_config(args.config, args=[])

    # Override with command-line arguments
    config = override_config(config, args)

    # Set seed
    L.seed_everything(config.SEED)

    # Print configuration
    print("\n" + "="*80)
    print("DETECTION FINETUNING CONFIGURATION")
    print("="*80)
    print(f"Experiment: {config.EXPERIMENT.NAME}")
    print(f"Backbone: {config.MODEL.BACKBONE.NAME}")
    print(f"Pretrained weights: {config.MODEL.BACKBONE.WEIGHTS}")
    print(f"Blocks to unfreeze: {config.MODEL.NUM_BLOCKS_TO_UNFREEZE}")
    print(f"Dataset: {config.DATA.DATA_PATH}")
    print(f"Num classes: {config.DATA.NUM_CLASSES}")
    print(f"Batch size: {config.TRAIN.BATCH_SIZE}")
    print(f"Learning rate: {config.TRAIN.LR}")
    print(f"Epochs: {config.TRAIN.EPOCHS}")
    print(f"GPUs: {config.TRAIN.GPUS}")
    print("="*80 + "\n")

    # Create dataloaders
    print("Creating dataloaders...")
    train_dataloader, val_dataloader = create_detection_dataloaders(config)
    print(f"Train samples: {len(train_dataloader.dataset)}")
    print(f"Val samples: {len(val_dataloader.dataset)}")

    # Create model
    print(f"\nCreating model: {config.MODEL.NAME}")
    model = YOLODetector(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")

    # Setup logging
    log_dir = config.LOGGING.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=config.EXPERIMENT.NAME,
        version=config.EXPERIMENT.VERSION
    )
    print(f"\nTensorBoard logs: {logger.log_dir}")

    # Setup callbacks
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, 'checkpoints'),
        filename='epoch={epoch:02d}-mAP={val/mAP_50_95:.4f}',
        monitor='val/mAP_50_95',
        mode='max',
        save_top_k=config.LOGGING.SAVE_TOP_K,
        save_last=True,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)

    # Visualization callback
    viz_callback = ValidationVisualizationCallback(
        num_images=3,
        save_to_disk=True,
        log_to_tensorboard=True
    )
    callbacks.append(viz_callback)

    # Create trainer
    trainer = L.Trainer(
        max_epochs=config.TRAIN.EPOCHS,
        accelerator='auto',
        devices=config.TRAIN.GPUS,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=config.TRAIN.LOG_FREQ,
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches is not None else 1.0,
        limit_val_batches=args.limit_val_batches if args.limit_val_batches is not None else 1.0,
        precision='16-mixed' if torch.cuda.is_available() else 32,
        gradient_clip_val=1.0,  # Gradient clipping for stability
    )

    # Train
    print("\nStarting training...")
    print(f"Device: {trainer.strategy.root_device}")

    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=args.resume
        )
    else:
        trainer.fit(
            model=model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader
        )

    # Print results
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    if not args.fast_dev_run:
        print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
        if checkpoint_callback.best_model_score is not None:
            print(f"Best mAP@0.5:0.95: {checkpoint_callback.best_model_score:.4f}")
    print(f"TensorBoard logs: {logger.log_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
