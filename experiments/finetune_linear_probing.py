"""
Finetuning script for linear probing on classification tasks.

This script finetunes a pretrained backbone on a classification task by:
1. Loading a pretrained backbone from checkpoint
2. Adding a single linear layer for classification
3. Optionally unfreezing last N blocks of the backbone
4. Training with PyTorch Lightning

Usage:
    python experiments/finetune_linear_probing.py \
        --backbone_path checkpoints/pretrain/backbone_final.pth \
        --dataset_path data/classification/train \
        --num_classes 10 \
        --num_blocks_unfreeze 0 \
        --batch_size 32 \
        --epochs 100 \
        --learning_rate 1e-3
"""

import argparse
import os
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from linear_probing import LinearProbingClassifier


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Finetune pretrained backbone with linear probing for classification"
    )

    # Model arguments
    parser.add_argument(
        "--backbone_path",
        type=str,
        required=True,
        help="Path to pretrained backbone weights (.pth file)"
    )
    parser.add_argument(
        "--backbone_name",
        type=str,
        default="swinv2_tiny_window8_256",
        help="Backbone architecture name from timm (default: swinv2_tiny_window8_256)"
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        required=True,
        help="Number of output classes"
    )
    parser.add_argument(
        "--num_blocks_unfreeze",
        type=int,
        default=0,
        help="Number of blocks to unfreeze from the end (0 = only linear layer trainable)"
    )
    parser.add_argument(
        "--in_chans",
        type=int,
        default=1,
        help="Number of input channels (1 for SAR, 3 for RGB)"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="folder",
        choices=["folder", "custom"],
        help="Type of dataset (folder = ImageFolder structure, custom = custom implementation)"
    )
    parser.add_argument(
        "--val_split",
        type=float,
        default=0.2,
        help="Validation split ratio (default: 0.2)"
    )

    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="Weight decay for AdamW optimizer"
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of warmup epochs for learning rate scheduler"
    )

    # Data augmentation
    parser.add_argument(
        "--img_size",
        type=int,
        default=256,
        help="Input image size (default: 256)"
    )
    parser.add_argument(
        "--no_augmentation",
        action="store_true",
        help="Disable data augmentation during training"
    )

    # Training configuration
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Training precision (32, 16-mixed, bf16-mixed)"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        help="Accelerator type (auto, gpu, cpu)"
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices to use"
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=1.0,
        help="Gradient clipping value (default: 1.0)"
    )

    # Checkpointing and logging
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/linear_probing",
        help="Directory to save checkpoints and logs"
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name for logging (default: auto-generated)"
    )
    parser.add_argument(
        "--save_top_k",
        type=int,
        default=3,
        help="Save top k checkpoints based on validation accuracy"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=15,
        help="Early stopping patience in epochs (0 to disable)"
    )

    # Resume training
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    return parser.parse_args()


def create_dataloaders(args):
    """
    Create train and validation dataloaders.

    This is a placeholder implementation that uses torchvision ImageFolder.
    You should replace this with your custom dataset implementation.
    """
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import random_split

    # Define transforms
    if args.no_augmentation:
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.Grayscale(num_output_channels=args.in_chans),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*args.in_chans, std=[0.5]*args.in_chans)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Grayscale(num_output_channels=args.in_chans),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*args.in_chans, std=[0.5]*args.in_chans)
        ])

    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.Grayscale(num_output_channels=args.in_chans),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*args.in_chans, std=[0.5]*args.in_chans)
    ])

    # Load dataset
    if args.dataset_type == "folder":
        # Assume ImageFolder structure: dataset_path/class1/, dataset_path/class2/, etc.
        full_dataset = ImageFolder(args.dataset_path, transform=train_transform)

        # Split into train and validation
        val_size = int(len(full_dataset) * args.val_split)
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        # Update validation dataset transform
        val_dataset.dataset.transform = val_transform

        print(f"[Dataset] Loaded {len(full_dataset)} images")
        print(f"[Dataset] Train: {len(train_dataset)}, Val: {len(val_dataset)}")
        print(f"[Dataset] Classes: {full_dataset.classes}")

    else:
        # Custom dataset implementation
        raise NotImplementedError(
            "Custom dataset type not implemented. "
            "Please implement your own dataset class and add it here."
        )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    return train_loader, val_loader


def main():
    """Main training function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiment name if not provided
    if args.experiment_name is None:
        backbone_name = Path(args.backbone_path).stem
        args.experiment_name = f"{backbone_name}_unfreeze{args.num_blocks_unfreeze}_cls{args.num_classes}"

    print(f"\n{'='*60}")
    print(f"Linear Probing Finetuning")
    print(f"{'='*60}")
    print(f"Backbone: {args.backbone_name}")
    print(f"Pretrained weights: {args.backbone_path}")
    print(f"Dataset: {args.dataset_path}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Blocks to unfreeze: {args.num_blocks_unfreeze}")
    print(f"Experiment: {args.experiment_name}")
    print(f"{'='*60}\n")

    # Create dataloaders
    print("[Setup] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(args)

    # Create model
    print("\n[Setup] Creating model...")
    model = LinearProbingClassifier(
        backbone_path=args.backbone_path,
        backbone_name=args.backbone_name,
        num_classes=args.num_classes,
        num_blocks_unfreeze=args.num_blocks_unfreeze,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        in_chans=args.in_chans,
    )

    # Setup callbacks
    callbacks = []

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir / args.experiment_name,
        filename='epoch{epoch:03d}-val_acc{val/acc:.4f}',
        monitor='val/acc',
        mode='max',
        save_top_k=args.save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Early stopping
    if args.early_stopping_patience > 0:
        early_stop = EarlyStopping(
            monitor='val/acc',
            patience=args.early_stopping_patience,
            mode='max',
            verbose=True
        )
        callbacks.append(early_stop)

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name=args.experiment_name,
        default_hp_metric=False
    )

    # Create trainer
    print("\n[Setup] Creating trainer...")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=args.gradient_clip_val,
        log_every_n_steps=10,
        enable_model_summary=True,
    )

    # Train
    print(f"\n[Training] Starting training for {args.epochs} epochs...")
    print(f"[Training] Logs will be saved to: {output_dir / args.experiment_name}")

    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume_from
    )

    # Print best model info
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best validation accuracy: {checkpoint_callback.best_model_score:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
