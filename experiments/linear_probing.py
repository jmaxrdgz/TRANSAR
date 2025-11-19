"""
Linear Probing for Classification Finetuning

This module implements a PyTorch Lightning module for finetuning a pretrained
backbone on a classification task using linear probing. It supports:
- Loading pretrained backbone weights
- Adding a single dense layer for classification
- Freezing/unfreezing specific backbone blocks
- Standard classification training with cross-entropy loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm
from typing import Optional, Dict, Any
from torchmetrics import Accuracy, F1Score


class LinearProbingClassifier(pl.LightningModule):
    """
    Linear probing classifier for finetuning pretrained backbones.

    Architecture:
        - Pretrained backbone (frozen or partially frozen)
        - Global average pooling
        - Single linear layer for classification

    Args:
        backbone_path: Path to pretrained backbone weights
        backbone_name: Name of the backbone architecture (from timm)
        num_classes: Number of output classes
        num_blocks_unfreeze: Number of blocks to unfreeze from the end (0 = only linear layer)
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for AdamW
        warmup_epochs: Number of warmup epochs for LR scheduler
        max_epochs: Maximum number of training epochs
        in_chans: Number of input channels (1 for SAR, 3 for RGB)
    """

    def __init__(
        self,
        backbone_path: str,
        backbone_name: str = "swinv2_tiny_window8_256",
        num_classes: int = 10,
        num_blocks_unfreeze: int = 0,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.05,
        warmup_epochs: int = 5,
        max_epochs: int = 100,
        in_chans: int = 3,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.num_blocks_unfreeze = num_blocks_unfreeze

        # Create backbone
        print(f"[LinearProbing] Creating backbone: {backbone_name}")
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            num_classes=0,  # Remove head
            global_pool='',  # Remove pooling
            in_chans=in_chans
        )

        # Load pretrained weights
        print(f"[LinearProbing] Loading pretrained weights from: {backbone_path}")
        state_dict = torch.load(backbone_path, map_location='cpu')

        # Handle potential state_dict wrapping (e.g., from Lightning checkpoints)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Remove 'encoder.' or 'backbone.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('encoder.', '').replace('backbone.', '')
            new_state_dict[new_key] = v

        self.backbone.load_state_dict(new_state_dict, strict=True)
        print(f"[LinearProbing] Successfully loaded pretrained weights")

        # Get feature dimension from backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, in_chans, 256, 256)
            dummy_features = self.backbone(dummy_input)
            if isinstance(dummy_features, (list, tuple)):
                # If features_only, take last feature map
                feature_dim = dummy_features[-1].shape[1]
            else:
                feature_dim = dummy_features.shape[1]

        print(f"[LinearProbing] Backbone feature dimension: {feature_dim}")

        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Linear probing layer
        self.classifier = nn.Linear(feature_dim, num_classes)

        # Initialize classifier
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        if self.classifier.bias is not None:
            nn.init.constant_(self.classifier.bias, 0)

        # Freeze/unfreeze blocks
        self._configure_frozen_layers()

        # Metrics
        task = "multiclass" if num_classes > 2 else "binary"
        self.train_acc = Accuracy(task=task, num_classes=num_classes)
        self.val_acc = Accuracy(task=task, num_classes=num_classes)
        self.val_f1 = F1Score(task=task, num_classes=num_classes, average='macro')

    def _configure_frozen_layers(self):
        """Freeze backbone and unfreeze specified number of blocks from the end."""
        # First, freeze all backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Always keep classifier trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

        if self.num_blocks_unfreeze > 0:
            # Get all layer groups that can be unfrozen
            # For Swin Transformer, these are typically 'layers' or 'stages'
            unfrozen_blocks = []

            # Try to find layers/stages in the backbone
            if hasattr(self.backbone, 'layers'):
                # Swin Transformer v1/v2 structure
                layers = self.backbone.layers
                num_layers = len(layers)

                # Unfreeze last N layers
                for i in range(max(0, num_layers - self.num_blocks_unfreeze), num_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True
                    unfrozen_blocks.append(f'layers.{i}')

                print(f"[LinearProbing] Unfrozen {len(unfrozen_blocks)} blocks: {unfrozen_blocks}")

            elif hasattr(self.backbone, 'stages'):
                # Alternative naming
                stages = self.backbone.stages
                num_stages = len(stages)

                for i in range(max(0, num_stages - self.num_blocks_unfreeze), num_stages):
                    for param in stages[i].parameters():
                        param.requires_grad = True
                    unfrozen_blocks.append(f'stages.{i}')

                print(f"[LinearProbing] Unfrozen {len(unfrozen_blocks)} blocks: {unfrozen_blocks}")
            else:
                print(f"[LinearProbing] Warning: Could not find 'layers' or 'stages' in backbone. Only linear layer will be trainable.")
        else:
            print(f"[LinearProbing] All backbone frozen. Only linear probing layer is trainable.")

        # Print trainable parameters summary
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LinearProbing] Total parameters: {total_params:,}")
        print(f"[LinearProbing] Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")

    def forward(self, x):
        """Forward pass through backbone and classifier."""
        # Extract features
        features = self.backbone(x)

        # Handle features_only output (list of feature maps)
        if isinstance(features, (list, tuple)):
            features = features[-1]

        # Global average pooling
        features = self.pool(features)
        features = features.flatten(1)

        # Classification
        logits = self.classifier(features)
        return logits

    def training_step(self, batch, batch_idx):
        """Training step."""
        if isinstance(batch, dict):
            images = batch['image']
            labels = batch['label']
        else:
            images, labels = batch

        # Forward pass
        logits = self(images)

        # Compute loss
        loss = F.cross_entropy(logits, labels)

        # Compute accuracy
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)

        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        if isinstance(batch, dict):
            images = batch['image']
            labels = batch['label']
        else:
            images, labels = batch

        # Forward pass
        logits = self(images)

        # Compute loss
        loss = F.cross_entropy(logits, labels)

        # Compute metrics
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)
        self.val_f1(preds, labels)

        # Log metrics
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # AdamW optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999)
        )

        # Cosine annealing with warmup
        warmup_steps = self.warmup_epochs * self.trainer.estimated_stepping_batches // self.max_epochs
        total_steps = self.trainer.estimated_stepping_batches

        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return step / warmup_steps
            else:
                # Cosine decay
                progress = (step - warmup_steps) / (total_steps - warmup_steps)
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
