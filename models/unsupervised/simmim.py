"""
SimMIM: Simple Framework for Masked Image Modeling

Implements SimMIM with block-wise masking as used in TRANSAR paper.

Key features:
- Learnable mask token (replaces masked patches)
- Block-wise masking (contiguous 8×8 blocks)
- Full sequence encoding (all patches including masked)
- Simple linear decoder
- L1 loss on masked patches only

References:
- SimMIM paper: "SimMIM: A Simple Framework for Masked Image Modeling"
- TRANSAR paper: Uses block-wise masking with mask_size=8
"""

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import lightning as L
import matplotlib.pyplot as plt
from pathlib import Path

from models.unsupervised.backbones import create_backbone


class SimMIM(L.LightningModule):
    """
    SimMIM: A Simple Framework for Masked Image Modeling

    Key differences from MAE:
    1. Keeps all patches (replaces masked with learnable token)
    2. Uses L1 loss instead of MSE
    3. Simple linear decoder (no complex decoder)
    4. Block-wise masking (TRANSAR variant)
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Create backbone
        self.encoder, self.backbone_cfg = create_backbone(
            config.MODEL.BACKBONE,
            in_chans=config.MODEL.IN_CHANS,
            pretrained=True,
            img_size=config.DATA.IMG_SIZE
        )

        self.patch_size = self.backbone_cfg['patch_size']
        self.embed_dim = self.backbone_cfg['embed_dim']
        self.in_chans = config.MODEL.IN_CHANS
        self.mask_ratio = config.MODEL.MASK_RATIO

        # Block-wise masking (TRANSAR paper)
        self.mask_size = config.MODEL.MASK_SIZE if hasattr(config.MODEL, 'MASK_SIZE') else 1

        # Learnable mask token (CRITICAL for SimMIM)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Simple linear decoder (SimMIM uses minimal decoder)
        self.decoder = nn.Linear(
            self.encoder.num_features,  # Final feature dimension
            self.patch_size ** 2 * self.in_chans  # Reconstruct patch pixels
        )

        # L1 loss (better than MSE for pixel reconstruction)
        self.loss_fn = nn.L1Loss(reduction='none')

        print(f"[SimMIM] Backbone: {config.MODEL.BACKBONE}")
        print(f"[SimMIM] Patch size: {self.patch_size}, Embed dim: {self.embed_dim}")
        print(f"[SimMIM] Mask ratio: {self.mask_ratio}")
        print(f"[SimMIM] Mask size (block-wise): {self.mask_size}x{self.mask_size}")
        print(f"[SimMIM] Input channels: {self.in_chans}")

    def patchify(self, imgs):
        """
        Convert images to patches.

        Args:
            imgs: [B, C, H, W]
        Returns:
            patches: [B, N, patch_size^2 * C] where N = (H/p) * (W/p)
        """
        p = self.patch_size
        B, C, H, W = imgs.shape
        assert H % p == 0 and W % p == 0, f"Image size ({H}, {W}) not divisible by patch size {p}"

        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
        return x

    def unpatchify(self, x, H, W):
        """
        Convert patches back to images.

        Args:
            x: [B, N, patch_size^2 * C]
            H, W: Original image height and width
        Returns:
            imgs: [B, C, H, W]
        """
        p = self.patch_size
        B, N, _ = x.shape
        C = self.in_chans

        h, w = H // p, W // p
        assert N == h * w, f"Mismatch: N={N}, h*w={h*w}"

        x = x.reshape(B, h, w, p, p, C)
        imgs = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, H, W)
        return imgs

    def random_mask(self, B, N, device):
        """
        Generate random binary mask with block-wise masking.

        Block-wise masking (TRANSAR paper):
        Instead of masking individual patches randomly, we mask contiguous
        blocks of patches. This creates more structured masking that the
        model must reconstruct, forcing it to learn better representations.

        Args:
            B: Batch size
            N: Number of patches (should be H*W where H=W for square images)
            device: Device to create mask on

        Returns:
            mask: [B, N] where 1 = masked, 0 = visible
        """
        # Calculate number of patches per dimension (assumes square)
        H = W = int(N ** 0.5)
        assert H * W == N, f"Number of patches must be square, got N={N}"

        if self.mask_size > 1:
            # Block-wise masking (TRANSAR approach)
            assert H % self.mask_size == 0, \
                f"Patch grid size {H} must be divisible by mask_size {self.mask_size}"

            # Number of blocks per dimension
            num_blocks_per_dim = H // self.mask_size

            # Total number of blocks
            num_blocks = num_blocks_per_dim ** 2

            # Number of blocks to mask
            num_blocks_to_mask = int(num_blocks * self.mask_ratio)

            # Create block mask [B, num_blocks]
            block_mask = torch.zeros(B, num_blocks, device=device)

            for b in range(B):
                # Randomly select which blocks to mask
                mask_indices = torch.randperm(num_blocks, device=device)[:num_blocks_to_mask]
                block_mask[b, mask_indices] = 1

            # Expand block mask to patch mask
            # Reshape to [B, num_blocks_per_dim, num_blocks_per_dim]
            block_mask = block_mask.reshape(B, num_blocks_per_dim, num_blocks_per_dim)

            # Repeat each block to cover mask_size × mask_size patches
            # [B, num_blocks_per_dim, num_blocks_per_dim] -> [B, H, W]
            mask = block_mask.repeat_interleave(self.mask_size, dim=1)  # Expand rows
            mask = mask.repeat_interleave(self.mask_size, dim=2)  # Expand cols

            # Flatten to [B, N]
            mask = mask.reshape(B, N)

        else:
            # Patch-wise masking (original SimMIM)
            num_masked = int(N * self.mask_ratio)

            # Random shuffle
            noise = torch.rand(B, N, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)

            # Create mask: 1 for masked patches, 0 for visible
            mask = torch.zeros([B, N], device=device)
            mask[:, :num_masked] = 1

            # Unshuffle to get mask in original order
            mask = torch.gather(mask, dim=1, index=torch.argsort(ids_shuffle, dim=1))

        return mask

    def forward(self, imgs):
        """
        Forward pass with masking and reconstruction.

        Args:
            imgs: [B, C, H, W]

        Returns:
            loss_masked: Reconstruction loss on masked patches
            loss_visible: Reconstruction loss on visible patches
            pred: Reconstructed patches [B, N, patch_dim]
            mask: Binary mask [B, N]
        """
        B, C, H, W = imgs.shape

        # 1. Patchify image
        patches = self.patchify(imgs)  # [B, N, patch_dim]
        B, N, patch_dim = patches.shape

        # 2. Generate random mask
        mask = self.random_mask(B, N, imgs.device)  # [B, N]

        # 3. Get patch embeddings from encoder's patch_embed
        # Note: Swin models handle this differently than ViT
        # We need to pass the full image through patch_embed
        x = self.encoder.patch_embed(imgs)  # [B, H', W', C] for Swin

        # Flatten spatial dimensions to get sequence
        if x.ndim == 4:  # Swin format: [B, H', W', C]
            B, H_p, W_p, C_emb = x.shape
            x = x.reshape(B, H_p * W_p, C_emb)  # [B, N, C]

        assert x.shape[1] == N, f"Patch count mismatch: {x.shape[1]} vs {N}"

        # 4. Replace masked patches with learnable mask token
        mask_expanded = mask.unsqueeze(-1)  # [B, N, 1]
        mask_tokens = self.mask_token.expand(B, N, -1)  # [B, N, embed_dim]

        # Ensure dimensions match before masking
        if x.shape[-1] != mask_tokens.shape[-1]:
            # Project to match dimensions if needed
            if not hasattr(self, 'proj'):
                self.proj = nn.Linear(x.shape[-1], self.embed_dim).to(x.device)
            x = self.proj(x)

        x = x * (1 - mask_expanded) + mask_tokens * mask_expanded

        # 5. Pass through encoder (FULL SEQUENCE with mask tokens)
        # For Swin, we need to reshape back to spatial format
        x = x.reshape(B, H_p, W_p, -1)  # [B, H', W', C]

        # Forward through Swin stages (skip patch_embed since we already did it)
        # Call layers directly instead of forward_features to avoid double patch_embed
        for layer in self.encoder.layers:
            x = layer(x)
        features = self.encoder.norm(x)  # [B, H'', W'', C_final]

        # Flatten back to sequence
        if features.ndim == 4:
            B, H_f, W_f, C_f = features.shape
            features = features.reshape(B, H_f * W_f, C_f)

        # Handle potential downsampling in encoder
        if features.shape[1] != N:
            # Upsample features to match original patch count
            features = F.interpolate(
                features.permute(0, 2, 1).reshape(B, -1, H_f, W_f),
                size=(H_p, W_p),
                mode='bilinear',
                align_corners=False
            )
            features = features.reshape(B, -1, H_p * W_p).permute(0, 2, 1)

        # 6. Decode to reconstruct pixels
        pred = self.decoder(features)  # [B, N, patch_dim]

        # 7. Compute loss on masked and visible patches
        mask_bool = mask.bool()
        loss_masked = self.loss_fn(pred[mask_bool], patches[mask_bool]).sum() / mask_bool.sum() / self.in_chans
        loss_visible = self.loss_fn(pred[~mask_bool], patches[~mask_bool]).sum() / (~mask_bool).sum() / self.in_chans

        return loss_masked, loss_visible, pred, mask

    def training_step(self, batch, batch_idx):
        """Training step for Lightning."""
        imgs, _ = batch  # Unlabeled data (ignore labels)

        loss_masked, loss_visible, pred, mask = self(imgs)

        # Log metrics
        self.log("train_loss", loss_masked, prog_bar=True, sync_dist=True)
        self.log("loss_visible", loss_visible, prog_bar=False, sync_dist=True)
        self.log("lr", self.optimizers().param_groups[0]['lr'], prog_bar=True, sync_dist=True)

        return loss_masked

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norm before optimizer step."""
        # Compute gradient norm across all parameters
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        self.log("grad_norm", total_norm, prog_bar=False, sync_dist=True)

    def configure_optimizers(self):
        """Configure optimizer with cosine annealing and warmup."""
        # AdamW optimizer
        optimizer = AdamW(
            self.parameters(),
            lr=self.config.TRAIN.LR,
            weight_decay=self.config.TRAIN.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )

        # Cosine annealing with warmup
        def lr_lambda(current_step):
            warmup = self.config.TRAIN.WARMUP_EPOCHS * self.trainer.num_training_batches
            total = self.config.TRAIN.EPOCHS * self.trainer.num_training_batches

            if current_step < warmup:
                # Linear warmup
                return float(current_step) / float(max(1, warmup))
            else:
                # Cosine annealing
                progress = float(current_step - warmup) / float(max(1, total - warmup))
                return 0.5 * (1.0 + math.cos(progress * math.pi))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Update every step, not epoch
                'frequency': 1
            }
        }
