import platform
import torch
from torch import nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary
import timm

from configs import build_config
from data.data_pretrain import build_loader


# -----------------------------
# --- Backbone Factory ---
# -----------------------------
BACKBONE_CONFIGS = {
    # Swin v1 - Fixed resolution
    'swin_tiny': {
        'model_name': 'swin_tiny_patch4_window7_224',
        'img_size': 224,
        'window_size': 7,
        'patch_size': 4,
        'embed_dim': 96
    },
    'swin_small': {
        'model_name': 'swin_small_patch4_window7_224',
        'img_size': 224,
        'window_size': 7,
        'patch_size': 4,
        'embed_dim': 96
    },

    # Swin v2 - Variable resolution support
    'swinv2_tiny_w8': {
        'model_name': 'swinv2_tiny_window8_256',
        'img_size': 256,
        'window_size': 8,
        'patch_size': 4,
        'embed_dim': 96
    },
    'swinv2_tiny_w16': {
        'model_name': 'swinv2_tiny_window16_256',
        'img_size': 256,
        'window_size': 16,
        'patch_size': 4,
        'embed_dim': 96
    },
    'swinv2_small_w8': {
        'model_name': 'swinv2_small_window8_256',
        'img_size': 256,
        'window_size': 8,
        'patch_size': 4,
        'embed_dim': 96
    },
    'swinv2_small_w16': {
        'model_name': 'swinv2_small_window16_256',
        'img_size': 256,
        'window_size': 16,
        'patch_size': 4,
        'embed_dim': 96
    }
}


def create_backbone(backbone_name, in_chans=3, pretrained=False):
    """
    Create a Swin backbone from configuration.

    Args:
        backbone_name: Name from BACKBONE_CONFIGS
        in_chans: Number of input channels (1 for SAR, 3 for RGB)
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        backbone: Swin model
        config: Backbone configuration dict
    """
    if backbone_name not in BACKBONE_CONFIGS:
        raise ValueError(f"Unknown backbone: {backbone_name}. Choose from {list(BACKBONE_CONFIGS.keys())}")

    backbone_cfg = BACKBONE_CONFIGS[backbone_name]

    # Create model without classification head
    model = timm.create_model(
        backbone_cfg['model_name'],
        pretrained=pretrained,
        num_classes=0,  # Remove classification head
        global_pool='',  # Remove global pooling
        in_chans=in_chans
    )

    return model, backbone_cfg


# -----------------------------
# --- SimMIM Architecture ---
# -----------------------------
class SimMIM(L.LightningModule):
    """
    SimMIM: A Simple Framework for Masked Image Modeling

    Key differences from MAE:
    1. Keeps all patches (replaces masked with learnable token)
    2. Uses L1 loss instead of MSE
    3. Simple linear decoder (no complex decoder)
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        # Create backbone
        self.encoder, self.backbone_cfg = create_backbone(
            config.MODEL.BACKBONE,
            in_chans=config.MODEL.IN_CHANS,
            pretrained=False  # Always train from scratch for pretraining
        )

        self.patch_size = self.backbone_cfg['patch_size']
        self.embed_dim = self.backbone_cfg['embed_dim']
        self.in_chans = config.MODEL.IN_CHANS
        self.mask_ratio = config.MODEL.MASK_RATIO

        # Learnable mask token (CRITICAL for SimMIM)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Simple linear decoder (SimMIM uses minimal decoder)
        self.decoder = nn.Linear(
            self.encoder.num_features,  # Final feature dimension
            self.patch_size ** 2 * self.in_chans  # Reconstruct patch pixels
        )

        # L1 loss (better than MSE for pixel reconstruction)
        self.loss_fn = nn.L1Loss()

        print(f"[SimMIM] Backbone: {config.MODEL.BACKBONE}")
        print(f"[SimMIM] Patch size: {self.patch_size}, Embed dim: {self.embed_dim}")
        print(f"[SimMIM] Mask ratio: {self.mask_ratio}")
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
        Generate random binary mask.

        Args:
            B: Batch size
            N: Number of patches
            device: Device to create mask on

        Returns:
            mask: [B, N] where 1 = masked, 0 = visible
        """
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
            loss: Reconstruction loss on masked patches
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

        # Forward through Swin stages
        features = self.encoder.forward_features(x)  # [B, H'', W'', C_final]

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

        # 7. Compute loss ONLY on masked patches
        mask_bool = mask.bool()
        loss = self.loss_fn(pred[mask_bool], patches[mask_bool])

        return loss, pred, mask

    def training_step(self, batch, batch_idx):
        """Training step for Lightning."""
        imgs, _ = batch  # Unlabeled data (ignore labels)

        loss, pred, mask = self(imgs)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("mask_ratio", mask.float().mean(), prog_bar=False, sync_dist=True)

        return loss

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
                return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)))

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',  # Update every step, not epoch
                'frequency': 1
            }
        }


# -----------------------------
# --- Training Script ---
# -----------------------------
if __name__ == "__main__":
    if platform.system() == "Darwin":
        mp.set_start_method("spawn", force=True)

    # Load config
    config = build_config(pretrain=True)

    # Set seed
    L.seed_everything(config.SEED, workers=True)

    # Build data loader
    train_loader = build_loader(config)

    # Create or resume model
    if hasattr(config.MODEL, 'RESUME') and config.MODEL.RESUME is not None:
        print(f"Resuming from checkpoint: {config.MODEL.RESUME}")
        model = SimMIM.load_from_checkpoint(
            config.MODEL.RESUME,
            config=config,
            map_location="cpu"
        )
    else:
        print("Training from scratch")
        model = SimMIM(config)

    # Setup callbacks
    callbacks = [
        ModelSummary(max_depth=2),

        # Save checkpoints
        ModelCheckpoint(
            dirpath=f"checkpoints/pretrain/{config.MODEL.BACKBONE}",
            filename="simmim-{epoch:03d}-{train_loss:.4f}",
            save_top_k=3,
            monitor="train_loss",
            mode="min",
            save_last=True,
            every_n_epochs=10
        ),

        # Save backbone weights separately (for easy loading in finetuning)
        ModelCheckpoint(
            dirpath=f"checkpoints/pretrain/{config.MODEL.BACKBONE}/backbone",
            filename="backbone-{epoch:03d}",
            save_top_k=1,
            monitor="train_loss",
            mode="min",
            save_last=True,
            every_n_epochs=25,
            save_weights_only=True
        )
    ]

    # Create trainer
    trainer = L.Trainer(
        callbacks=callbacks,
        gradient_clip_val=config.TRAIN.CLIP_GRAD,
        precision="16-mixed",
        devices=config.TRAIN.N_GPU,
        accelerator="auto",
        max_epochs=config.TRAIN.EPOCHS,
        log_every_n_steps=50,
        deterministic=True,
        enable_progress_bar=True,
        default_root_dir=f"logs/pretrain/{config.MODEL.BACKBONE}"
    )

    # Train
    print("\n" + "="*60)
    print(f"Starting SimMIM pretraining")
    print(f"Backbone: {config.MODEL.BACKBONE}")
    print(f"Input size: {config.DATA.IMG_SIZE}")
    print(f"Batch size: {config.TRAIN.BATCH_SIZE}")
    print(f"Epochs: {config.TRAIN.EPOCHS}")
    print(f"Learning rate: {config.TRAIN.LR}")
    print("="*60 + "\n")

    trainer.fit(model, train_loader)

    # Save final backbone weights
    print("\nSaving final backbone weights...")
    torch.save(
        model.encoder.state_dict(),
        f"checkpoints/pretrain/{config.MODEL.BACKBONE}/backbone_final.pth"
    )
    print(f"Saved to: checkpoints/pretrain/{config.MODEL.BACKBONE}/backbone_final.pth")
