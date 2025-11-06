"""
SimMIM Pretraining Script

Trains a Swin Transformer backbone using SimMIM (Simple Masked Image Modeling)
with block-wise masking as specified in the TRANSAR paper.

Usage:
    python pretrain.py
    python pretrain.py --override MODEL.BACKBONE=swinv2_tiny_w16
    python pretrain.py --override DATA.IMG_SIZE=384 TRAIN.BATCH_SIZE=32

For more information, see docs/pretraining/
"""

import platform
import torch
import torch.multiprocessing as mp
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

from configs import build_config
from data.data_pretrain import build_loader
from models.unsupervised import SimMIM


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
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/pretrain/{config.MODEL.BACKBONE}",
        filename="simmim-{epoch:03d}-{train_loss:.4f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        save_last=True,
        every_n_epochs=config.TRAIN.CHECKPOINT_FREQUENCY
    )

    callbacks = [
        ModelSummary(max_depth=2),
        checkpoint_callback
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

    # Save backbone weights from best checkpoint
    print("\nExtracting backbone from best checkpoint...")
    best_model_path = checkpoint_callback.best_model_path

    if best_model_path:
        print(f"Loading best checkpoint: {best_model_path}")
        best_model = SimMIM.load_from_checkpoint(
            best_model_path,
            config=config,
            map_location="cpu"
        )

        backbone_path = f"checkpoints/pretrain/{config.MODEL.BACKBONE}/backbone_final.pth"
        torch.save(best_model.encoder.state_dict(), backbone_path)
        print(f"Saved backbone weights to: {backbone_path}")
    else:
        print("Warning: No best checkpoint found, using final model state")
        backbone_path = f"checkpoints/pretrain/{config.MODEL.BACKBONE}/backbone_final.pth"
        torch.save(model.encoder.state_dict(), backbone_path)
        print(f"Saved to: {backbone_path}")
