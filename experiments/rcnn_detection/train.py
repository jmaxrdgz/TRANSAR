import sys
import platform
import torch.multiprocessing as mp
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.config import load_config
from backbones.factory import get_backbone
from rcnn import FasterRCNNLit
from data.data_finetune import build_dataloaders

if __name__ == "__main__":
    if platform.system() == "Darwin":
            mp.set_start_method("spawn", force=True)

    config = load_config('experiments/rcnn_detection/config_experiment.yaml')

    L.seed_everything(config.SEED, workers=True)

    train_dataloader, val_dataloader = build_dataloaders(config)

    backbone = get_backbone(
        config.MODEL.BACKBONE.NAME, 
        config.MODEL.BACKBONE.PRETRAINED,
        config.MODEL.BACKBONE.WEIGHTS
        )

    model = FasterRCNNLit(
        backbone=backbone,
        num_classes=config.DATA.NUM_CLASSES,
        lr=config.TRAIN.LR,
        image_size=config.MODEL.IN_SIZE
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/pretrain/{config.MODEL.BACKBONE}",
        filename="rcnn-detection-{epoch:03d}-{train_loss:.4f}",
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        save_last=True,
        every_n_epochs=config.TRAIN.LOG_FREQ
    )

    callbacks = [
        ModelSummary(max_depth=2),
        checkpoint_callback
    ]

    trainer = L.Trainer(
        precision="16-mixed",
        max_epochs=config.TRAIN.EPOCHS,
        accelerator="gpu",
        devices=config.TRAIN.GPUS,
        log_every_n_steps=config.TRAIN.LOG_FREQ,
        callbacks=callbacks,
        deterministic=True,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
