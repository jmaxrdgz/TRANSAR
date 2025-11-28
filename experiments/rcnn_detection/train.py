import os
import sys
import platform
import argparse
import torch.multiprocessing as mp
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from utils.config import load_config
from backbones.factory import get_backbone
from rcnn import FasterRCNNLit
from data.data_finetune import build_dataloaders

if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description='Train Faster R-CNN with SwinV2 backbone')
    parser.add_argument('--limit_train_batches', type=float, default=1.0,
                        help='Percentage of training batches to use (0.0-1.0 for percentage, >1.0 for absolute number of batches)')
    parser.add_argument('--limit_val_batches', type=float, default=1.0,
                        help='Percentage of validation batches to use (0.0-1.0 for percentage, >1.0 for absolute number of batches)')
    args = parser.parse_args()

    # macOS multiprocessing fix
    if platform.system() == "Darwin":
            mp.set_start_method("spawn", force=True)


    config = load_config('experiments/rcnn_detection/config_experiment.yaml')

    L.seed_everything(config.SEED, workers=True)

    train_dataloader, val_dataloader = build_dataloaders(config)

    backbone = get_backbone(
        config.MODEL.BACKBONE.NAME,
        config.MODEL.BACKBONE.PRETRAINED,
        config.MODEL.BACKBONE.WEIGHTS,
        config.MODEL.NUM_BLOCKS_TO_UNFREEZE
        )

    model = FasterRCNNLit(
        backbone=backbone,
        num_classes=config.DATA.NUM_CLASSES,
        lr=config.TRAIN.LR,
        image_size=config.MODEL.IN_SIZE
    )

    # Setup logging
    log_dir = config.LOGGING.LOG_DIR
    os.makedirs(log_dir, exist_ok=True)

    logger = TensorBoardLogger(
        save_dir=log_dir,
        name=config.EXPERIMENT.NAME,
        version=config.EXPERIMENT.VERSION
    )
    print(f"TensorBoard logs: {logger.log_dir}")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(logger.log_dir, 'checkpoints'),
        filename='epoch={epoch:02d}-train_loss={train_loss:.4f}',
        monitor='train_loss',
        mode='min',
        save_top_k=config.LOGGING.SAVE_TOP_K,
        save_last=True,
        auto_insert_metric_name=False
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    callbacks = [
        ModelSummary(max_depth=2),
        checkpoint_callback,
        lr_monitor
    ]

    trainer = L.Trainer(
        precision="16-mixed",
        max_epochs=config.TRAIN.EPOCHS,
        accelerator="gpu",
        devices=config.TRAIN.GPUS,
        logger=logger,
        log_every_n_steps=config.TRAIN.LOG_FREQ,
        callbacks=callbacks,
        deterministic=True,
        enable_progress_bar=True,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
