import torch
import timm
import lightning as L
from torch.utils.data import DataLoader

from configs import build_config
from data.data_finetune import build_dataloaders, yolo_collate_fn
from models.supervised.head import ESPCN
from models.supervised.transar import TRANSAR
from models.supervised.adaptive_sampler import AdaptiveSampler
from models.supervised.callbacks import DataLoaderRecreationCallback, AdaptiveSamplingCallback


if __name__ == "__main__":
    config = build_config(pretrain=False)

    L.seed_everything(config.SEED)

    #-----------
    #   Model
    #-----------
    if config.MODEL.BACKBONE.WEIGHTS is None:
        backbone = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=True,
            features_only=True
        )
        print("Using ImageNet pretrained backbone weights")
    else:
        backbone = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=False,
            features_only=True
        )
        backbone.load_state_dict(torch.load(config.MODEL.BACKBONE.WEIGHTS, map_location="cpu"))
    in_channels = backbone.feature_info[-1]['num_chs']

    # Determine output channels based on num_classes
    # For binary classification (num_classes=2): 1 channel (foreground heatmap)
    # For multi-class: num_classes channels
    out_channels = 1 if config.DATA.NUM_CLASS == 2 else config.DATA.NUM_CLASS

    head = ESPCN(
        in_channels=in_channels,
        upscale_factor=config.MODEL.SCALE_FACTOR,
        out_channels=out_channels
    )
    model = TRANSAR(config, backbone, head)

    #-----------------
    #   DataLoaders
    #-----------------
    train_dataloader, val_dataloader = build_dataloaders(config)

    #-------------------------
    #   Adaptive Sampling
    #-------------------------
    if hasattr(config.TRAIN, 'ADAPTIVE_SAMPLING') and config.TRAIN.ADAPTIVE_SAMPLING.ENABLED:
        print(f"Initializing Adaptive Sampling with {config.TRAIN.ADAPTIVE_SAMPLING.SCHEDULER_TYPE} scheduler")

        # Get training dataset from dataloader
        train_dataset = train_dataloader.dataset

        # Create adaptive sampler
        adaptive_sampler = AdaptiveSampler(
            dataset=train_dataset,
            num_classes=config.DATA.NUM_CLASS,  # Use NUM_CLASS from DATA config
            scheduler_type=config.TRAIN.ADAPTIVE_SAMPLING.SCHEDULER_TYPE,
            alpha=config.TRAIN.ADAPTIVE_SAMPLING.ALPHA,
            total_epochs=config.TRAIN.EPOCHS,
            device='cpu'  # Keep on CPU for sampler
        )

        # Print initial class distribution
        info = adaptive_sampler.get_distribution_info()
        print(f"Class distribution: {info['class_counts']}")
        print(f"Initial d_train: {info['d_train']}")

        # Set sampler on model
        model.set_adaptive_sampler(adaptive_sampler)

        # Create initial train dataloader with weighted sampler for epoch 0
        initial_sampler = adaptive_sampler.get_weighted_sampler(epoch=0)

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.TRAIN.BATCH_SIZE,
            sampler=initial_sampler,  # Use sampler instead of shuffle
            num_workers=config.TRAIN.NUM_WORKERS,
            persistent_workers=True,
            pin_memory=True,
            collate_fn=yolo_collate_fn
        )

        print("Adaptive Sampling initialized successfully")
    else:
        print("Adaptive Sampling disabled")

    #-------------
    #   Trainer
    #-------------
    # Setup callbacks
    callbacks = []

    if hasattr(config.TRAIN, 'ADAPTIVE_SAMPLING') and config.TRAIN.ADAPTIVE_SAMPLING.ENABLED:
        # Add DataLoader callback to update sampler weights each epoch
        dataloader_callback = DataLoaderRecreationCallback()
        callbacks.append(dataloader_callback)

        # Optionally add monitoring callback for visualization
        # adaptive_monitoring_callback = AdaptiveSamplingCallback(
        #     log_dir="./logs/adaptive_sampling",
        #     plot_frequency=10
        # )
        # callbacks.append(adaptive_monitoring_callback)

    trainer = L.Trainer(
        max_epochs=config.TRAIN.EPOCHS,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
        callbacks=callbacks,
    )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

