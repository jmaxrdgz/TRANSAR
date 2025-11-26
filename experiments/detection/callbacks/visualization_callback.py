"""
Callback for epoch-end visualization of sampled validation images.
"""

import io
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lightning.pytorch.callbacks import Callback


class ValidationVisualizationCallback(Callback):
    """
    Visualize sampled validation images at the end of each epoch.

    Images are randomly selected at the start of each validation epoch
    and the same images are visualized consistently across epochs.

    Saves visualizations to disk and logs to TensorBoard.
    """

    def __init__(
        self,
        num_images: int = 3,
        save_to_disk: bool = True,
        log_to_tensorboard: bool = True,
        epoch_interval: int = 1
    ):
        """
        Args:
            num_images: Number of images to visualize (default: 3)
            save_to_disk: Whether to save visualizations to disk
            log_to_tensorboard: Whether to log to TensorBoard
            epoch_interval: Plot images only every N epochs (default: 1, i.e., every epoch)
        """
        super().__init__()
        self.num_images = num_images
        self.save_to_disk = save_to_disk
        self.log_to_tensorboard = log_to_tensorboard
        self.epoch_interval = epoch_interval

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of validation epoch."""
        # Only run on global rank 0 in distributed training
        if not trainer.is_global_zero:
            # Wait for rank 0 to finish visualization in distributed training
            if hasattr(trainer.strategy, 'barrier'):
                trainer.strategy.barrier()
            return

        # Check if we should visualize this epoch
        if (trainer.current_epoch + 1) % self.epoch_interval != 0:
            # Ensure barrier is called even when skipping visualization
            if hasattr(trainer.strategy, 'barrier'):
                trainer.strategy.barrier()
            return

        # Use sampled validation data (val_viz_*) instead of full validation set
        if not hasattr(pl_module, 'val_viz_images') or len(pl_module.val_viz_images) == 0:
            # Ensure barrier is called even when no images available
            if hasattr(trainer.strategy, 'barrier'):
                trainer.strategy.barrier()
            return

        images = pl_module.val_viz_images
        predictions = pl_module.val_viz_predictions
        targets = pl_module.val_viz_targets

        # Limit to num_images
        num_to_visualize = min(self.num_images, len(images))

        # Get save directory
        if self.save_to_disk:
            save_dir = self._get_save_directory(trainer)
        else:
            save_dir = None

        # Visualize sampled images
        for idx in range(num_to_visualize):
            self._visualize_and_save(
                image=images[idx],
                prediction=predictions[idx],
                target=targets[idx],
                title_prefix=f"Sample #{idx+1}",
                save_path=save_dir / f"sample_{idx+1}.png" if save_dir else None,
                trainer=trainer,
                tag=f"validation/sample_{idx+1}"
            )

        # Clear visualization buffers after use
        pl_module.val_viz_images = []
        pl_module.val_viz_predictions = []
        pl_module.val_viz_targets = []

        # Synchronize with other ranks in distributed training
        if hasattr(trainer.strategy, 'barrier'):
            trainer.strategy.barrier()

    def _get_save_directory(self, trainer) -> Path:
        """Get directory for saving visualizations."""
        # Use logger's log_dir
        log_dir = Path(trainer.logger.log_dir)
        viz_dir = log_dir / "visualizations" / f"epoch_{trainer.current_epoch:03d}"
        viz_dir.mkdir(parents=True, exist_ok=True)
        return viz_dir

    def _visualize_and_save(
        self,
        image: torch.Tensor,
        prediction: Dict,
        target: Dict,
        title_prefix: str,
        save_path: Path = None,
        trainer = None,
        tag: str = None
    ):
        """Visualize single image with predictions and ground truth."""
        from experiments.detection.visualization_utils import plot_detection_predictions

        # Create visualization (without loss_info)
        fig = plot_detection_predictions(
            image=image,
            prediction=prediction,
            target=target,
            title=title_prefix
        )

        # Save to disk
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")

        # Log to TensorBoard
        if self.log_to_tensorboard and trainer and tag:
            if hasattr(trainer.logger, 'experiment'):
                img_tensor = self._fig_to_tensor(fig)
                trainer.logger.experiment.add_image(
                    tag, img_tensor, global_step=trainer.current_epoch
                )

        plt.close(fig)

    def _fig_to_tensor(self, fig: plt.Figure) -> torch.Tensor:
        """Convert matplotlib figure to tensor for TensorBoard."""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)

        img = Image.open(buf)
        img_array = np.array(img)

        # Convert RGBA to RGB if needed
        if img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]

        # [H, W, C] -> [C, H, W]
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)

        # Normalize to [0, 1]
        if img_tensor.max() > 1.0:
            img_tensor = img_tensor.float() / 255.0

        buf.close()
        return img_tensor
