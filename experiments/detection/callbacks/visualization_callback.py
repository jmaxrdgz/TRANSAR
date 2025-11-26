"""
Callback for epoch-end visualization of random validation images.
"""

import os
import random
import io
from pathlib import Path
from typing import Dict, List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lightning.pytorch.callbacks import Callback


class ValidationVisualizationCallback(Callback):
    """
    Visualize random validation images at the end of each epoch.

    Saves visualizations to disk and logs to TensorBoard.
    The same random images are shown consistently across all epochs.
    """

    def __init__(
        self,
        num_images: int = 3,
        seed: int = 42,
        save_to_disk: bool = True,
        log_to_tensorboard: bool = True
    ):
        """
        Args:
            num_images: Number of random images to visualize (default: 3)
            seed: Random seed for reproducible image selection (default: 42)
            save_to_disk: Whether to save visualizations to disk
            log_to_tensorboard: Whether to log to TensorBoard
        """
        super().__init__()
        self.num_images = num_images
        self.seed = seed
        self.save_to_disk = save_to_disk
        self.log_to_tensorboard = log_to_tensorboard

        # Random number generator with fixed seed
        self.rng = random.Random(seed)

        # Will be initialized on first validation epoch
        self.selected_indices: Optional[List[int]] = None
        self._indices_initialized = False

    def on_validation_epoch_start(self, trainer, pl_module):
        """Called at the start of validation epoch. Select random indices on first epoch."""
        # Only initialize indices once
        if self._indices_initialized:
            return

        # Only run on global rank 0 in distributed training
        if not trainer.is_global_zero:
            return

        # We'll initialize indices at the end of first validation epoch
        # when we know the total dataset size

    def on_validation_epoch_end(self, trainer, pl_module):
        """Called at the end of validation epoch."""
        # Only run on global rank 0 in distributed training
        if not trainer.is_global_zero:
            return

        # Access stored validation data
        if not hasattr(pl_module, 'val_images') or len(pl_module.val_images) == 0:
            return

        images = pl_module.val_images
        predictions = pl_module.val_predictions
        targets = pl_module.val_targets

        # Initialize random indices on first epoch
        if not self._indices_initialized:
            total_images = len(images)
            # Handle case where num_images exceeds available images
            actual_num_images = min(self.num_images, total_images)
            if actual_num_images < self.num_images:
                print(f"Warning: Requested {self.num_images} images but only {total_images} available. "
                      f"Using {actual_num_images} images.")

            # Select random indices using fixed seed
            self.selected_indices = self.rng.sample(range(total_images), actual_num_images)
            self._indices_initialized = True
            print(f"Selected random validation image indices: {self.selected_indices}")

        # Get save directory
        if self.save_to_disk:
            save_dir = self._get_save_directory(trainer)
        else:
            save_dir = None

        # Visualize randomly selected images
        for rank, idx in enumerate(self.selected_indices):
            self._visualize_and_save(
                image=images[idx],
                prediction=predictions[idx],
                target=targets[idx],
                title_prefix=f"Random Sample #{rank+1}",
                save_path=save_dir / f"random_sample_{rank+1}.png" if save_dir else None,
                trainer=trainer,
                tag=f"validation/random_sample_{rank+1}"
            )


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

        # Create visualization (no loss info)
        fig = plot_detection_predictions(
            image=image,
            prediction=prediction,
            target=target,
            loss_info=None,
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

