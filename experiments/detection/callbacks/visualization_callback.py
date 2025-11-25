"""
Callback for epoch-end visualization of best/worst validation images.
"""

import os
import json
import io
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from lightning.pytorch.callbacks import Callback


class ValidationVisualizationCallback(Callback):
    """
    Visualize best and worst validation images at the end of each epoch.

    Saves visualizations to disk and logs to TensorBoard.
    """

    def __init__(
        self,
        num_images: int = 3,
        save_to_disk: bool = True,
        log_to_tensorboard: bool = True
    ):
        """
        Args:
            num_images: Number of best/worst images to visualize (default: 3)
            save_to_disk: Whether to save visualizations to disk
            log_to_tensorboard: Whether to log to TensorBoard
        """
        super().__init__()
        self.num_images = num_images
        self.save_to_disk = save_to_disk
        self.log_to_tensorboard = log_to_tensorboard

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

        # Compute per-image losses
        per_image_losses = self._compute_per_image_losses(
            pl_module, images, predictions, targets
        )

        # Select best and worst images
        best_indices, worst_indices = self._select_images(per_image_losses)

        # Get save directory
        if self.save_to_disk:
            save_dir = self._get_save_directory(trainer)
        else:
            save_dir = None

        # Visualize best images
        for rank, idx in enumerate(best_indices):
            self._visualize_and_save(
                image=images[idx],
                prediction=predictions[idx],
                target=targets[idx],
                loss_info=per_image_losses[idx],
                title_prefix=f"Best Loss #{rank+1}",
                save_path=save_dir / f"best_loss_{rank+1}.png" if save_dir else None,
                trainer=trainer,
                tag=f"validation/best_loss_{rank+1}"
            )

        # Visualize worst images
        for rank, idx in enumerate(worst_indices):
            self._visualize_and_save(
                image=images[idx],
                prediction=predictions[idx],
                target=targets[idx],
                loss_info=per_image_losses[idx],
                title_prefix=f"Worst Loss #{rank+1}",
                save_path=save_dir / f"worst_loss_{rank+1}.png" if save_dir else None,
                trainer=trainer,
                tag=f"validation/worst_loss_{rank+1}"
            )

        # Save metadata
        if self.save_to_disk and save_dir:
            self._save_metadata(
                save_dir, trainer.current_epoch,
                per_image_losses, best_indices, worst_indices
            )

    def _compute_per_image_losses(
        self,
        pl_module,
        images: List[torch.Tensor],
        predictions: List[Dict],
        targets: List[Dict]
    ) -> List[Dict[str, float]]:
        """
        Compute loss for each validation image.

        Returns:
            List of dicts with 'total_loss', 'box_loss', 'obj_loss', 'cls_loss'
        """
        per_image_losses = []

        for img, pred, target in zip(images, predictions, targets):
            # Move to model device
            img = img.to(pl_module.device)
            target = {k: v.to(pl_module.device) if isinstance(v, torch.Tensor) else v
                     for k, v in target.items()}

            # Forward pass to get raw predictions (before NMS)
            with torch.no_grad():
                # Get features from backbone
                img_batch = img.unsqueeze(0)
                if pl_module.needs_channel_conversion and img_batch.shape[1] == 3:
                    img_batch = img_batch[:, 0:1, :, :]

                features = pl_module.backbone(img_batch)
                features_list = [features[str(i)] for i in sorted(map(int, features.keys()))]

                # Get raw predictions from detection head
                raw_predictions = pl_module.detection_head(features_list)

                # Compute loss
                loss_dict = pl_module.compute_loss(
                    raw_predictions,
                    [target],
                    img.shape[-2:]
                )

            per_image_losses.append({
                'total_loss': float(loss_dict['loss']),
                'box_loss': float(loss_dict['box_loss']),
                'obj_loss': float(loss_dict['obj_loss']),
                'cls_loss': float(loss_dict['cls_loss'])
            })

        return per_image_losses

    def _select_images(
        self,
        per_image_losses: List[Dict]
    ) -> Tuple[List[int], List[int]]:
        """
        Select best and worst images based on total loss.

        Returns:
            Tuple of (best_indices, worst_indices)
        """
        # Sort by total loss
        sorted_indices = sorted(
            range(len(per_image_losses)),
            key=lambda i: per_image_losses[i]['total_loss']
        )

        # Best = lowest loss
        best_indices = sorted_indices[:self.num_images]
        # Worst = highest loss
        worst_indices = sorted_indices[-self.num_images:][::-1]  # Reverse for descending order

        return best_indices, worst_indices

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
        loss_info: Dict,
        title_prefix: str,
        save_path: Path = None,
        trainer = None,
        tag: str = None
    ):
        """Visualize single image with predictions and ground truth."""
        from experiments.detection.visualization_utils import plot_detection_predictions

        # Create visualization
        fig = plot_detection_predictions(
            image=image,
            prediction=prediction,
            target=target,
            loss_info=loss_info,
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

    def _save_metadata(
        self,
        save_dir: Path,
        epoch: int,
        per_image_losses: List[Dict],
        best_indices: List[int],
        worst_indices: List[int]
    ):
        """Save metadata JSON file."""
        metadata = {
            'epoch': epoch,
            'best_images': [
                {
                    'rank': rank + 1,
                    'index': idx,
                    **per_image_losses[idx]
                }
                for rank, idx in enumerate(best_indices)
            ],
            'worst_images': [
                {
                    'rank': rank + 1,
                    'index': idx,
                    **per_image_losses[idx]
                }
                for rank, idx in enumerate(worst_indices)
            ]
        }

        metadata_path = save_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
