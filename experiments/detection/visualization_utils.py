"""
Shared visualization utilities for detection experiments.
"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, Optional
from matplotlib.lines import Line2D


def plot_detection_predictions(
    image: torch.Tensor,
    prediction: Dict,
    target: Dict,
    loss_info: Optional[Dict] = None,
    title: str = ""
) -> plt.Figure:
    """
    Plot predictions and ground truth boxes on an image.

    Args:
        image: Image tensor [3, H, W] or [1, H, W]
        prediction: Dict with 'boxes' [N,4], 'labels' [N], 'scores' [N]
        target: Dict with 'boxes' [M,4], 'labels' [M]
        loss_info: Optional dict with loss components ('total_loss', 'box_loss', 'obj_loss', 'cls_loss')
        title: Title prefix

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Convert image to numpy (take first channel for SAR)
    if image.shape[0] == 1:
        img_np = image[0].cpu().numpy()
    else:
        img_np = image[0].cpu().numpy()  # Take first channel

    # Normalize for display
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    # Display image
    ax.imshow(img_np, cmap='gray')

    # Draw ground truth boxes (green)
    gt_boxes = target['boxes'].cpu()
    gt_labels = target['labels'].cpu()

    for box, label in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='green', facecolor='none'
        )
        ax.add_patch(rect)

    # Draw predicted boxes (red dashed)
    pred_boxes = prediction['boxes'].cpu()
    pred_labels = prediction['labels'].cpu()
    pred_scores = prediction['scores'].cpu()

    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2, edgecolor='red', facecolor='none',
            linestyle='--'
        )
        ax.add_patch(rect)

        # Add score text
        ax.text(
            x1, y1 - 5, f'{score:.2f}',
            color='red', fontsize=8,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
        )

    # Build title
    title_text = title
    if loss_info:
        title_text += f"\nLoss: {loss_info['total_loss']:.3f} | "
        title_text += f"Box: {loss_info['box_loss']:.3f} | "
        title_text += f"Obj: {loss_info['obj_loss']:.3f} | "
        title_text += f"Cls: {loss_info['cls_loss']:.3f}"
        title_text += f"\nGT boxes: {len(gt_boxes)} | Pred boxes: {len(pred_boxes)}"
    else:
        title_text += f"\nGT boxes: {len(gt_boxes)} | Pred boxes: {len(pred_boxes)}"

    ax.set_title(title_text, fontsize=10)
    ax.axis('off')

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='green', linewidth=2, label='Ground Truth'),
        Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Predictions')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    return fig
