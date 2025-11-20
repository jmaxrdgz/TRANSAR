"""
Test script for YOLO detection model.
Evaluates model on test set, computes metrics, and visualizes best/worst predictions.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from tqdm import tqdm

from utils.config import load_config
from experiments.detection.models import YOLODetector
from experiments.detection.data import create_detection_dataloaders
from data.data_finetune import SARDetYoloDataset, SARNormalization
from experiments.detection.data.detection_transforms import detection_collate_fn
from functools import partial


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Test Faster R-CNN detection model on test set'
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (.ckpt file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file (if None, uses config from checkpoint)'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default=None,
        help='Path to dataset (overrides config)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='test_results',
        help='Directory to save test results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=4,
        help='Batch size for testing'
    )
    parser.add_argument(
        '--num_workers',
        type=int,
        default=4,
        help='Number of data loading workers'
    )
    parser.add_argument(
        '--ranking_metric',
        type=str,
        default='all',
        choices=['f1', 'map', 'iou', 'all'],
        help='Metric to use for ranking images (default: all)'
    )
    parser.add_argument(
        '--num_visualize',
        type=int,
        default=5,
        help='Number of best/worst images to visualize per metric'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for inference'
    )

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path: str, config=None, device='cuda'):
    """
    Load YOLO model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Config object (if None, loads from checkpoint)
        device: Device to load model on

    Returns:
        model: Loaded model in eval mode
    """
    print(f"\nLoading model from: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get config from checkpoint if not provided
    if config is None:
        if 'hyper_parameters' in checkpoint:
            # Load config from hyperparameters
            # Note: This may not work if config wasn't saved properly
            raise ValueError("Config must be provided when loading from checkpoint")
        else:
            raise ValueError("Config must be provided")

    # Create model
    model = YOLODetector(config)

    # Load state dict
    model.load_state_dict(checkpoint['state_dict'])

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    print("Model loaded successfully")
    return model


def create_test_dataloader(config, batch_size=4, num_workers=4):
    """
    Create test dataloader.

    Args:
        config: Configuration object
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        test_dataloader
    """
    print("\nCreating test dataloader...")

    # Build transforms (same as validation)
    val_transforms = []

    # SAR normalization
    if config.DATA.SAR_NORM_SCALE is not None:
        from data.data_finetune import SARNormalization
        sar_norm = SARNormalization(
            s_norm=config.DATA.SAR_NORM_SCALE,
            global_std=config.DATA.GLOBAL_STD
        )
        val_transforms.append(sar_norm)

    from torchvision import transforms as T
    val_transform = T.Compose(val_transforms) if val_transforms else None

    # Create test dataset
    test_dataset = SARDetYoloDataset(
        root_dir=config.DATA.DATA_PATH,
        split='test',
        num_classes=config.DATA.NUM_CLASSES - 1,  # Exclude background
        transform=val_transform
    )

    # Create collate function with target size
    collate_fn_with_size = partial(detection_collate_fn, target_size=config.MODEL.IN_SIZE)

    # Create dataloader
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_with_size,
        pin_memory=True
    )

    print(f"Test samples: {len(test_dataset)}")
    return test_dataloader


def run_inference(model, dataloader, device='cuda'):
    """
    Run inference on test set.

    Args:
        model: Faster R-CNN model
        dataloader: Test dataloader
        device: Device for inference

    Returns:
        predictions: List of prediction dicts
        targets: List of target dicts
        images_list: List of image tensors for visualization
    """
    print("\nRunning inference on test set...")

    predictions = []
    targets = []
    images_list = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Testing"):
            images = batch['images']
            batch_targets = batch['targets']

            # Move to device
            images = [img.to(device) for img in images]

            # Run inference
            batch_predictions = model(images)

            # Store results
            predictions.extend(batch_predictions)
            targets.extend(batch_targets)

            # Store images for visualization (convert back to CPU)
            images_list.extend([img.cpu() for img in images])

    print(f"Inference complete. Processed {len(predictions)} images.")
    return predictions, targets, images_list


def compute_per_image_metrics(predictions: List[Dict], targets: List[Dict]) -> Dict[int, Dict]:
    """
    Compute metrics for each image.

    Args:
        predictions: List of prediction dicts with 'boxes', 'labels', 'scores'
        targets: List of target dicts with 'boxes', 'labels'

    Returns:
        Dict mapping image_id to metrics dict with 'f1', 'map', 'iou'
    """
    print("\nComputing per-image metrics...")

    per_image_metrics = {}

    for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
        pred_boxes = pred['boxes']
        pred_labels = pred['labels']
        pred_scores = pred['scores']
        gt_boxes = target['boxes']
        gt_labels = target['labels']

        # Initialize metrics
        metrics = {
            'f1': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'map_50': 0.0,
            'map_75': 0.0,
            'map_50_95': 0.0,
            'avg_iou': 0.0,
            'num_gt': len(gt_boxes),
            'num_pred': len(pred_boxes)
        }

        # Compute F1 score
        if len(gt_boxes) == 0 and len(pred_boxes) == 0:
            # Perfect match for empty images
            metrics['f1'] = 1.0
            metrics['precision'] = 1.0
            metrics['recall'] = 1.0
        elif len(gt_boxes) == 0:
            # All predictions are false positives
            metrics['f1'] = 0.0
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
        elif len(pred_boxes) == 0:
            # All ground truths are false negatives
            metrics['f1'] = 0.0
            metrics['precision'] = 0.0
            metrics['recall'] = 0.0
        else:
            # Compute IoU matrix
            iou_matrix = box_iou(pred_boxes, gt_boxes)  # [num_pred, num_gt]

            # Compute F1 with IoU threshold 0.5
            tp = 0
            fp = 0
            gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)

            for pred_idx in range(len(pred_boxes)):
                ious = iou_matrix[pred_idx]
                max_iou, max_idx = ious.max(dim=0)

                # Check if labels match and IoU is sufficient
                if max_iou >= 0.5 and pred_labels[pred_idx] == gt_labels[max_idx]:
                    if not gt_matched[max_idx]:
                        tp += 1
                        gt_matched[max_idx] = True
                    else:
                        fp += 1
                else:
                    fp += 1

            fn = (~gt_matched).sum().item()

            # Compute F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            metrics['f1'] = f1
            metrics['precision'] = precision
            metrics['recall'] = recall

            # Compute average IoU (for matched boxes)
            if tp > 0:
                matched_ious = []
                for pred_idx in range(len(pred_boxes)):
                    ious = iou_matrix[pred_idx]
                    max_iou, max_idx = ious.max(dim=0)
                    if max_iou >= 0.5 and pred_labels[pred_idx] == gt_labels[max_idx]:
                        matched_ious.append(max_iou.item())
                metrics['avg_iou'] = np.mean(matched_ious) if matched_ious else 0.0
            else:
                metrics['avg_iou'] = 0.0

            # Compute mAP for this image (simplified single-image AP)
            # For a single image, we compute precision-recall curve
            if len(pred_boxes) > 0:
                # Sort predictions by score
                sorted_indices = torch.argsort(pred_scores, descending=True)
                sorted_pred_boxes = pred_boxes[sorted_indices]
                sorted_pred_labels = pred_labels[sorted_indices]

                # Compute TP/FP for each prediction
                tp_list = []
                fp_list = []
                gt_matched_map = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)

                for pred_idx in range(len(sorted_pred_boxes)):
                    if len(gt_boxes) > 0:
                        ious = box_iou(sorted_pred_boxes[pred_idx:pred_idx+1], gt_boxes)[0]
                        max_iou, max_idx = ious.max(dim=0)

                        # Check multiple IoU thresholds
                        for iou_thresh in [0.5, 0.75]:
                            if max_iou >= iou_thresh and sorted_pred_labels[pred_idx] == gt_labels[max_idx]:
                                if not gt_matched_map[max_idx]:
                                    tp_list.append(1)
                                    fp_list.append(0)
                                    gt_matched_map[max_idx] = True
                                else:
                                    tp_list.append(0)
                                    fp_list.append(1)
                                break
                        else:
                            tp_list.append(0)
                            fp_list.append(1)
                    else:
                        tp_list.append(0)
                        fp_list.append(1)

                # Compute AP (simplified)
                if len(tp_list) > 0:
                    tp_cumsum = np.cumsum(tp_list)
                    fp_cumsum = np.cumsum(fp_list)
                    recalls = tp_cumsum / len(gt_boxes) if len(gt_boxes) > 0 else np.zeros_like(tp_cumsum)
                    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

                    # Compute AP using 11-point interpolation
                    ap = 0.0
                    for t in np.linspace(0, 1, 11):
                        mask = recalls >= t
                        if mask.any():
                            ap += precisions[mask].max()
                    ap = ap / 11.0

                    metrics['map_50_95'] = ap
                    metrics['map_50'] = ap  # Simplified
                    metrics['map_75'] = ap  # Simplified

        per_image_metrics[img_idx] = metrics

    return per_image_metrics


def compute_overall_metrics(predictions: List[Dict], targets: List[Dict], model) -> Dict:
    """
    Compute overall metrics using model's metric functions.

    Args:
        predictions: List of predictions
        targets: List of targets
        model: FasterRCNNDetector model

    Returns:
        Dict with overall metrics
    """
    print("\nComputing overall metrics...")

    # Use model's metric computation
    map_50, map_75, map_50_95 = model._compute_map(predictions, targets)
    f1, precision, recall = model._compute_f1(predictions, targets, iou_threshold=0.5)

    metrics = {
        'mAP_50': map_50,
        'mAP_75': map_75,
        'mAP_50_95': map_50_95,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }

    # Print results
    print("\n" + "="*80)
    print("OVERALL TEST RESULTS")
    print("="*80)
    print(f"mAP@0.5:        {metrics['mAP_50']:.4f}")
    print(f"mAP@0.75:       {metrics['mAP_75']:.4f}")
    print(f"mAP@0.5:0.95:   {metrics['mAP_50_95']:.4f}")
    print(f"F1 Score:       {metrics['F1']:.4f}")
    print(f"Precision:      {metrics['Precision']:.4f}")
    print(f"Recall:         {metrics['Recall']:.4f}")
    print("="*80 + "\n")

    return metrics


def rank_images_by_metric(per_image_metrics: Dict[int, Dict], metric_name: str, num_top: int = 5) -> Tuple[List[int], List[int]]:
    """
    Rank images by a specific metric.

    Args:
        per_image_metrics: Dict mapping image_id to metrics
        metric_name: Name of metric to rank by ('f1', 'map_50_95', 'avg_iou')
        num_top: Number of top/bottom images to return

    Returns:
        (best_image_ids, worst_image_ids)
    """
    # Get metric values for all images
    image_ids = list(per_image_metrics.keys())
    metric_values = [per_image_metrics[img_id][metric_name] for img_id in image_ids]

    # Sort by metric value
    sorted_indices = np.argsort(metric_values)

    # Get best and worst
    worst_indices = sorted_indices[:num_top]
    best_indices = sorted_indices[-num_top:][::-1]  # Reverse to get highest first

    best_image_ids = [image_ids[i] for i in best_indices]
    worst_image_ids = [image_ids[i] for i in worst_indices]

    return best_image_ids, worst_image_ids


def visualize_predictions(
    images_list: List[torch.Tensor],
    predictions: List[Dict],
    targets: List[Dict],
    image_ids: List[int],
    per_image_metrics: Dict[int, Dict],
    metric_name: str,
    title_prefix: str,
    output_path: str
):
    """
    Visualize predictions for selected images.

    Args:
        images_list: List of image tensors [3, H, W]
        predictions: List of all predictions
        targets: List of all targets
        image_ids: List of image IDs to visualize
        per_image_metrics: Dict with per-image metrics
        metric_name: Name of metric being visualized
        title_prefix: Prefix for plot title ("Best" or "Worst")
        output_path: Path to save figure
    """
    num_images = len(image_ids)
    fig, axes = plt.subplots(1, num_images, figsize=(5*num_images, 5))

    if num_images == 1:
        axes = [axes]

    for idx, (ax, img_id) in enumerate(zip(axes, image_ids)):
        # Get image, predictions, and targets
        img = images_list[img_id]
        pred = predictions[img_id]
        target = targets[img_id]
        metrics = per_image_metrics[img_id]

        # Convert image to numpy (take first channel for SAR)
        img_np = img[0].numpy()  # Take first channel

        # Normalize for display
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

        # Display image
        ax.imshow(img_np, cmap='gray')

        # Draw ground truth boxes (green)
        gt_boxes = target['boxes'].cpu().numpy()
        gt_labels = target['labels'].cpu().numpy()

        for box, label in zip(gt_boxes, gt_labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='green', facecolor='none',
                label='GT' if idx == 0 else None
            )
            ax.add_patch(rect)

        # Draw predicted boxes (red)
        pred_boxes = pred['boxes'].cpu().numpy()
        pred_labels = pred['labels'].cpu().numpy()
        pred_scores = pred['scores'].cpu().numpy()

        for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2, edgecolor='red', facecolor='none',
                linestyle='--',
                label='Pred' if idx == 0 else None
            )
            ax.add_patch(rect)

            # Add score text
            ax.text(x1, y1-5, f'{score:.2f}', color='red', fontsize=8,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        # Set title with metrics
        title = f"{title_prefix} #{idx+1}\n"
        title += f"F1: {metrics['f1']:.3f} | mAP: {metrics['map_50_95']:.3f}\n"
        title += f"IoU: {metrics['avg_iou']:.3f}"
        ax.set_title(title, fontsize=10)
        ax.axis('off')

    # Add legend
    if num_images > 0:
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_path}")
    plt.close()


def save_results(
    overall_metrics: Dict,
    per_image_metrics: Dict[int, Dict],
    output_dir: str
):
    """
    Save test results to JSON file.

    Args:
        overall_metrics: Overall metrics dict
        per_image_metrics: Per-image metrics dict
        output_dir: Output directory
    """
    results = {
        'overall_metrics': overall_metrics,
        'per_image_metrics': {str(k): v for k, v in per_image_metrics.items()}
    }

    output_path = os.path.join(output_dir, 'test_results.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to: {output_path}")


def main():
    """Main testing function."""
    args = parse_args()

    # Load config
    if args.config is not None:
        print(f"Loading config from: {args.config}")
        config = load_config(args.config, args=[])
    else:
        # Try to get config path from checkpoint directory
        checkpoint_dir = Path(args.checkpoint).parent.parent
        config_candidates = [
            checkpoint_dir / 'config.yaml',
            checkpoint_dir.parent / 'config.yaml',
            'experiments/detection/configs/config_experiment.yaml'
        ]

        config_path = None
        for candidate in config_candidates:
            if candidate.exists():
                config_path = str(candidate)
                break

        if config_path is None:
            raise ValueError("Config file not found. Please specify with --config")

        print(f"Loading config from: {config_path}")
        config = load_config(config_path, args=[])

    # Override config with command-line arguments
    if args.data_path is not None:
        config.DATA.DATA_PATH = args.data_path

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, config, device=args.device)

    # Create test dataloader
    test_dataloader = create_test_dataloader(config, args.batch_size, args.num_workers)

    # Run inference
    predictions, targets, images_list = run_inference(model, test_dataloader, device=args.device)

    # Compute overall metrics
    overall_metrics = compute_overall_metrics(predictions, targets, model)

    # Compute per-image metrics
    per_image_metrics = compute_per_image_metrics(predictions, targets)

    # Save results
    save_results(overall_metrics, per_image_metrics, args.output_dir)

    # Determine which metrics to visualize
    metrics_to_visualize = []
    if args.ranking_metric == 'all':
        metrics_to_visualize = [
            ('f1', 'F1 Score'),
            ('map_50_95', 'mAP@0.5:0.95'),
            ('avg_iou', 'Average IoU')
        ]
    else:
        metric_map = {
            'f1': ('f1', 'F1 Score'),
            'map': ('map_50_95', 'mAP@0.5:0.95'),
            'iou': ('avg_iou', 'Average IoU')
        }
        metrics_to_visualize = [metric_map[args.ranking_metric]]

    # Visualize best and worst for each metric
    print("\nGenerating visualizations...")
    for metric_key, metric_display_name in metrics_to_visualize:
        print(f"\nProcessing metric: {metric_display_name}")

        # Rank images
        best_ids, worst_ids = rank_images_by_metric(
            per_image_metrics,
            metric_key,
            num_top=args.num_visualize
        )

        print(f"Best {metric_display_name} images: {best_ids}")
        print(f"Worst {metric_display_name} images: {worst_ids}")

        # Visualize best
        visualize_predictions(
            images_list,
            predictions,
            targets,
            best_ids,
            per_image_metrics,
            metric_key,
            f"Best {metric_display_name}",
            os.path.join(args.output_dir, f'best_{metric_key}.png')
        )

        # Visualize worst
        visualize_predictions(
            images_list,
            predictions,
            targets,
            worst_ids,
            per_image_metrics,
            metric_key,
            f"Worst {metric_display_name}",
            os.path.join(args.output_dir, f'worst_{metric_key}.png')
        )

    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print(f"  - test_results.json: Detailed metrics")
    print(f"  - best_*.png: Visualizations of best performing images")
    print(f"  - worst_*.png: Visualizations of worst performing images")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
