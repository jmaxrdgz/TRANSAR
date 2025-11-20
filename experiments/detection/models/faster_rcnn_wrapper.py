"""
Lightning wrapper for Faster R-CNN with custom backbones.
Supports multi-scale detection with FPN and configurable backbones.
"""

import torch
import torch.nn as nn
import lightning as L
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import box_iou
from typing import Dict, List, Optional, Tuple
import numpy as np

from .backbone_adapter import TimmBackboneAdapter


class FasterRCNNDetector(L.LightningModule):
    """
    Lightning module wrapping Faster R-CNN with custom backbones for object detection.
    """

    def __init__(self, config):
        """
        Args:
            config: Configuration object with detection parameters
        """
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.config = config

        # Build backbone adapter
        self.backbone_adapter = TimmBackboneAdapter(
            backbone_name=config.MODEL.BACKBONE.NAME,
            pretrained=config.MODEL.BACKBONE.PRETRAINED,
            in_chans=config.MODEL.IN_CHANS,
            out_indices=(1, 2, 3, 4),  # Multi-scale features for FPN
            pretrained_weights_path=config.MODEL.BACKBONE.WEIGHTS,
        )

        # Get output channels for FPN
        backbone_out_channels = self.backbone_adapter.out_channels

        # Create FPN
        fpn = BackboneWithFPN(
            backbone=self.backbone_adapter,
            return_layers={str(i): str(i) for i in range(len(backbone_out_channels))},
            in_channels_list=backbone_out_channels,
            out_channels=256,  # FPN output channels
        )

        # Create anchor generator
        anchor_sizes = tuple(config.MODEL.RPN.ANCHOR_SIZES)
        aspect_ratios = tuple(config.MODEL.RPN.ASPECT_RATIOS)
        anchor_generator = AnchorGenerator(
            sizes=tuple([anchor_sizes for _ in range(len(backbone_out_channels))]),
            aspect_ratios=tuple([aspect_ratios for _ in range(len(backbone_out_channels))])
        )

        # Create ROI pooler
        roi_pooler = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2
        )

        # Create Faster R-CNN model
        self.model = FasterRCNN(
            backbone=fpn,
            num_classes=config.DATA.NUM_CLASSES,  # Including background
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            # RPN parameters
            rpn_pre_nms_top_n_train=config.MODEL.RPN.PRE_NMS_TOP_N_TRAIN,
            rpn_pre_nms_top_n_test=config.MODEL.RPN.PRE_NMS_TOP_N_TEST,
            rpn_post_nms_top_n_train=config.MODEL.RPN.POST_NMS_TOP_N_TRAIN,
            rpn_post_nms_top_n_test=config.MODEL.RPN.POST_NMS_TOP_N_TEST,
            rpn_nms_thresh=config.MODEL.RPN.NMS_THRESH,
            rpn_fg_iou_thresh=config.MODEL.RPN.FG_IOU_THRESH,
            rpn_bg_iou_thresh=config.MODEL.RPN.BG_IOU_THRESH,
            # Box parameters
            box_score_thresh=config.MODEL.BOX_SCORE_THRESH,
            box_nms_thresh=config.MODEL.BOX_NMS_THRESH,
            box_detections_per_img=config.MODEL.BOX_DETECTIONS_PER_IMG,
        )

        # Apply backbone freezing
        if hasattr(config.MODEL, 'NUM_BLOCKS_TO_UNFREEZE'):
            self.freeze_backbone(config.MODEL.NUM_BLOCKS_TO_UNFREEZE)

        # Validation metrics storage
        self.val_predictions = []
        self.val_targets = []

    def forward(self, images, targets=None):
        """
        Forward pass.

        Args:
            images: List of tensors or tensor [B, C, H, W]
            targets: List of dicts with 'boxes' and 'labels' (training only)

        Returns:
            During training: dict of losses
            During eval: list of predictions
        """
        return self.model(images, targets)

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: Dict with 'images' (list of tensors) and 'targets' (list of dicts)
            batch_idx: Batch index

        Returns:
            Total loss
        """
        images = batch['images']
        targets = batch['targets']

        # Forward pass - returns loss dict during training
        loss_dict = self.model(images, targets)

        # Sum all losses
        losses = sum(loss for loss in loss_dict.values())

        # Log individual losses
        self.log('train/loss', losses, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/loss_classifier', loss_dict['loss_classifier'], on_step=True, on_epoch=True)
        self.log('train/loss_box_reg', loss_dict['loss_box_reg'], on_step=True, on_epoch=True)
        self.log('train/loss_objectness', loss_dict['loss_objectness'], on_step=True, on_epoch=True)
        self.log('train/loss_rpn_box_reg', loss_dict['loss_rpn_box_reg'], on_step=True, on_epoch=True)

        return losses

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch: Dict with 'images' and 'targets'
            batch_idx: Batch index
        """
        images = batch['images']
        targets = batch['targets']

        # Forward pass - returns predictions during eval
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)

        # Store for epoch-end metric computation
        self.val_predictions.extend(predictions)
        self.val_targets.extend(targets)

    def on_validation_epoch_end(self):
        """
        Compute and log metrics at end of validation epoch.
        """
        if len(self.val_predictions) == 0:
            return

        # Compute mAP
        map_50, map_75, map_50_95 = self._compute_map(
            self.val_predictions,
            self.val_targets
        )

        # Compute F1 score
        f1_score, precision, recall = self._compute_f1(
            self.val_predictions,
            self.val_targets,
            iou_threshold=0.5  # Standard IoU threshold for F1
        )

        # Log metrics
        self.log('val/mAP_50', map_50, prog_bar=True)
        self.log('val/mAP_75', map_75)
        self.log('val/mAP_50_95', map_50_95, prog_bar=True)
        self.log('val/F1', f1_score, prog_bar=True)
        self.log('val/precision', precision)
        self.log('val/recall', recall)

        # Clear storage
        self.val_predictions = []
        self.val_targets = []

    def _compute_f1(
        self,
        predictions: List[Dict],
        targets: List[Dict],
        iou_threshold: float = 0.5
    ) -> Tuple[float, float, float]:
        """
        Compute F1 score, precision, and recall.

        Args:
            predictions: List of prediction dicts with 'boxes', 'labels', 'scores'
            targets: List of target dicts with 'boxes', 'labels'
            iou_threshold: IoU threshold for matching predictions to ground truth

        Returns:
            (F1 score, precision, recall)
        """
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for pred, target in zip(predictions, targets):
            pred_boxes = pred['boxes']
            pred_labels = pred['labels']
            gt_boxes = target['boxes']
            gt_labels = target['labels']

            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                continue
            elif len(gt_boxes) == 0:
                # All predictions are false positives
                total_fp += len(pred_boxes)
                continue
            elif len(pred_boxes) == 0:
                # All ground truths are false negatives
                total_fn += len(gt_boxes)
                continue

            # Compute IoU matrix
            iou_matrix = box_iou(pred_boxes, gt_boxes)  # [num_pred, num_gt]

            # Track which GT boxes have been matched
            gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)

            # For each prediction, find best matching GT
            for pred_idx in range(len(pred_boxes)):
                if len(gt_boxes) == 0:
                    total_fp += 1
                    continue

                # Get IoUs for this prediction
                ious = iou_matrix[pred_idx]
                max_iou, max_idx = ious.max(dim=0)

                # Check if labels match (class-aware matching)
                if max_iou >= iou_threshold and pred_labels[pred_idx] == gt_labels[max_idx]:
                    if not gt_matched[max_idx]:
                        # True positive
                        total_tp += 1
                        gt_matched[max_idx] = True
                    else:
                        # Already matched, this is a false positive
                        total_fp += 1
                else:
                    # No good match, false positive
                    total_fp += 1

            # Unmatched GT boxes are false negatives
            total_fn += (~gt_matched).sum().item()

        # Compute metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return f1, precision, recall

    def _compute_map(
        self,
        predictions: List[Dict],
        targets: List[Dict],
        iou_thresholds: List[float] = None
    ) -> Tuple[float, float, float]:
        """
        Compute mean Average Precision (mAP) at different IoU thresholds.

        Args:
            predictions: List of prediction dicts with 'boxes', 'labels', 'scores'
            targets: List of target dicts with 'boxes', 'labels'
            iou_thresholds: IoU thresholds for mAP computation

        Returns:
            (mAP@0.5, mAP@0.75, mAP@0.5:0.95)
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        num_classes = self.config.DATA.NUM_CLASSES
        aps_per_threshold = []

        for iou_thresh in iou_thresholds:
            class_aps = []

            for class_id in range(1, num_classes):  # Skip background class 0
                # Collect all predictions and targets for this class
                all_pred_boxes = []
                all_pred_scores = []
                all_gt_boxes = []
                all_gt_matched = []

                for pred, target in zip(predictions, targets):
                    # Get predictions for this class
                    class_mask = pred['labels'] == class_id
                    pred_boxes = pred['boxes'][class_mask]
                    pred_scores = pred['scores'][class_mask]

                    # Get ground truth for this class
                    gt_mask = target['labels'] == class_id
                    gt_boxes = target['boxes'][gt_mask]

                    all_pred_boxes.append(pred_boxes)
                    all_pred_scores.append(pred_scores)
                    all_gt_boxes.append(gt_boxes)
                    all_gt_matched.append(torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device))

                # Flatten predictions
                if len(all_pred_boxes) > 0 and all(len(x) > 0 for x in all_pred_boxes):
                    pred_boxes_flat = torch.cat(all_pred_boxes)
                    pred_scores_flat = torch.cat(all_pred_scores)
                else:
                    pred_boxes_flat = torch.empty((0, 4), device=self.device)
                    pred_scores_flat = torch.empty((0,), device=self.device)

                # Sort by score
                if len(pred_scores_flat) > 0:
                    sorted_indices = torch.argsort(pred_scores_flat, descending=True)
                    pred_boxes_flat = pred_boxes_flat[sorted_indices]
                    pred_scores_flat = pred_scores_flat[sorted_indices]

                    # Compute AP for this class
                    tp = torch.zeros(len(pred_boxes_flat), device=self.device)
                    fp = torch.zeros(len(pred_boxes_flat), device=self.device)

                    # Match predictions to ground truth
                    box_idx = 0
                    for pred_boxes_img, gt_boxes_img, gt_matched in zip(all_pred_boxes, all_gt_boxes, all_gt_matched):
                        num_pred_this_img = len(pred_boxes_img)

                        for _ in range(num_pred_this_img):
                            if box_idx >= len(pred_boxes_flat):
                                break

                            if len(gt_boxes_img) > 0:
                                # Compute IoU with all GT boxes
                                ious = box_iou(pred_boxes_flat[box_idx:box_idx+1], gt_boxes_img)[0]
                                max_iou, max_idx = ious.max(dim=0)

                                # Check if match
                                if max_iou >= iou_thresh and not gt_matched[max_idx]:
                                    tp[box_idx] = 1
                                    gt_matched[max_idx] = True
                                else:
                                    fp[box_idx] = 1
                            else:
                                fp[box_idx] = 1

                            box_idx += 1

                    # Compute precision-recall curve
                    tp_cumsum = torch.cumsum(tp, dim=0)
                    fp_cumsum = torch.cumsum(fp, dim=0)

                    # Count total GT boxes for this class
                    total_gt = sum(len(gt_boxes) for gt_boxes in all_gt_boxes)

                    if total_gt > 0:
                        recalls = tp_cumsum / total_gt
                        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)

                        # Compute AP using 11-point interpolation
                        ap = self._compute_ap(recalls.cpu().numpy(), precisions.cpu().numpy())
                        class_aps.append(ap)

            # Average over classes
            if len(class_aps) > 0:
                aps_per_threshold.append(np.mean(class_aps))
            else:
                aps_per_threshold.append(0.0)

        # Get mAP at specific thresholds
        map_50 = aps_per_threshold[0] if len(aps_per_threshold) > 0 else 0.0
        map_75 = aps_per_threshold[5] if len(aps_per_threshold) > 5 else 0.0
        map_50_95 = np.mean(aps_per_threshold) if len(aps_per_threshold) > 0 else 0.0

        return map_50, map_75, map_50_95

    @staticmethod
    def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
        """
        Compute Average Precision using 11-point interpolation.

        Args:
            recalls: Array of recall values
            precisions: Array of precision values

        Returns:
            Average Precision score
        """
        # 11-point interpolation
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            mask = recalls >= t
            if mask.any():
                ap += precisions[mask].max()
        return ap / 11.0

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.TRAIN.LR,
            weight_decay=self.config.TRAIN.WEIGHT_DECAY
        )

        # Optional: Learning rate scheduler
        if hasattr(self.config.TRAIN, 'SCHEDULER') and self.config.TRAIN.SCHEDULER.ENABLED:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.TRAIN.EPOCHS,
                eta_min=self.config.TRAIN.SCHEDULER.MIN_LR
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch',
                    'frequency': 1
                }
            }

        return optimizer

    def freeze_backbone(self, num_blocks_to_unfreeze: int = 0):
        """
        Freeze backbone parameters.

        Args:
            num_blocks_to_unfreeze: Number of blocks to keep trainable
        """
        self.backbone_adapter.freeze_backbone(num_blocks_to_unfreeze)

    def unfreeze_backbone(self, num_blocks: int = None):
        """
        Unfreeze backbone parameters.

        Args:
            num_blocks: Number of blocks to unfreeze (None = unfreeze all)
        """
        if num_blocks is None:
            # Unfreeze all
            for param in self.backbone_adapter.parameters():
                param.requires_grad = True
            print("Unfroze entire backbone")
        else:
            self.backbone_adapter.freeze_backbone(num_blocks)
