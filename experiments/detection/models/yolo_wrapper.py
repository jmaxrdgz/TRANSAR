"""
Lightning wrapper for YOLO detector with custom backbones.
Supports multi-scale detection with custom backbones via TimmBackboneAdapter.
"""

import torch
import torch.nn as nn
import lightning as L
from torchvision.ops import box_iou, box_convert
from typing import Dict, List, Optional, Tuple
import numpy as np

from .backbone_adapter import TimmBackboneAdapter
from .yolo_head import build_yolo_head


class YOLODetector(L.LightningModule):
    """
    Lightning module wrapping YOLO with custom backbones for object detection.
    """

    def __init__(self, config):
        """
        Args:
            config: Configuration object with detection parameters
        """
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.config = config

        # Input channels for backbone
        backbone_in_chans = config.MODEL.IN_CHANS

        # Build backbone adapter 
        self.backbone_adapter = TimmBackboneAdapter(
            backbone_name=config.MODEL.BACKBONE.NAME,
            pretrained=config.MODEL.BACKBONE.PRETRAINED,
            in_chans=backbone_in_chans,  # Use config channels (1 for SAR)
            out_indices=(3,),  # Only use last feature level
            pretrained_weights_path=config.MODEL.BACKBONE.WEIGHTS,
        )

        # If backbone expects 1 channel but we receive 3 (replicated), add conversion
        self.needs_channel_conversion = (backbone_in_chans == 1)
        if self.needs_channel_conversion:
            # Simple conversion: take first channel (all 3 are identical after replication)
            print("Note: Converting 3-channel input to 1-channel for backbone compatibility")

        # Get backbone output channels for last feature level
        backbone_out_channels = self.backbone_adapter.out_channels
        print(f"Backbone output channels (last feature only): {backbone_out_channels}")

        # Single-scale YOLO detection head
        # Anchors and strides should be configured for single scale in config
        anchors = config.MODEL.ANCHORS if hasattr(config.MODEL, 'ANCHORS') else [[(116, 90), (156, 198), (373, 326)]]
        strides = config.MODEL.STRIDES if hasattr(config.MODEL, 'STRIDES') else [32]

        # Build YOLO detection head for single scale
        self.detection_head = build_yolo_head(
            in_channels=backbone_out_channels,  # List with single element
            num_classes=config.DATA.NUM_CLASSES,
            anchors=anchors,
            strides=strides
        )

        # Apply backbone freezing
        if hasattr(config.MODEL, 'NUM_BLOCKS_TO_UNFREEZE'):
            self.freeze_backbone(config.MODEL.NUM_BLOCKS_TO_UNFREEZE)

        # Loss weights
        self.box_loss_weight = config.MODEL.LOSS_WEIGHTS.BOX if hasattr(config.MODEL, 'LOSS_WEIGHTS') else 0.05
        self.obj_loss_weight = config.MODEL.LOSS_WEIGHTS.OBJ if hasattr(config.MODEL, 'LOSS_WEIGHTS') else 1.0
        self.cls_loss_weight = config.MODEL.LOSS_WEIGHTS.CLS if hasattr(config.MODEL, 'LOSS_WEIGHTS') else 0.5

        # Confidence threshold for inference
        self.conf_threshold = config.MODEL.CONF_THRESHOLD if hasattr(config.MODEL, 'CONF_THRESHOLD') else 0.25
        self.nms_threshold = config.MODEL.NMS_THRESHOLD if hasattr(config.MODEL, 'NMS_THRESHOLD') else 0.45

        # Validation metrics storage
        self.val_predictions = []
        self.val_targets = []

    def forward(self, images, targets=None):
        """
        Forward pass.

        Args:
            images: List of tensors [C, H, W] or tensor [B, C, H, W]
            targets: List of dicts with 'boxes' and 'labels' (training only)

        Returns:
            During training: dict of losses
            During eval: list of predictions
        """
        # Convert list of images to batch tensor if needed
        if isinstance(images, list):
            images = torch.stack(images)

        # Convert channels if needed (3 channels -> 1 channel for SAR backbones)
        if self.needs_channel_conversion and images.shape[1] == 3:
            # Take first channel (all 3 are identical after replication in data loader)
            images = images[:, 0:1, :, :]

        # Forward through backbone
        features = self.backbone_adapter(images)

        # Convert OrderedDict to list (in order)
        features_list = [features[str(i)] for i in range(len(features))]

        # Forward through YOLO head
        predictions = self.detection_head(features_list)

        if self.training and targets is not None:
            # Compute losses
            losses = self.compute_loss(predictions, targets, images.shape[-2:])
            return losses
        else:
            # Decode predictions for inference
            detections = self.detection_head.decode_predictions(
                predictions,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold
            )
            return detections

    def compute_loss(self, predictions: List[torch.Tensor], targets: List[Dict], image_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """
        Compute YOLO loss (box, objectness, classification).

        Args:
            predictions: List of raw predictions from YOLO head
            targets: List of target dicts with 'boxes', 'labels'
            image_size: (H, W) of input images

        Returns:
            Dict with loss components
        """
        device = predictions[0].device
        batch_size = predictions[0].shape[0]

        total_box_loss = 0.0
        total_obj_loss = 0.0
        total_cls_loss = 0.0

        # Build target tensors for each scale
        for scale_idx, pred in enumerate(predictions):
            stride = self.detection_head.strides[scale_idx]
            anchors = getattr(self.detection_head, f'anchors_{scale_idx}')

            # Get grid size
            _, num_anchors, grid_h, grid_w, num_outputs = pred.shape

            # Create target tensors
            target_boxes = torch.zeros((batch_size, num_anchors, grid_h, grid_w, 4), device=device)
            target_obj = torch.zeros((batch_size, num_anchors, grid_h, grid_w), device=device)
            target_cls = torch.zeros((batch_size, num_anchors, grid_h, grid_w, self.detection_head.num_classes), device=device)

            # Process each image in batch
            for batch_idx in range(batch_size):
                if len(targets[batch_idx]['boxes']) == 0:
                    continue

                gt_boxes = targets[batch_idx]['boxes']  # [N, 4] in (x1, y1, x2, y2)
                gt_labels = targets[batch_idx]['labels']  # Already 0-indexed from YOLO format

                # Convert boxes to center format
                gt_boxes_cxcywh = box_convert(gt_boxes, 'xyxy', 'cxcywh')

                # Assign targets to grid cells and anchors
                for gt_idx in range(len(gt_boxes)):
                    gt_box = gt_boxes_cxcywh[gt_idx]
                    gt_label = gt_labels[gt_idx]

                    # Validate label is within valid range
                    assert 0 <= gt_label < self.detection_head.num_classes, \
                        f"Label {gt_label} out of range [0, {self.detection_head.num_classes}). " \
                        f"Original label: {targets[batch_idx]['labels'][gt_idx]}, NUM_CLASSES config: {self.config.DATA.NUM_CLASSES}"

                    # Get grid cell
                    cx, cy, w, h = gt_box
                    grid_x = int(cx / stride)
                    grid_y = int(cy / stride)

                    # Clip to grid bounds
                    grid_x = max(0, min(grid_x, grid_w - 1))
                    grid_y = max(0, min(grid_y, grid_h - 1))

                    # Normalize box to grid scale
                    cx_grid = cx / stride
                    cy_grid = cy / stride
                    w_grid = w / stride
                    h_grid = h / stride

                    # Find best matching anchor
                    gt_wh = torch.tensor([w_grid, h_grid], device=device)
                    anchor_ious = torch.min(gt_wh / (anchors + 1e-9), anchors / (gt_wh + 1e-9)).min(dim=1)[0]
                    best_anchor_idx = anchor_ious.argmax()

                    # Assign target
                    target_obj[batch_idx, best_anchor_idx, grid_y, grid_x] = 1.0

                    # Box target (offset from grid cell)
                    target_boxes[batch_idx, best_anchor_idx, grid_y, grid_x, 0] = cx_grid - grid_x
                    target_boxes[batch_idx, best_anchor_idx, grid_y, grid_x, 1] = cy_grid - grid_y
                    target_boxes[batch_idx, best_anchor_idx, grid_y, grid_x, 2] = torch.log(w_grid / (anchors[best_anchor_idx, 0] + 1e-9) + 1e-9)
                    target_boxes[batch_idx, best_anchor_idx, grid_y, grid_x, 3] = torch.log(h_grid / (anchors[best_anchor_idx, 1] + 1e-9) + 1e-9)

                    # Class target (one-hot)
                    target_cls[batch_idx, best_anchor_idx, grid_y, grid_x, gt_label] = 1.0

            # Parse predictions
            pred_xy = pred[..., 0:2]
            pred_wh = pred[..., 2:4]
            pred_obj = pred[..., 4]
            pred_cls = pred[..., 5:]

            # Compute objectness loss (BCE)
            obj_loss = nn.functional.binary_cross_entropy_with_logits(
                pred_obj,
                target_obj,
                reduction='mean'
            )

            # Only compute box and class loss for positive samples
            pos_mask = target_obj > 0.5

            if pos_mask.sum() > 0:
                # Box loss (MSE on offset)
                box_loss = nn.functional.mse_loss(
                    pred_xy[pos_mask],
                    target_boxes[..., 0:2][pos_mask],
                    reduction='mean'
                ) + nn.functional.mse_loss(
                    pred_wh[pos_mask],
                    target_boxes[..., 2:4][pos_mask],
                    reduction='mean'
                )

                # Class loss (BCE)
                cls_loss = nn.functional.binary_cross_entropy_with_logits(
                    pred_cls[pos_mask],
                    target_cls[pos_mask],
                    reduction='mean'
                )
            else:
                box_loss = torch.tensor(0.0, device=device)
                cls_loss = torch.tensor(0.0, device=device)

            # Accumulate losses
            total_box_loss += box_loss
            total_obj_loss += obj_loss
            total_cls_loss += cls_loss

        # Average over scales
        num_scales = len(predictions)
        total_box_loss /= num_scales
        total_obj_loss /= num_scales
        total_cls_loss /= num_scales

        # Weighted total loss
        total_loss = (
            self.box_loss_weight * total_box_loss +
            self.obj_loss_weight * total_obj_loss +
            self.cls_loss_weight * total_cls_loss
        )

        return {
            'loss': total_loss,
            'box_loss': total_box_loss,
            'obj_loss': total_obj_loss,
            'cls_loss': total_cls_loss
        }

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch: Dict with 'images' and 'targets'
            batch_idx: Batch index

        Returns:
            Total loss
        """
        images = batch['images']
        targets = batch['targets']

        # Forward pass - returns loss dict during training
        loss_dict = self(images, targets)

        # Log losses
        self.log('train/loss', loss_dict['loss'], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/box_loss', loss_dict['box_loss'], on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/obj_loss', loss_dict['obj_loss'], on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/cls_loss', loss_dict['cls_loss'], on_step=True, on_epoch=True, sync_dist=True)

        return loss_dict['loss']

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
        with torch.no_grad():
            predictions = self(images)

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
            iou_threshold=0.5
        )

        # Log metrics
        self.log('val/mAP_50', map_50, prog_bar=True, sync_dist=True)
        self.log('val/mAP_75', map_75, sync_dist=True)
        self.log('val/mAP_50_95', map_50_95, prog_bar=True, sync_dist=True)
        self.log('val/F1', f1_score, prog_bar=True, sync_dist=True)
        self.log('val/precision', precision, sync_dist=True)
        self.log('val/recall', recall, sync_dist=True)

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
        Reused from FasterRCNNDetector for consistency.
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
                total_fp += len(pred_boxes)
                continue
            elif len(pred_boxes) == 0:
                total_fn += len(gt_boxes)
                continue

            # Compute IoU matrix
            iou_matrix = box_iou(pred_boxes, gt_boxes)

            # Track which GT boxes have been matched
            gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)

            # For each prediction, find best matching GT
            for pred_idx in range(len(pred_boxes)):
                ious = iou_matrix[pred_idx]
                max_iou, max_idx = ious.max(dim=0)

                if max_iou >= iou_threshold and pred_labels[pred_idx] == gt_labels[max_idx]:
                    if not gt_matched[max_idx]:
                        total_tp += 1
                        gt_matched[max_idx] = True
                    else:
                        total_fp += 1
                else:
                    total_fp += 1

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
        Reused from FasterRCNNDetector for consistency.
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

        num_classes = self.config.DATA.NUM_CLASSES
        aps_per_threshold = []

        for iou_thresh in iou_thresholds:
            class_aps = []

            for class_id in range(num_classes):  # Labels are 0-indexed (0 to num_classes-1)
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

                # Flatten predictions (filter out empty tensors before concatenating)
                non_empty_pred_boxes = [x for x in all_pred_boxes if len(x) > 0]
                non_empty_pred_scores = [x for x in all_pred_scores if len(x) > 0]

                if len(non_empty_pred_boxes) > 0:
                    pred_boxes_flat = torch.cat(non_empty_pred_boxes)
                    pred_scores_flat = torch.cat(non_empty_pred_scores)
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
                                ious = box_iou(pred_boxes_flat[box_idx:box_idx+1], gt_boxes_img)[0]
                                max_iou, max_idx = ious.max(dim=0)

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
        """
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
            for param in self.backbone_adapter.parameters():
                param.requires_grad = True
            print("Unfroze entire backbone")
        else:
            self.backbone_adapter.freeze_backbone(num_blocks)
