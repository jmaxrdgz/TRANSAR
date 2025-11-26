"""
Lightning wrapper for YOLO detector with custom timm backbones (e.g., SwinV2).
"""

import torch
import torch.nn as nn
import lightning as L
from torchvision.ops import box_iou, box_convert
from typing import Dict, List, Optional, Tuple

from .backbone_adapter import TimmBackboneAdapter
from .yolo_head import build_yolo_head

# Try to use torchmetrics for more robust metric computation
try:
    from torchmetrics.detection import MeanAveragePrecision
    TORCHMETRICS_AVAILABLE = True
except ImportError:
    TORCHMETRICS_AVAILABLE = False
    print("Warning: torchmetrics not available, using custom mAP implementation")


class YOLODetector(L.LightningModule):
    """
    Lightning module wrapping YOLO with timm backbones for object detection.
    """

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(ignore=['config'])
        self.config = config

        backbone_in_chans = config.MODEL.IN_CHANS
        self.needs_channel_conversion = False
        if backbone_in_chans == 1:
            self.needs_channel_conversion = True

        # Build backbone adapter
        self.backbone = TimmBackboneAdapter(
            backbone_name=config.MODEL.BACKBONE.NAME,
            pretrained=config.MODEL.BACKBONE.PRETRAINED,
            in_chans=backbone_in_chans,
            out_indices=config.MODEL.BACKBONE.OUT_INDICES
                if hasattr(config.MODEL.BACKBONE, 'OUT_INDICES') else (3,),
            pretrained_weights_path=getattr(config.MODEL.BACKBONE, 'WEIGHTS', None)
        )

        backbone_out_channels = self.backbone.out_channels
        print(f"Backbone output channels: {backbone_out_channels}")

        anchors = getattr(config.MODEL, 'ANCHORS', [[(116, 90), (156, 198), (373, 326)]])
        strides = getattr(config.MODEL, 'STRIDES', [32])

        print(f"Using anchors: {len(anchors)}, strides: {len(strides)}, in_channels: {len(backbone_out_channels)}")

        self.detection_head = build_yolo_head(
            in_channels=backbone_out_channels,
            num_classes=config.DATA.NUM_CLASSES,
            anchors=anchors,
            strides=strides
        )

        # Freeze backbone if specified
        if hasattr(config.MODEL, 'NUM_BLOCKS_TO_UNFREEZE'):
            self.backbone.freeze_backbone(config.MODEL.NUM_BLOCKS_TO_UNFREEZE)

        # Loss weights
        lw = getattr(config.MODEL, 'LOSS_WEIGHTS', None)
        self.box_loss_weight = getattr(lw, 'BOX', 0.05) if lw else 0.05
        self.obj_loss_weight = getattr(lw, 'OBJ', 1.0) if lw else 1.0
        self.cls_loss_weight = getattr(lw, 'CLS', 0.5) if lw else 0.5

        self.conf_threshold = getattr(config.MODEL, 'CONF_THRESHOLD', 0.25)
        self.nms_threshold = getattr(config.MODEL, 'NMS_THRESHOLD', 0.45)

        # Metrics
        if TORCHMETRICS_AVAILABLE:
            self.map_metric = MeanAveragePrecision(
                box_format='xyxy',
                iou_type='bbox',
                class_metrics=False
            )
        else:
            self.map_metric = None

        self.val_predictions = []
        self.val_targets = []
        self.val_images = []

        # For visualization: sample a fixed set of images per epoch
        self.val_viz_sample_size = getattr(config, 'VIZ_SAMPLE_SIZE', 20)
        self.val_viz_epoch_interval = getattr(config, 'VIZ_EPOCH_INTERVAL', 1)
        self.val_viz_images = []
        self.val_viz_predictions = []
        self.val_viz_targets = []
        self.val_viz_indices = None  # Will be set on first validation step
        self.val_batch_counter = 0

    def forward(self, images, targets=None):
        if isinstance(images, list):
            images = torch.stack(images)

        if self.needs_channel_conversion and images.shape[1] == 3:
            images = images[:, 0:1, :, :]

        features = self.backbone(images)
        features_list = [features[str(i)] for i in sorted(map(int, features.keys()))]

        predictions = self.detection_head(features_list)

        if self.training and targets is not None:
            return self.compute_loss(predictions, targets, images.shape[-2:])
        else:
            return self.detection_head.decode_predictions(
                predictions,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold
            )

    def compute_loss(self, predictions: List[torch.Tensor], targets: List[Dict], image_size: Tuple[int, int]):
        device = predictions[0].device
        batch_size = predictions[0].shape[0]

        total_box_loss = 0.0
        total_obj_loss = 0.0
        total_cls_loss = 0.0

        for scale_idx, pred in enumerate(predictions):
            stride = self.detection_head.strides[scale_idx]
            anchors = getattr(self.detection_head, f'anchors_{scale_idx}')

            _, num_anchors, grid_h, grid_w, _ = pred.shape

            target_boxes = torch.zeros((batch_size, num_anchors, grid_h, grid_w, 4), device=device)
            target_obj = torch.zeros((batch_size, num_anchors, grid_h, grid_w), device=device)
            target_cls = torch.zeros((batch_size, num_anchors, grid_h, grid_w, self.detection_head.num_classes), device=device)

            for b in range(batch_size):
                if len(targets[b]['boxes']) == 0:
                    continue

                gt_boxes = targets[b]['boxes']
                gt_labels = targets[b]['labels']
                gt_boxes_cxcywh = box_convert(gt_boxes, 'xyxy', 'cxcywh')

                for gt_idx in range(len(gt_boxes)):
                    cx, cy, w, h = gt_boxes_cxcywh[gt_idx]
                    gt_label = gt_labels[gt_idx]

                    assert 0 <= gt_label < self.detection_head.num_classes

                    grid_x = max(0, min(int(cx / stride), grid_w - 1))
                    grid_y = max(0, min(int(cy / stride), grid_h - 1))

                    cx_grid = cx / stride
                    cy_grid = cy / stride
                    w_grid = w / stride
                    h_grid = h / stride

                    gt_wh = torch.tensor([w_grid, h_grid], device=device)
                    anchor_ious = torch.min(gt_wh / (anchors + 1e-9), anchors / (gt_wh + 1e-9)).min(dim=1)[0]
                    best_anchor_idx = anchor_ious.argmax()

                    target_obj[b, best_anchor_idx, grid_y, grid_x] = 1.0
                    target_boxes[b, best_anchor_idx, grid_y, grid_x, 0] = cx_grid - grid_x
                    target_boxes[b, best_anchor_idx, grid_y, grid_x, 1] = cy_grid - grid_y
                    target_boxes[b, best_anchor_idx, grid_y, grid_x, 2] = torch.log(w_grid / (anchors[best_anchor_idx, 0] + 1e-9) + 1e-9)
                    target_boxes[b, best_anchor_idx, grid_y, grid_x, 3] = torch.log(h_grid / (anchors[best_anchor_idx, 1] + 1e-9) + 1e-9)
                    target_cls[b, best_anchor_idx, grid_y, grid_x, gt_label] = 1.0

            pred_xy = pred[..., 0:2]
            pred_wh = pred[..., 2:4]
            pred_obj = pred[..., 4]
            pred_cls = pred[..., 5:]

            obj_loss = nn.functional.binary_cross_entropy_with_logits(pred_obj, target_obj, reduction='mean')

            pos_mask = target_obj > 0.5
            if pos_mask.sum() > 0:
                box_loss = nn.functional.mse_loss(pred_xy[pos_mask], target_boxes[..., 0:2][pos_mask], reduction='mean') + \
                           nn.functional.mse_loss(pred_wh[pos_mask], target_boxes[..., 2:4][pos_mask], reduction='mean')
                cls_loss = nn.functional.binary_cross_entropy_with_logits(pred_cls[pos_mask], target_cls[pos_mask], reduction='mean')
            else:
                box_loss = torch.tensor(0.0, device=device)
                cls_loss = torch.tensor(0.0, device=device)

            total_box_loss += box_loss
            total_obj_loss += obj_loss
            total_cls_loss += cls_loss

        num_scales = len(predictions)
        total_box_loss /= num_scales
        total_obj_loss /= num_scales
        total_cls_loss /= num_scales

        total_loss = self.box_loss_weight * total_box_loss + \
                     self.obj_loss_weight * total_obj_loss + \
                     self.cls_loss_weight * total_cls_loss

        return {'loss': total_loss, 'box_loss': total_box_loss, 'obj_loss': total_obj_loss, 'cls_loss': total_cls_loss}

    def training_step(self, batch, batch_idx):
        images, targets = batch['images'], batch['targets']
        loss_dict = self(images, targets)
        self.log('train/loss', loss_dict['loss'], prog_bar=True, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/box_loss', loss_dict['box_loss'], on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/obj_loss', loss_dict['obj_loss'], on_step=True, on_epoch=True, sync_dist=True)
        self.log('train/cls_loss', loss_dict['cls_loss'], on_step=True, on_epoch=True, sync_dist=True)
        return loss_dict['loss']

    def validation_step(self, batch, batch_idx):
        images, targets = batch['images'], batch['targets']

        # Compute validation loss by explicitly computing it
        if isinstance(images, list):
            images_tensor = torch.stack(images)
        else:
            images_tensor = images

        if self.needs_channel_conversion and images_tensor.shape[1] == 3:
            images_tensor = images_tensor[:, 0:1, :, :]

        features = self.backbone(images_tensor)
        features_list = [features[str(i)] for i in sorted(map(int, features.keys()))]
        predictions = self.detection_head(features_list)

        loss_dict = self.compute_loss(predictions, targets, images_tensor.shape[-2:])
        self.log('val/loss', loss_dict['loss'], prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/box_loss', loss_dict['box_loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/obj_loss', loss_dict['obj_loss'], on_step=False, on_epoch=True, sync_dist=True)
        self.log('val/cls_loss', loss_dict['cls_loss'], on_step=False, on_epoch=True, sync_dist=True)

        # Run inference for metrics
        with torch.no_grad():
            decoded_predictions = self.detection_head.decode_predictions(
                predictions,
                conf_threshold=self.conf_threshold,
                nms_threshold=self.nms_threshold
            )
        if TORCHMETRICS_AVAILABLE and self.map_metric is not None:
            self.map_metric.update(decoded_predictions, targets)
        self.val_predictions.extend(decoded_predictions)
        self.val_targets.extend(targets)

        # Sample images for visualization (fixed random sample per epoch)
        # Only collect images if they will be plotted this epoch
        # Only rank 0 needs to store images for visualization
        should_visualize_this_epoch = (self.current_epoch + 1) % self.val_viz_epoch_interval == 0

        if self.trainer.is_global_zero and should_visualize_this_epoch:
            batch_size = images_tensor.shape[0]

            # On first batch, determine which global indices to sample
            if self.val_viz_indices is None:
                # Set seed for reproducibility across epochs
                import random
                random.seed(self.current_epoch)

                # Estimate total validation samples (approximate)
                # We'll sample from early batches to avoid storing everything
                max_samples = min(self.val_viz_sample_size * 5, 100)  # Sample from first ~100 images
                self.val_viz_indices = set(random.sample(range(max_samples),
                                                          min(self.val_viz_sample_size, max_samples)))

            # Check if any images in this batch should be stored
            batch_start = self.val_batch_counter
            batch_end = batch_start + batch_size

            for i in range(batch_size):
                global_idx = batch_start + i
                if global_idx in self.val_viz_indices:
                    self.val_viz_images.append(images_tensor[i].cpu())
                    self.val_viz_predictions.append(decoded_predictions[i])
                    self.val_viz_targets.append(targets[i])

            self.val_batch_counter += batch_size

    def on_validation_epoch_end(self):
        if len(self.val_predictions) == 0:
            return

        if TORCHMETRICS_AVAILABLE and self.map_metric is not None:
            metric_dict = self.map_metric.compute()
            map_50 = float(metric_dict.get('map_50', 0.0))
            map_75 = float(metric_dict.get('map_75', 0.0))
            map_50_95 = float(metric_dict.get('map', 0.0))
            self.map_metric.reset()
        else:
            map_50 = map_75 = map_50_95 = 0.0

        f1, precision, recall = self._compute_f1(self.val_predictions, self.val_targets)

        self.log('val/mAP_50', map_50, prog_bar=True, sync_dist=True)
        self.log('val/mAP_75', map_75, sync_dist=True)
        self.log('val/mAP_50_95', map_50_95, prog_bar=True, sync_dist=True)
        self.log('val/F1', f1, prog_bar=True, sync_dist=True)
        self.log('val/precision', precision, sync_dist=True)
        self.log('val/recall', recall, sync_dist=True)

        # Reset validation buffers (for metrics)
        self.val_predictions = []
        self.val_targets = []

        # Reset visualization sample for next epoch
        # Note: val_viz_* will be used by callback before being cleared
        self.val_batch_counter = 0
        self.val_viz_indices = None

    def _compute_f1(self, predictions: List[Dict], targets: List[Dict], iou_threshold: float = 0.5):
        total_tp = total_fp = total_fn = 0
        for pred, target in zip(predictions, targets):
            pred_boxes, pred_labels, pred_scores = pred['boxes'], pred['labels'], pred['scores']
            gt_boxes, gt_labels = target['boxes'], target['labels']

            if len(gt_boxes) == 0 and len(pred_boxes) == 0:
                continue
            elif len(gt_boxes) == 0:
                total_fp += len(pred_boxes)
                continue
            elif len(pred_boxes) == 0:
                total_fn += len(gt_boxes)
                continue

            sorted_idx = torch.argsort(pred_scores, descending=True)
            pred_boxes, pred_labels = pred_boxes[sorted_idx], pred_labels[sorted_idx]

            iou_matrix = box_iou(pred_boxes, gt_boxes)
            gt_matched = torch.zeros(len(gt_boxes), dtype=torch.bool, device=gt_boxes.device)

            for p_idx in range(len(pred_boxes)):
                ious = iou_matrix[p_idx]
                max_iou, max_idx = ious.max(dim=0)
                if max_iou >= iou_threshold and pred_labels[p_idx] == gt_labels[max_idx]:
                    if not gt_matched[max_idx]:
                        total_tp += 1
                        gt_matched[max_idx] = True
                    else:
                        total_fp += 1
                else:
                    total_fp += 1
            total_fn += (~gt_matched).sum().item()

        precision = total_tp / (total_tp + total_fp) if total_tp + total_fp > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if total_tp + total_fn > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        return f1, precision, recall

    def configure_optimizers(self):
        """
        Configure optimizer with differential learning rates for head and backbone.

        Learning rate strategy:
        - Detection head: TRAIN.HEAD_LR (default: 1e-3)
        - Backbone (unfrozen): TRAIN.BACKBONE_LR (default: 1e-4)
        - Frozen backbone params: Excluded from optimizer
        """

        # Get learning rates from config with backward compatibility
        head_lr = getattr(self.config.TRAIN, 'HEAD_LR', self.config.TRAIN.LR)
        backbone_lr = getattr(self.config.TRAIN, 'BACKBONE_LR', self.config.TRAIN.LR)

        # Separate parameters into groups
        head_params = []
        backbone_params = []

        # Collect detection head parameters (only trainable)
        for name, param in self.detection_head.named_parameters():
            if param.requires_grad:
                head_params.append(param)

        # Collect backbone parameters (only unfrozen ones)
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                backbone_params.append(param)

        # Create parameter groups
        param_groups = []

        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr': backbone_lr,
                'name': 'backbone'
            })

        if head_params:
            param_groups.append({
                'params': head_params,
                'lr': head_lr,
                'name': 'head'
            })

        # Verify all trainable params are included
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_in_groups = sum(p.numel() for group in param_groups for p in group['params'])

        assert total_in_groups == total_trainable, \
            f"Parameter mismatch: {total_in_groups} in groups vs {total_trainable} trainable"

        # Create optimizer with parameter groups
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.TRAIN.WEIGHT_DECAY
        )

        # Log parameter group info
        print(f"[Optimizer] Head params: {len(head_params)} with LR={head_lr}")
        print(f"[Optimizer] Backbone params: {len(backbone_params)} with LR={backbone_lr}")

        # Add scheduler if enabled (works automatically with param groups)
        if getattr(self.config.TRAIN, 'SCHEDULER', None) and getattr(self.config.TRAIN.SCHEDULER, 'ENABLED', False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.TRAIN.EPOCHS,
                eta_min=self.config.TRAIN.SCHEDULER.MIN_LR
            )
            return {'optimizer': optimizer,
                    'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1}}
        return optimizer
