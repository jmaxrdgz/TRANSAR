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

from torchmetrics.detection import MeanAveragePrecision


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
        self.map_metric = MeanAveragePrecision(
            box_format='xyxy',
            iou_type='bbox',
            class_metrics=False
        )

        self.val_predictions = []
        self.val_targets = []
        self.val_images = []

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
        bs = predictions[0].shape[0]
        num_classes = self.detection_head.num_classes

        total_box_loss = 0.
        total_obj_loss = 0.
        total_cls_loss = 0.

        for scale_idx, pred in enumerate(predictions):
            stride = self.detection_head.strides[scale_idx]
            # Anchors are already in grid units (divided by stride in yolo_head.py)
            anchors_grid = getattr(self.detection_head, f'anchors_{scale_idx}').to(device)  # shape (A,2) in grid units

            B, A, GH, GW, _ = pred.shape

            obj_target = torch.zeros((B, A, GH, GW), device=device)
            cls_target = torch.zeros((B, A, GH, GW, num_classes), device=device)
            box_target = torch.zeros((B, A, GH, GW, 4), device=device)

            for b in range(bs):
                if len(targets[b]['boxes']) == 0:
                    continue

                boxes = targets[b]['boxes'].to(device)  # expected cx,cy,w,h normalized
                labels = targets[b]['labels'].to(device)

                for i, (cxn, cyn, wn, hn) in enumerate(boxes):
                    # NOTE: image_size is (H, W)
                    H, W = image_size
                    cx = cxn * W
                    cy = cyn * H
                    w  = wn  * W
                    h  = hn  * H

                    cxg, cyg = cx / stride, cy / stride
                    wg, hg = w / stride, h / stride

                    gi, gj = int(cxg), int(cyg)
                    if gi < 0 or gi >= GW or gj < 0 or gj >= GH:
                        continue

                    gt_wh = torch.tensor([wg, hg], device=device)

                    # --- True IoU anchor matching (anchors_grid vs gt_wh) ---
                    # anchors_grid is (A,2) where [:,0] is width, [:,1] is height (grid units)
                    inter_w = torch.min(anchors_grid[:,0], gt_wh[0])
                    inter_h = torch.min(anchors_grid[:,1], gt_wh[1])
                    inter = inter_w * inter_h
                    union = anchors_grid[:,0]*anchors_grid[:,1] + gt_wh[0]*gt_wh[1] - inter
                    ious = inter / (union + 1e-9)
                    best_a = int(ious.argmax().item())

                    obj_target[b, best_a, gj, gi] = 1.0
                    cls_target[b, best_a, gj, gi, int(labels[i])] = 1.0

                    box_target[b, best_a, gj, gi, 0] = cxg - gi
                    box_target[b, best_a, gj, gi, 1] = cyg - gj
                    box_target[b, best_a, gj, gi, 2] = torch.log(wg / (anchors_grid[best_a,0] + 1e-9) + 1e-9)
                    box_target[b, best_a, gj, gi, 3] = torch.log(hg / (anchors_grid[best_a,1] + 1e-9) + 1e-9)

            pred_xy = torch.sigmoid(pred[..., 0:2])   # tx,ty prob
            pred_wh = pred[..., 2:4]                  # tw,th (raw)
            pred_obj = pred[..., 4]                   # logits assumed
            pred_cls = pred[..., 5:]

            # objectness loss (logits)
            obj_loss = nn.functional.binary_cross_entropy_with_logits(pred_obj, obj_target)

            pos_mask = obj_target == 1

            if pos_mask.sum() > 0:
                tx = box_target[..., 0:2]
                tw = box_target[..., 2:4]

                # box xy: compare probabilities (sigmoid outputs) with targets in [0,1]
                box_loss_xy = nn.functional.binary_cross_entropy(pred_xy[pos_mask], tx[pos_mask])
                # wh: compare raw preds to target log-scale with MSE
                box_loss_wh = nn.functional.mse_loss(pred_wh[pos_mask], tw[pos_mask])

                box_loss = box_loss_xy + box_loss_wh

                cls_loss = nn.functional.binary_cross_entropy_with_logits(
                    pred_cls[pos_mask],
                    cls_target[pos_mask]
                )
            else:
                box_loss = torch.tensor(0., device=device)
                cls_loss = torch.tensor(0., device=device)

            total_box_loss += box_loss
            total_obj_loss += obj_loss
            total_cls_loss += cls_loss

        total_box_loss /= len(predictions)
        total_obj_loss /= len(predictions)
        total_cls_loss /= len(predictions)

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

        fixed_targets = []

        h, w = images_tensor.shape[-2:]

        for t in targets:
            gt_boxes = t['boxes']  # xywhn

            # xywhn -> xyxy absolute
            gt_boxes = box_convert(gt_boxes, in_fmt="cxcywh", out_fmt="xyxy").clone()
            gt_boxes[:, [0, 2]] *= w
            gt_boxes[:, [1, 3]] *= h

            fixed_targets.append({
                "boxes": gt_boxes,
                "labels": t["labels"]
            })

        self.map_metric.update(decoded_predictions, fixed_targets)

        self.val_predictions.extend(decoded_predictions)
        self.val_targets.extend(fixed_targets)
        self.val_images.extend([img.cpu() for img in images_tensor])

    def on_validation_epoch_end(self):
        if len(self.val_predictions) == 0:
            return

        metric_dict = self.map_metric.compute()
        map_50 = float(metric_dict.get('map_50', 0.0))
        map_75 = float(metric_dict.get('map_75', 0.0))
        map_50_95 = float(metric_dict.get('map', 0.0))
        self.map_metric.reset()

        f1, precision, recall = self._compute_f1(self.val_predictions, self.val_targets)

        self.log('val/mAP_50', map_50, prog_bar=True, sync_dist=True)
        self.log('val/mAP_75', map_75, sync_dist=True)
        self.log('val/mAP_50_95', map_50_95, prog_bar=True, sync_dist=True)
        self.log('val/F1', f1, prog_bar=True, sync_dist=True)
        self.log('val/precision', precision, sync_dist=True)
        self.log('val/recall', recall, sync_dist=True)

        self.val_predictions = []
        self.val_targets = []
        self.val_images = []

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
