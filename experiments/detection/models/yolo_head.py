"""
YOLOv5-style detection head (drop-in replacement).

Features:
- Per-scale anchor support (anchors can have different counts per scale)
- Cache grids for each (H, W, scale, device) to avoid recomputing
- YOLOv5-style bias initialization for objectness & class biases
- forward(..., inference=True) to directly return decoded detections
- decode_predictions() returns list[dict] per image: 'boxes' (x1,y1,x2,y2), 'scores', 'labels'
- Returns raw logits during training by default (so loss functions expect logits)
"""

from typing import List, Tuple, Optional, Dict
import math
import torch
import torch.nn as nn
from torchvision.ops import batched_nms


class YOLODetectionHead(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        anchors: Optional[List[List[Tuple[float, float]]]] = None,
        strides: Optional[List[int]] = None,
        anchor_tensors_device: Optional[torch.device] = None,
    ):
        """
        Args:
            in_channels: list of input channel counts per feature scale
            num_classes: number of object classes
            anchors: list (per scale) of list of (w,h) anchor tuples in pixel space.
                     If None, default YOLOv5-small anchors are used.
            strides: list of strides per scale (e.g. [8,16,32]). If None uses [8,16,32]
        """
        super().__init__()
        if anchors is None:
            anchors = [
                [(10, 13), (16, 30), (33, 23)],      # P3/8
                [(30, 61), (62, 45), (59, 119)],     # P4/16
                [(116, 90), (156, 198), (373, 326)]  # P5/32
            ]
        if strides is None:
            strides = [8, 16, 32]

        assert len(in_channels) == len(anchors) == len(strides), \
            "in_channels, anchors and strides must have same length (one per scale)"

        self.num_classes = int(num_classes)
        self.num_scales = len(in_channels)
        self.strides = list(strides)

        # allow different number of anchors per scale
        self.num_anchors_per_scale = [len(a) for a in anchors]
        self.num_outputs = 5 + self.num_classes  # [x,y,w,h,obj, classes...]

        # register anchor buffers normalized by stride (so they match decode math)
        # store also original anchors for clarity
        self.register_buffer('anchors_per_scale', torch.tensor([ [ahw for ahw in scale] for scale in anchors ], dtype=torch.float32))
        # normalized anchors buffers per scale (list, because different sizes)
        for i, (scale_anchors, s) in enumerate(zip(anchors, self.strides)):
            at = torch.tensor(scale_anchors, dtype=torch.float32) / float(s)  # normalized
            self.register_buffer(f'anchors_{i}', at)

        # create per-scale conv heads (1x1)
        heads: List[nn.Module] = []
        for i, ch in enumerate(in_channels):
            na = self.num_anchors_per_scale[i]
            out_ch = na * self.num_outputs
            conv = nn.Conv2d(ch, out_ch, kernel_size=1, stride=1, padding=0)
            heads.append(conv)
        self.detection_heads = nn.ModuleList(heads)

        # grid cache (not a buffer—depends on input shape & device). Keyed by (scale_idx, H, W, device_index)
        self._grid_cache: Dict[Tuple[int, int, int, int], torch.Tensor] = {}

        # initialize weights & biases (YOLOv5 heuristic for faster convergence)
        self._initialize_weights_and_biases()

    def _initialize_weights_and_biases(self):
        # conv weight init
        for m in self.detection_heads:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # YOLOv5-style bias init: objectness bias and class bias
        # Set bias for each head shaped (na, num_outputs)
        for idx, conv in enumerate(self.detection_heads):
            na = self.num_anchors_per_scale[idx]
            if conv.bias is None:
                continue
            # reshape bias to (na, num_outputs)
            b = conv.bias.detach().view(na, -1)  # view is safe because we know layout
            # objectness bias: encourage low objectness at start; typical value ~ -4.5
            b[:, 4] += -4.5
            # class bias: set to log(prior/(1-prior)) where prior ~ 0.6/(num_classes-0.99) (as used in YOLOv5)
            try:
                if self.num_classes > 1:
                    cls_prior = 0.01  # safer small prior if classes many; YOLOv5 uses a heuristic
                    # use a mild positive prior for classes to help early training (avoid -inf)
                    cls_bias_val = math.log(0.6 / (self.num_classes - 0.99)) if self.num_classes > 1 else 0.0
                else:
                    cls_bias_val = 0.0
            except Exception:
                cls_bias_val = 0.0
            # apply to class dims
            if self.num_classes > 0:
                b[:, 5:5 + self.num_classes] += cls_bias_val
            # write back
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)

    def _make_grid(self, H: int, W: int, device: torch.device, scale_idx: int) -> torch.Tensor:
        key = (scale_idx, H, W, device.index if device.index is not None else -1)
        if key in self._grid_cache:
            g = self._grid_cache[key]
            if g.device == device and g.shape[1] == H and g.shape[2] == W:
                return g
        # create new grid [1, H, W, 2] in float
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        grid = torch.stack((grid_x, grid_y), dim=-1).float()  # [H, W, 2]
        # expand later to [na, H, W, 2] when used; keep as [H,W,2] here
        self._grid_cache[key] = grid
        return grid

    def forward(self, features: List[torch.Tensor], inference: bool = False, conf_threshold: float = 0.25, nms_iou: float = 0.45, max_detections: int = 300):
        """
        Forward pass.

        Args:
            features: list of feature maps (one per scale), each [B, C, H, W]
            inference: if True, performs decode & NMS and returns detections (list[dict] per image).
                       If False, returns raw logits per scale in list as [B, na, H, W, 5+num_classes].
            conf_threshold / nms_iou / max_detections: used if inference=True
        Returns:
            If inference=False: List[torch.Tensor] (per scale) shaped [B, na, H, W, num_outputs] (raw logits)
            If inference=True: List[dict] per image with keys 'boxes' (x1,y1,x2,y2), 'scores', 'labels'
        """
        assert len(features) == self.num_scales, f"Expected {self.num_scales} feature maps, got {len(features)}"

        logits_per_scale: List[torch.Tensor] = []
        for i, (feat, conv) in enumerate(zip(features, self.detection_heads)):
            # apply 1x1 conv -> [B, na * num_outputs, H, W]
            pred = conv(feat)
            B, _, H, W = pred.shape
            na = self.num_anchors_per_scale[i]
            pred = pred.view(B, na, self.num_outputs, H, W).permute(0, 1, 3, 4, 2).contiguous()
            logits_per_scale.append(pred)  # raw logits

        if not inference:
            return logits_per_scale

        # inference path: decode + NMS
        return self.decode_predictions(logits_per_scale, conf_threshold=conf_threshold, apply_nms=True, nms_threshold=nms_iou, max_detections=max_detections)

    def decode_predictions(
        self,
        predictions: List[torch.Tensor],
        conf_threshold: float = 0.25,
        apply_nms: bool = True,
        nms_threshold: float = 0.45,
        max_detections: int = 300
    ) -> List[dict]:
        """
        Decode raw predictions and apply NMS.

        Args:
            predictions: list of raw logits (output of forward with inference=False),
                         each shape [B, na, H, W, 5+num_classes]
            conf_threshold: confidence threshold for final score (objectness * class_prob)
            apply_nms: whether to apply per-class NMS
            nms_threshold: IoU threshold
            max_detections: cap per image

        Returns:
            list of dicts per image: {'boxes': [N,4], 'scores':[N], 'labels':[N]}
            Boxes are in pixel space (x1,y1,x2,y2).
        """
        device = predictions[0].device
        batch_size = predictions[0].shape[0]

        batch_results = []

        for b_idx in range(batch_size):
            boxes_list = []
            scores_list = []
            labels_list = []

            for scale_idx, pred in enumerate(predictions):
                # pred: [B, na, H, W, num_outputs]
                pred_img = pred[b_idx]  # [na, H, W, num_outputs]
                na, H, W, no = pred_img.shape
                stride = self.strides[scale_idx]
                anchors = getattr(self, f'anchors_{scale_idx}')  # [na, 2] normalized (anchor/stride)

                # grid: [H, W, 2]
                grid = self._make_grid(H, W, device, scale_idx)  # [H, W, 2]

                # split components
                xy = pred_img[..., 0:2]     # [na, H, W, 2], logits
                wh = pred_img[..., 2:4]     # [na, H, W, 2]
                obj_logit = pred_img[..., 4]  # [na, H, W]
                cls_logits = pred_img[..., 5:]  # [na, H, W, num_classes]

                # apply YOLOv5 decode transforms
                # xy -> sigmoid*2 - 0.5 + grid, then * stride to go to pixels
                xy = (xy.sigmoid() * 2.0 - 0.5)
                # expand grid to shape [na, H, W, 2]
                grid_exp = grid.unsqueeze(0).expand(na, -1, -1, -1)
                xy = (xy + grid_exp) * stride  # [na, H, W, 2]

                # wh -> ((sigmoid*2)^2 * anchors) * stride
                wh = ((wh.sigmoid() * 2.0) ** 2) * anchors.view(na, 1, 1, 2) * stride  # [na, H, W, 2]

                # convert to x1,y1,x2,y2
                x1y1 = xy - wh / 2.0
                x2y2 = xy + wh / 2.0
                boxes = torch.cat((x1y1, x2y2), dim=-1)  # [na, H, W, 4]

                # confidences
                obj = obj_logit.sigmoid()  # [na, H, W]
                cls_prob = cls_logits.sigmoid()  # [na, H, W, num_classes]
                scores = obj.unsqueeze(-1) * cls_prob  # [na, H, W, num_classes]

                # flatten
                boxes = boxes.view(-1, 4)  # [na*H*W, 4]
                scores = scores.view(-1, self.num_classes)  # [na*H*W, num_classes]

                # get best class per box (class-wise scores)
                max_scores, labels = scores.max(dim=1)  # [na*H*W]

                # threshold
                keep_mask = max_scores > conf_threshold
                if not keep_mask.any():
                    continue

                boxes_kept = boxes[keep_mask]
                scores_kept = max_scores[keep_mask]
                labels_kept = labels[keep_mask]

                boxes_list.append(boxes_kept)
                scores_list.append(scores_kept)
                labels_list.append(labels_kept)

            # concat across scales
            if boxes_list:
                boxes_all = torch.cat(boxes_list, dim=0)
                scores_all = torch.cat(scores_list, dim=0)
                labels_all = torch.cat(labels_list, dim=0)
                # NMS per-class (batched_nms)
                if apply_nms and boxes_all.numel() > 0:
                    keep = batched_nms(boxes_all, scores_all, labels_all, nms_threshold)
                    if keep.numel() > max_detections:
                        keep = keep[:max_detections]
                    boxes_all = boxes_all[keep]
                    scores_all = scores_all[keep]
                    labels_all = labels_all[keep]
            else:
                boxes_all = torch.zeros((0, 4), device=device)
                scores_all = torch.zeros((0,), device=device)
                labels_all = torch.zeros((0,), device=device, dtype=torch.long)

            batch_results.append({
                'boxes': boxes_all,
                'scores': scores_all,
                'labels': labels_all
            })

        return batch_results


def build_yolo_head(
    in_channels: List[int],
    num_classes: int,
    anchors: Optional[List[List[Tuple[float, float]]]] = None,
    strides: Optional[List[int]] = None,
) -> YOLODetectionHead:
    """
    Factory to construct YOLODetectionHead with checks.
    """
    if num_classes is None:
        raise ValueError("num_classes cannot be None")
    return YOLODetectionHead(in_channels=in_channels, num_classes=num_classes, anchors=anchors, strides=strides)


if __name__ == "__main__":
    # quick test
    print("Testing YOLODetectionHead replacement...")

    batch_size = 2
    features = [
        torch.randn(batch_size, 256, 32, 32),
        torch.randn(batch_size, 512, 16, 16),
        torch.randn(batch_size, 1024, 8, 8),
    ]

    head = build_yolo_head(in_channels=[256, 512, 1024], num_classes=2)

    print("Head created. Forwarding raw logits...")
    preds = head(features, inference=False)
    for i, p in enumerate(preds):
        print(f"Scale {i}: {p.shape}")  # [B, na, H, W, 5+nc]

    print("Running inference decode (inference=True)...")
    detections = head(features, inference=True, conf_threshold=0.4)
    for i, det in enumerate(detections):
        print(f"Image {i}: {len(det['boxes'])} detections")

    print("Done.")
