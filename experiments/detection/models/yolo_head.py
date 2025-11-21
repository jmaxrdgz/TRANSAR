"""
YOLOv5-style detection head for multi-scale object detection.
Works with custom backbones via TimmBackboneAdapter.
"""

import torch
import torch.nn as nn
from typing import List, Tuple
import math


class YOLODetectionHead(nn.Module):
    """
    YOLOv5-style detection head that predicts boxes, objectness, and classes.

    Processes multiple feature scales and outputs predictions for each scale.
    """

    def __init__(
        self,
        in_channels: List[int],
        num_classes: int,
        anchors: List[List[Tuple[float, float]]],
        strides: List[int] = [8, 16, 32]
    ):
        """
        Args:
            in_channels: List of input channel counts for each scale [C3, C4, C5]
            num_classes: Number of classes (excluding background)
            anchors: Anchor boxes for each scale, shape [num_scales, num_anchors_per_scale, 2]
                     e.g., [[(10,13), (16,30), (33,23)], [(30,61), (62,45), (59,119)], ...]
            strides: Feature strides for each scale [8, 16, 32]
        """
        super().__init__()

        self.num_classes = num_classes
        self.num_outputs = num_classes + 5  # [x, y, w, h, objectness, ...classes]
        self.num_scales = len(in_channels)
        self.num_anchors_per_scale = len(anchors[0])
        self.strides = strides

        # Register anchors as buffers (not parameters)
        # Normalize anchors by stride for each scale
        for scale_idx, (scale_anchors, stride) in enumerate(zip(anchors, strides)):
            # Anchors are in pixel space, normalize by stride
            anchor_tensor = torch.tensor(scale_anchors, dtype=torch.float32) / stride
            self.register_buffer(f'anchors_{scale_idx}', anchor_tensor)

        # Create detection heads for each scale
        self.detection_heads = nn.ModuleList([
            nn.Conv2d(
                in_channels[i],
                self.num_anchors_per_scale * self.num_outputs,
                kernel_size=1,
                stride=1,
                padding=0
            )
            for i in range(self.num_scales)
        ])

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize detection head weights."""
        for m in self.detection_heads:
            nn.init.normal_(m.weight, mean=0.0, std=0.01)
            nn.init.constant_(m.bias, 0)

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass through detection heads.

        Args:
            features: List of feature maps from backbone, one per scale
                     [B, C3, H3, W3], [B, C4, H4, W4], [B, C5, H5, W5]

        Returns:
            List of predictions for each scale, each with shape:
            [B, num_anchors, H, W, num_outputs]
            where num_outputs = num_classes + 5 (x, y, w, h, objectness, classes)
        """
        outputs = []

        for scale_idx, (feat, head) in enumerate(zip(features, self.detection_heads)):
            # Apply detection head: [B, C, H, W] -> [B, num_anchors * num_outputs, H, W]
            pred = head(feat)

            # Reshape: [B, num_anchors * num_outputs, H, W] -> [B, num_anchors, num_outputs, H, W]
            B, _, H, W = pred.shape
            pred = pred.view(B, self.num_anchors_per_scale, self.num_outputs, H, W)

            # Permute to: [B, num_anchors, H, W, num_outputs]
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()

            outputs.append(pred)

        return outputs

    def decode_predictions(
        self,
        predictions: List[torch.Tensor],
        conf_threshold: float = 0.25,
        apply_nms: bool = True,
        nms_threshold: float = 0.45,
        max_detections: int = 300
    ) -> List[dict]:
        """
        Decode raw predictions to bounding boxes.

        Args:
            predictions: List of raw predictions from forward(), one per scale
            conf_threshold: Confidence threshold for filtering
            apply_nms: Whether to apply NMS
            nms_threshold: IoU threshold for NMS
            max_detections: Maximum detections per image

        Returns:
            List of dicts (one per image) with keys:
                'boxes': [N, 4] in (x1, y1, x2, y2) format
                'scores': [N]
                'labels': [N]
        """
        batch_size = predictions[0].shape[0]
        device = predictions[0].device

        batch_detections = []

        for batch_idx in range(batch_size):
            # Collect predictions from all scales
            all_boxes = []
            all_scores = []
            all_labels = []

            for scale_idx, pred in enumerate(predictions):
                # Get predictions for this image: [num_anchors, H, W, num_outputs]
                pred_img = pred[batch_idx]

                H, W = pred_img.shape[1:3]
                stride = self.strides[scale_idx]
                anchors = getattr(self, f'anchors_{scale_idx}')  # [num_anchors_per_scale, 2]

                # Create grid
                grid_y, grid_x = torch.meshgrid(
                    torch.arange(H, device=device),
                    torch.arange(W, device=device),
                    indexing='ij'
                )
                grid = torch.stack([grid_x, grid_y], dim=-1).float()  # [H, W, 2]

                # Expand grid and anchors for all anchor boxes
                grid = grid.unsqueeze(0).expand(self.num_anchors_per_scale, -1, -1, -1)  # [na, H, W, 2]
                anchors_expanded = anchors.view(self.num_anchors_per_scale, 1, 1, 2).expand(-1, H, W, -1)  # [na, H, W, 2]

                # Parse predictions
                xy = pred_img[..., 0:2]  # [na, H, W, 2]
                wh = pred_img[..., 2:4]  # [na, H, W, 2]
                objectness = pred_img[..., 4:5]  # [na, H, W, 1]
                class_probs = pred_img[..., 5:]  # [na, H, W, num_classes]

                # Decode boxes (YOLOv5 format)
                xy = (xy.sigmoid() * 2 - 0.5 + grid) * stride  # Center in pixel space
                wh = (wh.sigmoid() * 2) ** 2 * anchors_expanded * stride  # Width/height in pixel space

                # Convert to (x1, y1, x2, y2)
                x1y1 = xy - wh / 2
                x2y2 = xy + wh / 2
                boxes = torch.cat([x1y1, x2y2], dim=-1)  # [na, H, W, 4]

                # Compute confidence scores
                objectness = objectness.sigmoid()
                class_probs = class_probs.sigmoid()
                scores = objectness * class_probs  # [na, H, W, num_classes]

                # Get best class for each detection
                max_scores, labels = scores.max(dim=-1)  # [na, H, W]

                # Flatten spatial dimensions
                boxes = boxes.view(-1, 4)  # [na*H*W, 4]
                max_scores = max_scores.view(-1)  # [na*H*W]
                labels = labels.view(-1)  # [na*H*W]

                # Filter by confidence threshold
                mask = max_scores > conf_threshold
                boxes = boxes[mask]
                max_scores = max_scores[mask]
                labels = labels[mask]

                all_boxes.append(boxes)
                all_scores.append(max_scores)
                all_labels.append(labels)

            # Concatenate all scales
            if len(all_boxes) > 0:
                boxes = torch.cat(all_boxes, dim=0)
                scores = torch.cat(all_scores, dim=0)
                labels = torch.cat(all_labels, dim=0)

                # Apply NMS
                if apply_nms and len(boxes) > 0:
                    from torchvision.ops import nms, batched_nms

                    # Use batched NMS (per class)
                    keep = batched_nms(boxes, scores, labels, nms_threshold)

                    # Limit to max detections
                    if len(keep) > max_detections:
                        keep = keep[:max_detections]

                    boxes = boxes[keep]
                    scores = scores[keep]
                    labels = labels[keep]
            else:
                boxes = torch.zeros((0, 4), device=device)
                scores = torch.zeros((0,), device=device)
                labels = torch.zeros((0,), device=device, dtype=torch.long)

            # Add 1 to labels for torchvision compatibility (0 = background)
            labels = labels + 1

            batch_detections.append({
                'boxes': boxes,
                'scores': scores,
                'labels': labels
            })

        return batch_detections


def build_yolo_head(
    in_channels: List[int],
    num_classes: int,
    anchors: List[List[Tuple[float, float]]] = None,
    strides: List[int] = [8, 16, 32]
) -> YOLODetectionHead:
    """
    Factory function to build YOLO detection head.

    Args:
        in_channels: Input channels for each scale
        num_classes: Number of object classes (excluding background)
        anchors: Custom anchors, or None for default YOLOv5 anchors
        strides: Feature strides

    Returns:
        YOLODetectionHead instance
    """
    if anchors is None:
        # Default YOLOv5-small anchors (pixel space)
        anchors = [
            [(10, 13), (16, 30), (33, 23)],      # P3/8
            [(30, 61), (62, 45), (59, 119)],     # P4/16
            [(116, 90), (156, 198), (373, 326)]  # P5/32
        ]

    return YOLODetectionHead(
        in_channels=in_channels,
        num_classes=num_classes,
        anchors=anchors,
        strides=strides
    )


if __name__ == "__main__":
    # Test the YOLO head
    print("Testing YOLODetectionHead...")

    # Create dummy backbone features (3 scales)
    batch_size = 2
    features = [
        torch.randn(batch_size, 256, 32, 32),  # P3: stride 8
        torch.randn(batch_size, 512, 16, 16),  # P4: stride 16
        torch.randn(batch_size, 1024, 8, 8),   # P5: stride 32
    ]

    # Create YOLO head
    head = build_yolo_head(
        in_channels=[256, 512, 1024],
        num_classes=2
    )

    print(f"\nYOLO Head created:")
    print(f"  Num classes: {head.num_classes}")
    print(f"  Num scales: {head.num_scales}")
    print(f"  Anchors per scale: {head.num_anchors_per_scale}")

    # Forward pass
    predictions = head(features)

    print(f"\nPrediction shapes:")
    for i, pred in enumerate(predictions):
        print(f"  Scale {i}: {pred.shape}")

    # Decode predictions
    detections = head.decode_predictions(predictions)

    print(f"\nDecoded detections (batch_size={batch_size}):")
    for i, det in enumerate(detections):
        print(f"  Image {i}: {len(det['boxes'])} detections")
        if len(det['boxes']) > 0:
            print(f"    Box shape: {det['boxes'].shape}")
            print(f"    Score range: [{det['scores'].min():.3f}, {det['scores'].max():.3f}]")
