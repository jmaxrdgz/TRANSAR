"""
Detection models for Faster R-CNN experiments.
"""

from .backbone_adapter import TimmBackboneAdapter, build_detection_backbone
from .faster_rcnn_wrapper import FasterRCNNDetector

__all__ = [
    'TimmBackboneAdapter',
    'build_detection_backbone',
    'FasterRCNNDetector',
]
