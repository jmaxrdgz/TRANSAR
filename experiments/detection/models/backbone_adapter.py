"""
Backbone adapter to bridge timm models with torchvision detection models.
Converts timm feature extraction to torchvision-compatible format.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List
import timm


class TimmBackboneAdapter(nn.Module):
    """
    Adapter that wraps timm backbones for use with torchvision detection models.
    Extracts multi-scale features and returns them in torchvision format.
    """

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        in_chans: int = 1,
        out_indices: tuple = (0, 1, 2, 3),
        pretrained_weights_path: str = None
    ):
        """
        Args:
            backbone_name: Name of timm model (e.g., 'swinv2_tiny_window8_256')
            pretrained: Whether to use ImageNet pretrained weights
            in_chans: Number of input channels (1 for SAR, 3 for RGB)
            out_indices: Which feature levels to return (1-4, where 4 is deepest)
            pretrained_weights_path: Path to custom pretrained weights
        """
        super().__init__()

        self.out_indices = out_indices

        # Create timm model with feature extraction
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_chans,
            out_indices=out_indices
        )

        # Load custom pretrained weights if provided
        if pretrained_weights_path is not None:
            print(f"Loading pretrained weights from: {pretrained_weights_path}")
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')

            # Handle different state dict formats
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            # Remove any 'backbone.' prefix if present
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_state_dict[k[9:]] = v
                else:
                    new_state_dict[k] = v

            # Load with strict=False to handle missing/extra keys
            missing, unexpected = self.backbone.load_state_dict(new_state_dict, strict=False)
            if missing:
                print(f"Missing keys: {missing[:5]}...")  # Show first 5
            if unexpected:
                print(f"Unexpected keys: {unexpected[:5]}...")

        # Get feature info to determine output channels
        self.feature_info = self.backbone.feature_info

    @property
    def out_channels(self) -> List[int]:
        """Return number of output channels for each feature level."""
        # feature_info is already filtered by out_indices from timm
        return [info['num_chs'] for info in self.feature_info]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning multi-scale features.

        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            OrderedDict of features {name: tensor} for torchvision compatibility
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Convert to OrderedDict with string keys as expected by torchvision
        out = OrderedDict()
        for idx, feat in enumerate(features):
            # Name features as '0', '1', '2', '3' corresponding to different scales
            out[str(idx)] = feat

        return out

    def freeze_backbone(self, num_blocks_to_unfreeze: int = 0):
        """
        Freeze backbone parameters except for the last num_blocks_to_unfreeze blocks.

        Args:
            num_blocks_to_unfreeze: Number of blocks to keep trainable (0 = freeze all)
        """
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False

        if num_blocks_to_unfreeze == 0:
            print("Backbone fully frozen")
            return

        # Unfreeze specific layers based on backbone architecture
        # This is a heuristic and may need adjustment per architecture
        all_modules = list(self.backbone.named_modules())

        # Find layer/block modules (different naming conventions)
        layer_modules = [
            (name, module) for name, module in all_modules
            if any(x in name for x in ['layers', 'stages', 'blocks', 'layer'])
            and len(name.split('.')) <= 3  # Top-level blocks only
        ]

        if not layer_modules:
            # Fallback: unfreeze last N% of all parameters
            all_params = list(self.backbone.parameters())
            num_to_unfreeze = max(1, len(all_params) * num_blocks_to_unfreeze // 4)
            for param in all_params[-num_to_unfreeze:]:
                param.requires_grad = True
            print(f"Unfroze last {num_to_unfreeze} parameters")
            return

        # Unfreeze last N blocks
        blocks_to_unfreeze = layer_modules[-num_blocks_to_unfreeze:]
        for _, module in blocks_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True

        print(f"Unfroze {num_blocks_to_unfreeze} blocks: {[n for n, _ in blocks_to_unfreeze]}")


def build_detection_backbone(
    backbone_name: str,
    pretrained: bool = True,
    in_chans: int = 1,
    out_indices: tuple = (0, 1, 2, 3),
    pretrained_weights_path: str = None,
    num_blocks_to_unfreeze: int = 0
) -> TimmBackboneAdapter:
    """
    Factory function to build a detection backbone with optional freezing.

    Args:
        backbone_name: Name of timm model
        pretrained: Use ImageNet pretrained weights
        in_chans: Number of input channels
        out_indices: Feature levels to extract
        pretrained_weights_path: Path to custom weights
        num_blocks_to_unfreeze: Number of backbone blocks to keep trainable

    Returns:
        Configured TimmBackboneAdapter
    """
    backbone = TimmBackboneAdapter(
        backbone_name=backbone_name,
        pretrained=pretrained,
        in_chans=in_chans,
        out_indices=out_indices,
        pretrained_weights_path=pretrained_weights_path
    )

    # Apply freezing if requested
    if num_blocks_to_unfreeze >= 0:
        backbone.freeze_backbone(num_blocks_to_unfreeze)

    return backbone


if __name__ == "__main__":
    # Test the adapter
    print("Testing TimmBackboneAdapter...")

    # Test with Swin-V2-Tiny
    backbone = build_detection_backbone(
        backbone_name='swinv2_tiny_window8_256',
        pretrained=False,
        in_chans=1,
        num_blocks_to_unfreeze=1
    )

    # Test forward pass
    x = torch.randn(2, 1, 256, 256)
    features = backbone(x)

    print(f"\nBackbone out_channels: {backbone.out_channels}")
    print(f"\nFeature map shapes:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in backbone.parameters())
    trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
