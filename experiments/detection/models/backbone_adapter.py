"""
Backbone adapter to bridge timm models with torchvision detection models.
Converts timm feature extraction to torchvision-compatible format.
"""

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Dict, List, Tuple
import timm


class TimmBackboneAdapter(nn.Module):
    """
    Adapter that wraps timm backbones for use with torchvision detection models.
    Extracts multi-scale features and returns them in NCHW format for YOLO.
    """

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        in_chans: int = 1,
        out_indices: tuple = (0, 1, 2, 3),
        pretrained_weights_path: str = None
    ):
        super().__init__()

        self.out_indices = out_indices if isinstance(out_indices, (list, tuple)) else (out_indices,)

        # Build timm backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_chans,
            out_indices=self.out_indices
        )

        # Load custom weights if provided
        if pretrained_weights_path:
            state_dict = torch.load(pretrained_weights_path, map_location='cpu')
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.'):
                    new_state_dict[k[9:]] = v
                else:
                    new_state_dict[k] = v

            missing, unexpected = self.backbone.load_state_dict(new_state_dict, strict=False)
            if missing:
                print(f"Missing keys: {missing[:5]}...")
            if unexpected:
                print(f"Unexpected keys: {unexpected[:5]}...")

        self.feature_info = self.backbone.feature_info

    @property
    def out_channels(self) -> List[int]:
        """
        Return number of channels for each selected feature stage,
        strictly respecting out_indices.
        """
        return [self.feature_info[i]['num_chs'] for i in self.out_indices]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass returning OrderedDict of features in NCHW format.
        """
        features = self.backbone(x)  # list of tensors

        out = OrderedDict()
        for idx, feat in enumerate(features):
            # Convert NHWC -> NCHW if last dim > 3
            if feat.dim() == 4 and feat.shape[-1] > 3 and feat.shape[1] < feat.shape[-1]:
                feat = feat.permute(0, 3, 1, 2).contiguous()

            out[str(idx)] = feat

        return out

    def freeze_backbone(self, num_blocks_to_unfreeze: int = 0):
        # Freeze all
        for param in self.backbone.parameters():
            param.requires_grad = False

        if num_blocks_to_unfreeze == 0:
            print("Backbone fully frozen")
            return

        # Unfreeze last N blocks (heuristic)
        all_modules = list(self.backbone.named_modules())
        layer_modules = [
            (name, module) for name, module in all_modules
            if any(x in name for x in ['layers', 'stages', 'blocks', 'layer']) and len(name.split('.')) <= 3
        ]

        if not layer_modules:
            # fallback: last fraction of parameters
            all_params = list(self.backbone.parameters())
            num_to_unfreeze = max(1, len(all_params) * num_blocks_to_unfreeze // 4)
            for param in all_params[-num_to_unfreeze:]:
                param.requires_grad = True
            print(f"Unfroze last {num_to_unfreeze} parameters")
            return

        blocks_to_unfreeze = layer_modules[-num_blocks_to_unfreeze:]
        for _, module in blocks_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True

        print(f"Unfroze {num_blocks_to_unfreeze} blocks: {[n for n, _ in blocks_to_unfreeze]}")


def build_detection_backbone(
    backbone_name: str,
    pretrained: bool = True,
    in_chans: int = 1,
    out_indices: Tuple[int, ...] = (0, 1, 2, 3),
    pretrained_weights_path: str = None,
    num_blocks_to_unfreeze: int = 0
) -> TimmBackboneAdapter:
    """
    Factory function to build a detection backbone with optional freezing.
    """
    backbone = TimmBackboneAdapter(
        backbone_name=backbone_name,
        pretrained=pretrained,
        in_chans=in_chans,
        out_indices=out_indices,
        pretrained_weights_path=pretrained_weights_path
    )

    if num_blocks_to_unfreeze >= 0:
        backbone.freeze_backbone(num_blocks_to_unfreeze)

    return backbone


if __name__ == "__main__":
    # Quick test
    x = torch.randn(2, 1, 256, 256)
    backbone = build_detection_backbone(
        backbone_name='swinv2_tiny_window8_256',
        pretrained=False,
        in_chans=1,
        num_blocks_to_unfreeze=1
    )
    features = backbone(x)
    print("Out channels:", backbone.out_channels)
    for name, feat in features.items():
        print(f"{name}: {feat.shape}")
