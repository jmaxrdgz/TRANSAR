import timm
import torch.nn as nn

class SwinV2Backbone(nn.Module):
    def __init__(self, name, pretrained, weights=None):
        super().__init__()
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(3,)
        )

        # TODO: Load custom weights if provided

        self.out_channels = 768

    def forward(self, x):
        features = self.model(x)
        # SwinV2 outputs features in [B, H, W, C] format
        # Convert to [B, C, H, W] for PyTorch compatibility
        feature = features[0]
        if feature.dim() == 4 and feature.shape[-1] == self.out_channels:
            feature = feature.permute(0, 3, 1, 2)
        return {"0": feature}
