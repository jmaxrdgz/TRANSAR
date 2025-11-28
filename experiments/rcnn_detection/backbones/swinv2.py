import timm
import torch
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

        # Load custom weights if provided
        if weights is not None:
            state_dict = torch.load(weights, map_location='cpu')
            # Handle different checkpoint formats
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            elif 'model' in state_dict:
                state_dict = state_dict['model']

            # Filter out keys that don't match the feature extractor
            model_keys = set(self.model.state_dict().keys())
            filtered_state_dict = {}
            for k, v in state_dict.items():
                # Remove 'model.' prefix if present
                clean_key = k.replace('model.', '') if k.startswith('model.') else k
                if clean_key in model_keys:
                    filtered_state_dict[clean_key] = v

            self.model.load_state_dict(filtered_state_dict, strict=False)

        self.out_channels = 768

    def forward(self, x):
        features = self.model(x)
        # SwinV2 outputs features in [B, H, W, C] format
        # Convert to [B, C, H, W] for PyTorch compatibility
        feature = features[0]
        if feature.dim() == 4 and feature.shape[-1] == self.out_channels:
            feature = feature.permute(0, 3, 1, 2)
        return {"0": feature}
