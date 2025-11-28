import timm
import torch
import torch.nn as nn

class SwinV2Backbone(nn.Module):
    def __init__(self, name, pretrained, weights=None, num_blocks_to_unfreeze=0):
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

        # Apply layer freezing based on num_blocks_to_unfreeze
        self.freeze_layers(num_blocks_to_unfreeze)

    def freeze_layers(self, num_blocks_to_unfreeze: int = 0):
        """
        Freeze backbone layers based on num_blocks_to_unfreeze parameter.

        Args:
            num_blocks_to_unfreeze: Number of stages to unfreeze from the end
                - 0: Freeze all stages (only head trains)
                - 1-4: Unfreeze last N stages
        """
        # First, freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        if num_blocks_to_unfreeze == 0:
            print("✓ Backbone fully frozen - only RCNN head will train")
            return

        # Get all stage modules (layers_0, layers_1, layers_2, layers_3)
        all_modules = list(self.model.named_modules())
        stage_modules = [
            (name, module) for name, module in all_modules
            if name.startswith('layers_')
        ]

        # Unfreeze last N stages
        num_to_unfreeze = min(num_blocks_to_unfreeze, len(stage_modules))
        stages_to_unfreeze = stage_modules[-num_to_unfreeze:]

        for _, module in stages_to_unfreeze:
            for param in module.parameters():
                param.requires_grad = True

        unfrozen_names = [name for name, _ in stages_to_unfreeze]
        print(f"✓ Unfroze {num_to_unfreeze} stage(s): {unfrozen_names}")
        print(f"✓ RCNN head will also train")

    def forward(self, x):
        features = self.model(x)
        # SwinV2 outputs features in [B, H, W, C] format
        # Convert to [B, C, H, W] for PyTorch compatibility
        feature = features[0]
        if feature.dim() == 4 and feature.shape[-1] == self.out_channels:
            feature = feature.permute(0, 3, 1, 2)
        return {"0": feature}
