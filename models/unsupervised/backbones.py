"""
Backbone configurations for SimMIM pretraining.

Provides factory functions to create Swin Transformer backbones
with various configurations (v1/v2, different window sizes).
"""

import torch
import timm


# -----------------------------
# --- Backbone Factory ---
# -----------------------------
BACKBONE_CONFIGS = {
    # Swin v1 - Fixed resolution
    'swin_tiny': {
        'model_name': 'swin_tiny_patch4_window7_224',
        'img_size': 224,
        'window_size': 7,
        'patch_size': 4,
        'embed_dim': 96
    },
    'swin_small': {
        'model_name': 'swin_small_patch4_window7_224',
        'img_size': 224,
        'window_size': 7,
        'patch_size': 4,
        'embed_dim': 96
    },

    # Swin v2 - Variable resolution support
    'swinv2_tiny_w8': {
        'model_name': 'swinv2_tiny_window8_256',
        'img_size': 256,
        'window_size': 8,
        'patch_size': 4,
        'embed_dim': 96
    },
    'swinv2_tiny_w16': {
        'model_name': 'swinv2_tiny_window16_256',
        'img_size': 256,
        'window_size': 16,
        'patch_size': 4,
        'embed_dim': 96
    },
    'swinv2_small_w8': {
        'model_name': 'swinv2_small_window8_256',
        'img_size': 256,
        'window_size': 8,
        'patch_size': 4,
        'embed_dim': 96
    },
    'swinv2_small_w16': {
        'model_name': 'swinv2_small_window16_256',
        'img_size': 256,
        'window_size': 16,
        'patch_size': 4,
        'embed_dim': 96
    }
}


def create_backbone(backbone_name, in_chans=3, pretrained=False, img_size=None):
    """
    Create a Swin backbone from configuration.

    Handles channel mismatch when loading ImageNet weights for 1-channel SAR:
    - If in_chans=1 and pretrained=True: Average RGB conv weights to single channel
    - If in_chans=1 and pretrained=False: Create 1-channel model from scratch (efficient)
    - If in_chans=3: Standard RGB model

    Handles resolution mismatch when using pretrained weights at different resolution:
    - Swin v2 uses Log-CPB which smoothly transfers pretrained weights to different resolutions
    - Example: pretrained at 256x256 can be used at 512x512

    Args:
        backbone_name: Name from BACKBONE_CONFIGS
        in_chans: Number of input channels (1 for SAR, 3 for RGB)
        pretrained: Whether to use ImageNet pretrained weights
        img_size: Input image size (if None, uses default from config)

    Returns:
        backbone: Swin model
        config: Backbone configuration dict

    Raises:
        ValueError: If backbone_name is not in BACKBONE_CONFIGS
    """
    if backbone_name not in BACKBONE_CONFIGS:
        raise ValueError(f"Unknown backbone: {backbone_name}. Choose from {list(BACKBONE_CONFIGS.keys())}")

    backbone_cfg = BACKBONE_CONFIGS[backbone_name]
    target_img_size = img_size or backbone_cfg['img_size']

    if pretrained and in_chans != 3:
        # Load with 3 channels first, then adapt
        print(f"[Backbone] Loading ImageNet weights with channel adaptation (3→{in_chans})")
        model = timm.create_model(
            backbone_cfg['model_name'],
            pretrained=True,
            num_classes=0,
            global_pool='',
            img_size=target_img_size,
            in_chans=3  # Load with RGB
        )

        # Adapt first conv layer: average RGB weights to single channel
        first_conv = model.patch_embed.proj  # Swin's first conv layer
        with torch.no_grad():
            # Average across input channels: [out_ch, 3, h, w] -> [out_ch, 1, h, w]
            new_weight = first_conv.weight.mean(dim=1, keepdim=True)
            # Create new conv layer
            new_conv = torch.nn.Conv2d(
                in_channels=in_chans,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )
            new_conv.weight.copy_(new_weight)
            if first_conv.bias is not None:
                new_conv.bias.copy_(first_conv.bias)

            # Replace conv layer
            model.patch_embed.proj = new_conv

        print(f"[Backbone] Adapted first conv: 3 channels → {in_chans} channel (averaged weights)")

    else:
        # Create model directly with desired channels
        model = timm.create_model(
            backbone_cfg['model_name'],
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='',  # Remove global pooling
            img_size=target_img_size,
            in_chans=in_chans
        )

        if not pretrained and in_chans == 1:
            print(f"[Backbone] Created {backbone_name} from scratch with {in_chans} channel (efficient)")

    if img_size and img_size != backbone_cfg['img_size']:
        print(f"[Backbone] Using resolution {img_size}x{img_size} (pretrained at {backbone_cfg['img_size']}x{backbone_cfg['img_size']})")

    return model, backbone_cfg
