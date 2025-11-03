"""
Backbone configurations for SimMIM pretraining.

Provides factory functions to create Swin Transformer backbones
with various configurations (v1/v2, different window sizes).
"""

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


def create_backbone(backbone_name, in_chans=3, pretrained=False):
    """
    Create a Swin backbone from configuration.

    Args:
        backbone_name: Name from BACKBONE_CONFIGS
        in_chans: Number of input channels (1 for SAR, 3 for RGB)
        pretrained: Whether to use ImageNet pretrained weights

    Returns:
        backbone: Swin model
        config: Backbone configuration dict

    Raises:
        ValueError: If backbone_name is not in BACKBONE_CONFIGS
    """
    if backbone_name not in BACKBONE_CONFIGS:
        raise ValueError(f"Unknown backbone: {backbone_name}. Choose from {list(BACKBONE_CONFIGS.keys())}")

    backbone_cfg = BACKBONE_CONFIGS[backbone_name]

    # Create model without classification head
    model = timm.create_model(
        backbone_cfg['model_name'],
        pretrained=pretrained,
        num_classes=0,  # Remove classification head
        global_pool='',  # Remove global pooling
        in_chans=in_chans
    )

    return model, backbone_cfg
