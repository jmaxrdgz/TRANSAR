from .swinv2 import SwinV2Backbone
# from .hivit import HiViTBackbone

def get_backbone(name, pretrained, weights, num_blocks_to_unfreeze=0):
    if name == "swinv2_tiny_window8_256":
        return SwinV2Backbone(name, pretrained, weights, num_blocks_to_unfreeze)
    # elif name == "custom":
    #     return HiViTBackbone()
    else:
        raise ValueError(f"Unknown backbone: {name}")
