from .swinv2 import SwinV2Backbone
# from .hivit import HiViTBackbone

def get_backbone(name, pretrained, weights):
    if name == "swinv2_tiny_window8_256":
        return SwinV2Backbone(name, pretrained, weights)
    # elif name == "custom":
    #     return HiViTBackbone()
    else:
        raise ValueError(f"Unknown backbone: {name}")
