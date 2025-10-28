# config/__init__.py
from utils.config import load_config

def build_config(pretrain=True):
    if pretrain:
        config = load_config("configs/config_pretrain.yaml")
    else:
        config = load_config("configs/config_finetune.yaml")
    return config