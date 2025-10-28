import yaml
import argparse
from types import SimpleNamespace
from copy import deepcopy

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d

def namespace_to_dict(ns):
    if isinstance(ns, SimpleNamespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, list):
        return [namespace_to_dict(x) for x in ns]
    else:
        return ns

def update_dict(d, updates):
    """Update nested dict 'd' using dot notation keys in 'updates'."""
    for key, value in updates.items():
        keys = key.split(".")
        target = d
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = value
    return d

def load_config(default_path: str, args=None):
    # --- Load YAML base config ---
    with open(default_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # --- Parse CLI overrides ---
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_path, help="Path to config file")
    parser.add_argument("--override", nargs="*", default=[], help="Override config params (dot notation)")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    overrides = {}
    for item in args.override:
        key, value = item.split("=")
        try:
            value = eval(value)  # convert numbers, bools, lists, etc.
        except Exception:
            pass
        overrides[key] = value

    # --- Apply overrides ---
    cfg_dict = update_dict(cfg_dict, overrides)

    return dict_to_namespace(cfg_dict)