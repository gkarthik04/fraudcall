"""Shared helpers: config loading, seeding, small IO utilities."""
import json, os, random, yaml
import numpy as np

def load_config(path="configs/base.yaml", overrides=None):
    """Load a YAML config; if an experiment YAML is given it is merged over base.yaml (shallow per section)."""
    with open("configs/base.yaml", encoding="utf8") as f:
        cfg = yaml.safe_load(f)
    if path != "configs/base.yaml":
        with open(path, encoding="utf8") as f:
            over = yaml.safe_load(f) or {}
        for k, v in over.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed)
    except ImportError:
        pass

def write_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)

def read_json(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)
