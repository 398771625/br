"""YAML configuration utilities with no machine-specific Python paths."""
from pathlib import Path
import yaml


def load_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError(f"Configuration must be a mapping: {path}")
    return cfg


def ensure_dirs(cfg):
    for key in ("views_root", "preview_root", "descriptors_root", "results_root", "run_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
