"""Adapter for the already-validated CVTNet NCLT binary decoder."""
import importlib.util
from pathlib import Path
import numpy as np


def load_cvtnet_get_velo(cvtnet_root):
    module_path = Path(cvtnet_root) / "tools" / "gen_ri_bev.py"
    if not module_path.is_file():
        raise RuntimeError(f"Cannot import CVTNet NCLT decoder: missing {module_path}")
    spec = importlib.util.spec_from_file_location("fusionvlad_cvtnet_decoder", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot create import specification for {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "get_velo"):
        raise RuntimeError(f"Verified decoder file has no get_velo(): {module_path}")
    return module.get_velo


def make_raw_loader(cfg):
    if cfg.get("raw_loader") != "cvtnet_get_velo":
        raise ValueError("Only raw_loader=cvtnet_get_velo is supported; binary format will not be guessed")
    get_velo = load_cvtnet_get_velo(cfg["paths"]["cvtnet_root"])
    def load(path):
        points = np.asarray(get_velo(str(path)), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise RuntimeError(f"get_velo returned invalid shape {points.shape} for {path}")
        if points.shape[1] == 3:
            points = np.column_stack((points, np.zeros(len(points), dtype=np.float32)))
        return points[:, :4]
    return load
