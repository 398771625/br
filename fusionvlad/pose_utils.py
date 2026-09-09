"""Unambiguous NCLT 3x4 pose parsing and timestamp alignment."""
from pathlib import Path
import numpy as np


def parse_pose_rows(rows):
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim == 1: rows = rows[None, :]
    if rows.ndim != 2 or rows.shape[1] not in (12, 13):
        raise ValueError(f"Pose rows must be 3x4 row-major (12 values), optionally timestamped (13); got {rows.shape}")
    timestamps = rows[:, 0].astype(np.int64) if rows.shape[1] == 13 else None
    values = rows[:, 1:] if timestamps is not None else rows
    poses = np.tile(np.eye(4, dtype=np.float64), (len(rows), 1, 1))
    poses[:, :3, :4] = values.reshape(-1, 3, 4)
    return timestamps, poses


def load_pose_file(path):
    path = Path(path)
    first_data_line = next(
        (line for line in path.read_text(encoding="utf-8").splitlines()
         if line.strip() and not line.lstrip().startswith("#")),
        "",
    )
    rows = np.genfromtxt(path, delimiter="," if "," in first_data_line else None)
    rows = rows[~np.isnan(rows).all(axis=1)] if rows.ndim == 2 else rows
    return parse_pose_rows(rows)


def align_poses(scan_timestamps, pose_timestamps, poses, max_delta=50000):
    scans = np.asarray(scan_timestamps, dtype=np.int64)
    if pose_timestamps is None:
        if len(poses) != len(scans):
            raise RuntimeError(f"Untimestamped pose count {len(poses)} != manifest count {len(scans)}")
        return poses.copy()
    order = np.argsort(pose_timestamps)
    ts, sorted_poses = np.asarray(pose_timestamps)[order], poses[order]
    right = np.searchsorted(ts, scans)
    right = np.clip(right, 0, len(ts) - 1); left = np.clip(right - 1, 0, len(ts) - 1)
    choose_right = np.abs(ts[right] - scans) < np.abs(ts[left] - scans)
    indices = np.where(choose_right, right, left)
    deltas = np.abs(ts[indices] - scans)
    if np.any(deltas > max_delta):
        bad = int(np.argmax(deltas))
        raise RuntimeError(f"Pose alignment failed at manifest index {bad}: nearest timestamp delta {deltas[bad]} > {max_delta}")
    return sorted_poses[indices]


def load_aligned_poses(cfg, manifest):
    path = Path(cfg["paths"]["pose_root"]) / cfg["pose"]["file_pattern"].format(sequence=manifest.sequence)
    timestamps, poses = load_pose_file(path)
    return align_poses(manifest.scan_timestamps, timestamps, poses, cfg["pose"]["max_timestamp_delta"])


def planar_positions(poses):
    poses = np.asarray(poses)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4): raise ValueError("Expected poses [N,4,4]")
    return poses[:, :2, 3]
