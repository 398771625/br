"""Deterministic temporal local-map construction."""
import numpy as np
from .nclt_manifest import scan_path


def voxel_downsample(points, voxel_size):
    if not len(points): return points.astype(np.float32)
    keys = np.floor(points[:, :3] / voxel_size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(first)].astype(np.float32)


def build_local_map(index, manifest, poses, raw_root, raw_loader, window=2, voxel_size=0.2,
                    xy_min=-32.0, xy_max=32.0, max_range=30.0):
    center_inverse = np.linalg.inv(poses[index]); transformed = []
    lo, hi = max(0, index-window), min(len(manifest), index+window+1)
    for neighbor in range(lo, hi):
        points = raw_loader(scan_path(raw_root, manifest.sequence, manifest.source_filenames[neighbor]))
        transform = center_inverse @ poses[neighbor]
        xyz1 = np.column_stack((points[:, :3], np.ones(len(points))))
        xyz = (transform @ xyz1.T).T[:, :3]
        transformed.append(np.column_stack((xyz, points[:, 3])))
    points = voxel_downsample(np.concatenate(transformed), voxel_size)
    radius = np.linalg.norm(points[:, :3], axis=1)
    keep = ((points[:, 0] >= xy_min) & (points[:, 0] < xy_max) &
            (points[:, 1] >= xy_min) & (points[:, 1] < xy_max) & (radius < max_range))
    return points[keep].astype(np.float32)
