"""FusionVLAD top-down and range-sliced spherical projections."""
import numpy as np
from scipy.spatial import cKDTree


def topdown_projection(points, size=64, xy_min=-32.0, xy_max=32.0):
    out = np.zeros((2, size, size), np.float32)
    scale = size / (xy_max - xy_min)
    x = ((points[:, 0] - xy_min) * scale).astype(int); y = ((points[:, 1] - xy_min) * scale).astype(int)
    valid = (x >= 0) & (x < size) & (y >= 0) & (y < size)
    x, y, p = x[valid], y[valid], points[valid]
    max_h = np.full((size, size), -np.inf, np.float32); sums = np.zeros((size,size)); counts = np.zeros((size,size))
    np.maximum.at(max_h, (y, x), p[:, 2]); np.add.at(sums, (y,x), p[:,3]); np.add.at(counts, (y,x), 1)
    occupied = counts > 0; out[0, occupied] = max_h[occupied]; out[1, occupied] = sums[occupied] / counts[occupied]
    return out, int(occupied.sum())


def estimate_normals(xyz, radius=1.5, max_nn=30):
    normals = np.zeros_like(xyz, dtype=np.float32)
    if not len(xyz): return normals
    tree = cKDTree(xyz)
    for i, point in enumerate(xyz):
        ids = tree.query_ball_point(point, radius)[:max_nn]
        if len(ids) < 3: continue
        centered = xyz[ids] - np.mean(xyz[ids], axis=0)
        _, vectors = np.linalg.eigh(centered.T @ centered)
        normals[i] = vectors[:, 0]
    return normals


def spherical_projection(points, size=64, max_range=30.0, slices=5, normal_radius=1.5, normal_max_nn=30):
    output = np.zeros((2*slices, size, size), np.float32)
    xyz = points[:, :3]; ranges = np.linalg.norm(xyz, axis=1)
    valid = (ranges > 1e-6) & (ranges <= max_range); xyz, ranges = xyz[valid], ranges[valid]
    normals = estimate_normals(xyz, normal_radius, normal_max_nn)
    azimuth = np.arctan2(xyz[:,1], xyz[:,0]); elevation = np.arcsin(np.clip(xyz[:,2]/ranges, -1, 1))
    col = np.floor((azimuth + np.pi)/(2*np.pi)*size).astype(int) % size
    row = np.clip(np.floor((elevation + np.pi/2)/np.pi*size).astype(int), 0, size-1)
    layer = np.minimum((ranges / (max_range/slices)).astype(int), slices-1)
    rays = xyz / ranges[:, None]
    normal_norm = np.linalg.norm(normals, axis=1)
    cosine = np.zeros(len(ranges), dtype=np.float32)
    good = normal_norm > 1e-8
    cosine[good] = np.abs(np.sum(rays[good] * normals[good], axis=1) / normal_norm[good])
    angles = np.arccos(np.clip(cosine, 0, 1))
    # Stable nearest-point selection for every (range slice, elevation, azimuth) cell.
    flat = layer * size * size + row * size + col
    order = np.lexsort((ranges, flat)); chosen = order[np.unique(flat[order], return_index=True)[1]]
    output[2 * layer[chosen], row[chosen], col[chosen]] = ranges[chosen]
    output[2 * layer[chosen] + 1, row[chosen], col[chosen]] = angles[chosen]
    occupied = np.any(output[0::2] > 0, axis=0)
    return output, int(occupied.sum())


def make_views(points, settings):
    top, top_cells = topdown_projection(points, settings["size"], settings["xy_min"], settings["xy_max"])
    spherical, spherical_cells = spherical_projection(points, settings["size"], settings["max_range"],
        settings["range_slices"], settings["normal_radius"], settings["normal_max_nn"])
    return top, spherical, {"local_point_count": len(points), "valid_topdown_cells": top_cells,
                             "valid_spherical_cells": spherical_cells}
