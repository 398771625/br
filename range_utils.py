import cv2
import numpy as np


def load_range_png(path, depth_scale=100.0, max_range=80.0):
    depth_u16 = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if depth_u16 is None:
        raise RuntimeError('Failed to read range image: {}'.format(path))

    depth_m = depth_u16.astype(np.float32) / depth_scale
    mask = (depth_u16 > 0).astype(np.float32)

    depth_m[mask == 0] = 0.0
    depth_m = np.clip(depth_m, 0.0, max_range)
    depth_norm = depth_m / max_range

    out = np.stack([depth_norm, depth_norm, mask], axis=0).astype(np.float32)
    return out


def random_roll(img):
    shift = np.random.randint(0, img.shape[2])
    return np.roll(img, shift, axis=2)
