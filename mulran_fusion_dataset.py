import os
from os.path import join

import cv2
import faiss
import numpy as np
import torch.utils.data as data


def load_mulran_pose_file(pose_file):
    """
    Load MulRan matched pose file.

    Supported format:
    timestamp x y z r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz

    If the first line is a header, it will be skipped automatically.
    """
    with open(pose_file, "r") as f:
        first_line = f.readline().strip()

    first_token = first_line.split()[0]

    try:
        float(first_token)
        has_header = False
    except ValueError:
        has_header = True

    if has_header:
        poses = np.loadtxt(pose_file, skiprows=1)
    else:
        poses = np.loadtxt(pose_file)

    if poses.ndim == 1:
        poses = poses.reshape(1, -1)

    if poses.shape[1] < 3:
        raise ValueError(
            f"Pose file {pose_file} should contain at least timestamp, x, y columns, "
            f"but got shape {poses.shape}"
        )

    return poses

class InferFusionDataset(data.Dataset):
    def __init__(self, seq, dataset_path="./datasets/MulRan"):
        super().__init__()
        self.seq = seq

        bev_dir = join(dataset_path, seq, "bev_imgs")
        rng_dir = join(dataset_path, seq, "range_imgs")
        pose_file = join(dataset_path, "poses", f"{seq}.txt")

        bev_imgs = sorted(os.listdir(bev_dir))
        rng_imgs = sorted(os.listdir(rng_dir))

        self.bev_paths = [join(bev_dir, x) for x in bev_imgs]
        self.rng_paths = [join(rng_dir, x) for x in rng_imgs]

        if len(self.bev_paths) != len(self.rng_paths):
            raise RuntimeError(f"BEV and Range image counts do not match for {seq}")

        self.poses = load_mulran_pose_file(pose_file)

        if len(self.poses) != len(self.bev_paths):
            raise RuntimeError(
                f"{seq}: pose count {len(self.poses)} != image count {len(self.bev_paths)}. "
                "Please check whether BEV/Range images were generated from the same kept_indices as the pose file."
            )

    @staticmethod
    def _read_gray_3ch(path):
        img = cv2.imread(path, 0)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        img = img.astype(np.float32) / 256.0
        return img[np.newaxis, :, :].repeat(3, 0)

    def __getitem__(self, index):
        bev = self._read_gray_3ch(self.bev_paths[index])
        rng = self._read_gray_3ch(self.rng_paths[index])
        return bev, rng, index

    def __len__(self):
        return len(self.bev_paths)


def evaluateResults(global_descs, datasets, pos_thres=10.0):
    db_desc = global_descs[0]
    db_set = datasets[0]

    index = faiss.IndexFlatL2(db_desc.shape[1])
    index.add(db_desc)

    recalls_at1 = []
    recalls_at5 = []
    for i in range(1, len(datasets)):
        q_desc = global_descs[i]
        q_set = datasets[i]
        _, predictions = index.search(q_desc, 5)

        all_positives = 0
        tp1 = 0
        tp5 = 0
        for q_idx, pred in enumerate(predictions):
            qxy = q_set.poses[q_idx, [1, 2]]
            dbxy = db_set.poses[:, [1, 2]]
            d2 = np.sum((dbxy - qxy.reshape(1, 2)) ** 2, axis=1)
            positives = np.where(d2 < pos_thres ** 2)[0]
            if len(positives) == 0:
                continue

            all_positives += 1
            if pred[0] in positives:
                tp1 += 1
            if any(p in positives for p in pred[:5]):
                tp5 += 1

        r1 = tp1 / all_positives if all_positives > 0 else 0.0
        r5 = tp5 / all_positives if all_positives > 0 else 0.0
        recalls_at1.append(r1)
        recalls_at5.append(r5)

        print(f"DB {db_set.seq} -> Query {q_set.seq}: Recall@1={r1*100:.2f}, Recall@5={r5*100:.2f}")

    print(f"mean Recall@1: {np.mean(recalls_at1)*100:.2f}")
    print(f"mean Recall@5: {np.mean(recalls_at5)*100:.2f}")
    return recalls_at1, recalls_at5