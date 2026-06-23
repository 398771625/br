import os
from os.path import join

import cv2
import faiss
import numpy as np
import torch.utils.data as data


class InferFusionDataset(data.Dataset):
    def __init__(self, seq, dataset_path="./datasets/NCLT/"):
        super().__init__()

        bev_imgs = sorted(os.listdir(join(dataset_path, seq, "bev_imgs")))
        self.bev_paths = [join(dataset_path, seq, "bev_imgs", x) for x in bev_imgs]

        rng_dir = join(dataset_path, seq, "range_imgs")
        if not os.path.isdir(rng_dir):
            raise FileNotFoundError(f"Range image folder not found: {rng_dir}")
        rng_imgs = sorted(os.listdir(rng_dir))
        self.rng_paths = [join(rng_dir, x) for x in rng_imgs]

        if len(self.bev_paths) != len(self.rng_paths):
            raise RuntimeError("BEV and Range image counts do not match")

        self.poses = np.loadtxt(join(dataset_path, "poses", f"{seq}.txt"))

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


def evaluateResults(global_descs, datasets):
    gt_thres = 5
    faiss_index = faiss.IndexFlatL2(global_descs[0].shape[1])
    faiss_index.add(global_descs[0])

    recalls_nclt = []
    for i in range(1, len(datasets)):
        _, predictions = faiss_index.search(global_descs[i], 1)

        all_positives = 0
        tp = 0
        for q_idx, pred in enumerate(predictions):
            gt_dis = (datasets[i].poses[q_idx] - datasets[0].poses) ** 2
            positives = np.where(np.sum(gt_dis[:, [4, 8]], axis=1) < gt_thres ** 2)[0]
            if len(positives) > 0:
                all_positives += 1
                if pred[0] in positives:
                    tp += 1

        recalls_nclt.append(tp / all_positives)

    return recalls_nclt