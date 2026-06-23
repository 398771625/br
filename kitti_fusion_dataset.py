import os
from os.path import join

import cv2
import faiss
import h5py
import numpy as np
import torch
import torch.utils.data as data

from RANSAC import rigidRansac

kitti_seq_split_points = {"00": 3000, "02": 3400, "05": 1000, "06": 600, "08": 1000}


class InferFusionDataset(data.Dataset):
    def __init__(self, seq, dataset_path="./datasets/KITTI/", sample_inteval=1):
        super().__init__()
        self.sample_inteval = sample_inteval
        self.db_split_index = int(kitti_seq_split_points[seq] / sample_inteval)

        bev_imgs = sorted(os.listdir(join(dataset_path, seq, "bev_imgs")))
        self.bev_paths = [join(dataset_path, seq, "bev_imgs", bev_imgs[i]) for i in range(0, len(bev_imgs), sample_inteval)]

        rng_dir = join(dataset_path, seq, "range_imgs")
        if not os.path.isdir(rng_dir):
            raise FileNotFoundError(f"Range image folder not found: {rng_dir}")
        rng_imgs = sorted(os.listdir(rng_dir))
        self.rng_paths = [join(rng_dir, rng_imgs[i]) for i in range(0, len(rng_imgs), sample_inteval)]

        if len(self.bev_paths) != len(self.rng_paths):
            raise RuntimeError("BEV and Range image counts do not match")

        self.poses = np.loadtxt(join(dataset_path, "poses", f"{seq}.txt"))[::sample_inteval]

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


class TrainingFusionDataset(data.Dataset):
    def __init__(self, dataset_path="./datasets/KITTI/", seq="00"):
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

        self.poses = np.loadtxt(join(dataset_path, "poses", f"{seq}.txt"))[:3000]

        self.pos_thres = 5
        self.neg_thres = 7
        self.num_neg = 10

        self.positives = []
        self.negatives = []
        for qi in range(len(self.poses)):
            q_pose = self.poses[qi]
            dists = np.sqrt(np.sum(((q_pose - self.poses) ** 2)[:, [3, 7, 11]], axis=1))
            indexes = np.argsort(dists)

            pos_idx = indexes[np.where(dists[indexes] < self.pos_thres)[0]]
            pos_idx = pos_idx[1:]  # remove itself
            self.positives.append(pos_idx)

            neg_idx = indexes[np.where(dists[indexes] > self.neg_thres)[0]]
            self.negatives.append(neg_idx)

        self.mining = False
        self.cache = None

    def refreshCache(self):
        if self.cache is None:
            raise RuntimeError("cache path not set before refreshCache")
        with h5py.File(self.cache, mode="r") as h5:
            self.h5feat = np.array(h5.get("features"))

    @staticmethod
    def _aug_read(path):
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        mat = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), np.random.randint(0, 360), 1)
        img = cv2.warpAffine(img, mat, img.shape[:2])
        img = img.transpose(2, 0, 1)
        return img.astype(np.float32) / 256.0

    def __getitem__(self, index):
        if self.mining:
            q_feat = self.h5feat[index]

            # farthest positive by feature distance (same behavior as original style)
            pos_feat = self.h5feat[self.positives[index]]
            dis_pos = np.sqrt(np.sum((q_feat.reshape(1, -1) - pos_feat) ** 2, axis=1))
            _hard_pos_idx = np.where(dis_pos == np.max(dis_pos))[0][0]
            pos_idx = np.random.choice(self.positives[index], 1)[0]

            neg_feat = self.h5feat[self.negatives[index].tolist()]
            dis_neg = np.sqrt(np.sum((q_feat.reshape(1, -1) - neg_feat) ** 2, axis=1))
            dis_loss = (-dis_neg) + 0.3
            dis_inc_index_tmp = dis_loss.argsort()[:-self.num_neg - 1 : -1]
            neg_idx = self.negatives[index][dis_inc_index_tmp[: self.num_neg]]
        else:
            pos_idx = self.positives[index][0]
            neg_idx = np.random.choice(np.arange(len(self.negatives[index])).astype(int), self.num_neg)
            neg_idx = self.negatives[index][neg_idx]

        q_bev = self._aug_read(self.bev_paths[index])
        p_bev = self._aug_read(self.bev_paths[pos_idx])
        n_bev = torch.stack([torch.from_numpy(self._aug_read(self.bev_paths[n])) for n in neg_idx], 0)

        q_rng = self._aug_read(self.rng_paths[index])
        p_rng = self._aug_read(self.rng_paths[pos_idx])
        n_rng = torch.stack([torch.from_numpy(self._aug_read(self.rng_paths[n])) for n in neg_idx], 0)

        return q_bev, p_bev, n_bev, q_rng, p_rng, n_rng, index

    def __len__(self):
        return len(self.poses)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None, None, None, None, None, None

    q_bev, p_bev, n_bev, q_rng, p_rng, n_rng, indices = zip(*batch)

    q_bev = data.dataloader.default_collate(np.array(q_bev))
    p_bev = data.dataloader.default_collate(np.array(p_bev))
    q_rng = data.dataloader.default_collate(np.array(q_rng))
    p_rng = data.dataloader.default_collate(np.array(p_rng))

    n_bev = torch.cat(n_bev, 0)
    n_rng = torch.cat(n_rng, 0)

    return q_bev, p_bev, n_bev, q_rng, p_rng, n_rng, list(indices)


def evaluateResults(seq, global_descs, bev_local_feats, dataset, match_results_save_path=None):
    if match_results_save_path is not None:
        os.system("mkdir -p " + match_results_save_path)
        all_errs = []
        bev_local_feats = bev_local_feats.transpose(0, 2, 3, 1)

    gt_thres = 5
    faiss_index = faiss.IndexFlatL2(global_descs.shape[1])
    faiss_index.add(global_descs[: dataset.db_split_index])

    _, predictions = faiss_index.search(global_descs[dataset.db_split_index + int(200 / dataset.sample_inteval) :], 1)

    eval_start_split_point = dataset.db_split_index + int(200 / dataset.sample_inteval)
    all_positives = 0
    tp = 0
    for q_idx, pred in enumerate(predictions):
        query_idx = eval_start_split_point + q_idx
        gt_dis = (dataset.poses[query_idx] - dataset.poses[: dataset.db_split_index]) ** 2
        positives = np.where(np.sum(gt_dis[:, [3, 7, 11]], axis=1) < gt_thres ** 2)[0]
        if len(positives) > 0:
            all_positives += 1
            if pred[0] in positives:
                tp += 1

            if match_results_save_path is not None:
                index = pred[0]

                query_im = dataset[query_idx][0].transpose(1, 2, 0) * 256
                db_im = dataset[index][0].transpose(1, 2, 0) * 256
                query_im = query_im.astype(np.uint8)
                db_im = db_im.astype(np.uint8)

                fast = cv2.FastFeatureDetector_create()
                im_side = db_im.shape[0]

                query_kps = fast.detect(query_im, None)
                db_kps = fast.detect(db_im, None)

                query_des = [bev_local_feats[query_idx][int(kp.pt[1]), int(kp.pt[0])] for kp in query_kps]
                db_des = [bev_local_feats[index][int(kp.pt[1]), int(kp.pt[0])] for kp in db_kps]
                query_des = np.array(query_des)
                db_des = np.array(db_des)

                matcher = cv2.BFMatcher()
                matches = matcher.knnMatch(query_des, db_des, k=2)
                all_match = [m[0] for m in matches]
                points1 = np.float32([query_kps[m.queryIdx].pt for m in all_match])
                points2 = np.float32([db_kps[m.trainIdx].pt for m in all_match])

                H, mask, _ = rigidRansac(
                    (np.array([[im_side // 2, im_side // 2]] - points1) * 0.4),
                    ((np.array([[im_side // 2, im_side // 2]] - points2)) * 0.4),
                )

                q_pose = dataset.poses[query_idx]
                q_pose = np.hstack((q_pose[:12].reshape(3, 4)[:2, :2], q_pose[:12].reshape(3, 4)[:2, 3].reshape(-1, 1)))
                q_pose = np.vstack((q_pose, np.array([[0, 0, 1]])))

                db_pose = dataset.poses[index]
                db_pose = np.hstack((db_pose[:12].reshape(3, 4)[:2, :2], db_pose[:12].reshape(3, 4)[:2, 3].reshape(-1, 1)))
                db_pose = np.vstack((db_pose, np.array([[0, 0, 1]])))

                relative_gt = np.linalg.inv(db_pose).dot(q_pose)
                relative_H = np.vstack((H, np.array([[0, 0, 1]])))

                err = np.linalg.inv(relative_H).dot(relative_gt)
                err_theta = np.abs(np.arctan2(err[0, 1], err[0, 0]) / np.pi * 180)
                err_trans = np.sqrt(err[0, 2] ** 2 + err[1, 2] ** 2)

                all_errs.append([err_trans, err_theta])

    recall_top1 = tp / all_positives

    if match_results_save_path is not None:
        all_errs = np.array(all_errs)
        success_loc = (all_errs[:, 0] < 2) & (all_errs[:, 1] < 5)
        success_rate = np.sum(success_loc) / all_positives
        mean_trans_err = np.mean(all_errs[success_loc, 1])
        mean_rot_err = np.mean(all_errs[success_loc, 0])
        return recall_top1, success_rate, mean_trans_err, mean_rot_err

    return recall_top1
