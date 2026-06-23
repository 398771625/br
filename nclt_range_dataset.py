import os
import numpy as np
import torch.utils.data as data

import faiss

from range_utils import load_range_png


class InferDataset(data.Dataset):
    def __init__(self, seq, dataset_path='./datasets/NCLT/'):
        super().__init__()

        imgs_p = os.listdir(dataset_path + seq + '/range_imgs/')
        imgs_p.sort()
        self.imgs_path = [dataset_path + seq + '/range_imgs/' + i for i in imgs_p]

        self.poses = np.loadtxt(dataset_path + 'poses/' + seq + '.txt')

    def __getitem__(self, index):
        img = load_range_png(self.imgs_path[index])
        return img, index

    def __len__(self):
        return len(self.imgs_path)


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
            query_idx = q_idx
            gt_dis = (datasets[i].poses[query_idx] - datasets[0].poses) ** 2
            positives = np.where(np.sum(gt_dis[:, [4, 8]], axis=1) < gt_thres ** 2)[0]
            if len(positives) > 0:
                all_positives += 1
                if pred[0] in positives:
                    tp += 1

        recall_top1 = tp / all_positives
        recalls_nclt.append(recall_top1)

    return recalls_nclt