import argparse
from os.path import isfile, join

import faiss
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from REIN import REIN
from mulran_fusion_dataset import InferFusionDataset


def get_args():
    parser = argparse.ArgumentParser(description="BEVPlace++ MulRan Evaluation")
    parser.add_argument("--load_from", type=str, required=True)
    parser.add_argument("--ckpt", type=str, default="best", choices=["latest", "best"])
    parser.add_argument("--cacheBatchSize", type=int, default=128)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--mulran_path", type=str, default="./datasets/MulRan")
    parser.add_argument("--mulran_ref_seq", type=str, default="KAIST01")
    parser.add_argument("--mulran_query_seqs", type=str, default="KAIST02,KAIST03")
    parser.add_argument("--mulran_pos_thres", type=float, default=10.0)
    return parser.parse_args()


def infer_bev(eval_set, model, opt, device):
    loader = DataLoader(eval_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False)
    model.eval()
    all_desc = []
    with torch.no_grad():
        for bev, _rng, _idx in tqdm(loader):
            bev = bev.to(device)
            _out1, _local, desc = model(bev)
            all_desc.append(desc.detach().cpu().numpy())
    return np.concatenate(all_desc, axis=0)


def evaluate_mulran(descs, sets, thres=10.0):
    db_desc = descs[0]
    db_set = sets[0]
    index = faiss.IndexFlatL2(db_desc.shape[1])
    index.add(db_desc)

    r1s, r5s = [], []
    for i in range(1, len(sets)):
        q_desc = descs[i]
        q_set = sets[i]
        _, preds = index.search(q_desc, 5)

        tp1 = 0
        tp5 = 0
        total = 0
        db_xy = db_set.poses[:, [1, 2]]
        for q_idx, pred in enumerate(preds):
            q_xy = q_set.poses[q_idx, [1, 2]]
            d2 = np.sum((db_xy - q_xy.reshape(1, 2)) ** 2, axis=1)
            pos = np.where(d2 < thres ** 2)[0]
            if len(pos) == 0:
                continue
            total += 1
            if pred[0] in pos:
                tp1 += 1
            if any(p in pos for p in pred[:5]):
                tp5 += 1

        r1 = tp1 / total if total > 0 else 0.0
        r5 = tp5 / total if total > 0 else 0.0
        r1s.append(r1)
        r5s.append(r5)
        print(f"DB {db_set.seq} -> Query {q_set.seq}: Recall@1={r1*100:.2f}, Recall@5={r5*100:.2f}")

    print(f"mean Recall@1: {np.mean(r1s)*100:.2f}")
    print(f"mean Recall@5: {np.mean(r5s)*100:.2f}")


def main():
    opt = get_args()
    device = torch.device("cuda")

    model = REIN().to(device)
    ckpt_file = join(opt.load_from, "model_best.pth.tar" if opt.ckpt == "best" else "checkpoint.pth.tar")
    if not isfile(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")
    checkpoint = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    seqs = [opt.mulran_ref_seq] + [s.strip() for s in opt.mulran_query_seqs.split(",") if s.strip()]
    datasets = [InferFusionDataset(seq=s, dataset_path=opt.mulran_path) for s in seqs]
    descs = [infer_bev(ds, model, opt, device) for ds in datasets]
    evaluate_mulran(descs, datasets, thres=opt.mulran_pos_thres)


if __name__ == "__main__":
    main()