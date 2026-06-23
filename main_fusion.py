import argparse
import json
import random
import shutil
import os
from datetime import datetime
from os import makedirs
from os.path import exists, isfile, join

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import kitti_fusion_dataset
import mulran_fusion_dataset
import nclt_fusion_dataset
from fusion_model import FusionModel


class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = torch.sqrt((anchor - positive).pow(2).sum())
        neg_dist = torch.sqrt((anchor - negative).pow(2).sum(1))
        return F.relu(pos_dist - neg_dist + self.margin)


def get_args():
    parser = argparse.ArgumentParser(description="BEVPlace2 Fusion")
    parser.add_argument("--mode", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--fusion_mode", type=str, default="adaptive_weighted", choices=["weighted_sum", "adaptive_weighted"])

    parser.add_argument("--alpha_bev", type=float, default=0.6)
    parser.add_argument("--alpha_range", type=float, default=0.4)

    parser.add_argument("--batchSize", type=int, default=4)
    parser.add_argument("--cacheBatchSize", type=int, default=128)
    parser.add_argument("--nEpochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1024)

    parser.add_argument("--runsPath", type=str, default="./runs_fusion/")
    parser.add_argument("--cachePath", type=str, default="./cache_fusion/")

    parser.add_argument("--bev_ckpt", type=str, default="", help="Path to BEV run directory")
    parser.add_argument("--range_ckpt", type=str, default="", help="Path to Range run directory")
    parser.add_argument("--fusion_ckpt", type=str, default="", help="Path to Fusion run directory")
    parser.add_argument("--ckpt", type=str, default="best", choices=["latest", "best"])

    parser.add_argument("--freeze_branches", action="store_true", help="Freeze BEV/Range branches")
    parser.add_argument("--warmup_epochs", type=int, default=3)
    parser.add_argument("--lambda_teacher", type=float, default=0.2)
    parser.add_argument("--lambda_prior", type=float, default=0.05)
    parser.add_argument("--alpha_prior", type=float, default=0.6)
    parser.add_argument("--nclt_protocol", type=str, default="standard",
                        choices=["standard", "cross_season", "near_season", "custom"])
    parser.add_argument("--nclt_ref_seq", type=str, default="2012-02-04")
    parser.add_argument("--nclt_query_seqs", type=str, default="2012-06-15,2012-09-28,2012-11-16,2013-02-23")
    parser.add_argument("--print_nclt_protocol", action="store_true")
    parser.add_argument("--save_alpha", action="store_true")
    parser.add_argument("--alpha_save_dir", type=str, default="./alpha_stats")
    parser.add_argument("--eval_dataset", type=str, default="kitti_nclt",
                        choices=["kitti_nclt", "nclt", "mulran", "all"])
    parser.add_argument("--mulran_path", type=str, default="./datasets/MulRan")
    parser.add_argument("--mulran_ref_seq", type=str, default="KAIST01")
    parser.add_argument("--mulran_query_seqs", type=str, default="KAIST02,KAIST03")
    parser.add_argument("--mulran_pos_thres", type=float, default=10.0)

    return parser.parse_args()


def saveCheckpoint(state, is_best, model_out_path, filename="checkpoint.pth.tar"):
    filename = join(model_out_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(model_out_path, "model_best.pth.tar"))


def load_checkpoint_to_branch(branch, ckpt_dir, ckpt_type):
    if ckpt_dir == "":
        return False

    ckpt_file = join(ckpt_dir, "model_best.pth.tar" if ckpt_type == "best" else "checkpoint.pth.tar")
    if not isfile(ckpt_file):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_file}")

    checkpoint = torch.load(ckpt_file, map_location="cpu")
    state = checkpoint["state_dict"]

    # original checkpoints are single REIN model checkpoints
    branch.load_state_dict(state, strict=True)
    return True


def load_fusion_checkpoint(model, optimizer, ckpt_dir, ckpt_type, device):
    if ckpt_dir == "":
        return 0, 0.0

    ckpt_file = join(ckpt_dir, "model_best.pth.tar" if ckpt_type == "best" else "checkpoint.pth.tar")
    if not isfile(ckpt_file):
        raise FileNotFoundError(f"Fusion checkpoint not found: {ckpt_file}")

    checkpoint = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint.get("epoch", 0) + 1, checkpoint.get("best_score", 0.0)


def train_epoch(epoch, model, optimizer, train_set, opt, device, writer):
    criterion_triplet = TripletLoss().to(device)
    n_batches = (len(train_set) + opt.batchSize - 1) // opt.batchSize

    use_hard_mining = epoch >= opt.warmup_epochs
    if use_hard_mining:
        print("====> Building Cache for Hard Mining")
        train_set.mining = False
        train_set.cache = join(opt.cachePath, "train_fusion_feat_cache.hdf5")
        with h5py.File(train_set.cache, mode="w") as h5:
            pool_size = model.global_feat_dim
            h5feat = h5.create_dataset("features", [len(train_set), pool_size], dtype=np.float32)
            loader = DataLoader(
                dataset=train_set,
                num_workers=opt.threads,
                batch_size=opt.batchSize,
                shuffle=False,
                collate_fn=kitti_fusion_dataset.collate_fn,
            )
            model.eval()
            with torch.no_grad():
                for q_bev, _p_bev, _n_bev, q_rng, _p_rng, _n_rng, indices in loader:
                    q_bev = q_bev.to(device)
                    q_rng = q_rng.to(device)
                    out = model(q_bev, q_rng)
                    h5feat[indices, :] = out["fused_desc"].detach().cpu().numpy()

        train_set.mining = True
        train_set.refreshCache()
    else:
        train_set.mining = False
        print(f"====> Warmup epoch {epoch}: random negatives (hard mining off)")

    training_loader = DataLoader(
        dataset=train_set,
        num_workers=opt.threads,
        batch_size=opt.batchSize,
        shuffle=True,
        collate_fn=kitti_fusion_dataset.collate_fn,
    )

    model.train()
    if opt.freeze_branches:
        model.freeze_branches()

    epoch_loss = 0.0
    for it, batch in enumerate(training_loader, 1):
        q_bev, p_bev, n_bev, q_rng, p_rng, n_rng, _indices = batch

        B = q_bev.shape[0]
        bev_input = torch.cat([q_bev, p_bev, n_bev], dim=0).to(device)
        rng_input = torch.cat([q_rng, p_rng, n_rng], dim=0).to(device)

        out = model(bev_input, rng_input)
        fused_desc = out["fused_desc"]
        teacher_desc = out["teacher_desc"]
        alpha = out["alpha"]

        fused_q, fused_p, fused_n = torch.split(fused_desc, [B, B, n_bev.shape[0]])
        teacher_q, teacher_p, teacher_n = torch.split(teacher_desc, [B, B, n_bev.shape[0]])
        alpha_q, _, _ = torch.split(alpha, [B, B, n_bev.shape[0]])

        triplet_loss = 0.0
        num_negs = n_bev.shape[0] // B
        for i in range(B):
            tri = criterion_triplet(fused_q[i], fused_p[i], fused_n[num_negs * i : num_negs * (i + 1)])
            triplet_loss += torch.max(tri)
        triplet_loss = triplet_loss / B

        teacher_all = torch.cat([teacher_q, teacher_p, teacher_n], dim=0)
        teacher_align = (1.0 - F.cosine_similarity(fused_desc, teacher_all, dim=1)).mean()
        prior_loss = ((alpha_q - opt.alpha_prior) ** 2).mean()

        loss = triplet_loss + opt.lambda_teacher * teacher_align + opt.lambda_prior * prior_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        if it % 50 == 0 or n_batches <= 10:
            print(
                f"==> Epoch[{epoch}]({it}/{n_batches}): loss={loss.item():.4f}, "
                f"triplet={triplet_loss.item():.4f}, teacher={teacher_align.item():.4f}, prior={prior_loss.item():.4f}"
            )
            step = (epoch * n_batches) + it
            writer.add_scalar("Train/Loss", loss.item(), step)
            writer.add_scalar("Train/Triplet", triplet_loss.item(), step)
            writer.add_scalar("Train/Teacher", teacher_align.item(), step)
            writer.add_scalar("Train/Prior", prior_loss.item(), step)
            writer.add_scalar("Train/AlphaMean", alpha_q.mean().item(), step)

    avg_loss = epoch_loss / n_batches
    writer.add_scalar("Train/AvgLoss", avg_loss, epoch)
    print(f"===> Epoch {epoch} Complete: Avg. Loss {avg_loss:.4f}")


def infer(eval_set, model, opt, device, return_local_feats=False):
    loader = DataLoader(eval_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False)

    model.eval()
    all_global = []
    all_local = []
    with torch.no_grad():
        for bev, rng, _idx in tqdm(loader):
            bev = bev.to(device)
            rng = rng.to(device)
            out = model(bev, rng)
            all_global.append(out["fused_desc"].detach().cpu().numpy())
            if return_local_feats:
                all_local.append(out["bev_local_feats"].detach().cpu().numpy())

    all_global = np.concatenate(all_global, axis=0)
    if return_local_feats:
        return np.concatenate(all_local, axis=0), all_global
    return all_global

def infer_with_alpha(eval_set, model, opt, device):
    loader = DataLoader(eval_set, num_workers=opt.threads, batch_size=opt.cacheBatchSize, shuffle=False)

    model.eval()
    all_global = []
    all_alpha = []
    with torch.no_grad():
        for bev, rng, _idx in tqdm(loader):
            bev = bev.to(device)
            rng = rng.to(device)
            out = model(bev, rng)
            all_global.append(out["fused_desc"].detach().cpu().numpy())
            all_alpha.append(out["alpha"].detach().cpu().numpy())

    return np.concatenate(all_global, axis=0), np.concatenate(all_alpha, axis=0)

def build_nclt_eval_seq(opt):
    if opt.nclt_protocol == "standard":
        return [
            "2012-01-15",
            "2012-02-04",
            "2012-03-17",
            "2012-06-15",
            "2012-09-28",
            "2012-11-16",
            "2013-02-23",
        ]
    if opt.nclt_protocol == "cross_season":
        return ["2012-02-04", "2012-06-15", "2012-09-28", "2012-11-16", "2013-02-23"]
    if opt.nclt_protocol == "near_season":
        return ["2012-02-04", "2012-01-15", "2012-03-17"]
    # custom
    parsed_query_seqs = [x.strip() for x in opt.nclt_query_seqs.split(",") if x.strip()]
    return [opt.nclt_ref_seq] + parsed_query_seqs


def evaluate_kitti(model, opt, device, writer=None, epoch=None):
    recalls_kitti = []
    for seq in ["00", "02", "05", "06", "08"]:
        if seq == "08":
            test_set = kitti_fusion_dataset.InferFusionDataset(seq=seq, sample_inteval=5)
            local_feats, global_descs = infer(test_set, model, opt, device, return_local_feats=True)
            recall_top1, success_rate, mean_trans_err, mean_rot_err = kitti_fusion_dataset.evaluateResults(
                seq, global_descs, local_feats, test_set, "out_imgs/"
            )
        else:
            test_set = kitti_fusion_dataset.InferFusionDataset(seq=seq)
            global_descs = infer(test_set, model, opt, device, return_local_feats=False)
            recall_top1 = kitti_fusion_dataset.evaluateResults(seq, global_descs, None, test_set)

        recalls_kitti.append(recall_top1)
        if writer is not None:
            writer.add_scalars("val", {f"KITTI_{seq}": recall_top1}, epoch)

    if "success_rate" in locals():
        print(
            f"KITTI08 local refinement - success: {success_rate*100:.2f}, "
            f"mean trans err: {mean_trans_err:.2f}, mean rot err: {mean_rot_err:.2f}"
        )

    return recalls_kitti


def evaluate_nclt(model, opt, device, writer=None, epoch=None):
    eval_seq = build_nclt_eval_seq(opt)

    if opt.print_nclt_protocol:
        print(f"====> NCLT protocol: {opt.nclt_protocol}")
        print(f"Database: {eval_seq[0]}")
        print("Queries:")
        for seq in eval_seq[1:]:
            print(f"  - {seq}")

    eval_sets = []
    eval_descs = []
    alpha_stats = []
    for seq in eval_seq:
        ds = nclt_fusion_dataset.InferFusionDataset(seq=seq)
        if opt.save_alpha:
            desc, alpha = infer_with_alpha(ds, model, opt, device)
            alpha_stats.append((seq, float(alpha.mean()), float(alpha.std())))
        else:
            desc = infer(ds, model, opt, device, return_local_feats=False)
        eval_sets.append(ds)
        eval_descs.append(desc)

    if opt.save_alpha:
        os.makedirs(opt.alpha_save_dir, exist_ok=True)
        alpha_out = {
            "protocol": opt.nclt_protocol,
            "database": eval_seq[0],
            "stats": [{"seq": s, "alpha_mean": m, "alpha_std": sd} for s, m, sd in alpha_stats],
        }
        out_file = join(opt.alpha_save_dir, "nclt_alpha_stats.json")
        with open(out_file, "w") as f:
            json.dump(alpha_out, f, indent=2)
        print(f"Saved alpha stats to: {out_file}")

    recalls_nclt = nclt_fusion_dataset.evaluateResults(eval_descs, eval_sets)
    if writer is not None:
        for i, r in enumerate(recalls_nclt):
            writer.add_scalars("val", {f"NCLT_{eval_seq[i+1]}": r}, epoch)

    return recalls_nclt, eval_seq


def evaluate_mulran(model, opt, device):
    seqs = [opt.mulran_ref_seq] + [s.strip() for s in opt.mulran_query_seqs.split(",") if s.strip()]
    eval_sets = []
    eval_descs = []
    for seq in seqs:
        ds = mulran_fusion_dataset.InferFusionDataset(seq=seq, dataset_path=opt.mulran_path)
        desc = infer(ds, model, opt, device, return_local_feats=False)
        eval_sets.append(ds)
        eval_descs.append(desc)
    return mulran_fusion_dataset.evaluateResults(eval_descs, eval_sets, pos_thres=opt.mulran_pos_thres)


def main():
    opt = get_args()
    device = torch.device("cuda")

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    if not exists(opt.cachePath):
        makedirs(opt.cachePath)

    if abs((opt.alpha_bev + opt.alpha_range) - 1.0) > 1e-6:
        print("[Warn] alpha_bev + alpha_range != 1.0; fusion still uses given values.")

    model = FusionModel(
        fusion_mode=opt.fusion_mode,
        alpha_bev=opt.alpha_bev,
        alpha_range=opt.alpha_range,
        freeze_branches=opt.freeze_branches,
    ).to(device)

    if opt.bev_ckpt:
        load_checkpoint_to_branch(model.bev_branch, opt.bev_ckpt, opt.ckpt)
        print(f"Loaded BEV branch from {opt.bev_ckpt}")
    if opt.range_ckpt:
        load_checkpoint_to_branch(model.range_branch, opt.range_ckpt, opt.ckpt)
        print(f"Loaded Range branch from {opt.range_ckpt}")

    # enforce the requested setting for this version
    if opt.freeze_branches:
        model.freeze_branches()

    if opt.mode == "train":
        writer = SummaryWriter(log_dir=join(opt.runsPath, datetime.now().strftime("%b%d_%H-%M-%S")))
        logdir = writer.file_writer.get_logdir()
        if not exists(logdir):
            makedirs(logdir)

        with open(join(logdir, "flags.json"), "w") as f:
            f.write(json.dumps({k: v for k, v in vars(opt).items()}))

        train_set = kitti_fusion_dataset.TrainingFusionDataset()

        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.Adam(trainable_params, lr=opt.lr)

        start_epoch, best_score = load_fusion_checkpoint(model, optimizer, opt.fusion_ckpt, opt.ckpt, device)
        if opt.fusion_ckpt:
            print(f"Resumed fusion model from {opt.fusion_ckpt} at epoch {start_epoch}")

        for epoch in range(start_epoch, opt.nEpochs):
            train_epoch(epoch, model, optimizer, train_set, opt, device, writer)

            print("===> Validation")
            recalls_kitti = evaluate_kitti(model, opt, device, writer=writer, epoch=epoch)
            recalls_nclt, _ = evaluate_nclt(model, opt, device, writer=writer, epoch=epoch)

            mean_recall = float(np.mean(recalls_nclt))
            print(f"===> Mean Recall KITTI: {np.mean(recalls_kitti)*100:.2f}")
            print(f"===> Mean Recall NCLT : {np.mean(recalls_nclt)*100:.2f}")

            is_best = mean_recall > best_score
            if is_best:
                best_score = mean_recall

            saveCheckpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_score": best_score,
                    "recalls": mean_recall,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                logdir,
            )

        writer.close()

    else:
        # test
        if opt.fusion_mode == "adaptive_weighted":
            if opt.fusion_ckpt == "":
                raise ValueError("--fusion_ckpt is required for adaptive_weighted test mode")
            load_fusion_checkpoint(model, optimizer=None, ckpt_dir=opt.fusion_ckpt, ckpt_type=opt.ckpt, device=device)
            print(f"Loaded fusion checkpoint from {opt.fusion_ckpt}")

        if opt.eval_dataset in ["kitti_nclt", "all"]:
            print("===> Running KITTI evaluation")
            recalls_kitti = evaluate_kitti(model, opt, device)
            print("\n################# Recall @ top 1 on KITTI ########################")
            for seq, r in zip(["00", "02", "05", "06", "08"], recalls_kitti):
                print(f"{seq}: {r*100:.2f}")
            print(f"mean: {np.mean(recalls_kitti)*100:.2f}")

        if opt.eval_dataset in ["kitti_nclt", "nclt", "all"]:
            print("\n===> Running NCLT evaluation")
            recalls_nclt, eval_seq = evaluate_nclt(model, opt, device)
            protocol_label = opt.nclt_protocol.replace("_", "-")
            print(f"\n################# Recall @ top 1 on NCLT {protocol_label} ########################")
            db_seq = eval_seq[0]
            for seq, r in zip(eval_seq[1:], recalls_nclt):
                print(f"DB {db_seq} -> Query {seq}: {r*100:.2f}")
            print(f"mean: {np.mean(recalls_nclt)*100:.2f}")

        if opt.eval_dataset in ["mulran", "all"]:
            print("\n===> Running MulRan evaluation")
            evaluate_mulran(model, opt, device)


if __name__ == "__main__":
    main()