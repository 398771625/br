"""Cached-view datasets, source-only tuples, normalization, and paired yaw augmentation."""
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

REQUIRED_VIEW_KEYS=("topdown","spherical","frame_id","scan_timestamp","source_filename")

def load_view(path, return_metadata=False):
    with np.load(path,allow_pickle=False) as data:
        missing=[k for k in REQUIRED_VIEW_KEYS if k not in data]
        if missing: raise ValueError(f"{path}: missing cached-view keys {missing}")
        top=np.asarray(data["topdown"],np.float32); spherical=np.asarray(data["spherical"],np.float32)
        metadata = (int(data["frame_id"]), int(data["scan_timestamp"]), str(data["source_filename"]))
    if top.shape!=(2,64,64) or spherical.shape!=(10,64,64): raise ValueError(f"Invalid view shapes in {path}: {top.shape}, {spherical.shape}")
    return (top,spherical,metadata) if return_metadata else (top,spherical)

def periodic_shift(x, shift):
    base=int(np.floor(shift)); frac=float(shift-base)
    return (1-frac)*torch.roll(x,base,-1)+frac*torch.roll(x,base+1,-1)

def augment_views(top,spherical,yaw_degrees):
    angle=torch.tensor(-yaw_degrees*np.pi/180,dtype=top.dtype,device=top.device); c,s=torch.cos(angle),torch.sin(angle)
    theta=torch.stack((c,-s,torch.zeros_like(c),s,c,torch.zeros_like(c))).reshape(1,2,3)
    grid=F.affine_grid(theta,(1,*top.shape),align_corners=False)
    rotated=F.grid_sample(top[None],grid,mode="bilinear",padding_mode="zeros",align_corners=False)[0]
    return rotated,periodic_shift(spherical,yaw_degrees/360*spherical.shape[-1])

class ViewDataset(Dataset):
    def __init__(self,root,sequence,count,stats_path=None,manifest=None):
        self.paths=[Path(root)/sequence/f"{i:06d}.npz" for i in range(count)]
        self.manifest=manifest
        self.mean=self.std=None
        if stats_path:
            with np.load(stats_path) as stats:
                self.mean=torch.from_numpy(stats["mean"].astype(np.float32))[:,None,None]
                self.std=torch.from_numpy(stats["std"].astype(np.float32))[:,None,None]
    def __len__(self): return len(self.paths)
    def __getitem__(self,index):
        top,spherical,metadata=load_view(self.paths[index],return_metadata=True)
        if self.manifest is not None:
            expected=(int(self.manifest.frame_ids[index]),int(self.manifest.scan_timestamps[index]),str(self.manifest.source_filenames[index]))
            if metadata != expected: raise RuntimeError(f"{self.paths[index]} metadata {metadata} != manifest {expected}")
        combined=torch.from_numpy(np.concatenate((top,spherical)))
        if self.mean is not None: combined=(combined-self.mean)/self.std
        return combined[:2],combined[2:],index

class TupleDataset(Dataset):
    def __init__(self,views,positions,indices,positive_distance=5,negative_distance=10,positives=2,negatives=18,seed=1024,yaw_step=30):
        self.views=views; self.positions=np.asarray(positions); self.indices=np.asarray(indices); self.p=positives; self.n=negatives
        self.rng=np.random.default_rng(seed); self.yaws=np.arange(0,360,yaw_step)
        subset=self.positions[self.indices]; distances=np.linalg.norm(subset[:,None]-subset[None,:],axis=-1)
        pm=distances<positive_distance; np.fill_diagonal(pm,False); nm=distances>negative_distance
        self.pos=[self.indices[np.flatnonzero(r)] for r in pm]; self.neg=[self.indices[np.flatnonzero(r)] for r in nm]
        self.valid=[i for i in self.indices if len(self.pos[np.where(self.indices==i)[0][0]])>=self.p and len(self.neg[np.where(self.indices==i)[0][0]])>=self.n]
        if not self.valid: raise RuntimeError("No source-training tuples satisfy configured thresholds/counts")
    def _one(self,index):
        top,spherical,_=self.views[int(index)]; yaw=int(self.rng.choice(self.yaws)); return augment_views(top,spherical,yaw)
    def __len__(self): return len(self.valid)
    def __getitem__(self,item):
        anchor=self.valid[item]; local=int(np.where(self.indices==anchor)[0][0])
        pos=self.rng.choice(self.pos[local],self.p,replace=False); neg=self.rng.choice(self.neg[local],self.n,replace=False)
        at,asph=self._one(anchor); pv=[self._one(i) for i in pos]; nv=[self._one(i) for i in neg]
        return at,asph,torch.stack([x[0] for x in pv]),torch.stack([x[1] for x in pv]),torch.stack([x[0] for x in nv]),torch.stack([x[1] for x in nv])

def compute_normalization_stats(view_root,sequence,train_count,output):
    sums=np.zeros(12,np.float64); squares=np.zeros(12,np.float64); pixels=0
    for i in range(train_count):
        top,spherical=load_view(Path(view_root)/sequence/f"{i:06d}.npz"); values=np.concatenate((top,spherical)).reshape(12,-1)
        sums+=values.sum(1); squares+=(values*values).sum(1); pixels+=values.shape[1]
    mean=sums/pixels; std=np.sqrt(np.maximum(squares/pixels-mean*mean,1e-12))
    Path(output).parent.mkdir(parents=True,exist_ok=True); np.savez(output,mean=mean.astype('f'),std=std.astype('f'),train_count=train_count)
    return mean,std
