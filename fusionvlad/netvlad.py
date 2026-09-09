"""Differentiable NetVLAD with learnable centers and soft assignment."""
import torch
from torch import nn
from torch.nn import functional as F

class NetVLAD(nn.Module):
    def __init__(self, dim=512, num_clusters=64):
        super().__init__(); self.dim=dim; self.num_clusters=num_clusters
        self.assignment=nn.Conv2d(dim,num_clusters,1); self.centers=nn.Parameter(torch.empty(num_clusters,dim))
        nn.init.xavier_uniform_(self.centers)
    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.dim: raise ValueError(f"Expected [B,{self.dim},H,W], got {tuple(x.shape)}")
        b,_,h,w=x.shape; local=x.flatten(2).transpose(1,2); weights=F.softmax(self.assignment(x).flatten(2),dim=1).transpose(1,2)
        residual=local.unsqueeze(2)-self.centers.view(1,1,self.num_clusters,self.dim)
        vlad=(weights.unsqueeze(-1)*residual).sum(1); vlad=F.normalize(vlad,p=2,dim=-1)
        return F.normalize(vlad.flatten(1),p=2,dim=1)
