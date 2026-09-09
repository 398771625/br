"""Independent dual-view VGG16-style FusionVLAD network."""
import torch
from torch import nn
from torch.nn import functional as F
from .netvlad import NetVLAD

VGG16_CFG=(64,64,"M",128,128,"M",256,256,256,"M",512,512,512,"M",512,512,512,"M")

def make_vgg_features(in_channels):
    layers=[]; current=in_channels
    for value in VGG16_CFG:
        if value=="M": layers.append(nn.MaxPool2d(2,2))
        else: layers.extend((nn.Conv2d(current,value,3,padding=1),nn.ReLU(inplace=True))); current=value
    return nn.Sequential(*layers)

class ViewBranch(nn.Module):
    def __init__(self,in_channels,num_clusters=64,output_dim=512):
        super().__init__(); self.features=make_vgg_features(in_channels); self.vlad=NetVLAD(512,num_clusters)
        self.projection=nn.Linear(512*num_clusters,output_dim)
    def forward(self,x): return F.normalize(self.projection(self.vlad(self.features(x))),p=2,dim=1)

class FusionVLAD(nn.Module):
    descriptor_dim=2048
    def __init__(self,num_clusters=64,branch_dim=512,joint_dim=1024,imagenet_init=False):
        super().__init__(); self.top=ViewBranch(2,num_clusters,branch_dim); self.spherical=ViewBranch(10,num_clusters,branch_dim)
        self.fusion=nn.Linear(2*branch_dim,joint_dim)
        if 2*branch_dim+joint_dim != self.descriptor_dim: raise ValueError("Fusion dimensions must produce a 2048-D descriptor")
        if imagenet_init: self.initialize_from_imagenet()
    def initialize_from_imagenet(self):
        from torchvision.models import VGG16_Weights, vgg16
        source=vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        source_convs=[m for m in source if isinstance(m,nn.Conv2d)]
        for branch in (self.top,self.spherical):
            target_convs=[m for m in branch.features if isinstance(m,nn.Conv2d)]
            for i,(target,original) in enumerate(zip(target_convs,source_convs)):
                weights=original.weight.detach()
                if i==0: weights=weights.mean(1,keepdim=True).repeat(1,target.in_channels,1,1)
                target.weight.data.copy_(weights); target.bias.data.copy_(original.bias.data)
    def forward(self,topdown,spherical):
        f_top=self.top(topdown); f_spherical=self.spherical(spherical)
        f_joint=F.normalize(self.fusion(torch.cat((f_top,f_spherical),1)),p=2,dim=1)
        final=F.normalize(torch.cat((f_top,f_joint,f_spherical),1),p=2,dim=1)
        return {"top":f_top,"spherical":f_spherical,"joint":f_joint,"final":final}
