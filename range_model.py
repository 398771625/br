import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

from REIN import NetVLAD


class RangeEncoder(nn.Module):
    """Lightweight encoder for range-view images."""

    def __init__(self, from_scratch: bool = False):
        super().__init__()
        pretrain = not from_scratch
        encoder = models.resnet34(pretrained=pretrain)
        layers = list(encoder.children())[:-4]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        feat = self.encoder(x)

        # NetVLAD input feature map (1/4 spatial size of input)
        netvlad_input = F.interpolate(
            feat,
            scale_factor=0.5,
            mode="bilinear",
            align_corners=True,
            recompute_scale_factor=False,
        )
        netvlad_input = F.normalize(netvlad_input, dim=1)

        # Local feature map in input size for potential local usage
        local_feats = F.interpolate(
            feat,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        local_feats = F.normalize(local_feats, dim=1)

        return netvlad_input, local_feats


class RangeREIN(nn.Module):
    """Range-view counterpart of REIN with explicit branch outputs."""

    def __init__(self):
        super().__init__()
        self.encoder = RangeEncoder()
        self.pooling = NetVLAD()

        self.local_feat_dim = 128
        self.global_feat_dim = self.local_feat_dim * 64

    def forward(self, x):
        netvlad_input, local_feats = self.encoder(x)
        global_desc = self.pooling(netvlad_input)
        return netvlad_input, local_feats, global_desc