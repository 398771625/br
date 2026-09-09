"""Lazy hardest-positive/hardest-negative multi-head metric objective."""
import torch
from torch.nn import functional as F

def lazy_viewpoint_triplet(anchor, positives, negatives, margin_translation=.5, margin_rotation=.2):
    positive_dist=torch.linalg.vector_norm(anchor[:,None,:]-positives,dim=-1)
    negative_dist=torch.linalg.vector_norm(anchor[:,None,:]-negatives,dim=-1)
    hardest_positive=positive_dist.max(1).values; hardest_negative=negative_dist.min(1).values
    translation=F.relu(margin_translation+hardest_positive-hardest_negative)
    viewpoint=F.relu(hardest_positive-margin_rotation)
    return (translation+viewpoint).mean()

def fusionvlad_loss(anchor, positives, negatives, margin_translation=.5, margin_rotation=.2):
    losses={}
    for key in ("top","spherical","final"):
        losses[key]=lazy_viewpoint_triplet(anchor[key],positives[key],negatives[key],margin_translation,margin_rotation)
    losses["total"]=losses["top"]+losses["spherical"]+losses["final"]
    return losses
