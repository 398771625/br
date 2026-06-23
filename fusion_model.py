import torch
import torch.nn as nn
import torch.nn.functional as F

from REIN import REIN

from range_model import RangeREIN


class ScalarGate(nn.Module):
    """Small scalar gate predicting alpha in [0, 1] per sample."""

    def __init__(self, desc_dim: int, hidden_dim: int = 256):
        super().__init__()
        gate_in_dim = desc_dim * 4  # bev, range, abs diff, elementwise product
        self.net = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, bev_desc_norm: torch.Tensor, rng_desc_norm: torch.Tensor) -> torch.Tensor:
        gate_feat = torch.cat(
            [
                bev_desc_norm,
                rng_desc_norm,
                torch.abs(bev_desc_norm - rng_desc_norm),
                bev_desc_norm * rng_desc_norm,
            ],
            dim=1,
        )
        alpha = torch.sigmoid(self.net(gate_feat))
        return alpha


class FusionModel(nn.Module):
    """
    Fusion model for weighted_sum and adaptive_weighted modes.

    Returns explicit dict output to avoid tuple ordering ambiguity.
    """

    def __init__(
        self,
        fusion_mode: str = "adaptive_weighted",
        alpha_bev: float = 0.6,
        alpha_range: float = 0.4,
        freeze_branches: bool = True,
    ):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.alpha_bev = alpha_bev
        self.alpha_range = alpha_range

        # Reuse BEVPlace backbone style for both modalities unless a different
        # range backbone is swapped in later.
        self.bev_branch = REIN()
        self.range_branch = RangeREIN()

        desc_dim = self.bev_branch.global_feat_dim
        self.global_feat_dim = desc_dim
        self.local_feat_dim = self.bev_branch.local_feat_dim

        self.gate = ScalarGate(desc_dim=desc_dim)

        if freeze_branches:
            self.freeze_branches()

    def freeze_branches(self):
        for p in self.bev_branch.parameters():
            p.requires_grad = False
        for p in self.range_branch.parameters():
            p.requires_grad = False

    def unfreeze_branches(self):
        for p in self.bev_branch.parameters():
            p.requires_grad = True
        for p in self.range_branch.parameters():
            p.requires_grad = True

    @staticmethod
    def _l2_norm(desc: torch.Tensor) -> torch.Tensor:
        return F.normalize(desc, p=2, dim=1)

    def _forward_branches(self, bev_input: torch.Tensor, rng_input: torch.Tensor):
        bev_out1, bev_local_feats, bev_desc = self.bev_branch(bev_input)
        _rng_out1, _rng_local_feats, rng_desc = self.range_branch(rng_input)

        bev_desc_norm = self._l2_norm(bev_desc)
        rng_desc_norm = self._l2_norm(rng_desc)

        return bev_out1, bev_local_feats, bev_desc_norm, rng_desc_norm

    def _teacher_desc(self, bev_desc_norm: torch.Tensor, rng_desc_norm: torch.Tensor, teacher_alpha_bev: float = 0.6):
        teacher_desc = teacher_alpha_bev * bev_desc_norm + (1.0 - teacher_alpha_bev) * rng_desc_norm
        return self._l2_norm(teacher_desc)

    def forward(self, bev_input: torch.Tensor, rng_input: torch.Tensor):
        bev_out1, bev_local_feats, bev_desc_norm, rng_desc_norm = self._forward_branches(bev_input, rng_input)

        if self.fusion_mode == "weighted_sum":
            alpha = torch.full(
                (bev_desc_norm.size(0), 1),
                float(self.alpha_bev),
                dtype=bev_desc_norm.dtype,
                device=bev_desc_norm.device,
            )
        elif self.fusion_mode == "adaptive_weighted":
            alpha = self.gate(bev_desc_norm, rng_desc_norm)
        else:
            raise ValueError(f"Unsupported fusion_mode: {self.fusion_mode}")

        fused_desc = alpha * bev_desc_norm + (1.0 - alpha) * rng_desc_norm
        fused_desc = self._l2_norm(fused_desc)

        teacher_desc = self._teacher_desc(bev_desc_norm, rng_desc_norm)

        return {
            "bev_desc": bev_desc_norm,
            "rng_desc": rng_desc_norm,
            "alpha": alpha,
            "fused_desc": fused_desc,
            "teacher_desc": teacher_desc,
            "bev_local_feats": bev_local_feats,
            "bev_out1": bev_out1,
        }