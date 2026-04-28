"""
Vessel-aware losses for aneurysm generation  (v2 -- intrinsic).

All losses operate on reconstructed mesh vertices directly, without external
coordinate targets.  This avoids the coordinate-system mismatch between the
GHD-normalised canonical space and the original mesh space.

IntrinsicPlaneLoss        -- Ring vertices should lie on a plane (SVD-based)
IntrinsicPenetrationLoss  -- Dome vertices should be on one side of the ring plane
RingMatchLoss             -- Pred ring should match GT ring (same GHD space)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def _fit_plane(ring_verts: torch.Tensor):
    """
    Fit a plane to ring vertices via SVD.
    ring_verts: [B, N, 3]
    Returns:
        centroid: [B, 3]
        normal:   [B, 3]  (unit normal of best-fit plane)
    If SVD fails (degenerate early-training meshes), returns z-axis as normal.
    """
    centroid = ring_verts.mean(dim=1)                               # [B, 3]
    centered = ring_verts - centroid.unsqueeze(1)                   # [B, N, 3]
    # Add small jitter for numerical stability
    centered = centered + torch.randn_like(centered) * 1e-6
    try:
        U, S, Vh = torch.linalg.svd(centered, full_matrices=False) # Vh: [B, 3, 3]
        normal = Vh[:, -1, :]                                      # [B, 3]
    except torch._C._LinAlgError:
        # Fallback: return z-axis normal (harmless placeholder)
        normal = torch.zeros(ring_verts.shape[0], 3, device=ring_verts.device)
        normal[:, 2] = 1.0
    normal = F.normalize(normal, dim=-1)
    return centroid, normal


class IntrinsicPlaneLoss(nn.Module):
    """
    Ring vertices should lie on a plane.
    Loss = mean squared distance of each ring vertex to the SVD best-fit plane.
    Purely intrinsic (no external reference frame needed).
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_verts: torch.Tensor, opening_idx: torch.Tensor):
        """
        pred_verts: [B, V, 3]
        opening_idx: [N_ring]
        """
        ring = pred_verts[:, opening_idx, :]               # [B, N_ring, 3]
        centroid, normal = _fit_plane(ring)                 # [B, 3], [B, 3]
        centered = ring - centroid.unsqueeze(1)             # [B, N_ring, 3]
        dist = (centered * normal.unsqueeze(1)).sum(-1)     # [B, N_ring]
        return dist.pow(2).mean()


class IntrinsicPenetrationLoss(nn.Module):
    """
    All non-ring (dome) vertices should be on ONE side of the ring plane.
    Uses the ring-plane from the GROUND-TRUTH reconstructed mesh as reference,
    because the predicted mesh's ring plane may be noisy during early training.
    """
    def __init__(self, margin: float = 0.002):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        pred_verts: torch.Tensor,       # [B, V, 3]  predicted mesh
        gt_verts: torch.Tensor,          # [B, V, 3]  ground-truth mesh (same GHD space)
        opening_idx: torch.Tensor,       # [N_ring]
    ):
        # Fit plane from GT ring
        gt_ring = gt_verts[:, opening_idx, :]
        centroid, normal = _fit_plane(gt_ring)

        # Orient normal so GT dome center is on + side
        B, V, _ = gt_verts.shape
        mask = torch.ones(V, dtype=torch.bool, device=gt_verts.device)
        mask[opening_idx] = False
        dome_gt = gt_verts[:, mask, :]                               # [B, V-N_ring, 3]
        dome_center = dome_gt.mean(dim=1)                            # [B, 3]
        sign = ((dome_center - centroid) * normal).sum(-1).sign()    # [B]
        normal = normal * sign.unsqueeze(-1)                         # flip if needed

        # Compute signed distance of all pred vertices to plane
        rel = pred_verts - centroid.unsqueeze(1)                     # [B, V, 3]
        signed_dist = (rel * normal.unsqueeze(1)).sum(-1)            # [B, V]

        # Penalise dome vertices that penetrate (go to negative side)
        dome_dist = signed_dist[:, mask]                             # [B, V-N_ring]
        violation = F.relu(-dome_dist - self.margin)
        return violation.mean()


class RingMatchLoss(nn.Module):
    """
    Opening ring of predicted mesh should match GT ring (both in GHD space).
    Simple MSE on ring vertex positions.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred_verts: torch.Tensor, gt_verts: torch.Tensor,
                opening_idx: torch.Tensor):
        pred_ring = pred_verts[:, opening_idx, :]
        gt_ring = gt_verts[:, opening_idx, :]
        return F.mse_loss(pred_ring, gt_ring)
