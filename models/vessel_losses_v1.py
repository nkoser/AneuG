"""
Vessel-aware losses for aneurysm generation.

OstiumBoundaryLoss     — Chamfer between generated opening ring and vessel cut boundary
OstiumPlaneLoss        — Forces generated opening to lie on target ostium plane
VesselPenetrationLoss  — Penalises dome vertices that penetrate below the ostium plane
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
from typing import Optional


class OstiumPlaneLoss(nn.Module):
    """
    Ensures that the generated aneurysm's opening ring vertices lie on
    the target ostium plane  (defined by centroid + normal).

    The loss is the mean squared signed-distance of opening-ring vertices
    to the plane.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_verts_padded: torch.Tensor,   # [B, V, 3]  full reconstructed mesh verts
        opening_idx: torch.Tensor,          # [N_ring]  vertex indices of the opening ring on the canonical mesh
        target_centroid: torch.Tensor,      # [B, 3]
        target_normal: torch.Tensor,        # [B, 3]
    ) -> torch.Tensor:
        """Returns scalar loss."""
        ring_verts = pred_verts_padded[:, opening_idx, :]              # [B, N_ring, 3]
        rel = ring_verts - target_centroid.unsqueeze(1)                # [B, N_ring, 3]
        signed_dist = (rel * target_normal.unsqueeze(1)).sum(dim=-1)   # [B, N_ring]
        return signed_dist.pow(2).mean()


class OstiumBoundaryLoss(nn.Module):
    """
    Chamfer distance between the generated opening ring and the target
    ostium boundary.  Works with padded vertex tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred_verts_padded: torch.Tensor,   # [B, V, 3]
        opening_idx: torch.Tensor,          # [N_ring]
        target_centroid: torch.Tensor,      # [B, 3]
        target_normal: torch.Tensor,        # [B, 3]
        target_radius: torch.Tensor,        # [B, 1]
    ) -> torch.Tensor:
        """
        Because we don't have the exact target vessel-cut ring per sample at
        train time, we approximate it as a circle on the target plane.
        This is a soft constraint — the plane loss is the hard geometric one.

        Returns scalar loss.
        """
        ring_verts = pred_verts_padded[:, opening_idx, :]  # [B, N_ring, 3]

        # Build target ring on the plane (circle approximation)
        B = target_centroid.shape[0]
        N_ring = opening_idx.shape[0]
        device = ring_verts.device

        # Create N_ring evenly spaced angles
        angles = torch.linspace(0, 2 * 3.14159265, N_ring + 1, device=device)[:-1]  # [N_ring]

        # Build local coordinate frame from normal
        normal = F.normalize(target_normal, dim=-1)                 # [B, 3]
        # Arbitrary perpendicular vector
        helper = torch.zeros_like(normal)
        helper[:, 0] = 1.0
        # If normal is nearly parallel to helper, use y-axis instead
        dot = (normal * helper).sum(dim=-1).abs()
        helper[dot > 0.9] = torch.tensor([0., 1., 0.], device=device)
        u = F.normalize(torch.cross(normal, helper, dim=-1), dim=-1)   # [B, 3]
        v = torch.cross(normal, u, dim=-1)                             # [B, 3]

        # Circle points:  c + r * (cos(a)*u + sin(a)*v)
        cos_a = angles.cos().unsqueeze(0).unsqueeze(-1)   # [1, N_ring, 1]
        sin_a = angles.sin().unsqueeze(0).unsqueeze(-1)   # [1, N_ring, 1]
        r = target_radius.unsqueeze(1)                     # [B, 1, 1]
        target_ring = (target_centroid.unsqueeze(1)
                       + r * (cos_a * u.unsqueeze(1) + sin_a * v.unsqueeze(1)))  # [B, N_ring, 3]

        # Chamfer
        loss, _ = chamfer_distance(ring_verts, target_ring)
        return loss


class VesselPenetrationLoss(nn.Module):
    """
    Penalises aneurysm dome vertices that go *below* the ostium plane
    (i.e. back into the vessel lumen).

    Convention: the ostium normal points OUTWARD (away from vessel, into
    aneurysm dome).  Dome vertices should have positive signed distance.
    """

    def __init__(self, margin: float = 0.005):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        pred_verts_padded: torch.Tensor,   # [B, V, 3]
        target_centroid: torch.Tensor,      # [B, 3]
        target_normal: torch.Tensor,        # [B, 3]
    ) -> torch.Tensor:
        rel = pred_verts_padded - target_centroid.unsqueeze(1)
        signed_dist = (rel * target_normal.unsqueeze(1)).sum(dim=-1)   # [B, V]
        penetration = F.relu(-signed_dist - self.margin)                # only negative side
        return penetration.mean()
