"""
Vessel & Ostium Conditioning for Aneurysm Generation.

Provides:
  - OstiumFeatureExtractor:  offline per-case feature extraction from prepared_meshes_3
  - VesselPointEncoder:      PointNet-style encoder for local vessel surface points
  - OstiumConditioner:       combines ostium plane params + vessel context into condition vector
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from typing import Tuple, Optional, Dict, List


# ---------------------------------------------------------------------------
# 1.  Offline Feature Extraction (runs once per case, results cached)
# ---------------------------------------------------------------------------

class OstiumFeatureExtractor:
    """
    Given a case directory in prepared_meshes_3, extract:
      - ostium_centroid  [3]
      - ostium_normal    [3]
      - ostium_radius    [1]   mean dist of ostium verts from centroid
      - ostium_ecc       [1]   eccentricity (ratio of PCA eigenvalues)
      - vessel_local_pts [N, 3] vessel surface samples near the ostium
    """

    def __init__(self, data_root: str, num_vessel_pts: int = 256, radius_factor: float = 3.0):
        self.data_root = data_root
        self.num_vessel_pts = num_vessel_pts
        self.radius_factor = radius_factor

    def extract_case(self, case_name: str) -> Dict[str, np.ndarray]:
        case_dir = os.path.join(self.data_root, case_name)
        centroid = np.load(os.path.join(case_dir, '07_other', 'centroid_ostium.npy')).astype(np.float32)
        normal   = np.load(os.path.join(case_dir, '07_other', 'normal_vector.npy')).astype(np.float32)

        # Full mesh + labels for ostium ring
        full_mesh = trimesh.load(os.path.join(case_dir, '01_mesh', 'mesh.obj'), process=False)
        labels    = np.load(os.path.join(case_dir, '02_labels', 'labels.npy'))
        ostium_verts = np.array(full_mesh.vertices[labels == 1], dtype=np.float32)

        # Ostium radius & eccentricity
        if len(ostium_verts) >= 3:
            dists = np.linalg.norm(ostium_verts - centroid, axis=1)
            ostium_radius = float(dists.mean())
            # PCA for eccentricity
            centered = ostium_verts - centroid
            try:
                _, svals, _ = np.linalg.svd(centered, full_matrices=False)
                ecc = float(svals[1] / (svals[0] + 1e-8))
            except np.linalg.LinAlgError:
                ecc = 1.0
        else:
            ostium_radius = 0.1
            ecc = 1.0

        # Vessel submesh: sample points near the ostium
        vessel_mesh = trimesh.load(
            os.path.join(case_dir, '05_submeshes', 'vessel_submesh.obj'), process=False
        )
        vessel_verts = np.array(vessel_mesh.vertices, dtype=np.float32)
        vessel_dists = np.linalg.norm(vessel_verts - centroid, axis=1)
        cutoff = self.radius_factor * ostium_radius
        near_mask = vessel_dists < cutoff
        local_verts = vessel_verts[near_mask]

        # Sub-/over-sample to fixed size
        if len(local_verts) == 0:
            local_verts = vessel_verts  # fallback: use all
        if len(local_verts) >= self.num_vessel_pts:
            idx = np.random.choice(len(local_verts), self.num_vessel_pts, replace=False)
        else:
            idx = np.random.choice(len(local_verts), self.num_vessel_pts, replace=True)
        vessel_local_pts = local_verts[idx]

        return {
            'ostium_centroid':   centroid,                                    # [3]
            'ostium_normal':     normal / (np.linalg.norm(normal) + 1e-8),   # [3]
            'ostium_radius':     np.array([ostium_radius], dtype=np.float32),# [1]
            'ostium_ecc':        np.array([ecc], dtype=np.float32),          # [1]
            'ostium_verts':      ostium_verts,                                # [N_ostium, 3]
            'vessel_local_pts':  vessel_local_pts,                            # [N, 3]
        }

    def extract_all(self, case_names: List[str], verbose: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
        all_features = {}
        skipped = 0
        for i, case in enumerate(case_names):
            try:
                all_features[case] = self.extract_case(case)
            except Exception as e:
                if verbose:
                    print(f"[OstiumFeatureExtractor] skip {case}: {e}")
                skipped += 1
        if verbose:
            print(f"Extracted ostium features for {len(all_features)}/{len(case_names)} cases (skipped {skipped})")
        return all_features


# ---------------------------------------------------------------------------
# 2.  PointNet-style Vessel Encoder (for local vessel points)
# ---------------------------------------------------------------------------

class VesselPointEncoder(nn.Module):
    """
    Simple PointNet: per-point MLP → max-pool → global feature.
    Input:  [B, N, 3]
    Output: [B, feat_dim]
    """

    def __init__(self, feat_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feat_dim),
        )

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        """pts: [B, N, 3] → [B, feat_dim]"""
        h = self.mlp(pts)          # [B, N, feat_dim]
        return h.max(dim=1)[0]     # [B, feat_dim]


# ---------------------------------------------------------------------------
# 3.  Ostium Conditioner — produces the full condition vector
# ---------------------------------------------------------------------------

class OstiumConditioner(nn.Module):
    """
    Combines:
      - ostium plane params (centroid, normal, radius, eccentricity) → MLP → [B, ostium_feat_dim]
      - vessel local points → VesselPointEncoder → [B, vessel_feat_dim]
    Fuses into a single condition vector  [B, cond_out_dim].
    """

    def __init__(self, vessel_feat_dim: int = 64, ostium_plane_dim: int = 8,
                 ostium_feat_dim: int = 16, cond_out_dim: int = 32):
        super().__init__()
        self.vessel_encoder = VesselPointEncoder(feat_dim=vessel_feat_dim)

        # ostium plane: centroid(3) + normal(3) + radius(1) + ecc(1) = 8
        self.ostium_plane_encoder = nn.Sequential(
            nn.Linear(ostium_plane_dim, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, ostium_feat_dim),
        )

        self.fuse = nn.Sequential(
            nn.Linear(vessel_feat_dim + ostium_feat_dim, cond_out_dim),
            nn.ReLU(inplace=True),
        )
        self.cond_out_dim = cond_out_dim

    def forward(self, vessel_pts: torch.Tensor, ostium_params: torch.Tensor) -> torch.Tensor:
        """
        vessel_pts:    [B, N, 3]
        ostium_params: [B, 8]  (centroid_3 + normal_3 + radius_1 + ecc_1)
        returns:       [B, cond_out_dim]
        """
        v_feat = self.vessel_encoder(vessel_pts)           # [B, vessel_feat_dim]
        o_feat = self.ostium_plane_encoder(ostium_params)  # [B, ostium_feat_dim]
        return self.fuse(torch.cat([v_feat, o_feat], dim=-1))
