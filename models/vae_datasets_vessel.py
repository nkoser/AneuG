"""
Vessel-aware GHD Dataset.

Extends GHDDataset with per-case ostium / vessel conditioning data loaded from
prepared_meshes_3.  Each sample returns:
    ghd:           [D]          normalised GHD coefficients  (same as original)
    ostium_params: [8]          centroid(3) + normal(3) + radius(1) + ecc(1)
    vessel_pts:    [N_vessel,3] local vessel surface points near the ostium

By default, conditioning geometry stays in the raw prepared_meshes_3 frame for
backward compatibility.  With condition_space="ghd_local", ostium/vessel points
are transformed into the same local canonical GHD frame as the coefficients.
"""
import os
import io
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from models.vessel_conditioner import OstiumFeatureExtractor


class _TorchCPUUnpickler(pickle.Unpickler):
    """Load pickle files containing torch CUDA storages on CPU-only sessions."""

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
        return super().find_class(module, name)


def _load_checkpoint_cpu(path: str):
    with open(path, 'rb') as f:
        return _TorchCPUUnpickler(f).load()


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _axis_angle_to_matrix_np(axis_angle) -> np.ndarray:
    """Rodrigues formula matching pytorch3d.transforms.axis_angle_to_matrix."""
    vec = _to_numpy(axis_angle).reshape(-1)[:3].astype(np.float64)
    theta = float(np.linalg.norm(vec))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)

    axis = vec / theta
    x, y, z = axis
    k = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    return eye * np.cos(theta) + (1.0 - np.cos(theta)) * np.outer(axis, axis) + np.sin(theta) * k


def _apply_homogeneous(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    return vec / (np.linalg.norm(vec) + 1e-12)


def _ostium_radius_ecc(ostium_verts: np.ndarray, centroid: np.ndarray) -> Tuple[float, float]:
    if ostium_verts is None or len(ostium_verts) < 3:
        return 0.1, 1.0

    centered = np.asarray(ostium_verts, dtype=np.float64) - centroid.reshape(1, 3)
    radius = float(np.linalg.norm(centered, axis=1).mean())
    try:
        _, svals, _ = np.linalg.svd(centered, full_matrices=False)
        ecc = float(svals[1] / (svals[0] + 1e-8))
    except np.linalg.LinAlgError:
        ecc = 1.0
    return radius, ecc


def _canonical_norm(canonical_mesh: str, norm_factor: float) -> float:
    import trimesh

    mesh = trimesh.load(canonical_mesh, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    return float(np.linalg.norm(verts, axis=1).max() * norm_factor)


def _condition_to_ghd_local(
    vf: Dict[str, np.ndarray],
    chk: Dict,
    prealign_transform: np.ndarray,
    canonical_norm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert raw prepared_meshes_3 condition geometry into GHD-local coordinates.

    Fitting forward path:
        p_target_norm = (p_ghd_local @ R.T) * s + T

    Therefore inverse path:
        p_ghd_local = ((p_target_norm - T) / s) @ R
    """
    r_mat = _axis_angle_to_matrix_np(chk['R'])
    s = float(np.abs(_to_numpy(chk['s']).reshape(-1)[0])) + 1e-12
    t = _to_numpy(chk['T']).reshape(-1, 3)[0].astype(np.float64)

    def points_to_local(points: np.ndarray) -> np.ndarray:
        target = _apply_homogeneous(points, prealign_transform) / canonical_norm
        return ((target - t.reshape(1, 3)) / s) @ r_mat

    def normal_to_local(normal: np.ndarray) -> np.ndarray:
        target_normal = np.asarray(normal, dtype=np.float64).reshape(3) @ prealign_transform[:3, :3].T
        return _normalize_vector(target_normal @ r_mat)

    vessel_pts = points_to_local(vf['vessel_local_pts']).astype(np.float32)
    ostium_centroid = points_to_local(vf['ostium_centroid'].reshape(1, 3))[0]
    ostium_normal = normal_to_local(vf['ostium_normal'])

    ostium_verts_raw = vf.get('ostium_verts')
    if ostium_verts_raw is not None and len(ostium_verts_raw) >= 3:
        ostium_verts = points_to_local(ostium_verts_raw)
        ostium_radius, ostium_ecc = _ostium_radius_ecc(ostium_verts, ostium_centroid)
    else:
        ostium_radius = float(np.asarray(vf['ostium_radius']).reshape(-1)[0]) / canonical_norm / s
        ostium_ecc = float(np.asarray(vf['ostium_ecc']).reshape(-1)[0])

    ostium_params = np.concatenate([
        ostium_centroid.astype(np.float32),
        ostium_normal.astype(np.float32),
        np.array([ostium_radius], dtype=np.float32),
        np.array([ostium_ecc], dtype=np.float32),
    ])
    return ostium_params, vessel_pts


class VesselAwareGHDDataset(Dataset):
    """
    Loads:
      1) GHD coefficients from ghd checkpoint root  (same logic as GHDDataset)
      2) Ostium/vessel features from prepared_meshes_3 via OstiumFeatureExtractor
    Only cases present in BOTH roots are kept.
    """

    def __init__(
        self,
        ghd_chk_root: str,
        ghd_run: str,
        ghd_chk_name: str,
        data_root: str,                # e.g. /data/prepared_meshes_3
        cases: List[str],
        num_vessel_pts: int = 256,
        vessel_radius_factor: float = 3.0,
        normalize: bool = True,
        withscale: bool = False,
        condition_space: str = 'raw',
        aligned_data_root: Optional[str] = None,
        canonical_mesh: Optional[str] = None,
        canonical_norm_factor: float = 1.10,
        canonical_norm: Optional[float] = None,
    ):
        self.ghd_chk_root = ghd_chk_root
        self.ghd_run = ghd_run
        self.ghd_chk_name = ghd_chk_name
        self.data_root = data_root
        self.normalize_flag = normalize
        self.withscale = withscale
        self.num_vessel_pts = num_vessel_pts
        self.condition_space = condition_space
        self.aligned_data_root = aligned_data_root

        if self.condition_space not in ('raw', 'ghd_local'):
            raise ValueError("condition_space must be 'raw' or 'ghd_local'")
        if self.condition_space == 'ghd_local':
            if aligned_data_root is None:
                raise ValueError("condition_space='ghd_local' requires aligned_data_root")
            if canonical_norm is None:
                if canonical_mesh is None:
                    raise ValueError("condition_space='ghd_local' requires canonical_mesh or canonical_norm")
                canonical_norm = _canonical_norm(canonical_mesh, canonical_norm_factor)
            self.condition_canonical_norm = float(canonical_norm)
            print(
                "VesselAwareGHDDataset: transforming vessel/ostium conditions "
                f"to GHD-local frame (canonical_norm={self.condition_canonical_norm:.8f})"
            )
        else:
            self.condition_canonical_norm = None

        # ---- 1) extract ostium features for all candidate cases ----
        extractor = OstiumFeatureExtractor(data_root, num_vessel_pts=num_vessel_pts,
                                           radius_factor=vessel_radius_factor)
        vessel_feats_all = extractor.extract_all(cases, verbose=True)

        # ---- 2) load GHD checkpoints, keep only matched cases ----
        self.case_names: List[str] = []
        self.ghd_list: List[torch.Tensor] = []
        self.scale_list: List[torch.Tensor] = []
        self.alignment_list: List[torch.Tensor] = []
        self.ostium_params_list: List[torch.Tensor] = []    # [8]
        self.vessel_pts_list: List[torch.Tensor] = []       # [N, 3]
        skipped_transform = 0

        for case in cases:
            ghd_path = os.path.join(ghd_chk_root, case, ghd_run, ghd_chk_name)
            if not os.path.exists(ghd_path):
                continue
            if case not in vessel_feats_all:
                continue
            # load ghd
            chk = _load_checkpoint_cpu(ghd_path)
            ghd_coeff = chk['GHD_coefficient'].view(-1)
            if torch.isnan(ghd_coeff).any() or torch.isinf(ghd_coeff).any():
                continue  # skip corrupted GHD fits
            R, s, T = chk['R'], chk['s'].abs(), chk['T']
            alignment = torch.cat((R.view(-1), s.view(-1), T.view(-1))).detach()
            scale = s.view(-1).detach()

            # load vessel/ostium
            vf = vessel_feats_all[case]
            if self.condition_space == 'ghd_local':
                prealign_path = os.path.join(aligned_data_root, case, 'prealign_transform.npy')
                if not os.path.exists(prealign_path):
                    skipped_transform += 1
                    continue
                try:
                    prealign_transform = np.load(prealign_path).astype(np.float64)
                    ostium_params, vessel_pts = _condition_to_ghd_local(
                        vf, chk, prealign_transform, self.condition_canonical_norm
                    )
                except Exception as e:
                    print(f"[VesselAwareGHDDataset] skip {case}: condition transform failed: {e}")
                    skipped_transform += 1
                    continue
            else:
                ostium_params = np.concatenate([
                    vf['ostium_centroid'],   # 3
                    vf['ostium_normal'],     # 3
                    vf['ostium_radius'],     # 1
                    vf['ostium_ecc'],        # 1
                ])  # → [8]
                vessel_pts = vf['vessel_local_pts']

            self.case_names.append(case)
            self.ghd_list.append(ghd_coeff)
            self.scale_list.append(scale)
            self.alignment_list.append(alignment)
            self.ostium_params_list.append(torch.from_numpy(ostium_params))
            self.vessel_pts_list.append(torch.from_numpy(vessel_pts))

        print(f"VesselAwareGHDDataset: {len(self.case_names)} cases loaded "
              f"(from {len(cases)} candidates)")
        if skipped_transform:
            print(f"VesselAwareGHDDataset: skipped {skipped_transform} cases during condition transform")

        # ---- 3) normalise GHD (same as original GHDDataset) ----
        if self.withscale:
            stacked = torch.stack([torch.cat([g, s]) for g, s in
                                   zip(self.ghd_list, self.scale_list)], dim=0)
        else:
            stacked = torch.stack(self.ghd_list, dim=0)
        self.ghd_mean = stacked.mean(dim=0, keepdim=True)
        self.ghd_std  = stacked.std(dim=0, keepdim=True) + 0.01

        # ---- 4) normalise ostium params ----
        ostium_stack = torch.stack(self.ostium_params_list, dim=0)  # [N_cases, 8]
        self.ostium_mean = ostium_stack.mean(dim=0, keepdim=True)
        self.ostium_std  = ostium_stack.std(dim=0, keepdim=True) + 1e-6

        # ---- 5) normalise vessel points (per dataset, global centering + scale) ----
        all_vpts = torch.cat(self.vessel_pts_list, dim=0)           # [N_total, 3]
        self.vessel_center = all_vpts.mean(dim=0, keepdim=True)     # [1, 3]
        self.vessel_scale  = all_vpts.std() + 1e-6                  # scalar

    # ----- public API (mirrors GHDDataset) ----- #

    def __len__(self):
        return len(self.case_names)

    def __getitem__(self, idx):
        ghd = self.ghd_list[idx]
        scale = self.scale_list[idx]
        x = torch.cat([ghd, scale]) if self.withscale else ghd
        if self.normalize_flag:
            x = (x - self.ghd_mean) / self.ghd_std

        ostium_params = self.ostium_params_list[idx]
        if self.normalize_flag:
            ostium_params = (ostium_params - self.ostium_mean) / self.ostium_std

        vessel_pts = self.vessel_pts_list[idx]
        if self.normalize_flag:
            vessel_pts = (vessel_pts - self.vessel_center) / self.vessel_scale

        return {
            'ghd':           x.view(-1),
            'ostium_params': ostium_params.view(-1),     # [8]
            'vessel_pts':    vessel_pts,                  # [N_vessel, 3]
        }

    def get_dim(self) -> int:
        x = self.ghd_list[0]
        return x.shape[0]  # 196*3 = 588

    def get_mean_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.withscale:
            return self.ghd_mean[:, :-1], self.ghd_std[:, :-1]
        return self.ghd_mean, self.ghd_std

    def get_scale_mean_std(self):
        if self.withscale:
            return self.ghd_mean[:, -1], self.ghd_std[:, -1]
        return None, None

    def get_ostium_mean_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.ostium_mean, self.ostium_std

    def de_normalize_ghd(self, x: torch.Tensor) -> torch.Tensor:
        if not self.normalize_flag:
            return x
        return x * self.ghd_std.to(x.device) + self.ghd_mean.to(x.device)

    def de_normalize_ostium(self, o: torch.Tensor) -> torch.Tensor:
        if not self.normalize_flag:
            return o
        return o * self.ostium_std.to(o.device) + self.ostium_mean.to(o.device)
