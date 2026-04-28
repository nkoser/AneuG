import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__),'.','..'))
from pytorch3d.loss import chamfer_distance,mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss
from pytorch3d.ops import sample_points_from_meshes, cot_laplacian, padded_to_packed
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.structures import Meshes
# from .rigid_deform import RigidLoss
from .mesh_loss import Rigid_Loss
from .diceloss import BinaryDiceLoss, BinaryDiceLoss_Weighted
from ghd.base.graph_operators import Laplacain, Normal_consistence
from torch_geometric.utils import to_undirected
from typing import Union


class Mesh_loss(nn.Module):
    def __init__(self, mesh_std: Meshes, sample_num=5000, resolution=128, length=2, ):
        super(Mesh_loss, self).__init__()
        self.sample_num = sample_num
        self.mesh_std = mesh_std
        cotweight_std, _ = cot_laplacian(mesh_std.verts_packed(), mesh_std.faces_packed())  # laplace matrix
        self.connection_std = cotweight_std.coalesce().indices()
        self.cotweight_std = cotweight_std.coalesce().values()
        self.connection_std, self.cotweight_std = to_undirected(self.connection_std, edge_attr=self.cotweight_std)
        self.rigidloss = Rigid_Loss(mesh_std)
        self.lap = Laplacain()
        self.normal_consistence = Normal_consistence()
        self.edge_lenth = torch.norm(self.mesh_std.verts_packed().index_select(0,self.connection_std[0]) - self.mesh_std.verts_packed().index_select(0,self.connection_std[1]),dim=-1).mean()
        self.resolution = resolution
        self.length = length
        # loss modules
        self.dice_loss = BinaryDiceLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.dice_loss_attention = BinaryDiceLoss_Weighted(weights_normalize=False)  # conservative: not using weights_normalization

    def _vertex_fallback_sample(self, meshes: Meshes, num_samples: int, return_normals: bool = True):
        verts = meshes.verts_padded()
        B = verts.shape[0]
        dtype = verts.dtype
        device = verts.device
        normals_all = None
        if return_normals:
            try:
                normals_all = meshes.verts_normals_padded()
            except Exception:
                normals_all = None

        samples_b = []
        normals_b = []
        for b in range(B):
            verts_b = verts[b]
            finite_mask = torch.isfinite(verts_b).all(dim=-1)
            if torch.any(finite_mask):
                verts_valid = verts_b[finite_mask]
                n_valid = int(verts_valid.shape[0])
                idx = torch.randint(0, n_valid, (int(num_samples),), device=device)
                sampled = verts_valid.index_select(0, idx)
                samples_b.append(sampled)
                if return_normals:
                    if normals_all is not None:
                        normals_valid = normals_all[b][finite_mask]
                        sampled_normals = normals_valid.index_select(0, idx)
                        sampled_normals = torch.where(
                            torch.isfinite(sampled_normals),
                            sampled_normals,
                            torch.zeros_like(sampled_normals),
                        )
                    else:
                        sampled_normals = torch.zeros_like(sampled)
                    normals_b.append(sampled_normals)
            else:
                sampled = torch.zeros((int(num_samples), 3), dtype=dtype, device=device)
                samples_b.append(sampled)
                if return_normals:
                    normals_b.append(torch.zeros_like(sampled))

        samples = torch.stack(samples_b, dim=0)
        if return_normals:
            normals = torch.stack(normals_b, dim=0)
            return samples, normals
        return samples

    def _opening_mode_uses_boundary(self, mode: str) -> bool:
        mode_key = str(mode).strip().lower()
        return mode_key in (
            "boundary",
            "rim",
            "edge",
            "rim_plane",
            "rim_ordered",
            "ordered_rim",
            "cyclic_rim",
        )

    def _opening_mode_uses_ordered_rim(self, mode: str) -> bool:
        mode_key = str(mode).strip().lower()
        return mode_key in ("rim_ordered", "ordered_rim", "cyclic_rim")

    def _safe_sample_points_from_meshes(self, meshes: Meshes, num_samples: int, return_normals: bool = True):
        """
        sample_points_from_meshes can fail when a mesh becomes degenerate
        (e.g., zero/invalid face areas). Fallback to sampled vertices so training
        can continue instead of crashing.
        """
        try:
            sampled = sample_points_from_meshes(meshes, num_samples, return_normals=return_normals)
            if return_normals:
                points, normals = sampled
                if (not torch.isfinite(points).all()) or (not torch.isfinite(normals).all()):
                    raise ValueError("sampled points or normals are non-finite")
                return points, normals
            if not torch.isfinite(sampled).all():
                raise ValueError("sampled points are non-finite")
            return sampled
        except Exception as e:
            print(f"[Mesh_loss] sample_points_from_meshes failed, using vertex fallback: {e}")
            return self._vertex_fallback_sample(meshes, num_samples, return_normals=return_normals)

    def _sample_opening_boundary_points(self, mesh: Meshes, num_samples: int):
        """
        Sample points from opening boundary vertices (rim) instead of the full
        opening cap surface. This gives stronger supervision for ostium edge fit.
        Returns shape [1, N, 3].
        """
        n = max(1, int(num_samples))
        try:
            verts = mesh.verts_packed()
            faces = mesh.faces_packed().long()
            if verts.numel() == 0:
                raise ValueError("opening mesh has no vertices")
            if faces.numel() == 0:
                boundary_pts = verts
            else:
                edges = torch.cat(
                    (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
                    dim=0,
                )
                edges, _ = torch.sort(edges, dim=1)
                unique_edges, counts = torch.unique(edges, dim=0, return_counts=True)
                boundary_edges = unique_edges[counts == 1]
                if boundary_edges.numel() == 0:
                    boundary_pts = verts
                else:
                    boundary_idx = torch.unique(boundary_edges.reshape(-1))
                    boundary_pts = verts.index_select(0, boundary_idx)
            if boundary_pts.shape[0] <= 0:
                boundary_pts = verts
            sample_idx = torch.randint(0, boundary_pts.shape[0], (n,), device=boundary_pts.device)
            sampled = boundary_pts.index_select(0, sample_idx)
            return sampled.unsqueeze(0)
        except Exception as e:
            print(f"[Mesh_loss] boundary sampling failed, fallback to surface sampling: {e}")
            return self._safe_sample_points_from_meshes(mesh, n, return_normals=False)

    def _sample_point_cloud(self, points, num_samples: int):
        pts = torch.as_tensor(points, dtype=torch.float32)
        if pts.ndim == 3 and pts.shape[0] == 1:
            pts = pts[0]
        if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 1:
            raise ValueError("point cloud for sampling must have shape [N, 3]")
        pts = pts.to(self.mesh_std.device)
        n = max(1, int(num_samples))
        idx = torch.randint(0, pts.shape[0], (n,), device=pts.device)
        return pts.index_select(0, idx).unsqueeze(0)

    def _plane_from_points(self, points: torch.Tensor):
        if points is None:
            return None, None
        pts = points
        if pts.ndim == 3 and pts.shape[0] == 1:
            pts = pts[0]
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 3:
            return None, None
        finite_mask = torch.isfinite(pts).all(dim=-1)
        pts = pts[finite_mask]
        if pts.shape[0] < 3:
            return None, None
        centroid = pts.mean(dim=0)
        centered = pts - centroid
        try:
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
        except Exception:
            cov = centered.transpose(0, 1).matmul(centered)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            normal = eigvecs[:, torch.argmin(eigvals)]
        normal_norm = normal.norm()
        if (not torch.isfinite(normal_norm)) or float(normal_norm.detach().cpu().item()) <= 1e-8:
            return centroid, None
        return centroid, normal / normal_norm

    def _opening_planarity_loss(self, mesh: Meshes, target_plane=None):
        verts = mesh.verts_packed()
        centroid_self, normal_self = self._plane_from_points(verts)
        if centroid_self is None or normal_self is None:
            return torch.zeros(1, device=verts.device, dtype=verts.dtype).squeeze(0)

        self_dist = (verts - centroid_self.unsqueeze(0)).matmul(normal_self.unsqueeze(-1)).squeeze(-1)
        loss_self = torch.mean(self_dist.square())

        if target_plane is None:
            return loss_self

        target_centroid, target_normal = target_plane
        if target_centroid is None or target_normal is None:
            return loss_self
        target_centroid = target_centroid.to(device=verts.device, dtype=verts.dtype)
        target_normal = target_normal.to(device=verts.device, dtype=verts.dtype)
        target_dist = (verts - target_centroid.unsqueeze(0)).matmul(target_normal.unsqueeze(-1)).squeeze(-1)
        loss_target = torch.mean(target_dist.square())
        return 0.5 * (loss_self + loss_target)

    def _opening_plane_normal_loss(self, mesh: Meshes, target_plane=None, sign_invariant: bool = True):
        verts = mesh.verts_packed()
        _, normal_self = self._plane_from_points(verts)
        if normal_self is None:
            return torch.zeros(1, device=verts.device, dtype=verts.dtype).squeeze(0)
        if target_plane is None or target_plane[1] is None:
            return torch.zeros(1, device=verts.device, dtype=verts.dtype).squeeze(0)
        target_normal = target_plane[1].to(device=verts.device, dtype=verts.dtype)
        dot = torch.sum(normal_self * target_normal)
        if sign_invariant:
            dot = torch.abs(dot)
        dot = torch.clamp(dot, -1.0, 1.0)
        return 1.0 - dot

    def _opening_centroid_axis_loss(self, mesh: Meshes, target_plane=None,
                                    axis_weight: float = 0.5,
                                    sign_invariant: bool = True):
        """Robust low-frequency opening loss: centroid distance + plane-axis alignment.
        Cheap, well-conditioned, helps when the rim is far from the target.
        Uses verts_packed() so gradients flow through GHD parameters via the warped
        opening mesh.
        """
        verts = mesh.verts_packed()
        centroid_self, normal_self = self._plane_from_points(verts)
        if centroid_self is None:
            return torch.zeros(1, device=verts.device, dtype=verts.dtype).squeeze(0)
        if target_plane is None or target_plane[0] is None:
            return torch.zeros(1, device=verts.device, dtype=verts.dtype).squeeze(0)
        tc = target_plane[0].to(device=verts.device, dtype=verts.dtype)
        loss_c = (centroid_self - tc).pow(2).sum()
        loss_axis = torch.zeros((), device=verts.device, dtype=verts.dtype)
        if normal_self is not None and target_plane[1] is not None:
            tn = target_plane[1].to(device=verts.device, dtype=verts.dtype)
            dot = torch.sum(normal_self * tn)
            if sign_invariant:
                dot = torch.abs(dot)
            dot = torch.clamp(dot, -1.0, 1.0)
            loss_axis = 1.0 - dot
        return loss_c + float(axis_weight) * loss_axis

    def _order_loop_by_plane_angle(self, points: torch.Tensor):
        """Reorder a closed-loop point set by polar angle around the centroid in the
        best-fit plane. Returns points in cyclic order. Used to build a canonical
        parameterisation for both warped-rim and target-rim point sets so that
        ordered cyclic correspondence losses become well-defined even when the
        target rim is supplied as an unordered vertex set.
        """
        pts = points
        if pts.ndim == 3 and pts.shape[0] == 1:
            pts = pts[0]
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 3:
            return pts
        finite_mask = torch.isfinite(pts).all(dim=-1)
        pts = pts[finite_mask]
        if pts.shape[0] < 3:
            return pts
        centroid, normal = self._plane_from_points(pts)
        if centroid is None or normal is None:
            return pts
        centered = pts - centroid
        helper = torch.tensor([1.0, 0.0, 0.0], device=pts.device, dtype=pts.dtype)
        if torch.abs(torch.dot(helper, normal)) > 0.9:
            helper = torch.tensor([0.0, 1.0, 0.0], device=pts.device, dtype=pts.dtype)
        u = helper - normal * torch.dot(helper, normal)
        u_norm = u.norm()
        if (not torch.isfinite(u_norm)) or float(u_norm.detach().cpu().item()) <= 1e-8:
            return pts
        u = u / u_norm
        v = torch.linalg.cross(normal, u)
        x = centered @ u
        y = centered @ v
        angles = torch.atan2(y, x)
        order = torch.argsort(angles)
        return pts.index_select(0, order)

    def _ordered_boundary_loop_points(self, mesh: Meshes):
        verts = mesh.verts_packed()
        faces = mesh.faces_packed().long()
        if verts.shape[0] < 3:
            return verts
        if faces.numel() == 0:
            return verts
        try:
            faces_np = faces.detach().cpu().numpy()
            edges = torch.cat(
                (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
                dim=0,
            )
            edges, _ = torch.sort(edges, dim=1)
            edges_np = edges.detach().cpu().numpy()
            uniq_edges, counts = torch.unique(edges, dim=0, return_counts=True)
            boundary_edges = uniq_edges[counts == 1]
            if boundary_edges.numel() == 0:
                return verts
            boundary_edges_np = boundary_edges.detach().cpu().numpy()

            adjacency = {}
            for a, b in boundary_edges_np.tolist():
                adjacency.setdefault(int(a), []).append(int(b))
                adjacency.setdefault(int(b), []).append(int(a))
            if len(adjacency) < 3:
                return verts.index_select(0, torch.unique(boundary_edges.reshape(-1)))

            start = min(adjacency.keys())
            order = [start]
            prev = None
            curr = start
            max_steps = max(len(adjacency) + 2, 8)
            for _ in range(max_steps):
                neigh = adjacency.get(curr, [])
                if len(neigh) == 0:
                    break
                if prev is None:
                    nxt = neigh[0]
                else:
                    nxt = neigh[0] if neigh[0] != prev else (neigh[1] if len(neigh) > 1 else neigh[0])
                if nxt == start:
                    break
                if nxt in order:
                    break
                order.append(nxt)
                prev, curr = curr, nxt

            if len(order) < 3:
                boundary_idx = torch.unique(boundary_edges.reshape(-1))
                return verts.index_select(0, boundary_idx)
            idx = torch.tensor(order, device=verts.device, dtype=torch.long)
            return verts.index_select(0, idx)
        except Exception:
            edges = torch.cat(
                (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
                dim=0,
            )
            edges, _ = torch.sort(edges, dim=1)
            uniq_edges, counts = torch.unique(edges, dim=0, return_counts=True)
            boundary_edges = uniq_edges[counts == 1]
            if boundary_edges.numel() == 0:
                return verts
            boundary_idx = torch.unique(boundary_edges.reshape(-1))
            return verts.index_select(0, boundary_idx)

    def _resample_closed_curve_torch(self, points: torch.Tensor, num_samples: int):
        pts = points
        if pts.ndim == 3 and pts.shape[0] == 1:
            pts = pts[0]
        if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3:
            return pts
        n = max(8, int(num_samples))
        ring = torch.cat((pts, pts[:1]), dim=0)
        seg = torch.norm(ring[1:] - ring[:-1], dim=1)
        total = seg.sum()
        if (not torch.isfinite(total)) or float(total.detach().cpu().item()) <= 1e-8:
            return pts
        cum = torch.cat((torch.zeros(1, device=pts.device, dtype=pts.dtype), torch.cumsum(seg, dim=0)), dim=0)
        query = torch.linspace(0.0, 1.0, steps=n + 1, device=pts.device, dtype=pts.dtype)[:-1] * total
        idx = torch.searchsorted(cum, query, right=True) - 1
        idx = torch.clamp(idx, 0, pts.shape[0] - 1)
        local = query - cum.index_select(0, idx)
        seg_sel = seg.index_select(0, idx).clamp_min(1e-8)
        alpha = (local / seg_sel).unsqueeze(-1)
        p0 = ring.index_select(0, idx)
        p1 = ring.index_select(0, idx + 1)
        return (1.0 - alpha) * p0 + alpha * p1

    def _cyclic_sequence_l2_loss(self, seq_a: torch.Tensor, seq_b: torch.Tensor, allow_flip: bool = True):
        if seq_a is None or seq_b is None:
            device = self.mesh_std.device
            dtype = self.mesh_std.verts_packed().dtype
            return torch.zeros(1, device=device, dtype=dtype).squeeze(0)
        a = seq_a
        b = seq_b
        if a.ndim == 3 and a.shape[0] == 1:
            a = a[0]
        if b.ndim == 3 and b.shape[0] == 1:
            b = b[0]
        if a.ndim == 1:
            a = a.unsqueeze(-1)
        if b.ndim == 1:
            b = b.unsqueeze(-1)
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] < 2 or b.shape[0] < 2:
            if a.shape == b.shape and a.numel() > 0:
                return torch.mean((a - b) ** 2)
            device = a.device if hasattr(a, "device") else self.mesh_std.device
            dtype = a.dtype if hasattr(a, "dtype") else self.mesh_std.verts_packed().dtype
            return torch.zeros(1, device=device, dtype=dtype).squeeze(0)
        if a.shape != b.shape:
            if a.shape[1] != b.shape[1]:
                return torch.zeros(1, device=a.device, dtype=a.dtype).squeeze(0)
            n = min(int(a.shape[0]), int(b.shape[0]))
            a = a[:n]
            b = b[:n]
        n = int(a.shape[0])
        base_idx = torch.arange(n, device=a.device)
        shift_idx = (base_idx.unsqueeze(0) + base_idx.unsqueeze(1)) % n

        def _shift_losses(base_seq: torch.Tensor):
            shifted = base_seq[shift_idx]
            sq = (a.unsqueeze(0) - shifted).square().sum(dim=-1)
            return sq.mean(dim=-1)

        losses = [_shift_losses(b)]
        if allow_flip:
            losses.append(_shift_losses(torch.flip(b, dims=[0])))
        loss_all = torch.cat(losses, dim=0)
        return torch.min(loss_all) if loss_all.numel() > 0 else torch.zeros(1, device=a.device, dtype=a.dtype).squeeze(0)

    def _rim_curvature_profile(self, points: torch.Tensor):
        pts = points
        if pts.ndim == 3 and pts.shape[0] == 1:
            pts = pts[0]
        if pts.ndim != 2 or pts.shape[0] < 4 or pts.shape[1] != 3:
            return None
        prev = torch.roll(pts, shifts=1, dims=0)
        nxt = torch.roll(pts, shifts=-1, dims=0)
        e_prev = F.normalize(pts - prev, dim=-1, eps=1e-8)
        e_next = F.normalize(nxt - pts, dim=-1, eps=1e-8)
        dot = torch.clamp(torch.sum(e_prev * e_next, dim=-1), -1.0, 1.0)
        return 1.0 - dot

    def _cyclic_profile_l2_loss(self, profile_a: torch.Tensor, profile_b: torch.Tensor):
        if profile_a is None or profile_b is None:
            device = self.mesh_std.device
            dtype = self.mesh_std.verts_packed().dtype
            return torch.zeros(1, device=device, dtype=dtype).squeeze(0)
        return self._cyclic_sequence_l2_loss(profile_a.reshape(-1, 1), profile_b.reshape(-1, 1), allow_flip=True)

    def _opening_ordered_rim_loss(self, mesh: Meshes, target_rim_points, num_samples: int = 96):
        warped_loop = self._ordered_boundary_loop_points(mesh)
        # If topological walk failed (returned an unordered vertex subset), reorder
        # by plane angle so we still get a meaningful cyclic parameterisation.
        if warped_loop.shape[0] < 3:
            warped_loop = self._order_loop_by_plane_angle(mesh.verts_packed())
        target_loop = torch.as_tensor(target_rim_points, dtype=warped_loop.dtype, device=warped_loop.device)
        if target_loop.ndim == 3 and target_loop.shape[0] == 1:
            target_loop = target_loop[0]
        if target_loop.ndim != 2 or target_loop.shape[0] < 3 or target_loop.shape[1] != 3:
            fallback_warped = self._sample_opening_boundary_points(mesh, num_samples)
            fallback_target = self._sample_point_cloud(target_rim_points, num_samples)
            loss_p, _ = chamfer_distance(fallback_warped, fallback_target, x_normals=None, y_normals=None)
            return loss_p, fallback_warped, fallback_target
        # Target rim points are typically supplied as an unordered vertex set.
        # Sort them around the best-fit plane so arc-length resampling produces a
        # consistent cyclic order matching the warped loop after shift/flip search.
        target_loop = self._order_loop_by_plane_angle(target_loop)
        warped_loop = self._order_loop_by_plane_angle(warped_loop)
        sample_n = max(16, int(num_samples))
        warped_rs = self._resample_closed_curve_torch(warped_loop, sample_n)
        target_rs = self._resample_closed_curve_torch(target_loop, sample_n)
        loss_p = self._cyclic_sequence_l2_loss(warped_rs, target_rs, allow_flip=True)
        return loss_p, warped_rs.unsqueeze(0), target_rs.unsqueeze(0)

    def _opening_rim_curvature_loss(self, mesh: Meshes, target_rim_points, num_samples: int = 64):
        warped_loop = self._ordered_boundary_loop_points(mesh)
        if warped_loop.shape[0] < 3:
            warped_loop = self._order_loop_by_plane_angle(mesh.verts_packed())
        target_loop = torch.as_tensor(target_rim_points, dtype=warped_loop.dtype, device=warped_loop.device)
        if target_loop.ndim == 3 and target_loop.shape[0] == 1:
            target_loop = target_loop[0]
        target_loop = self._order_loop_by_plane_angle(target_loop)
        warped_loop = self._order_loop_by_plane_angle(warped_loop)
        warped_rs = self._resample_closed_curve_torch(warped_loop, num_samples)
        target_rs = self._resample_closed_curve_torch(target_loop, num_samples)
        warped_profile = self._rim_curvature_profile(warped_rs)
        target_profile = self._rim_curvature_profile(target_rs)
        return self._cyclic_profile_l2_loss(warped_profile, target_profile)
    
    def forward(self, meshes_scr: Meshes, trg: Union[Meshes, torch.Tensor], loss_list:dict, B=1):
        # calcualte all types of losses for self and meshes_scr
        loss_dict = {}
        # Tier-A speedup: only run the (very expensive) global surface chamfer
        # when a caller actually requested loss_p0 or loss_n1. The decap path
        # in meshloss_do.forward_opa_do calls this method twice per epoch -
        # once for surface losses, once for pure regularisation - and the old
        # code sampled 2 * sample_num points unconditionally each time.
        if any(k in loss_list for k in ('loss_p0', 'loss_n1')):
            sample_scr, normals_scr = self._safe_sample_points_from_meshes(meshes_scr, self.sample_num, return_normals=True)
            if isinstance(trg, Meshes):
                sample_trg, normals_trg = self._safe_sample_points_from_meshes(trg, self.sample_num, return_normals=True)
                loss_p0, loss_n1 = chamfer_distance(sample_scr, sample_trg, x_normals=normals_scr, y_normals=normals_trg)
            else:
                sample_trg = trg
                loss_p0, loss_n1 = chamfer_distance(sample_scr, sample_trg, x_normals=None, y_normals=None)
                loss_n1 = 1e-5
            if 'loss_p0' in loss_list:
                loss_dict['loss_p0'] = loss_p0 if not torch.isnan(loss_p0) else torch.Tensor([0.0]).to(self.device)
            if 'loss_n1' in loss_list:
                loss_dict['loss_n1'] = loss_n1 if not torch.isnan(loss_n1) else torch.Tensor([0.0]).to(self.device)
        if 'loss_laplacian' in loss_list:
            laplacain_vect = mesh_laplacian_smoothing(meshes_scr, method="cot")
            loss_dict['loss_laplacian'] = laplacain_vect if not torch.isnan(laplacain_vect) else torch.Tensor([0.0]).to(self.device)
        if 'loss_edge' in loss_list:
            mesh_edge_loss_item = mesh_edge_loss(meshes_scr,self.edge_lenth.to(meshes_scr.device))
            loss_dict['loss_edge'] = mesh_edge_loss_item if not torch.isnan(mesh_edge_loss_item) else torch.Tensor([0.0]).to(self.device)
        if 'loss_consistency' in loss_list:
            mesh_normal_consistency_item = mesh_normal_consistency(meshes_scr)
            loss_dict['loss_consistency'] = mesh_normal_consistency_item if not torch.isnan(mesh_normal_consistency_item) else torch.Tensor([0.0]).to(self.device)
        if 'loss_rigid' in loss_list:
            verts_scr = meshes_scr.verts_padded()
            verts_std = self.mesh_std.verts_packed()
            loss_dict['loss_rigid'] = torch.zeros_like(verts_std[:,:1])
            for i in range(B):
                rigid_i = self.rigidloss.forward(verts_scr[i])
                loss_dict['loss_rigid'] += rigid_i if not torch.isnan(rigid_i) else torch.Tensor([0.0]).to(self.device)
            loss_dict['loss_rigid'] = loss_dict['loss_rigid'].mean()/B
        return loss_dict
