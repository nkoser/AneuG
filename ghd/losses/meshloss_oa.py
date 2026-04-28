import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__),'.','..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pytorch3d.loss import chamfer_distance,mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss
from pytorch3d.ops import sample_points_from_meshes, cot_laplacian, padded_to_packed
from .meshloss import Mesh_loss
from ghd.fitting.registration import RegistrationwOpeningAlignment
import torch
from pytorch3d.structures import Meshes
import numpy as np
import itertools


class Mesh_loss_opening_alignment(Mesh_loss):
    def __init__(self, args, oa_class_canonical: RegistrationwOpeningAlignment, oa_class_target: RegistrationwOpeningAlignment):
        self.device = torch.device(args.device)
        base_shape = getattr(oa_class_canonical, "mesh_target_p3d").to(self.device)  # transform o3d mesh to pytorch3d mesh
        self.target_mesh = getattr(oa_class_target, "mesh_target_p3d").to(self.device)
        # TODO: check if chamfer loss cares about normal direction
        # register static meshes for the target mesh
        self.target_openings = oa_class_target.return_opening_Meshes_static(register_normal=False)
        self.sample_num = args.sample_num
        self.op_sample_num = args.op_sample_num
        self.opening_match_mode = str(getattr(args, "opening_match_mode", "permutation")).strip().lower()
        self.opening_normal_bidirectional = bool(int(getattr(args, "opening_normal_bidirectional", 1)))
        self.opening_loss_mode = str(getattr(args, "opening_loss_mode", "surface")).strip().lower()
        self.preview_opening_points_warped = []
        self.preview_opening_points_target = []
        self.preview_opening_normal_points_warped = []
        self.preview_opening_normals_warped = []
        self.preview_opening_normal_points_target = []
        self.preview_opening_normals_target = []
        if hasattr(oa_class_target, "return_opening_rim_pointclouds_static"):
            self.target_opening_rims = oa_class_target.return_opening_rim_pointclouds_static(prefer_source=True)
        else:
            self.target_opening_rims = [opening.verts_padded().detach().cpu() for opening in self.target_openings]
        if hasattr(oa_class_target, "return_opening_planes_static"):
            self.target_opening_planes = oa_class_target.return_opening_planes_static(prefer_source=True)
        else:
            self.target_opening_planes = []
            for opening in self.target_openings:
                centroid, normal = self._plane_from_points(opening.verts_packed().to(self.device))
                self.target_opening_planes.append((centroid, normal))
        super(Mesh_loss_opening_alignment, self).__init__(mesh_std=base_shape, sample_num=self.sample_num)

    def _solve_opening_assignment(self, warped_openings):
        num_pairs = min(len(warped_openings), len(self.target_openings))
        if num_pairs <= 1 or self.opening_match_mode in ("none", "index", "identity"):
            return list(range(num_pairs))

        warped_points = []
        target_points = []
        for i in range(num_pairs):
            if self._opening_mode_uses_boundary(self.opening_loss_mode):
                warped_points.append(self._sample_opening_boundary_points(warped_openings[i], self.op_sample_num))
                target_points.append(self._sample_point_cloud(self.target_opening_rims[i], self.op_sample_num))
            else:
                warped_points.append(
                    self._safe_sample_points_from_meshes(
                        warped_openings[i], self.op_sample_num, return_normals=False
                    )
                )
                target_points.append(
                    self._safe_sample_points_from_meshes(
                        self.target_openings[i].to(self.device), self.op_sample_num, return_normals=False
                    )
                )

        cost = np.zeros((num_pairs, num_pairs), dtype=np.float64)
        for i in range(num_pairs):
            for j in range(num_pairs):
                loss_p, _ = chamfer_distance(warped_points[i], target_points[j], x_normals=None, y_normals=None)
                cost[i, j] = float(loss_p.detach().cpu().item())

        if num_pairs <= 8:
            best_perm = None
            best_score = np.inf
            for perm in itertools.permutations(range(num_pairs), num_pairs):
                score = float(np.sum([cost[i, perm[i]] for i in range(num_pairs)]))
                if score < best_score:
                    best_score = score
                    best_perm = list(perm)
            return best_perm if best_perm is not None else list(range(num_pairs))

        used = set()
        assign = []
        for i in range(num_pairs):
            row = np.argsort(cost[i])
            pick = None
            for j in row:
                jj = int(j)
                if jj not in used:
                    pick = jj
                    break
            if pick is None:
                pick = int(row[0])
            used.add(pick)
            assign.append(pick)
        return assign

    def forward_opening_alignment(self, warped_mesh, warped_openings, loss_weighting: dict, B=1):
        # TODO: write a switch so children classes can skip opa losses
        loss_dict = self.forward(meshes_scr=warped_mesh, trg=self.target_mesh, loss_list=loss_weighting, B=B)
        loss_p_list = []
        loss_surface_p_list = []
        loss_n_list = []
        loss_plane_list = []
        self.preview_opening_points_warped = []
        self.preview_opening_points_target = []
        self.preview_opening_normal_points_warped = []
        self.preview_opening_normals_warped = []
        self.preview_opening_normal_points_target = []
        self.preview_opening_normals_target = []
        if any(k in loss_weighting for k in ('loss_openings_p', 'loss_openings_n', 'loss_openings_plane', 'loss_openings_surface_p', 'loss_openings_rim_curvature')):
            num_pairs = min(len(warped_openings), len(self.target_openings))
            assignment = self._solve_opening_assignment(warped_openings)
            loss_rim_curve_list = []
            for idx in range(num_pairs):
                trg_idx = int(assignment[idx]) if idx < len(assignment) else idx
                target_opening = self.target_openings[trg_idx].to(self.device)
                target_plane = self.target_opening_planes[trg_idx] if trg_idx < len(self.target_opening_planes) else None
                if self._opening_mode_uses_boundary(self.opening_loss_mode):
                    if self._opening_mode_uses_ordered_rim(self.opening_loss_mode):
                        ordered_samples = max(64, min(160, int(self.op_sample_num // 12)))
                        loss_p, pcd_wo, pcd_to = self._opening_ordered_rim_loss(
                            warped_openings[idx],
                            self.target_opening_rims[trg_idx],
                            num_samples=ordered_samples,
                        )
                    else:
                        pcd_wo = self._sample_opening_boundary_points(
                            warped_openings[idx], self.op_sample_num
                        )
                        pcd_to = self._sample_point_cloud(self.target_opening_rims[trg_idx], self.op_sample_num)
                        loss_p, _ = chamfer_distance(pcd_wo, pcd_to, x_normals=None, y_normals=None)

                    pcd_wo_n = pcd_wo
                    pcd_to_n = pcd_to
                    _, warped_normal = self._plane_from_points(pcd_wo[0])
                    _, target_normal = target_plane if target_plane is not None else (None, None)
                    if warped_normal is None:
                        nor_wo = torch.zeros_like(pcd_wo_n)
                    else:
                        nor_wo = warped_normal.reshape(1, 1, 3).repeat(1, pcd_wo_n.shape[1], 1)
                    if target_normal is None:
                        nor_to = torch.zeros_like(pcd_to_n)
                    else:
                        target_normal = target_normal.to(device=self.device, dtype=pcd_to_n.dtype)
                        nor_to = target_normal.reshape(1, 1, 3).repeat(1, pcd_to_n.shape[1], 1)
                    loss_n = self._opening_plane_normal_loss(
                        warped_openings[idx],
                        target_plane=target_plane,
                        sign_invariant=self.opening_normal_bidirectional,
                    )
                else:
                    pcd_wo, nor_wo = self._safe_sample_points_from_meshes(
                        warped_openings[idx], self.op_sample_num, return_normals=True
                    )
                    pcd_to, nor_to = self._safe_sample_points_from_meshes(
                        target_opening, self.op_sample_num, return_normals=True
                    )
                    loss_p, loss_n = chamfer_distance(pcd_wo, pcd_to, x_normals=nor_wo, y_normals=nor_to)
                    if self.opening_normal_bidirectional:
                        _, loss_n_flip = chamfer_distance(
                            pcd_wo, pcd_to, x_normals=nor_wo, y_normals=(-1.0 * nor_to)
                        )
                        if torch.isfinite(loss_n_flip):
                            loss_n = torch.minimum(loss_n, loss_n_flip)
                if 'loss_openings_surface_p' in loss_weighting:
                    pcd_wo_surface = self._safe_sample_points_from_meshes(
                        warped_openings[idx], self.op_sample_num, return_normals=False
                    )
                    pcd_to_surface = self._safe_sample_points_from_meshes(
                        target_opening, self.op_sample_num, return_normals=False
                    )
                    loss_surface_p, _ = chamfer_distance(
                        pcd_wo_surface,
                        pcd_to_surface,
                        x_normals=None,
                        y_normals=None,
                    )
                    loss_surface_p_list.append(
                        loss_surface_p if torch.isfinite(loss_surface_p) else torch.tensor(0.0, device=self.device)
                    )
                self.preview_opening_points_warped.append(pcd_wo.detach().cpu())
                self.preview_opening_points_target.append(pcd_to.detach().cpu())
                self.preview_opening_normal_points_warped.append(pcd_wo_n.detach().cpu())
                self.preview_opening_normals_warped.append(nor_wo.detach().cpu())
                self.preview_opening_normal_points_target.append(pcd_to_n.detach().cpu())
                self.preview_opening_normals_target.append(nor_to.detach().cpu())
                if 'loss_openings_plane' in loss_weighting:
                    loss_plane = self._opening_planarity_loss(warped_openings[idx], target_plane=target_plane)
                    loss_plane_list.append(
                        loss_plane if torch.isfinite(loss_plane) else torch.tensor(0.0, device=self.device)
                    )
                if 'loss_openings_rim_curvature' in loss_weighting:
                    loss_rim_curve = self._opening_rim_curvature_loss(
                        warped_openings[idx],
                        self.target_opening_rims[trg_idx],
                        num_samples=max(48, min(128, int(self.op_sample_num // 16))),
                    )
                    loss_rim_curve_list.append(
                        loss_rim_curve if torch.isfinite(loss_rim_curve) else torch.tensor(0.0, device=self.device)
                    )
                loss_p_list.append(loss_p if torch.isfinite(loss_p) else torch.Tensor([0.0]).to(self.device))
                loss_n_list.append(loss_n if torch.isfinite(loss_n) else torch.Tensor([0.0]).to(self.device))
            if 'loss_openings_p' in loss_weighting:
                loss_dict['loss_openings_p'] = loss_p_list
            if 'loss_openings_surface_p' in loss_weighting and len(loss_surface_p_list) > 0:
                loss_dict['loss_openings_surface_p'] = torch.mean(torch.stack(loss_surface_p_list))
            if 'loss_openings_n' in loss_weighting:
                loss_dict['loss_openings_n'] = loss_n_list
            if 'loss_openings_plane' in loss_weighting and len(loss_plane_list) > 0:
                loss_dict['loss_openings_plane'] = torch.mean(torch.stack(loss_plane_list))
            if 'loss_openings_rim_curvature' in loss_weighting and len(loss_rim_curve_list) > 0:
                loss_dict['loss_openings_rim_curvature'] = torch.mean(torch.stack(loss_rim_curve_list))
        return loss_dict


def o3d_mesh_to_pytorch3d(o3d_mesh) -> Meshes:
    verts = torch.Tensor(np.asarray(o3d_mesh.vertices))
    faces = torch.Tensor(np.asarray(o3d_mesh.triangles))
    pytorch3d_mesh = Meshes(verts=[verts], faces=[faces])
    return pytorch3d_mesh
