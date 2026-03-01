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
        super(Mesh_loss_opening_alignment, self).__init__(mesh_std=base_shape, sample_num=self.sample_num)

    def _solve_opening_assignment(self, warped_openings):
        num_pairs = min(len(warped_openings), len(self.target_openings))
        if num_pairs <= 1 or self.opening_match_mode in ("none", "index", "identity"):
            return list(range(num_pairs))

        warped_points = []
        target_points = []
        for i in range(num_pairs):
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
        loss_n_list = []
        if ('loss_openings_p' in loss_weighting) or ('loss_openings_n' in loss_weighting):
            num_pairs = min(len(warped_openings), len(self.target_openings))
            assignment = self._solve_opening_assignment(warped_openings)
            for idx in range(num_pairs):
                trg_idx = int(assignment[idx]) if idx < len(assignment) else idx
                pcd_wo, nor_wo = self._safe_sample_points_from_meshes(
                    warped_openings[idx], self.op_sample_num, return_normals=True
                )
                pcd_to, nor_to = self._safe_sample_points_from_meshes(
                    self.target_openings[trg_idx].to(self.device), self.op_sample_num, return_normals=True
                )
                loss_p, loss_n = chamfer_distance(pcd_wo, pcd_to, x_normals=nor_wo, y_normals=nor_to)
                if self.opening_normal_bidirectional:
                    _, loss_n_flip = chamfer_distance(
                        pcd_wo, pcd_to, x_normals=nor_wo, y_normals=(-1.0 * nor_to)
                    )
                    if torch.isfinite(loss_n_flip):
                        loss_n = torch.minimum(loss_n, loss_n_flip)
                loss_p_list.append(loss_p if torch.isfinite(loss_p) else torch.Tensor([0.0]).to(self.device))
                loss_n_list.append(loss_n if torch.isfinite(loss_n) else torch.Tensor([0.0]).to(self.device))
            if 'loss_openings_p' in loss_weighting:
                loss_dict['loss_openings_p'] = loss_p_list
            if 'loss_openings_n' in loss_weighting:
                loss_dict['loss_openings_n'] = loss_n_list
        return loss_dict


def o3d_mesh_to_pytorch3d(o3d_mesh) -> Meshes:
    verts = torch.Tensor(np.asarray(o3d_mesh.vertices))
    faces = torch.Tensor(np.asarray(o3d_mesh.triangles))
    pytorch3d_mesh = Meshes(verts=[verts], faces=[faces])
    return pytorch3d_mesh
