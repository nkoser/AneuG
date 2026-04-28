import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__),'.','..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pytorch3d.loss import chamfer_distance,mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss
from pytorch3d.ops import sample_points_from_meshes, cot_laplacian, padded_to_packed
from ghd.losses.meshloss import Mesh_loss
from ghd.fitting.registration import RegistrationwOpeningAlignment
import torch
from pytorch3d.structures import Meshes
import numpy as np
from pytorch3d.loss import chamfer_distance
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from ghd.base.mesh_geometry3 import Winding_Occupancy
from ghd.losses.diceloss import BinaryDiceLoss, BinaryDiceLoss_Weighted
import math
import pyvista as pv
import trimesh
import itertools


class Mesh_loss_differentiable_occupancy(Mesh_loss):
    def __init__(self, args, oa_class_canonical: RegistrationwOpeningAlignment, oa_class_target: RegistrationwOpeningAlignment):
        self.device = torch.device(args.device)
        base_shape = getattr(oa_class_canonical, "mesh_target_p3d").to(self.device)  # transform o3d mesh to pytorch3d mesh
        self.target_mesh = getattr(oa_class_target, "mesh_target_p3d").to(self.device)
        # TODO: check if chamfer loss cares about normal direction
        # register static meshes for the target mesh
        self.target_openings = oa_class_target.return_opening_Meshes_static(register_normal=False)
        self.sample_num = args.sample_num
        self.op_sample_num = args.op_sample_num
        self.do_dpi = getattr(args, "do_dpi", 32)
        self.do_style = None
        self.do_module = None  # upsampling module if using uniform
        self.do_loss_type = getattr(args, "do_loss_type", "dice_loss")
        self.do_number = getattr(args, "do_number", 10000)
        self.redo_do_points = bool(int(getattr(args, "redo_do_points", 0)))
        self.do_sampling_max_loops = max(8, int(getattr(args, "do_sampling_max_loops", 128)))
        self.root_target = args.root_target
        self.name_target = args.name_target
        self.weights_attention = None
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

        # ------------------------------------------------------------------
        # Build *decapped* face masks so that loss_p0 (global surface
        # chamfer) does not sample from cap triangles.  Cap samples create
        # conflicting gradients: the chamfer pulls them toward the dome
        # surface while the opening-specific losses want them planar.
        # ------------------------------------------------------------------
        self._decap_enabled = bool(int(getattr(args, "decap_chamfer", 1)))
        self._canon_cap_face_mask = None   # bool [F] – True = keep
        self._target_cap_face_mask = None
        self._target_mesh_decapped = None
        if self._decap_enabled:
            self._canon_cap_face_mask = self._build_decap_mask(
                base_shape, oa_class_canonical)
            self._target_cap_face_mask = self._build_decap_mask(
                self.target_mesh, oa_class_target)
            if self._target_cap_face_mask is not None:
                tgt_faces = self.target_mesh.faces_packed()
                tgt_verts = self.target_mesh.verts_padded()
                kept = tgt_faces[self._target_cap_face_mask]
                self._target_mesh_decapped = Meshes(
                    verts=tgt_verts, faces=kept.unsqueeze(0)
                ).to(self.device)
                n_removed = int((~self._target_cap_face_mask).sum())
                print(f"[Decap] Target: removed {n_removed} cap faces from chamfer sampling")

        super(Mesh_loss_differentiable_occupancy, self).__init__(mesh_std=base_shape, sample_num=self.sample_num)

    @staticmethod
    def _build_decap_mask(mesh: Meshes, oa_class) -> torch.Tensor:
        """Return a boolean mask [F] that is True for non-cap faces.

        Cap faces are identified as faces whose **all three** vertices
        belong to the set of opening rim vertex indices.  This is robust
        regardless of how the face map is stored internally.
        """
        op_v_indices = getattr(oa_class, "op_v_indices", [])
        if not op_v_indices:
            return None
        # Collect all opening rim vertex indices across all openings
        rim_verts = set()
        for vlist in op_v_indices:
            for vi in vlist:
                rim_verts.add(int(vi))
        if not rim_verts:
            return None
        faces = mesh.faces_packed().cpu().numpy()  # (F, 3)
        n_faces = faces.shape[0]
        mask = np.ones(n_faces, dtype=bool)
        for fi in range(n_faces):
            if int(faces[fi, 0]) in rim_verts and \
               int(faces[fi, 1]) in rim_verts and \
               int(faces[fi, 2]) in rim_verts:
                mask[fi] = False
        n_removed = int((~mask).sum())
        if n_removed == 0:
            return None
        return torch.from_numpy(mask)

    def _decapped_warped_mesh(self, warped_mesh: Meshes) -> Meshes:
        """Build a decapped version of the warped mesh using the canonical cap mask."""
        if self._canon_cap_face_mask is None:
            return warped_mesh
        faces = warped_mesh.faces_packed()
        verts = warped_mesh.verts_padded()
        kept = faces[self._canon_cap_face_mask.to(faces.device)]
        return Meshes(verts=verts, faces=kept.unsqueeze(0))

    def _target_occupancy_prob(self, query_points, device, warned_nonfinite=False):
        winding_val = Winding_Occupancy(self.target_mesh.to(device), query_points)
        finite_mask = torch.isfinite(winding_val)
        if (not warned_nonfinite) and (not bool(finite_mask.all())):
            invalid = int((~finite_mask).sum().item())
            print(
                f"[DO] Non-finite winding values for {self.name_target}: "
                f"{invalid}/{int(winding_val.numel())}. Replacing invalid values before thresholding."
            )
            warned_nonfinite = True
        winding_val = torch.nan_to_num(winding_val, nan=0.0, posinf=1.0, neginf=-1.0)
        do_gt = torch.sigmoid((winding_val.abs() - 0.5) * 100)
        return do_gt, warned_nonfinite

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

        # Greedy fallback for larger opening counts.
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

    def get_static_mask_probabilistic(self):
        device = torch.device('cpu')
        bound_box1 = self.mesh_std.get_bounding_boxes().squeeze(0).to(device)
        bound_box2 = self.target_mesh.get_bounding_boxes().squeeze(0).to(device)
        box_min = torch.min(torch.cat((bound_box1, bound_box2), dim=1), dim=-1)[0]
        box_max = torch.max(torch.cat((bound_box1, bound_box2), dim=1), dim=-1)[0]
        bound_box = torch.cat((box_min.unsqueeze(-1), box_max.unsqueeze(-1)), dim=-1)
        radius = bound_box[:, 1] - bound_box[:, 0]
        bound_box[:, 0] -= 0.125 * radius
        bound_box[:, 1] += 0.125 * radius
        dpi = torch.max((bound_box[:, 1]-bound_box[:, 0]) / torch.min((bound_box[:, 1]-bound_box[:, 0])/self.do_dpi)).int().item()
        scale = []
        for axis in range(3):
            scale.append(torch.linspace(bound_box[axis, 0], bound_box[axis, 1], dpi))
        x_grid, y_grid, z_grid = torch.meshgrid(scale)
        voxel = torch.stack([x_grid, y_grid, z_grid], dim=-1)
        voxel = voxel.view(-1, 3).to(device)
        batch_size = round(voxel.shape[0]/10)
        voxel_set = TensorDataset(voxel)
        loader = DataLoader(voxel_set, batch_size=batch_size, shuffle=False)
        distance_list = []
        for batch in iter(loader):
            reference = self.mesh_std.verts_packed().unsqueeze(0).expand(batch[0].shape[0], -1, -1).to(device)
            distances, _ = chamfer_distance(batch[0].unsqueeze(1).to(device), reference, batch_reduction=None)
            distance_list.append(distances)
        distances = torch.cat(distance_list, dim=0)  # zero dim tensor indicating distance to the mesh
        sorted_indices = torch.argsort(distances).to(device)
        min_prob = 0.1
        max_prob = 0.9
        prob_step = (max_prob - min_prob) / (distances.size(0) - 1)
        probabilities = torch.linspace(min_prob, max_prob, steps=distances.size(0)).to(device)
        sorted_probabilities = probabilities[sorted_indices]
        mask = torch.rand_like(distances) > sorted_probabilities
        masked_tensor = torch.cat((voxel, distances.unsqueeze(-1)), dim=-1)
        masked_tensor = masked_tensor[mask, :3]
        print('static query points registered.')
        return masked_tensor  # [N, 3]

    def get_static_mask_uniform(self):
        device = torch.device('cpu')
        bound_box1 = self.mesh_std.get_bounding_boxes().squeeze(0).to(device)
        bound_box2 = self.target_mesh.get_bounding_boxes().squeeze(0).to(device)
        box_min = torch.min(torch.cat((bound_box1, bound_box2), dim=1), dim=-1)[0]
        box_max = torch.max(torch.cat((bound_box1, bound_box2), dim=1), dim=-1)[0]
        bound_box = torch.cat((box_min.unsqueeze(-1), box_max.unsqueeze(-1)), dim=-1)
        radius = bound_box[:, 1] - bound_box[:, 0]
        bound_box[:, 0] -= 0.125 * radius
        bound_box[:, 1] += 0.125 * radius
        dpi = torch.max((bound_box[:, 1] - bound_box[:, 0]) / torch.min(
            (bound_box[:, 1] - bound_box[:, 0]) / self.do_dpi)).int().item()
        scale = []
        for axis in range(3):
            scale.append(torch.linspace(bound_box[axis, 0], bound_box[axis, 1], dpi))
        x_grid, y_grid, z_grid = torch.meshgrid(scale)
        voxel = torch.stack([x_grid, y_grid, z_grid], dim=-1).to(device)  # [N,3]
        voxel = voxel.view(-1, 3).to(device)
        # mod = torch.nn.Upsample(scale_factor=2, mode='trilinear')
        # voxel = mod(voxel)
        # voxel = voxel.view(-1, 3)  #
        return voxel

    def get_static_mask_number_control(self, num_in=10000, num_out=10000):
        device = torch.device('cpu')
        batch_size = num_in + num_out
        # get bounding box
        bound_box1 = self.mesh_std.get_bounding_boxes().squeeze(0).to(device)
        bound_box2 = self.target_mesh.get_bounding_boxes().squeeze(0).to(device)
        box_min = torch.min(torch.cat((bound_box1, bound_box2), dim=1), dim=-1)[0]
        box_max = torch.max(torch.cat((bound_box1, bound_box2), dim=1), dim=-1)[0]
        bound_box = torch.cat((box_min.unsqueeze(-1), box_max.unsqueeze(-1)), dim=-1)
        radius = bound_box[:, 1] - bound_box[:, 0]
        bound_box[:, 0] -= 0.125 * radius
        bound_box[:, 1] += 0.125 * radius
        # iteratively sample until enough
        count_num_in = 0
        count_num_out = 0
        query_points_in = []
        query_points_out = []
        for batch in range(20):
            x = np.random.uniform(bound_box[0, 0], bound_box[0, 1], size=batch_size)
            y = np.random.uniform(bound_box[1, 0], bound_box[1, 1], size=batch_size)
            z = np.random.uniform(bound_box[2, 0], bound_box[2, 1], size=batch_size)
            query_points = torch.tensor(np.stack([x, y, z], axis=1)).float()
            do_gt, _ = self._target_occupancy_prob(query_points, device)
            indices_in = torch.where(do_gt > 0.95)[0]
            indices_out = torch.where(do_gt < 0.05)[0]
            count_num_in += indices_in.shape[0]
            count_num_out += indices_out.shape[0]
            query_points_in.append(query_points[indices_in, :].numpy())
            query_points_out.append(query_points[indices_out, :].numpy())
            if batch < 3 or batch == 19 or (batch + 1) % 5 == 0:
                print(
                    f"[DO] static sampling {batch + 1}/20: "
                    f"found {count_num_in} inside and {count_num_out} outside points."
                )
            if count_num_in >= num_in and count_num_out >= num_out:
                break
        query_points_in = torch.Tensor(np.concatenate(query_points_in, axis=0))[:num_in, :]
        query_points_out = torch.Tensor(np.concatenate(query_points_out, axis=0))[:num_out, :]
        query_points = torch.cat((query_points_in, query_points_out), dim=0)
        do_gt = torch.cat((torch.ones(num_in), torch.zeros(num_out)), dim=0)
        return query_points, do_gt

    def get_static_mask_number_control_v2(self, num_in=10000, num_out=10000, expand_ratio=5, smooth=0.02, inspect=False, redo=False):
        """
        :param num_in: point number inside the shape
        :param num_out:
        :param expand_ratio: 1 / expand_ratio of the points will be retained
        :return:
        """
        device = torch.device('cpu')
        batch_size = round((num_in + num_out) / 4)
        # get bounding box
        bound_box1 = self.mesh_std.get_bounding_boxes().squeeze(0).to(device)
        bound_box2 = self.target_mesh.get_bounding_boxes().squeeze(0).to(device)
        box_min = torch.min(torch.cat((bound_box1, bound_box2), dim=1), dim=-1)[0]
        box_max = torch.max(torch.cat((bound_box1, bound_box2), dim=1), dim=-1)[0]
        bound_box = torch.cat((box_min.unsqueeze(-1), box_max.unsqueeze(-1)), dim=-1)
        radius = bound_box[:, 1] - bound_box[:, 0]
        bound_box[:, 0] -= 0.125 * radius
        bound_box[:, 1] += 0.125 * radius
        # iteratively sample until enough
        count_num_in = 0
        count_num_out = 0
        query_points_in = []
        query_points_out = []
        points_path = os.path.join(self.root_target, self.name_target, "do_points.pt")
        need_regen = bool((not os.path.exists(points_path)) or redo)
        if not need_regen:
            try:
                print("do points successfully loaded!")
                dict_pt = torch.load(points_path, map_location="cpu")
                query_points_in = dict_pt['query_points_in']
                query_points_out = dict_pt['query_points_out']
                if not torch.is_tensor(query_points_in):
                    query_points_in = torch.as_tensor(query_points_in).float()
                if not torch.is_tensor(query_points_out):
                    query_points_out = torch.as_tensor(query_points_out).float()
                query_points_in = query_points_in.detach().cpu().float()
                query_points_out = query_points_out.detach().cpu().float()
                if query_points_in.ndim != 2 or query_points_in.shape[1] != 3:
                    raise ValueError("query_points_in has invalid shape")
                if query_points_out.ndim != 2 or query_points_out.shape[1] != 3:
                    raise ValueError("query_points_out has invalid shape")
                if query_points_in.shape[0] < num_in:
                    raise ValueError("query_points_in has too few points")
                if query_points_out.shape[0] < num_out:
                    raise ValueError("query_points_out has too few points")
                if not torch.isfinite(query_points_in).all():
                    raise ValueError("query_points_in contains non-finite values")
                if not torch.isfinite(query_points_out).all():
                    raise ValueError("query_points_out contains non-finite values")
            except Exception as e:
                print(f"do_points load failed, redoing points searching: {type(e).__name__}: {e}")
                need_regen = True
        if need_regen:
            print("do_points no found, redoing points searching")
            query_points_in = torch.empty((0, 3)).float()
            query_points_out = torch.empty((0, 3)).float()
            loop_idx = 0
            warned_nonfinite = False
            while loop_idx < self.do_sampling_max_loops:
                loop_idx += 1
                x = np.random.uniform(bound_box[0, 0], bound_box[0, 1], size=batch_size)
                y = np.random.uniform(bound_box[1, 0], bound_box[1, 1], size=batch_size)
                z = np.random.uniform(bound_box[2, 0], bound_box[2, 1], size=batch_size)
                query_points = torch.tensor(np.stack([x, y, z], axis=1)).float()

                do_gt, warned_nonfinite = self._target_occupancy_prob(
                    query_points,
                    device,
                    warned_nonfinite=warned_nonfinite,
                )
                indices_in = torch.where(do_gt > 0.95)[0]
                indices_out = torch.where(do_gt < 0.05)[0]
                if count_num_in <= expand_ratio * num_in:
                    count_num_in += indices_in.shape[0]
                    query_points_in = torch.cat((query_points_in, query_points[indices_in, :]), dim=0)

                if count_num_out <= expand_ratio * num_out:
                    count_num_out += indices_out.shape[0]
                    query_points_out = torch.cat((query_points_out, query_points[indices_out, :]), dim=0)

                if loop_idx <= 3 or loop_idx == self.do_sampling_max_loops or loop_idx % 10 == 0:
                    print(
                        f"[DO] static sampling {loop_idx}/{self.do_sampling_max_loops}: "
                        f"found {count_num_in} inside and {count_num_out} outside points."
                    )
                if count_num_in >= expand_ratio * num_in and count_num_out >= expand_ratio * num_out:
                    break
            if count_num_in < expand_ratio * num_in or count_num_out < expand_ratio * num_out:
                raise RuntimeError(
                    "Static DO point sampling did not converge for "
                    f"{self.name_target} after {self.do_sampling_max_loops} loops "
                    f"(inside={count_num_in}, outside={count_num_out}, "
                    f"required_inside={expand_ratio * num_in}, required_outside={expand_ratio * num_out}). "
                    "This usually indicates invalid winding/mesh geometry or an unusable cached target mesh."
                )
            # calculate field strength
            query_points_in = query_points_in[: expand_ratio * num_in, :]
            query_points_out = query_points_out[: expand_ratio * num_out, :]
            torch.save({'query_points_in': query_points_in, 'query_points_out': query_points_out}, points_path)
        # dict_pt = torch.load('points.pt')
        # query_points_in, query_points_out = dict_pt['query_points_in'], dict_pt['query_points_out0']
        source_verts = torch.cat((self.mesh_std.verts_packed().detach().cpu().float(),
                                          self.target_mesh.verts_packed().detach().cpu().float()), dim=0)
        
        # Batch processing to avoid OOM
        batch_size = 1000
        strength_in_list = []
        for i in range(0, query_points_in.shape[0], batch_size):
             qp_batch = query_points_in[i:i + batch_size]
             # cdist handles efficient distance calculation
             d_batch = torch.cdist(source_verts.unsqueeze(0), qp_batch.unsqueeze(0)).squeeze(0) + smooth
             s_batch = torch.sum((1 / torch.pow(d_batch, 2)), dim=0)
             strength_in_list.append(s_batch)
        strength_in = torch.cat(strength_in_list, dim=0) if len(strength_in_list) > 0 else torch.tensor([])

        strength_out_list = []
        for i in range(0, query_points_out.shape[0], batch_size):
             qp_batch = query_points_out[i:i + batch_size]
             d_batch = torch.cdist(source_verts.unsqueeze(0), qp_batch.unsqueeze(0)).squeeze(0) + smooth
             s_batch = torch.sum((1 / torch.pow(d_batch, 2)), dim=0)
             strength_out_list.append(s_batch)
        strength_out = torch.cat(strength_out_list, dim=0) if len(strength_out_list) > 0 else torch.tensor([])

        strength = torch.cat((strength_in, strength_out), dim=0)
        _, topk_indices_in = torch.topk(strength_in, num_in)
        _, topk_indices_out = torch.topk(strength_out, num_out)
        # query_points_in_filtered = query_points_in[topk_indices_in, :]
        query_points_in_filtered = query_points_in[:num_in, :]  # thin vessels don't need control for inside
        query_points_out_filtered = query_points_out[topk_indices_out, :]
        query_points = torch.cat((query_points_in_filtered, query_points_out_filtered), dim=0)
        # # visualize
        if inspect:
            p = pv.Plotter()
            trimesh_canonical = trimesh.Trimesh(self.mesh_std.verts_packed().detach().cpu().numpy(),
                                                self.mesh_std.faces_packed().detach().cpu().numpy())
            trimesh_target = trimesh.Trimesh(self.target_mesh.verts_packed().detach().cpu().numpy(),
                                             self.target_mesh.faces_packed().detach().cpu().numpy())
            p.add_mesh(trimesh_canonical, color='blue', opacity=0.5, pickable=False)
            p.add_mesh(trimesh_target, color='red', opacity=0.5, pickable=False)
            p.add_points(query_points_out_filtered.numpy(), color='black')
            # p.add_points(query_points_out.numpy(), color='grey')
            p.show()
        # return points and ground truth of differentiable voxel masks
        do_gt = torch.cat((torch.ones(num_in), torch.zeros(num_out)), dim=0)
        return query_points, do_gt

    def get_static_mask_and_gt(self, style='number_control_v2'):
        device = torch.device('cpu')
        if style == 'uniform':
            query_points = self.get_static_mask_uniform()
            # query_points_upsample = query_points.permute((3, 0, 1, 2)).unsqueeze(0)
            # self.do_module = torch.nn.Upsample(scale_factor=2, mode='trilinear')
            # query_points_upsample = self.do_module(query_points_upsample)
            do_gt, _ = self._target_occupancy_prob(query_points, device)
            query_points = query_points.view(-1, 3)
        elif style == 'number_control':
            query_points, do_gt = self.get_static_mask_number_control(num_in=self.do_number, num_out=self.do_number)
        elif style == 'number_control_v2':
            query_points, do_gt = self.get_static_mask_number_control_v2(num_in=self.do_number, num_out=self.do_number,
                                                                         expand_ratio=2, smooth=0.02, redo=self.redo_do_points)
        else:
            query_points = self.get_static_mask_probabilistic()
            do_gt, _ = self._target_occupancy_prob(query_points, device)

        self.do_style = style

        return query_points, do_gt

    def get_weights_attention(self, query_points, min_w=1.0, max_w=3.0, smooth=0.01, inspect=False):
        # calculate field strength
        batch_size = int(1000)
        batch_num = math.ceil(query_points.shape[0] / batch_size)
        source_verts = self.target_mesh.verts_packed().detach().cpu().float()
        strength = []
        for i in range(batch_num):
            query_points_batch = query_points[i*batch_size: (i+1)*batch_size].cpu().float()
            distance_batch = torch.norm(source_verts.unsqueeze(1) - query_points_batch.unsqueeze(0), dim=2) + smooth
            strength_batch = torch.sum((1 / torch.pow(distance_batch, 2)), dim=0)
            strength.append(strength_batch)
            print('field strength has been caluclated for batch {}'.format(i))
        strength = torch.cat(strength, dim=0)
        # map to certain range
        ratio = (max_w - min_w) / (torch.max(strength) - torch.min(strength))
        strength = (strength - torch.min(strength)) * ratio + min_w
        # visualize
        if inspect:
            index = np.argsort(strength.detach().cpu().numpy())[::-1]
            ratio = 1.0  # for inspecting dropping threshold
            index = index[:round(ratio * len(index))]
            cloud = pv.PolyData(query_points.detach().cpu().numpy()[index, :])
            cloud['attention'] = strength.cpu().numpy()[index]  # just use z coordinate
            pv.plot(cloud, scalars='attention', cmap='jet', show_bounds=True, cpos='yz')
        # p = pv.Plotter()
        # p.add_points(query_points.detach().cpu().numpy(), render_points_as_spheres=False, color='red', point_size=0.1)
        # p.add_points(source_verts.numpy(), render_points_as_spheres=True, color='blue', point_size=1)
        # p.show()
        self.weights_attention = strength.unsqueeze(0).detach().to(self.device)
        print('field strength assigned')

    def forward_opa_do(self, warped_mesh, warped_openings, loss_weighting: dict, query_points, do_gt, do_index, B=1):
        # Use decapped meshes for loss_p0 / loss_n1 to avoid cap-surface
        # samples pulling cap vertices toward the dome.  Regularisation
        # losses (laplacian, edge, consistency, rigid) still use the full
        # mesh so that cap triangles stay well-conditioned.
        if self._decap_enabled and self._canon_cap_face_mask is not None:
            warped_decap = self._decapped_warped_mesh(warped_mesh)
            target_decap = self._target_mesh_decapped if self._target_mesh_decapped is not None else self.target_mesh
            # Compute surface chamfer on decapped meshes
            loss_dict_surface = self.forward(
                meshes_scr=warped_decap, trg=target_decap,
                loss_list={k: v for k, v in loss_weighting.items() if k in ('loss_p0', 'loss_n1')},
                B=B,
            )
            # Compute regularisation on full mesh
            reg_keys = {k for k in loss_weighting if k not in ('loss_p0', 'loss_n1')}
            loss_dict_reg = self.forward(
                meshes_scr=warped_mesh, trg=self.target_mesh,
                loss_list={k: v for k, v in loss_weighting.items() if k in reg_keys},
                B=B,
            )
            loss_dict = {**loss_dict_surface, **loss_dict_reg}
        else:
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
                    warped_centroid, warped_normal = self._plane_from_points(pcd_wo[0])
                    target_centroid, target_normal = target_plane if target_plane is not None else (None, None)
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
        if 'loss_do' in loss_weighting:
            if self.do_style == 'uniform_upsample':
                winding_field = torch.sigmoid((Winding_Occupancy(warped_mesh, query_points) - 0.5) * 100)
                resolution = int(np.cbrt(query_points.shape[0]))
                winding_field = winding_field.view((resolution, resolution, resolution)).unsqueeze(0).unsqueeze(0)
                winding_field = self.do_module(winding_field)
                winding_field = winding_field.squeeze(0).squeeze(0).view(-1)
            else:
                winding_field = torch.sigmoid((Winding_Occupancy(warped_mesh, query_points) - 0.5) * 100)
                # winding_field = Winding_Occupancy(warped_mesh, query_points)
            # loss_do = dice_loss.forward(winding_field.unsqueeze(0), do_gt.unsqueeze(0))
            if self.do_loss_type == "mse_loss":
                loss_do = self.mse_loss(winding_field, do_gt)
            elif self.do_loss_type == "dice_loss":
                loss_do = self.dice_loss.forward(winding_field.unsqueeze(0), do_gt.unsqueeze(0))
            elif self.do_loss_type == "dice_loss_attention":
                loss_do = self.dice_loss_attention.forward(winding_field.unsqueeze(0), do_gt.unsqueeze(0), self.weights_attention[:, do_index])
            else:
                raise NotImplementedError("do loss type not implemented")
            loss_dict['loss_do'] = loss_do if not torch.isnan(loss_do) else torch.Tensor([0.0]).to(self.device)
        return loss_dict



def o3d_mesh_to_pytorch3d(o3d_mesh) -> Meshes:
    verts = torch.Tensor(np.asarray(o3d_mesh.vertices))
    faces = torch.Tensor(np.asarray(o3d_mesh.triangles))
    pytorch3d_mesh = Meshes(verts=[verts], faces=[faces])
    return pytorch3d_mesh
