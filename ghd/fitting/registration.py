"""
Resigration classes for ghd fitting.
Truth mesh is registered to enable opening alignment & differentiable centreline losses during ghd fitting.
"""
import open3d as o3d
import numpy as np
import numpy
import logging
import os
import itertools
import trimesh
import shapely
from utils import utils_registration as u_register
import pickle
from pytorch3d.structures import Meshes
import torch
import sys
from utils.utils import o3d_mesh_to_pytorch3d
import vtk
import pytorch3d as p3d
import igraph as ig
from tqdm import tqdm
try:
    from skeletor.utilities import make_trimesh
except Exception:
    def make_trimesh(mesh, validate=False):
        return mesh
import pyvista as pv
from pytorch3d.io import save_obj, load_objs_as_meshes
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from typing import Dict, List, Optional, Tuple


class RegistrationwOpeningAlignment(object):
    def __init__(self, args, root, target, num_op=3, suffix=None):
        self.device = torch.device(args.device)
        self.root = root
        self.target = target
        self.suffix = suffix if suffix is not None else '.obj'
        assert self.suffix == '.obj', 'Not implemented for mesh file other than .obj'
        # mesh objects of true complexes
        mesh_path = os.path.join(self.root, self.target + self.suffix)
        if not os.path.exists(mesh_path):
             # Try checking inside the folder for part_aligned.obj
             alt_path = os.path.join(self.root, self.target, "part_aligned.obj")
             if os.path.exists(alt_path):
                 mesh_path = alt_path
        
        print(f"Loading mesh from: {mesh_path}")
        mesh_trimesh = trimesh.load(mesh_path, process=False)
        if isinstance(mesh_trimesh, trimesh.Scene):
            if len(mesh_trimesh.geometry) == 0:
                raise ValueError("Loaded trimesh scene is empty.")
            mesh_trimesh = trimesh.util.concatenate(tuple(mesh_trimesh.geometry.values()))
        if not isinstance(mesh_trimesh, trimesh.Trimesh):
            mesh_o3d = o3d.io.read_triangle_mesh(mesh_path)
            mesh_trimesh = trimesh.Trimesh(
                vertices=np.asarray(mesh_o3d.vertices),
                faces=np.asarray(mesh_o3d.triangles),
                process=False,
            )

        verts = np.asarray(mesh_trimesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh_trimesh.faces, dtype=np.int64)
        self.mesh_target_trimesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)

        # Keep open3d and trimesh in the same vertex/face indexing order.
        self.mesh_target = o3d.geometry.TriangleMesh()
        self.mesh_target.vertices = o3d.utility.Vector3dVector(verts)
        self.mesh_target.triangles = o3d.utility.Vector3iVector(faces)
        self.mesh_target.compute_vertex_normals()

        verts_t = torch.from_numpy(verts).float()
        faces_t = torch.from_numpy(faces).long()
        self.mesh_target_p3d = Meshes(verts=[verts_t], faces=[faces_t])
        self.num_op = num_op  # number of openings
        # assembly of opening v indices (num_op, N), v coordinates (num_op, N, 3), n (num_op, N, 3)
        self.op_v_indices, self.op_v_coords, self.op_v_normal, self.op_n_mean = [], [], [], []
        # assembly of newly reconstructed plane meshes
        self.op_rec_v, self.op_rec_f = [], []
        # assembly of mapped reconstructed plane meshes
        self.op_rec_v_indices_map, self.op_rec_f_map = [], []
        # optional metadata for automatic registration and repaired target supervision
        self.op_tangent, self.op_cut_points = [], []
        self.op_target_rim_v, self.op_target_rec_v, self.op_target_rec_f = [], [], []
        self.op_target_plane_center, self.op_target_plane_normal = [], []
        self.op_source_kind, self.op_source_surface_v, self.op_source_surface_f = [], [], []
        self.op_target_debug = []
        self.auto_centreline_summary = None
        self.auto_registration_debug = {}
        self._auto_reg_pending_debug = None
        self._mesh_graph_cache = None

    def _reset_opening_state(self):
        self.op_v_indices, self.op_v_coords, self.op_v_normal, self.op_n_mean = [], [], [], []
        self.op_rec_v, self.op_rec_f = [], []
        self.op_rec_v_indices_map, self.op_rec_f_map = [], []
        self.op_tangent, self.op_cut_points = [], []
        self.op_target_rim_v, self.op_target_rec_v, self.op_target_rec_f = [], [], []
        self.op_target_plane_center, self.op_target_plane_normal = [], []
        self.op_source_kind, self.op_source_surface_v, self.op_source_surface_f = [], [], []
        self.op_target_debug = []

    def _get_mesh_graph_cache(self) -> Dict[str, object]:
        if self._mesh_graph_cache is not None:
            return self._mesh_graph_cache

        mesh = self.mesh_target_trimesh
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                raise ValueError("Loaded trimesh scene is empty.")
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
            self.mesh_target_trimesh = mesh
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = trimesh.Trimesh(
                vertices=np.asarray(self.mesh_target.vertices),
                faces=np.asarray(self.mesh_target.triangles),
                process=False,
            )
            self.mesh_target_trimesh = mesh

        verts = np.asarray(mesh.vertices, dtype=np.float64)
        edges = np.asarray(mesh.edges_unique, dtype=np.int64)
        if edges.size == 0:
            raise ValueError("Mesh has no edges; cannot build graph.")
        edge_lengths = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
        graph = ig.Graph(n=verts.shape[0], edges=edges.tolist(), directed=False)
        graph.es["weight"] = edge_lengths.tolist()

        self._mesh_graph_cache = {
            "verts": verts,
            "edges": edges,
            "edge_lengths": edge_lengths,
            "median_edge_length": float(np.median(edge_lengths)),
            "graph": graph,
            "distance_cache": {},
        }
        return self._mesh_graph_cache

    def _distances_from_vertex(self, vertex_idx: int) -> np.ndarray:
        cache = self._get_mesh_graph_cache()
        distance_cache = cache["distance_cache"]
        vertex_idx = int(vertex_idx)
        if vertex_idx not in distance_cache:
            dist = np.asarray(
                cache["graph"].distances(
                    source=[vertex_idx],
                    target=None,
                    mode="all",
                    weights="weight",
                ),
                dtype=np.float64,
            )[0]
            distance_cache[vertex_idx] = dist
        return distance_cache[vertex_idx]

    def _select_endpoint_indices_fps(self, count: int) -> List[int]:
        cache = self._get_mesh_graph_cache()
        verts = cache["verts"]
        centroid = np.mean(verts, axis=0, keepdims=True)

        start_idx = int(np.argmax(np.linalg.norm(verts - centroid, axis=1)))
        start_dist = self._distances_from_vertex(start_idx)
        finite_start = np.isfinite(start_dist)
        if np.any(finite_start):
            start_idx = int(np.argmax(np.where(finite_start, start_dist, -np.inf)))

        endpoints = []
        min_dist = np.full(verts.shape[0], np.inf, dtype=np.float64)
        for step in range(int(count)):
            if step == 0:
                candidate = start_idx
            else:
                finite_mask = np.isfinite(min_dist)
                if not np.any(finite_mask):
                    break
                candidate = int(np.argmax(np.where(finite_mask, min_dist, -np.inf)))
            if candidate in endpoints:
                finite_mask = np.isfinite(min_dist)
                ranked = np.argsort(np.where(finite_mask, min_dist, -np.inf))[::-1]
                candidate = None
                for idx in ranked:
                    idx = int(idx)
                    if idx not in endpoints and np.isfinite(min_dist[idx]):
                        candidate = idx
                        break
                if candidate is None:
                    break
            endpoints.append(candidate)
            min_dist = np.minimum(min_dist, self._distances_from_vertex(candidate))

        if len(endpoints) < count:
            raise RuntimeError(f"Could only find {len(endpoints)} endpoints, expected {count}.")
        return [int(idx) for idx in endpoints[:count]]

    def _estimate_bifurcation_index(self, endpoint_indices: List[int]) -> int:
        dist_stack = np.vstack([self._distances_from_vertex(idx) for idx in endpoint_indices])
        finite_mask = np.all(np.isfinite(dist_stack), axis=0)
        if not np.any(finite_mask):
            raise RuntimeError("No finite geodesic distances found while estimating bifurcation.")
        score = np.sum(dist_stack, axis=0)
        score[~finite_mask] = np.inf
        return int(np.argmin(score))

    def _sort_endpoint_indices(self, endpoint_indices: List[int], center_idx: int) -> List[int]:
        cache = self._get_mesh_graph_cache()
        verts = cache["verts"]
        center = verts[int(center_idx)]
        directions = verts[np.asarray(endpoint_indices, dtype=np.int64)] - center.reshape(1, 3)
        azimuth = np.arctan2(directions[:, 1], directions[:, 0])
        elevation = np.arctan2(
            directions[:, 2],
            np.linalg.norm(directions[:, :2], axis=1) + 1e-12,
        )
        order = np.lexsort((elevation, azimuth))
        return [int(endpoint_indices[i]) for i in order]

    def _branch_paths_from_endpoints(self, endpoint_indices: List[int], center_idx: int) -> List[np.ndarray]:
        cache = self._get_mesh_graph_cache()
        graph = cache["graph"]
        paths = []
        for endpoint_idx in endpoint_indices:
            path = graph.get_shortest_paths(
                v=int(center_idx),
                to=int(endpoint_idx),
                mode="all",
                weights="weight",
                output="vpath",
            )[0]
            if len(path) < 2:
                continue
            paths.append(np.asarray(path, dtype=np.int64))
        return paths

    def _endpoint_tangent_from_path(self, path: np.ndarray, step: int = 6) -> np.ndarray:
        cache = self._get_mesh_graph_cache()
        verts = cache["verts"]
        if len(path) < 2:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        tip = verts[int(path[-1])]
        anchor_idx = max(0, len(path) - 1 - int(step))
        anchor = verts[int(path[anchor_idx])]
        tangent = tip - anchor
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-12 and len(path) > 1:
            tangent = tip - verts[int(path[-2])]
            tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return tangent / tangent_norm

    def _path_turn_curvature(self, path: np.ndarray, tip_window_frac: float = 0.45) -> Dict[str, float]:
        path = np.asarray(path, dtype=np.int64).reshape(-1)
        if path.size < 4:
            return {
                "turn_mean": 0.0,
                "turn_p90": 0.0,
                "tip_turn_mean": 0.0,
                "tip_turn_p90": 0.0,
            }
        cache = self._get_mesh_graph_cache()
        pts = cache["verts"][path]
        seg = np.diff(pts, axis=0)
        seg_norm = np.linalg.norm(seg, axis=1, keepdims=True)
        valid = seg_norm.reshape(-1) > 1e-12
        if np.count_nonzero(valid) < 3:
            return {
                "turn_mean": 0.0,
                "turn_p90": 0.0,
                "tip_turn_mean": 0.0,
                "tip_turn_p90": 0.0,
            }
        unit = np.zeros_like(seg)
        unit[valid] = seg[valid] / seg_norm[valid]
        dot = np.sum(unit[:-1] * unit[1:], axis=1)
        dot = np.clip(dot, -1.0, 1.0)
        ang = np.arccos(dot)
        if ang.size == 0:
            return {
                "turn_mean": 0.0,
                "turn_p90": 0.0,
                "tip_turn_mean": 0.0,
                "tip_turn_p90": 0.0,
            }
        tip_count = max(3, int(np.ceil(float(tip_window_frac) * float(ang.size))))
        tip_count = min(tip_count, int(ang.size))
        tip_ang = ang[-tip_count:]
        return {
            "turn_mean": float(np.mean(ang)),
            "turn_p90": float(np.quantile(ang, 0.90)),
            "tip_turn_mean": float(np.mean(tip_ang)),
            "tip_turn_p90": float(np.quantile(tip_ang, 0.90)),
        }

    def _path_elbow_feature(
        self,
        path: np.ndarray,
        distal_frac: float = 0.40,
    ) -> Dict[str, float]:
        path = np.asarray(path, dtype=np.int64).reshape(-1)
        if path.size < 6:
            return {
                "elbow_angle": 0.0,
                "elbow_axis_ratio": 1.0,
            }
        cache = self._get_mesh_graph_cache()
        pts = cache["verts"][path]
        seg = np.diff(pts, axis=0)
        seg_norm = np.linalg.norm(seg, axis=1, keepdims=True)
        valid = seg_norm.reshape(-1) > 1e-12
        if np.count_nonzero(valid) < 4:
            return {
                "elbow_angle": 0.0,
                "elbow_axis_ratio": 1.0,
            }
        unit = np.zeros_like(seg)
        unit[valid] = seg[valid] / seg_norm[valid]
        n_seg = unit.shape[0]
        distal_count = max(3, int(np.ceil(float(distal_frac) * float(n_seg))))
        distal_count = min(distal_count, n_seg - 1)
        prox_end = max(1, n_seg - distal_count)
        prox_start = max(0, prox_end - distal_count)
        prox_slice = unit[prox_start:prox_end]
        dist_slice = unit[-distal_count:]
        if prox_slice.shape[0] < 1 or dist_slice.shape[0] < 1:
            return {
                "elbow_angle": 0.0,
                "elbow_axis_ratio": 1.0,
            }

        prox_vec = np.mean(prox_slice, axis=0)
        dist_vec = np.mean(dist_slice, axis=0)
        prox_norm = np.linalg.norm(prox_vec)
        dist_norm = np.linalg.norm(dist_vec)
        if prox_norm < 1e-12 or dist_norm < 1e-12:
            elbow_angle = 0.0
        else:
            prox_vec = prox_vec / prox_norm
            dist_vec = dist_vec / dist_norm
            elbow_angle = float(np.arccos(np.clip(np.dot(prox_vec, dist_vec), -1.0, 1.0)))

        window = unit[prox_start:]
        if window.shape[0] < 3:
            elbow_axis_ratio = 1.0
        else:
            tangent_change = np.diff(window, axis=0)
            tangent_change = tangent_change[np.linalg.norm(tangent_change, axis=1) > 1e-12]
            if tangent_change.shape[0] < 3:
                elbow_axis_ratio = 1.0
            else:
                centered = tangent_change - np.mean(tangent_change, axis=0, keepdims=True)
                try:
                    _, svals, _ = np.linalg.svd(centered, full_matrices=False)
                    if svals.shape[0] >= 2:
                        elbow_axis_ratio = float(svals[1] / (svals[0] + 1e-12))
                    else:
                        elbow_axis_ratio = 1.0
                except np.linalg.LinAlgError:
                    elbow_axis_ratio = 1.0

        return {
            "elbow_angle": float(elbow_angle),
            "elbow_axis_ratio": float(np.clip(elbow_axis_ratio, 0.0, 5.0)),
        }

    def _sort_indices_around_normal(self, indices: np.ndarray, normal: np.ndarray) -> np.ndarray:
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if indices.size <= 2:
            return indices
        cache = self._get_mesh_graph_cache()
        coords = cache["verts"][indices]
        centroid = np.mean(coords, axis=0)
        normal = np.asarray(normal, dtype=np.float64).reshape(3)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            return indices
        normal = normal / normal_norm
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(helper, normal)) > 0.95:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        basis_x = np.cross(normal, helper)
        basis_x_norm = np.linalg.norm(basis_x)
        if basis_x_norm < 1e-12:
            return indices
        basis_x = basis_x / basis_x_norm
        basis_y = np.cross(normal, basis_x)
        rel = coords - centroid.reshape(1, 3)
        angles = np.arctan2(rel @ basis_y, rel @ basis_x)
        order = np.argsort(angles)
        return indices[order]

    def _regularize_loop_indices_angular(
        self,
        loop_idx: np.ndarray,
        normal: np.ndarray,
        bins_min: int = 24,
        bins_max: int = 72,
        radius_quantile: float = 0.60,
    ) -> np.ndarray:
        loop_idx = np.unique(np.asarray(loop_idx, dtype=np.int64).reshape(-1))
        if loop_idx.size < 16:
            return loop_idx
        basis = self._angle_basis_from_normal(normal)
        if basis is None:
            return loop_idx
        _, basis_x, basis_y = basis
        cache = self._get_mesh_graph_cache()
        coords = cache["verts"][loop_idx]
        centroid = np.mean(coords, axis=0)
        rel = coords - centroid.reshape(1, 3)
        angles = np.arctan2(rel @ basis_y, rel @ basis_x)
        radial = np.linalg.norm(rel, axis=1)

        n_bins = int(np.clip(loop_idx.size // 2, int(bins_min), int(bins_max)))
        if n_bins < 12:
            return loop_idx
        edges = np.linspace(-np.pi, np.pi, n_bins + 1)
        picked = []
        q = float(np.clip(radius_quantile, 0.05, 0.95))
        for b in range(n_bins):
            if b == n_bins - 1:
                mask = (angles >= edges[b]) & (angles <= edges[b + 1])
            else:
                mask = (angles >= edges[b]) & (angles < edges[b + 1])
            local = np.where(mask)[0]
            if local.size == 0:
                continue
            local_r = radial[local]
            target_r = float(np.quantile(local_r, q))
            pick_local = int(local[int(np.argmin(np.abs(local_r - target_r)))])
            picked.append(int(loop_idx[pick_local]))
        if len(picked) < 12:
            return loop_idx
        picked = np.unique(np.asarray(picked, dtype=np.int64))
        if picked.size < 12:
            return loop_idx
        return self._sort_indices_around_normal(picked, normal)

    def _map_points_to_mesh_vertices(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        if points.size == 0:
            return np.array([], dtype=np.int64)
        cache = self._get_mesh_graph_cache()
        verts = cache["verts"]
        try:
            _, nearest_idx = self.mesh_target_trimesh.kdtree.query(points)
            nearest_idx = np.asarray(nearest_idx, dtype=np.int64).reshape(-1)
        except Exception:
            sq_dist = np.sum(
                (points[:, None, :] - verts[None, :, :]) ** 2,
                axis=-1,
            )
            nearest_idx = np.argmin(sq_dist, axis=1).astype(np.int64)
        return nearest_idx

    def _estimate_loop_normal(self, loop_coords: np.ndarray) -> np.ndarray:
        loop_coords = np.asarray(loop_coords, dtype=np.float64)
        if loop_coords.ndim != 2 or loop_coords.shape[0] < 3:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        centered = loop_coords - np.mean(loop_coords, axis=0, keepdims=True)
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
            normal = vh[-1]
        except np.linalg.LinAlgError:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=np.float64)
        return normal / norm

    def _extract_boundary_loops(self, min_loop_vertices: int = 12) -> List[np.ndarray]:
        mesh = self.mesh_target_trimesh
        if not isinstance(mesh, trimesh.Trimesh):
            return []
        # trimesh API differs across versions; compute boundary edges robustly.
        if hasattr(mesh, "edges_boundary"):
            boundary_edges = np.asarray(mesh.edges_boundary, dtype=np.int64)
        else:
            faces = np.asarray(mesh.faces, dtype=np.int64)
            if faces.size == 0:
                return []
            all_edges = np.concatenate(
                (
                    faces[:, [0, 1]],
                    faces[:, [1, 2]],
                    faces[:, [2, 0]],
                ),
                axis=0,
            )
            all_edges = np.sort(all_edges, axis=1)
            uniq_edges, counts = np.unique(all_edges, axis=0, return_counts=True)
            boundary_edges = uniq_edges[counts == 1]
        if boundary_edges.size == 0:
            return []
        boundary_vertices = np.unique(boundary_edges.reshape(-1))
        if boundary_vertices.size == 0:
            return []
        lut = -np.ones(mesh.vertices.shape[0], dtype=np.int64)
        lut[boundary_vertices] = np.arange(boundary_vertices.shape[0], dtype=np.int64)
        local_edges = lut[boundary_edges]
        graph = ig.Graph(n=boundary_vertices.shape[0], edges=local_edges.tolist(), directed=False)
        loops = []
        for component in graph.components():
            comp = np.asarray(component, dtype=np.int64)
            if comp.size < int(min_loop_vertices):
                continue
            loop_idx = boundary_vertices[comp]
            loops.append(np.asarray(loop_idx, dtype=np.int64))
        return loops

    def _assign_loops_to_branches(self, score: np.ndarray) -> Optional[List[int]]:
        score = np.asarray(score, dtype=np.float64)
        if score.ndim != 2:
            return None
        n_branches, n_loops = score.shape
        if n_loops < n_branches:
            return None
        loop_ids = list(range(n_loops))
        best_perm = None
        best_score = np.inf
        # Exact assignment for a small number of loops, greedy fallback otherwise.
        if n_loops <= 9:
            for perm in itertools.permutations(loop_ids, n_branches):
                val = float(np.sum([score[i, perm[i]] for i in range(n_branches)]))
                if val < best_score:
                    best_score = val
                    best_perm = list(perm)
        else:
            used = set()
            greedy = []
            for i in range(n_branches):
                row = np.argsort(score[i])
                pick = None
                for j in row:
                    if int(j) not in used:
                        pick = int(j)
                        break
                if pick is None:
                    return None
                used.add(pick)
                greedy.append(pick)
            best_perm = greedy
        return best_perm

    def _extract_normal_graph_cluster_candidates(
        self,
        normal_dot_min: float = 0.90,
        min_loop_vertices: int = 12,
    ) -> List[Dict[str, object]]:
        mesh = self.mesh_target_trimesh
        if not isinstance(mesh, trimesh.Trimesh):
            return []
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if faces.size == 0:
            return []
        face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
        if face_normals.shape[0] != faces.shape[0]:
            return []
        face_adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
        if face_adj.size == 0:
            return []

        cache = self._get_mesh_graph_cache()
        verts = cache["verts"]
        median_edge = cache["median_edge_length"]
        global_centroid = np.mean(verts, axis=0)

        # Keep face-neighbor edges whose normals are similarly oriented.
        keep_edges = []
        ndot = np.abs(np.einsum("ij,ij->i", face_normals[face_adj[:, 0]], face_normals[face_adj[:, 1]]))
        valid = ndot >= float(normal_dot_min)
        if np.any(valid):
            keep_edges = face_adj[valid].tolist()
        if len(keep_edges) == 0:
            return []

        fgraph = ig.Graph(n=faces.shape[0], edges=keep_edges, directed=False)
        fcomps = fgraph.components()
        comp_face_min = max(6, int(min_loop_vertices) // 2)
        candidates = []

        for comp in fcomps:
            comp = np.asarray(comp, dtype=np.int64)
            if comp.size < comp_face_min:
                continue
            comp_faces = faces[comp]
            tri_edges = np.concatenate(
                (
                    comp_faces[:, [0, 1]],
                    comp_faces[:, [1, 2]],
                    comp_faces[:, [2, 0]],
                ),
                axis=0,
            )
            tri_edges = np.sort(tri_edges, axis=1)
            uniq_edges, counts = np.unique(tri_edges, axis=0, return_counts=True)
            boundary_edges = uniq_edges[counts == 1]
            if boundary_edges.size == 0:
                continue

            boundary_vertices = np.unique(boundary_edges.reshape(-1))
            if boundary_vertices.size < max(6, min_loop_vertices):
                continue
            lut = -np.ones(verts.shape[0], dtype=np.int64)
            lut[boundary_vertices] = np.arange(boundary_vertices.shape[0], dtype=np.int64)
            local_edges = lut[boundary_edges]
            bgraph = ig.Graph(n=boundary_vertices.shape[0], edges=local_edges.tolist(), directed=False)

            for bcomp in bgraph.components():
                bcomp = np.asarray(bcomp, dtype=np.int64)
                if bcomp.size < max(6, min_loop_vertices):
                    continue
                loop_idx = boundary_vertices[bcomp]
                coords = verts[loop_idx]
                center = np.mean(coords, axis=0)
                normal = self._estimate_loop_normal(coords)
                radial = np.linalg.norm(coords - center.reshape(1, 3), axis=1)
                coverage = self._loop_angular_coverage(coords, normal, bins=18)
                if coverage < 0.45:
                    continue

                planarity_pen = 0.0
                centered = coords - center.reshape(1, 3)
                try:
                    _, svals, _ = np.linalg.svd(centered, full_matrices=False)
                    if svals.shape[0] >= 3:
                        planarity_pen = 2.2 * float(svals[-1] / (svals[0] + 1e-6))
                except np.linalg.LinAlgError:
                    planarity_pen = 0.0
                circ_pen = 0.25 * float(np.std(radial) / (np.mean(radial) + 1e-6))
                cov_pen = max(0.0, 0.65 - coverage) * 4.0
                size_pen = max(0.0, float(min_loop_vertices - loop_idx.size)) * 0.05
                # Prefer terminal/outward regions over central sidewall patches.
                center_idx = int(self._map_points_to_mesh_vertices(center.reshape(1, 3))[0])
                dist_center = self._distances_from_vertex(center_idx)
                finite_mask = np.isfinite(dist_center)
                ecc_bonus = 0.0
                if np.any(finite_mask):
                    ecc_bonus = -0.015 * float(np.max(dist_center[finite_mask]) / (median_edge + 1e-12))
                dist_bonus = -0.04 * float(np.linalg.norm(center - global_centroid) / (median_edge + 1e-12))
                score = float(size_pen + circ_pen + cov_pen + planarity_pen + ecc_bonus + dist_bonus)

                candidates.append(
                    {
                        "loop_idx": np.asarray(loop_idx, dtype=np.int64),
                        "center": center,
                        "normal": normal,
                        "tangent": normal.copy(),
                        "source": "normal_graph_cluster",
                        "score": score,
                    }
                )
        return candidates

    def _extract_normal_patch_loop(
        self,
        path: np.ndarray,
        tangent: np.ndarray,
        min_loop_vertices: int = 12,
        normal_dot_min: float = 0.72,
        face_dot_min: float = 0.90,
    ) -> Optional[np.ndarray]:
        mesh = self.mesh_target_trimesh
        if not isinstance(mesh, trimesh.Trimesh):
            return None
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if faces.size == 0:
            return None
        face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
        if face_normals.shape[0] != faces.shape[0]:
            return None

        cache = self._get_mesh_graph_cache()
        verts = cache["verts"]
        path = np.asarray(path, dtype=np.int64).reshape(-1)
        if path.size == 0:
            return None
        tip_idx = int(path[-1])
        tip = verts[tip_idx]
        dist_tip, geo_min, geo_max, _ = self._opening_geo_limits(path, tip_idx)

        face_geo = np.median(dist_tip[faces], axis=1)
        face_centroids = np.mean(verts[faces], axis=1)
        tangent = np.asarray(tangent, dtype=np.float64).reshape(3)
        tan_norm = np.linalg.norm(tangent)
        if tan_norm < 1e-12:
            return None
        tangent = tangent / tan_norm

        valid = np.isfinite(face_geo) & (face_geo >= 0.5 * geo_min) & (face_geo <= 1.4 * geo_max)
        if not np.any(valid):
            valid = np.isfinite(face_geo)
        if not np.any(valid):
            return None

        align = np.abs(face_normals @ tangent)
        dist = np.linalg.norm(face_centroids - tip.reshape(1, 3), axis=1)
        cand = np.where(valid & (align >= float(normal_dot_min)))[0]
        if cand.size == 0:
            cand = np.where(valid)[0]
        if cand.size == 0:
            return None
        seed_score = align[cand] - 0.06 * dist[cand]
        seed = int(cand[int(np.argmax(seed_score))])

        face_adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
        if face_adj.size == 0:
            return None
        neigh = [[] for _ in range(faces.shape[0])]
        for a, b in face_adj:
            a = int(a)
            b = int(b)
            neigh[a].append(b)
            neigh[b].append(a)

        selected = {seed}
        queue = [seed]
        seed_normal = face_normals[seed]
        max_tip_dist = max(geo_max * 1.25, 8.0 * cache["median_edge_length"])
        while queue:
            curr = queue.pop()
            for nb in neigh[curr]:
                if nb in selected:
                    continue
                if not np.isfinite(face_geo[nb]) or face_geo[nb] > 1.5 * geo_max:
                    continue
                if np.linalg.norm(face_centroids[nb] - tip) > max_tip_dist:
                    continue
                if np.abs(np.dot(face_normals[nb], tangent)) < float(normal_dot_min):
                    continue
                if np.abs(np.dot(face_normals[nb], seed_normal)) < float(face_dot_min):
                    continue
                selected.add(nb)
                queue.append(nb)

        if len(selected) < 4:
            return None
        selected_faces = faces[np.asarray(sorted(selected), dtype=np.int64)]
        tri_edges = np.concatenate(
            (
                selected_faces[:, [0, 1]],
                selected_faces[:, [1, 2]],
                selected_faces[:, [2, 0]],
            ),
            axis=0,
        )
        tri_edges = np.sort(tri_edges, axis=1)
        uniq_edges, counts = np.unique(tri_edges, axis=0, return_counts=True)
        boundary_edges = uniq_edges[counts == 1]
        if boundary_edges.size == 0:
            return None

        boundary_vertices = np.unique(boundary_edges.reshape(-1))
        lut = -np.ones(verts.shape[0], dtype=np.int64)
        lut[boundary_vertices] = np.arange(boundary_vertices.shape[0], dtype=np.int64)
        local_edges = lut[boundary_edges]
        loop_graph = ig.Graph(n=boundary_vertices.shape[0], edges=local_edges.tolist(), directed=False)

        best_loop, best_score = None, np.inf
        for component in loop_graph.components():
            comp = np.asarray(component, dtype=np.int64)
            if comp.size < max(6, int(min_loop_vertices) // 2):
                continue
            loop_idx = boundary_vertices[comp]
            loop_idx = self._filter_loop_by_tip_geodesic(
                loop_idx, tip_idx, 0.5 * geo_min, 1.6 * geo_max
            )
            if loop_idx.size < 6:
                continue
            coords = verts[loop_idx]
            centroid = np.mean(coords, axis=0)
            radial = np.linalg.norm(coords - centroid.reshape(1, 3), axis=1)
            coverage = self._loop_angular_coverage(coords, tangent, bins=16)
            score = (
                np.linalg.norm(centroid - tip)
                + 0.20 * np.median(radial)
                + 0.15 * float(np.std(radial) / (np.mean(radial) + 1e-6))
                + max(0.0, 0.65 - coverage) * 4.0
            )
            if score < best_score:
                best_score = score
                best_loop = np.asarray(loop_idx, dtype=np.int64)
        return best_loop

    def _extract_normal_patch_loop_from_tip(
        self,
        tip_idx: int,
        tangent: np.ndarray,
        min_loop_vertices: int = 12,
        normal_dot_min: float = 0.72,
        face_dot_min: float = 0.90,
    ) -> Optional[np.ndarray]:
        cache = self._get_mesh_graph_cache()
        verts = cache["verts"]
        median_edge = cache["median_edge_length"]
        mesh = self.mesh_target_trimesh
        if not isinstance(mesh, trimesh.Trimesh):
            return None
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if faces.size == 0:
            return None
        face_normals = np.asarray(mesh.face_normals, dtype=np.float64)
        if face_normals.shape[0] != faces.shape[0]:
            return None

        tangent = np.asarray(tangent, dtype=np.float64).reshape(3)
        tan_norm = np.linalg.norm(tangent)
        if tan_norm < 1e-12:
            return None
        tangent = tangent / tan_norm

        dist_tip = self._distances_from_vertex(int(tip_idx))
        geo_min = max(2.0 * median_edge, 1e-6)
        geo_max = max(24.0 * median_edge, geo_min + 3.0 * median_edge)
        face_geo = np.median(dist_tip[faces], axis=1)
        valid = np.isfinite(face_geo) & (face_geo >= geo_min) & (face_geo <= geo_max)
        if not np.any(valid):
            valid = np.isfinite(face_geo)
        if not np.any(valid):
            return None

        face_centroids = np.mean(verts[faces], axis=1)
        align = np.abs(face_normals @ tangent)
        cand = np.where(valid & (align >= float(normal_dot_min)))[0]
        if cand.size == 0:
            cand = np.where(valid)[0]
        if cand.size == 0:
            return None
        tip = verts[int(tip_idx)]
        seed_score = align[cand] - 0.05 * np.linalg.norm(face_centroids[cand] - tip.reshape(1, 3), axis=1)
        seed = int(cand[int(np.argmax(seed_score))])

        face_adj = np.asarray(mesh.face_adjacency, dtype=np.int64)
        if face_adj.size == 0:
            return None
        neigh = [[] for _ in range(faces.shape[0])]
        for a, b in face_adj:
            a = int(a)
            b = int(b)
            neigh[a].append(b)
            neigh[b].append(a)

        selected = {seed}
        queue = [seed]
        seed_normal = face_normals[seed]
        max_tip_dist = max(geo_max * 1.35, 8.0 * median_edge)
        while queue:
            curr = queue.pop()
            for nb in neigh[curr]:
                if nb in selected:
                    continue
                if not np.isfinite(face_geo[nb]) or face_geo[nb] > 1.8 * geo_max:
                    continue
                if np.linalg.norm(face_centroids[nb] - tip) > max_tip_dist:
                    continue
                if np.abs(np.dot(face_normals[nb], tangent)) < float(normal_dot_min):
                    continue
                if np.abs(np.dot(face_normals[nb], seed_normal)) < float(face_dot_min):
                    continue
                selected.add(nb)
                queue.append(nb)

        if len(selected) < 4:
            return None
        selected_faces = faces[np.asarray(sorted(selected), dtype=np.int64)]
        tri_edges = np.concatenate(
            (
                selected_faces[:, [0, 1]],
                selected_faces[:, [1, 2]],
                selected_faces[:, [2, 0]],
            ),
            axis=0,
        )
        tri_edges = np.sort(tri_edges, axis=1)
        uniq_edges, counts = np.unique(tri_edges, axis=0, return_counts=True)
        boundary_edges = uniq_edges[counts == 1]
        if boundary_edges.size == 0:
            return None

        boundary_vertices = np.unique(boundary_edges.reshape(-1))
        lut = -np.ones(verts.shape[0], dtype=np.int64)
        lut[boundary_vertices] = np.arange(boundary_vertices.shape[0], dtype=np.int64)
        local_edges = lut[boundary_edges]
        loop_graph = ig.Graph(n=boundary_vertices.shape[0], edges=local_edges.tolist(), directed=False)

        best_loop, best_score = None, np.inf
        for component in loop_graph.components():
            comp = np.asarray(component, dtype=np.int64)
            if comp.size < max(6, int(min_loop_vertices) // 2):
                continue
            loop_idx = boundary_vertices[comp]
            loop_idx = self._filter_loop_by_tip_geodesic(loop_idx, int(tip_idx), geo_min * 0.8, geo_max * 1.2)
            if loop_idx.size < 6:
                continue
            coords = verts[loop_idx]
            coverage = self._loop_angular_coverage(coords, tangent, bins=16)
            centroid = np.mean(coords, axis=0)
            radial = np.linalg.norm(coords - centroid.reshape(1, 3), axis=1)
            score = (
                np.linalg.norm(centroid - tip)
                + 0.20 * np.median(radial)
                + 0.15 * float(np.std(radial) / (np.mean(radial) + 1e-6))
                + max(0.0, 0.65 - coverage) * 4.0
            )
            if score < best_score:
                best_score = score
                best_loop = np.asarray(loop_idx, dtype=np.int64)
        return best_loop

    def _angle_basis_from_normal(self, normal: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        normal = np.asarray(normal, dtype=np.float64).reshape(3)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            return None
        normal = normal / norm
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(helper, normal)) > 0.95:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        basis_x = np.cross(normal, helper)
        bx_norm = np.linalg.norm(basis_x)
        if bx_norm < 1e-12:
            return None
        basis_x = basis_x / bx_norm
        basis_y = np.cross(normal, basis_x)
        return normal, basis_x, basis_y

    def _loop_angular_coverage(self, loop_coords: np.ndarray, normal: np.ndarray, bins: int = 18) -> float:
        loop_coords = np.asarray(loop_coords, dtype=np.float64)
        if loop_coords.shape[0] < 6:
            return 0.0
        basis = self._angle_basis_from_normal(normal)
        if basis is None:
            return 0.0
        _, basis_x, basis_y = basis
        centroid = np.mean(loop_coords, axis=0)
        rel = loop_coords - centroid.reshape(1, 3)
        angles = np.arctan2(rel @ basis_y, rel @ basis_x)
        if angles.size == 0:
            return 0.0
        hist, _ = np.histogram(angles, bins=int(bins), range=(-np.pi, np.pi))
        occupied = np.count_nonzero(hist)
        return float(occupied) / float(bins)

    def _connected_components_from_subset(self, vertex_indices: np.ndarray) -> List[np.ndarray]:
        vertex_indices = np.unique(np.asarray(vertex_indices, dtype=np.int64).reshape(-1))
        if vertex_indices.size == 0:
            return []
        cache = self._get_mesh_graph_cache()
        edges = cache["edges"]
        if edges.size == 0:
            return []
        vertex_mask = np.zeros(cache["verts"].shape[0], dtype=bool)
        vertex_mask[vertex_indices] = True
        edge_mask = vertex_mask[edges[:, 0]] & vertex_mask[edges[:, 1]]
        sub_edges = edges[edge_mask]
        if sub_edges.size == 0:
            return [np.asarray([idx], dtype=np.int64) for idx in vertex_indices.tolist()]
        lut = -np.ones(cache["verts"].shape[0], dtype=np.int64)
        lut[vertex_indices] = np.arange(vertex_indices.size, dtype=np.int64)
        local_edges = lut[sub_edges]
        graph = ig.Graph(n=vertex_indices.size, edges=local_edges.tolist(), directed=False)
        return [
            vertex_indices[np.asarray(component, dtype=np.int64)]
            for component in graph.components()
        ]

    def _filter_loop_by_tip_geodesic(
        self,
        loop_idx: np.ndarray,
        tip_idx: int,
        geo_min: Optional[float],
        geo_max: Optional[float],
    ) -> np.ndarray:
        loop_idx = np.unique(np.asarray(loop_idx, dtype=np.int64).reshape(-1))
        if loop_idx.size == 0:
            return loop_idx
        if geo_min is None and geo_max is None:
            return loop_idx
        dist_tip = self._distances_from_vertex(int(tip_idx))
        geo = dist_tip[loop_idx]
        valid = np.isfinite(geo)
        if geo_min is not None:
            valid &= geo >= float(geo_min)
        if geo_max is not None:
            valid &= geo <= float(geo_max)
        filtered = loop_idx[valid]
        if filtered.size >= 6:
            return filtered
        if filtered.size == 0:
            return filtered
        # keep nearest geodesic vertices when thresholding gets too strict
        geo_f = dist_tip[filtered]
        order = np.argsort(geo_f)
        return filtered[order[: min(24, filtered.size)]]

    def _opening_geo_limits(self, path: np.ndarray, tip_idx: int) -> Tuple[np.ndarray, float, float, float]:
        cache = self._get_mesh_graph_cache()
        median_edge = cache["median_edge_length"]
        path = np.asarray(path, dtype=np.int64).reshape(-1)
        dist_tip = self._distances_from_vertex(int(tip_idx))
        branch_length = float(dist_tip[int(path[0])]) if path.size > 0 and np.isfinite(dist_tip[int(path[0])]) else np.nan
        if not np.isfinite(branch_length) or branch_length <= 0:
            branch_length = max(25.0 * median_edge, 1e-6)

        geo_min = max(2.5 * median_edge, 0.03 * branch_length)
        geo_max = min(0.42 * branch_length, 26.0 * median_edge)
        geo_max = max(geo_max, geo_min + 4.0 * median_edge)
        return dist_tip, float(geo_min), float(geo_max), float(branch_length)

    def _extract_geodesic_ring_loop(
        self,
        tip_idx: int,
        tangent: np.ndarray,
        dist_tip: np.ndarray,
        desired_geo: float,
        geo_min: float,
        geo_max: float,
        min_vertices: int,
    ) -> Optional[np.ndarray]:
        cache = self._get_mesh_graph_cache()
        verts = cache["verts"]
        median_edge = cache["median_edge_length"]
        finite = np.isfinite(dist_tip)
        if not np.any(finite):
            return None

        target_radii = sorted(
            set(
                [
                    float(desired_geo),
                    float(max(geo_min + 1.5 * median_edge, 0.75 * desired_geo)),
                    float(min(geo_max - 1.5 * median_edge, 1.25 * desired_geo)),
                ]
            )
        )
        band = max(1.75 * median_edge, 0.02 * (geo_max + geo_min))
        tip = verts[int(tip_idx)]

        best_loop, best_score = None, np.inf
        for radius in target_radii:
            if radius <= geo_min or radius >= geo_max:
                continue
            cand_mask = (
                finite
                & (dist_tip >= geo_min)
                & (dist_tip <= geo_max)
                & (np.abs(dist_tip - radius) <= band)
            )
            candidate = np.where(cand_mask)[0]
            if candidate.size < max(8, min_vertices // 2):
                continue
            for comp in self._connected_components_from_subset(candidate):
                if comp.size < max(8, min_vertices // 2):
                    continue
                coords = verts[comp]
                coverage = self._loop_angular_coverage(coords, tangent, bins=18)
                if coverage < 0.45:
                    continue
                centroid = np.mean(coords, axis=0)
                radial = np.linalg.norm(coords - centroid.reshape(1, 3), axis=1)
                geo_med = float(np.median(dist_tip[comp]))
                score = (
                    abs(geo_med - radius)
                    + 0.35 * np.linalg.norm(centroid - tip)
                    + 0.15 * float(np.std(radial) / (np.mean(radial) + 1e-6))
                    + max(0.0, min_vertices - comp.size) * 0.03
                    + max(0.0, 0.70 - coverage) * 3.5
                )
                if score < best_score:
                    best_score = score
                    best_loop = comp
        if best_loop is None:
            return None
        return np.asarray(best_loop, dtype=np.int64)

    def _extract_section_loop(
        self,
        plane_origin: np.ndarray,
        plane_normal: np.ndarray,
        tip_idx: int,
        geo_min: Optional[float] = None,
        geo_max: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        try:
            section = self.mesh_target_trimesh.section(
                plane_origin=np.asarray(plane_origin, dtype=np.float64),
                plane_normal=np.asarray(plane_normal, dtype=np.float64),
            )
        except Exception:
            return None
        if section is None:
            return None
        curves = getattr(section, "discrete", None)
        if curves is None or len(curves) == 0:
            return None

        cache = self._get_mesh_graph_cache()
        tip = cache["verts"][int(tip_idx)]
        best_curve = None
        best_score = np.inf
        for curve in curves:
            curve = np.asarray(curve, dtype=np.float64)
            if curve.ndim != 2 or curve.shape[0] < 6:
                continue
            loop_idx = np.unique(self._map_points_to_mesh_vertices(curve))
            loop_idx = self._filter_loop_by_tip_geodesic(loop_idx, tip_idx, geo_min, geo_max)
            if loop_idx.size < 6:
                continue
            loop_coords = cache["verts"][loop_idx]
            centroid = np.mean(loop_coords, axis=0)
            radius = np.median(np.linalg.norm(loop_coords - centroid.reshape(1, 3), axis=1))
            coverage = self._loop_angular_coverage(loop_coords, plane_normal, bins=16)
            score = (
                np.linalg.norm(centroid - tip)
                + 0.30 * radius
                + max(0.0, 0.60 - coverage) * 4.0
            )
            if score < best_score:
                best_curve = curve
                best_score = score
        if best_curve is None:
            return None

        loop_idx = np.unique(self._map_points_to_mesh_vertices(best_curve))
        loop_idx = self._filter_loop_by_tip_geodesic(loop_idx, tip_idx, geo_min, geo_max)
        if loop_idx.size < 6:
            return None
        return loop_idx.astype(np.int64)

    def _extract_sign_loop(
        self,
        plane_origin: np.ndarray,
        plane_normal: np.ndarray,
        tip_idx: int,
        min_vertices: int = 12,
        geo_min: Optional[float] = None,
        geo_max: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        cache = self._get_mesh_graph_cache()
        verts, edges = cache["verts"], cache["edges"]
        plane_origin = np.asarray(plane_origin, dtype=np.float64).reshape(1, 3)
        plane_normal = np.asarray(plane_normal, dtype=np.float64).reshape(3)
        plane_normal_norm = np.linalg.norm(plane_normal)
        if plane_normal_norm < 1e-12:
            return None
        plane_normal = plane_normal / plane_normal_norm

        signed = np.dot(verts - plane_origin, plane_normal)
        side = signed >= 0.0
        crossing_mask = np.logical_xor(side[edges[:, 0]], side[edges[:, 1]])
        boundary_edges = edges[crossing_mask]
        if boundary_edges.size == 0:
            return None
        boundary_vertices = np.unique(boundary_edges)
        boundary_vertices = self._filter_loop_by_tip_geodesic(boundary_vertices, tip_idx, geo_min, geo_max)
        if boundary_vertices.size < 6:
            return None
        vertex_mask = np.zeros(verts.shape[0], dtype=bool)
        vertex_mask[boundary_vertices] = True
        edge_mask = vertex_mask[boundary_edges[:, 0]] & vertex_mask[boundary_edges[:, 1]]
        boundary_edges = boundary_edges[edge_mask]
        if boundary_edges.size == 0:
            return None
        lut = -np.ones(verts.shape[0], dtype=np.int64)
        lut[boundary_vertices] = np.arange(boundary_vertices.shape[0], dtype=np.int64)
        local_edges = lut[boundary_edges]
        loop_graph = ig.Graph(
            n=boundary_vertices.shape[0],
            edges=local_edges.tolist(),
            directed=False,
        )
        components = loop_graph.components()
        if len(components) == 0:
            return None

        tip = verts[int(tip_idx)]
        best_loop = None
        best_score = np.inf
        for component in components:
            local_idx = np.asarray(component, dtype=np.int64)
            if local_idx.size < 4:
                continue
            orig_idx = boundary_vertices[local_idx]
            centroid = np.mean(verts[orig_idx], axis=0)
            radial = np.linalg.norm(verts[orig_idx] - centroid.reshape(1, 3), axis=1)
            coverage = self._loop_angular_coverage(verts[orig_idx], plane_normal, bins=16)
            size_penalty = 0.0 if orig_idx.size >= int(min_vertices) else 1000.0
            score = (
                np.linalg.norm(centroid - tip)
                + 0.35 * np.median(radial)
                + 0.12 * float(np.std(radial) / (np.mean(radial) + 1e-6))
                + max(0.0, 0.60 - coverage) * 4.5
                + size_penalty
            )
            if score < best_score:
                best_score = score
                best_loop = orig_idx
        if best_loop is None:
            return None
        return np.asarray(best_loop, dtype=np.int64)

    def _extract_opening_loop_indices(
        self,
        path: np.ndarray,
        tangent: np.ndarray,
        min_loop_vertices: int = 24,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        cache = self._get_mesh_graph_cache()
        verts = cache["verts"]
        path = np.asarray(path, dtype=np.int64).reshape(-1)
        if path.size < 3:
            return None, None
        tip_idx = int(path[-1])
        tip = verts[tip_idx]
        dist_tip, geo_min, geo_max, _ = self._opening_geo_limits(path, tip_idx)
        desired_geo = 0.5 * (geo_min + geo_max)

        best_loop, best_cut = None, None
        best_score = np.inf

        # 1) First try a local geodesic ring near endpoint.
        loop_geo = self._extract_geodesic_ring_loop(
            tip_idx=tip_idx,
            tangent=tangent,
            dist_tip=dist_tip,
            desired_geo=desired_geo,
            geo_min=geo_min,
            geo_max=geo_max,
            min_vertices=int(min_loop_vertices),
        )
        if loop_geo is not None and loop_geo.size >= 6:
            centroid_geo = np.mean(verts[loop_geo], axis=0)
            coverage_geo = self._loop_angular_coverage(verts[loop_geo], tangent, bins=18)
            geo_med = float(np.median(dist_tip[loop_geo]))
            score_geo = (
                abs(geo_med - desired_geo)
                + 0.30 * np.linalg.norm(centroid_geo - tip)
                + max(0.0, 0.70 - coverage_geo) * 3.5
                + max(0.0, min_loop_vertices - loop_geo.size) * 0.03
            )
            best_loop = loop_geo
            best_score = score_geo
            target_idx = int(path[np.argmin(np.abs(dist_tip[path] - geo_med))])
            best_cut = verts[target_idx]

        # 2) Plane-based methods, but constrained by endpoint geodesic distance.
        path_geo = dist_tip[path]
        geo_targets = np.linspace(geo_min * 1.10, geo_max * 0.95, num=4)
        for target_geo in geo_targets:
            cut_local = int(np.argmin(np.abs(path_geo - target_geo)))
            cut_idx = int(path[cut_local])
            cut_point = verts[cut_idx]
            loop_idx = self._extract_section_loop(
                cut_point,
                tangent,
                tip_idx,
                geo_min=geo_min,
                geo_max=geo_max,
            )
            if loop_idx is None:
                loop_idx = self._extract_sign_loop(
                    cut_point,
                    tangent,
                    tip_idx,
                    min_vertices=max(8, min_loop_vertices // 2),
                    geo_min=geo_min,
                    geo_max=geo_max,
                )
            if loop_idx is None or loop_idx.size < 6:
                continue
            loop_centroid = np.mean(verts[loop_idx], axis=0)
            loop_radius = np.median(np.linalg.norm(verts[loop_idx] - loop_centroid.reshape(1, 3), axis=1))
            coverage = self._loop_angular_coverage(verts[loop_idx], tangent, bins=18)
            geo_med = float(np.median(dist_tip[loop_idx]))
            score = (
                abs(geo_med - target_geo)
                + 0.25 * np.linalg.norm(loop_centroid - tip)
                + 0.20 * loop_radius
                + max(0.0, 0.65 - coverage) * 4.0
                + max(0.0, min_loop_vertices - loop_idx.size) * 0.04
            )
            if score < best_score:
                best_score = score
                best_loop = loop_idx
                best_cut = cut_point

        if best_loop is None:
            return None, None
        return np.asarray(best_loop, dtype=np.int64), np.asarray(best_cut, dtype=np.float64)

    def _fallback_opening_indices(self, path: np.ndarray, tangent: np.ndarray, count: int = 48) -> np.ndarray:
        cache = self._get_mesh_graph_cache()
        verts = cache["verts"]
        path = np.asarray(path, dtype=np.int64).reshape(-1)
        if path.size == 0:
            raise RuntimeError("Cannot fallback opening extraction without a valid path.")
        anchor_idx = max(0, path.size - 1 - max(2, path.size // 10))
        anchor = verts[int(path[anchor_idx])]
        tangent = np.asarray(tangent, dtype=np.float64).reshape(3)
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm < 1e-12:
            tangent = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            tangent_norm = 1.0
        tangent = tangent / tangent_norm
        signed = np.dot(verts - anchor.reshape(1, 3), tangent)
        projected = verts - np.outer(signed, tangent)
        radial = np.linalg.norm(projected - anchor.reshape(1, 3), axis=1)
        mask = signed >= -0.5 * cache["median_edge_length"]
        candidate_idx = np.where(mask)[0]
        if candidate_idx.size < count:
            candidate_idx = np.arange(verts.shape[0])
        ranked = candidate_idx[np.argsort(radial[candidate_idx])]
        return np.asarray(ranked[: min(int(count), ranked.shape[0])], dtype=np.int64)

    def extract_centreline_from_mesh(self, num_endpoints: Optional[int] = None, sort_branches: bool = True) -> Dict[str, object]:
        num_endpoints = int(self.num_op if num_endpoints is None else num_endpoints)
        if num_endpoints < 1:
            raise ValueError("num_endpoints must be >= 1")

        endpoint_indices = self._select_endpoint_indices_fps(num_endpoints)
        center_idx = self._estimate_bifurcation_index(endpoint_indices)
        if sort_branches:
            endpoint_indices = self._sort_endpoint_indices(endpoint_indices, center_idx)
        paths = self._branch_paths_from_endpoints(endpoint_indices, center_idx)
        if len(paths) != len(endpoint_indices):
            raise RuntimeError(
                f"Failed to recover all branch paths. Expected {len(endpoint_indices)}, got {len(paths)}."
            )

        tangents = np.stack([self._endpoint_tangent_from_path(path) for path in paths], axis=0)
        cache = self._get_mesh_graph_cache()
        verts = cache["verts"]
        centreline_points = [verts[path] for path in paths]
        path_lengths = [float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))) for points in centreline_points]
        summary = {
            "endpoint_indices": [int(i) for i in endpoint_indices],
            "center_index": int(center_idx),
            "paths": [path.astype(np.int64) for path in paths],
            "tangents": tangents.astype(np.float64),
            "centreline_points": centreline_points,
            "endpoints": verts[np.asarray(endpoint_indices, dtype=np.int64)].copy(),
            "center_point": verts[int(center_idx)].copy(),
            "path_lengths": path_lengths,
        }
        self.auto_centreline_summary = summary
        return summary

    def _centreline_from_endpoint_indices(
        self,
        endpoint_indices: List[int],
        sort_branches: bool = True,
    ) -> Dict[str, object]:
        endpoint_indices = [int(i) for i in endpoint_indices]
        if len(endpoint_indices) < 1:
            raise ValueError("endpoint_indices must contain at least one vertex id.")
        center_idx = self._estimate_bifurcation_index(endpoint_indices)
        if sort_branches:
            endpoint_indices = self._sort_endpoint_indices(endpoint_indices, center_idx)
        paths = self._branch_paths_from_endpoints(endpoint_indices, center_idx)
        if len(paths) != len(endpoint_indices):
            raise RuntimeError(
                f"Failed to recover all branch paths. Expected {len(endpoint_indices)}, got {len(paths)}."
            )
        tangents = np.stack([self._endpoint_tangent_from_path(path) for path in paths], axis=0)
        cache = self._get_mesh_graph_cache()
        verts = cache["verts"]
        centreline_points = [verts[path] for path in paths]
        path_lengths = [float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))) for points in centreline_points]
        summary = {
            "endpoint_indices": [int(i) for i in endpoint_indices],
            "center_index": int(center_idx),
            "paths": [path.astype(np.int64) for path in paths],
            "tangents": tangents.astype(np.float64),
            "centreline_points": centreline_points,
            "endpoints": verts[np.asarray(endpoint_indices, dtype=np.int64)].copy(),
            "center_point": verts[int(center_idx)].copy(),
            "path_lengths": path_lengths,
        }
        self.auto_centreline_summary = summary
        return summary

    def _map_points_to_mesh_vertices_unique(self, points: np.ndarray, k: int = 8) -> List[int]:
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[0] == 0:
            return []
        mesh = self.mesh_target_trimesh
        used = set()
        out = []
        for p in points:
            candidate = None
            try:
                _, idx = mesh.kdtree.query(p, k=max(1, int(k)))
                idx = np.asarray(idx).reshape(-1).astype(np.int64)
            except Exception:
                cache = self._get_mesh_graph_cache()
                verts = cache["verts"]
                sq = np.sum((verts - p.reshape(1, 3)) ** 2, axis=1)
                idx = np.argsort(sq)[: max(1, int(k))].astype(np.int64)
            for v in idx:
                vv = int(v)
                if vv not in used:
                    candidate = vv
                    break
            if candidate is None:
                candidate = int(idx[0])
            used.add(candidate)
            out.append(candidate)
        return out

    def register_openings_auto(self, min_loop_vertices: int = 24) -> Dict[str, object]:
        self._reset_opening_state()
        pending_debug = self._auto_reg_pending_debug if isinstance(self._auto_reg_pending_debug, dict) else None
        self._auto_reg_pending_debug = None
        self.auto_registration_debug = {
            "method": "register_openings_auto",
            "fallback_from": None,
            "fallback_reason": None,
            "min_loop_vertices": int(min_loop_vertices),
        }
        if pending_debug is not None:
            self.auto_registration_debug.update(pending_debug)
        centreline_summary = self.extract_centreline_from_mesh(num_endpoints=self.num_op, sort_branches=True)
        cache = self._get_mesh_graph_cache()
        verts = cache["verts"]

        mesh_normals = None
        try:
            self.mesh_target.compute_vertex_normals()
            mesh_normals = np.asarray(self.mesh_target.vertex_normals)
            if mesh_normals.shape[0] != verts.shape[0]:
                mesh_normals = None
        except Exception:
            mesh_normals = None

        for path, tangent in zip(centreline_summary["paths"], centreline_summary["tangents"]):
            loop_idx, cut_point = self._extract_opening_loop_indices(
                path=path,
                tangent=tangent,
                min_loop_vertices=int(min_loop_vertices),
            )
            if loop_idx is None or loop_idx.size < 6:
                loop_idx = self._fallback_opening_indices(
                    path=path,
                    tangent=tangent,
                    count=max(48, int(min_loop_vertices)),
                )
                cut_point = verts[int(path[-1])]
            loop_idx = np.unique(loop_idx.astype(np.int64))
            loop_idx = self._sort_indices_around_normal(loop_idx, tangent)
            loop_coords = verts[loop_idx]
            n_mean = np.asarray(tangent, dtype=np.float64).reshape(3)
            tip = verts[int(path[-1])]
            if np.dot(n_mean, tip - np.mean(loop_coords, axis=0)) < 0:
                n_mean = -1.0 * n_mean

            self.op_v_indices.append(loop_idx.tolist())
            self.op_v_coords.append(loop_coords)
            if mesh_normals is not None:
                self.op_v_normal.append(mesh_normals[loop_idx])
            else:
                self.op_v_normal.append(np.repeat(n_mean.reshape(1, 3), loop_idx.size, axis=0))
            self.op_n_mean.append(n_mean)
            self.op_tangent.append(np.asarray(tangent, dtype=np.float64))
            self.op_cut_points.append(np.asarray(cut_point, dtype=np.float64))

        if len(self.op_v_indices) != self.num_op:
            raise RuntimeError(
                f"Automatic opening registration produced {len(self.op_v_indices)} openings, expected {self.num_op}."
            )
        self.auto_registration_debug.update(
            {
                "num_openings": len(self.op_v_indices),
                "opening_sizes": [len(v) for v in self.op_v_indices],
                "endpoint_indices": [int(i) for i in centreline_summary.get("endpoint_indices", [])],
            }
        )
        return centreline_summary

    def register_openings_auto_normals(
        self,
        min_loop_vertices: int = 24,
        normal_dot_min: float = 0.72,
        face_dot_min: float = 0.90,
    ) -> Dict[str, object]:
        """
        Automatic opening registration using surface-normal cues:
        1) prefer explicit boundary loops (open meshes),
        2) fallback to face-graph normal patch extraction,
        3) derive centreline endpoints from opening centers.
        """
        self._reset_opening_state()
        self.auto_registration_debug = {
            "method": "register_openings_auto_normals",
            "fallback_from": None,
            "fallback_reason": None,
            "min_loop_vertices": int(min_loop_vertices),
            "normal_dot_min": float(normal_dot_min),
            "face_dot_min": float(face_dot_min),
        }
        cache = self._get_mesh_graph_cache()
        verts = cache["verts"]
        median_edge = cache["median_edge_length"]
        global_centroid = np.mean(verts, axis=0)

        mesh_normals = None
        try:
            self.mesh_target.compute_vertex_normals()
            mesh_normals = np.asarray(self.mesh_target.vertex_normals)
            if mesh_normals.shape[0] != verts.shape[0]:
                mesh_normals = None
        except Exception:
            mesh_normals = None

        candidates = []
        anchor_indices = []
        anchor_center_idx = None
        try:
            anchor_indices = self._select_endpoint_indices_fps(int(self.num_op))
        except Exception:
            anchor_indices = []
        if len(anchor_indices) >= int(self.num_op):
            anchor_indices = [int(i) for i in anchor_indices[: int(self.num_op)]]
            try:
                anchor_center_idx = int(self._estimate_bifurcation_index(anchor_indices))
            except Exception:
                anchor_center_idx = None
        self.auto_registration_debug["anchor_endpoint_indices"] = [int(i) for i in anchor_indices]
        self.auto_registration_debug["anchor_center_index"] = (
            None if anchor_center_idx is None else int(anchor_center_idx)
        )

        # Stage A: boundary loops (open meshes).
        boundary_loops = self._extract_boundary_loops(min_loop_vertices=max(6, int(min_loop_vertices) // 2))
        self.auto_registration_debug["boundary_loop_count"] = int(len(boundary_loops))
        for loop_idx in boundary_loops:
            loop_idx = np.unique(np.asarray(loop_idx, dtype=np.int64))
            if loop_idx.size < 6:
                continue
            coords = verts[loop_idx]
            center = np.mean(coords, axis=0)
            normal = self._estimate_loop_normal(coords)
            tip_local = int(np.argmax(np.linalg.norm(coords - center.reshape(1, 3), axis=1)))
            tangent = coords[tip_local] - center
            if np.linalg.norm(tangent) < 1e-12:
                tangent = normal.copy()
            candidates.append(
                {
                    "loop_idx": loop_idx,
                    "center": center,
                    "normal": normal,
                    "tangent": tangent / (np.linalg.norm(tangent) + 1e-12),
                    "source": "boundary",
                }
            )

        # Stage B: global face-graph normal clustering (with adaptive retries).
        def _deduplicate_candidates(cand_list):
            dedup_list = []
            for cand in cand_list:
                loop_idx = np.asarray(cand["loop_idx"], dtype=np.int64)
                keep = True
                for j, prev in enumerate(dedup_list):
                    cand_anchor = cand.get("anchor_id", None)
                    prev_anchor = prev.get("anchor_id", None)
                    if cand_anchor is not None and prev_anchor is not None and int(cand_anchor) != int(prev_anchor):
                        continue
                    inter = np.intersect1d(loop_idx, prev["loop_idx"]).size
                    denom = max(1, min(loop_idx.size, prev["loop_idx"].size))
                    overlap = inter / float(denom)
                    center_dist = np.linalg.norm(cand["center"] - prev["center"])
                    if overlap > 0.65 or center_dist < 2.0 * median_edge:
                        # Keep the larger loop for stability.
                        if loop_idx.size > prev["loop_idx"].size:
                            dedup_list[j] = cand
                        keep = False
                        break
                if keep:
                    dedup_list.append(cand)
            return dedup_list

        base_thr = float(max(normal_dot_min, face_dot_min))
        tried_thr = set()
        cluster_attempts = []
        cluster_candidates_total = 0

        def _run_cluster_thresholds(label: str, thresholds: List[float]):
            nonlocal cluster_candidates_total
            local_thresholds = []
            local_added = 0
            for thr in thresholds:
                thr = float(np.clip(thr, 0.50, 0.9995))
                key = round(thr, 6)
                if key in tried_thr:
                    continue
                tried_thr.add(key)
                local_thresholds.append(thr)
                cands_thr = self._extract_normal_graph_cluster_candidates(
                    normal_dot_min=thr,
                    min_loop_vertices=int(min_loop_vertices),
                )
                local_added += len(cands_thr)
                cluster_candidates_total += len(cands_thr)
                candidates.extend(cands_thr)
                if len(cands_thr) >= int(self.num_op):
                    break
            if len(local_thresholds) > 0:
                cluster_attempts.append(
                    {
                        "label": str(label),
                        "thresholds": [float(t) for t in local_thresholds],
                        "added_candidates": int(local_added),
                    }
                )
            return local_added

        # First pass.
        _run_cluster_thresholds(
            "base",
            [
                base_thr,
                max(0.82, base_thr - 0.06),
                max(0.75, base_thr - 0.12),
            ],
        )
        deduped = _deduplicate_candidates(candidates)

        # Retry pass: lower threshold when too few candidates survived.
        if len(deduped) < int(self.num_op):
            _run_cluster_thresholds(
                "lower_retry",
                [
                    max(0.78, base_thr - 0.10),
                    max(0.70, base_thr - 0.16),
                    max(0.62, base_thr - 0.22),
                    max(0.55, base_thr - 0.28),
                ],
            )
            deduped = _deduplicate_candidates(candidates)

        # Optional rescue: in some cases everything merges into one giant component;
        # stricter thresholds can split it into usable terminal patches.
        if len(deduped) < int(self.num_op):
            strict_base = float(min(0.995, max(base_thr + 0.08, 0.94)))
            strict_thresholds = [
                strict_base,
                max(0.92, strict_base - 0.03),
                max(0.90, strict_base - 0.06),
            ]
            _run_cluster_thresholds(
                "strict_retry",
                strict_thresholds,
            )
            deduped = _deduplicate_candidates(candidates)

        self.auto_registration_debug["cluster_thresholds"] = [
            float(t) for attempt in cluster_attempts for t in attempt["thresholds"]
        ]
        self.auto_registration_debug["cluster_threshold_attempts"] = cluster_attempts
        self.auto_registration_debug["cluster_retry_used"] = bool(len(cluster_attempts) > 1)
        self.auto_registration_debug["cluster_candidates_raw"] = int(cluster_candidates_total)

        candidate_count_raw_pre_rescue = int(len(candidates))
        candidate_count_dedup_pre_rescue = int(len(deduped))

        # Stage C: endpoint-anchored local normal patches.
        # This rescues cases where global clustering yields too few loops.
        tip_patch_attempts = []
        tip_patch_added = 0
        if len(deduped) < int(self.num_op) and len(anchor_indices) > 0:
            if anchor_center_idx is not None and 0 <= int(anchor_center_idx) < verts.shape[0]:
                center_ref = verts[int(anchor_center_idx)]
                center_ref_idx = int(anchor_center_idx)
            else:
                center_ref = global_centroid
                center_ref_idx = None

            rescue_anchor_pool = [int(i) for i in anchor_indices]
            rescue_anchor_meta = {}
            rescue_pool_count = max(8, int(3 * int(self.num_op)))
            try:
                pool_indices = self._select_endpoint_indices_fps(rescue_pool_count)
            except Exception:
                pool_indices = rescue_anchor_pool.copy()
            pool_indices = [int(i) for i in pool_indices]
            pool_indices = list(dict.fromkeys(pool_indices))
            if center_ref_idx is None and len(pool_indices) >= 2:
                try:
                    center_ref_idx = int(self._estimate_bifurcation_index(pool_indices[: max(2, int(self.num_op))]))
                    center_ref = verts[center_ref_idx]
                except Exception:
                    center_ref_idx = None
            if center_ref_idx is not None:
                dist_from_center = self._distances_from_vertex(center_ref_idx)
                ranked = []
                for idx in pool_indices:
                    geo = float(dist_from_center[int(idx)]) if np.isfinite(dist_from_center[int(idx)]) else -np.inf
                    direction = verts[int(idx)] - center_ref
                    dnorm = float(np.linalg.norm(direction))
                    if dnorm > 1e-12:
                        direction = direction / dnorm
                    else:
                        direction = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    ranked.append((int(idx), geo, direction))
                ranked = sorted(ranked, key=lambda x: x[1], reverse=True)
                diverse = []
                diverse_dirs = []
                cos_same_branch = 0.86
                target_pool = max(int(self.num_op) + 3, 6)
                for idx, geo, direction in ranked:
                    is_same = any(float(np.dot(direction, dprev)) > cos_same_branch for dprev in diverse_dirs)
                    if is_same:
                        continue
                    diverse.append((idx, geo, direction))
                    diverse_dirs.append(direction)
                    if len(diverse) >= target_pool:
                        break
                if len(diverse) < int(self.num_op):
                    used_idx = {int(v[0]) for v in diverse}
                    for item in ranked:
                        if int(item[0]) in used_idx:
                            continue
                        diverse.append(item)
                        if len(diverse) >= target_pool:
                            break
                if len(diverse) > 0:
                    rescue_anchor_pool = [int(v[0]) for v in diverse]
                    max_geo = max([float(v[1]) for v in diverse if np.isfinite(v[1])] + [1e-6])
                    for idx, geo, _ in diverse:
                        geo_val = float(geo) if np.isfinite(geo) else 0.0
                        rescue_anchor_meta[int(idx)] = {
                            "geo_distance": geo_val,
                            "geo_norm": float(np.clip(geo_val / (max_geo + 1e-12), 0.0, 1.0)),
                        }
            self.auto_registration_debug["tip_patch_anchor_pool_size"] = int(len(rescue_anchor_pool))
            self.auto_registration_debug["tip_patch_anchor_pool_indices"] = [int(v) for v in rescue_anchor_pool]

            patch_thresholds = [
                (
                    float(np.clip(max(normal_dot_min + 0.10, 0.82), 0.50, 0.995)),
                    float(np.clip(max(face_dot_min + 0.02, 0.92), 0.55, 0.999)),
                ),
                (
                    float(np.clip(max(normal_dot_min + 0.04, 0.74), 0.50, 0.99)),
                    float(np.clip(max(face_dot_min - 0.02, 0.86), 0.55, 0.995)),
                ),
                (
                    float(np.clip(max(normal_dot_min - 0.04, 0.66), 0.50, 0.98)),
                    float(np.clip(max(face_dot_min - 0.08, 0.78), 0.55, 0.99)),
                ),
                (
                    float(np.clip(max(normal_dot_min - 0.10, 0.58), 0.50, 0.95)),
                    float(np.clip(max(face_dot_min - 0.16, 0.70), 0.55, 0.98)),
                ),
            ]
            patch_thresholds_unique = []
            seen_thresholds = set()
            for nd_thr, fd_thr in patch_thresholds:
                key = (round(float(nd_thr), 6), round(float(fd_thr), 6))
                if key in seen_thresholds:
                    continue
                seen_thresholds.add(key)
                patch_thresholds_unique.append((float(nd_thr), float(fd_thr)))

            for anchor_id, tip_idx in enumerate(rescue_anchor_pool):
                tip_idx = int(tip_idx)
                tangent = verts[tip_idx] - center_ref
                tan_norm = np.linalg.norm(tangent)
                if tan_norm < 1e-12:
                    tangent = verts[tip_idx] - global_centroid
                    tan_norm = np.linalg.norm(tangent)
                if tan_norm < 1e-12:
                    tangent = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    tan_norm = 1.0
                tangent = tangent / tan_norm

                attempt = {
                    "anchor_id": int(anchor_id),
                    "tip_idx": int(tip_idx),
                    "thresholds_tried": [],
                    "accepted": False,
                }
                for nd_thr, fd_thr in patch_thresholds_unique:
                    attempt["thresholds_tried"].append(
                        {
                            "normal_dot_min": float(nd_thr),
                            "face_dot_min": float(fd_thr),
                        }
                    )
                    loop_idx = self._extract_normal_patch_loop_from_tip(
                        tip_idx=tip_idx,
                        tangent=tangent,
                        min_loop_vertices=int(min_loop_vertices),
                        normal_dot_min=float(nd_thr),
                        face_dot_min=float(fd_thr),
                    )
                    if loop_idx is None or loop_idx.size < 6:
                        continue
                    loop_idx = np.unique(np.asarray(loop_idx, dtype=np.int64))
                    if loop_idx.size < 6:
                        continue
                    coords = verts[loop_idx]
                    center = np.mean(coords, axis=0)
                    normal = self._estimate_loop_normal(coords)
                    if np.dot(normal, tangent) < 0:
                        normal = -1.0 * normal

                    radius_growth = None
                    path_turn_mean = None
                    path_turn_p90 = None
                    tip_turn_mean = None
                    tip_turn_p90 = None
                    elbow_angle = None
                    elbow_axis_ratio = None
                    if center_ref_idx is not None:
                        try:
                            path_tip = cache["graph"].get_shortest_paths(
                                v=int(center_ref_idx),
                                to=int(tip_idx),
                                mode="all",
                                weights="weight",
                                output="vpath",
                            )[0]
                            path_tip = np.asarray(path_tip, dtype=np.int64)
                            if path_tip.size >= 2:
                                curvature = self._path_turn_curvature(path_tip, tip_window_frac=0.45)
                                path_turn_mean = float(curvature["turn_mean"])
                                path_turn_p90 = float(curvature["turn_p90"])
                                tip_turn_mean = float(curvature["tip_turn_mean"])
                                tip_turn_p90 = float(curvature["tip_turn_p90"])
                                elbow = self._path_elbow_feature(path_tip, distal_frac=0.40)
                                elbow_angle = float(elbow["elbow_angle"])
                                elbow_axis_ratio = float(elbow["elbow_axis_ratio"])
                                dist_tip_rg, geo_min_rg, geo_max_rg, _ = self._opening_geo_limits(path_tip, int(tip_idx))
                                radii_rg = []
                                for alpha_rg in (0.30, 0.85):
                                    desired_rg = float(geo_min_rg + alpha_rg * (geo_max_rg - geo_min_rg))
                                    loop_rg = self._extract_geodesic_ring_loop(
                                        tip_idx=int(tip_idx),
                                        tangent=tangent,
                                        dist_tip=dist_tip_rg,
                                        desired_geo=desired_rg,
                                        geo_min=geo_min_rg,
                                        geo_max=geo_max_rg,
                                        min_vertices=max(8, int(min_loop_vertices) // 2),
                                    )
                                    if loop_rg is None or loop_rg.size < 6:
                                        radii_rg.append(None)
                                        continue
                                    loop_rg = np.unique(np.asarray(loop_rg, dtype=np.int64))
                                    coords_rg = verts[loop_rg]
                                    center_rg = np.mean(coords_rg, axis=0)
                                    radial_rg = np.linalg.norm(coords_rg - center_rg.reshape(1, 3), axis=1)
                                    radii_rg.append(float(np.median(radial_rg)))
                                if (
                                    radii_rg[0] is not None
                                    and radii_rg[1] is not None
                                    and radii_rg[0] > 1e-8
                                ):
                                    radius_growth = float(radii_rg[1] / radii_rg[0])
                        except Exception:
                            radius_growth = None
                            path_turn_mean = None
                            path_turn_p90 = None
                            tip_turn_mean = None
                            tip_turn_p90 = None
                            elbow_angle = None
                            elbow_axis_ratio = None
                    candidates.append(
                        {
                            "loop_idx": loop_idx,
                            "center": center,
                            "normal": normal,
                            "tangent": tangent.copy(),
                            "source": "normal_tip_patch",
                            "anchor_id": int(anchor_id),
                            "tip_idx": int(tip_idx),
                            "anchor_geo_distance": float(
                                rescue_anchor_meta.get(int(tip_idx), {}).get("geo_distance", 0.0)
                            ),
                            "anchor_geo_norm": float(
                                rescue_anchor_meta.get(int(tip_idx), {}).get("geo_norm", 0.0)
                            ),
                            "radius_growth": None if radius_growth is None else float(radius_growth),
                            "path_turn_mean": None if path_turn_mean is None else float(path_turn_mean),
                            "path_turn_p90": None if path_turn_p90 is None else float(path_turn_p90),
                            "tip_turn_mean": None if tip_turn_mean is None else float(tip_turn_mean),
                            "tip_turn_p90": None if tip_turn_p90 is None else float(tip_turn_p90),
                            "elbow_angle": None if elbow_angle is None else float(elbow_angle),
                            "elbow_axis_ratio": None if elbow_axis_ratio is None else float(elbow_axis_ratio),
                        }
                    )
                    tip_patch_added += 1
                    attempt["accepted"] = True
                    attempt["accepted_size"] = int(loop_idx.size)
                    attempt["accepted_threshold"] = {
                        "normal_dot_min": float(nd_thr),
                        "face_dot_min": float(fd_thr),
                    }
                    break
                tip_patch_attempts.append(attempt)

            if tip_patch_added > 0:
                deduped = _deduplicate_candidates(candidates)

        self.auto_registration_debug["tip_patch_attempts"] = tip_patch_attempts
        self.auto_registration_debug["tip_patch_added_candidates"] = int(tip_patch_added)
        self.auto_registration_debug["tip_patch_rescue_used"] = bool(tip_patch_added > 0)
        self.auto_registration_debug["candidate_count_raw_pre_rescue"] = int(candidate_count_raw_pre_rescue)
        self.auto_registration_debug["candidate_count_dedup_pre_rescue"] = int(candidate_count_dedup_pre_rescue)
        self.auto_registration_debug["candidate_count_raw"] = int(len(candidates))
        self.auto_registration_debug["candidate_count_dedup"] = int(len(deduped))

        if len(deduped) == 0:
            # Full fallback to legacy auto path-based method.
            self._auto_reg_pending_debug = {
                "fallback_from": "register_openings_auto_normals",
                "fallback_reason": "no_candidates_after_dedup",
                "normals_debug_snapshot": dict(self.auto_registration_debug),
            }
            return self.register_openings_auto(min_loop_vertices=min_loop_vertices)

        for cand in deduped:
            coords = verts[cand["loop_idx"]]
            center = cand["center"]
            center_idx = int(self._map_points_to_mesh_vertices(center.reshape(1, 3))[0])
            radial = np.linalg.norm(coords - center.reshape(1, 3), axis=1)
            coverage = self._loop_angular_coverage(coords, cand["normal"], bins=16)
            planarity_pen = 0.0
            if coords.shape[0] >= 6:
                centered = coords - np.mean(coords, axis=0, keepdims=True)
                try:
                    _, svals, _ = np.linalg.svd(centered, full_matrices=False)
                    if svals.shape[0] >= 3:
                        planarity_pen = 2.2 * float(svals[-1] / (svals[0] + 1e-6))
                except np.linalg.LinAlgError:
                    planarity_pen = 0.0
            size_pen = max(0.0, float(min_loop_vertices - len(cand["loop_idx"]))) * 0.05
            circ_pen = 0.25 * float(np.std(radial) / (np.mean(radial) + 1e-6))
            cov_pen = max(0.0, 0.65 - coverage) * 4.0
            dist_bonus = -0.04 * (
                np.linalg.norm(center - global_centroid) / (median_edge + 1e-12)
            )
            # Favor vertices with high geodesic eccentricity to avoid selecting sidewall aneurysm patches.
            dist_center = self._distances_from_vertex(center_idx)
            finite_mask = np.isfinite(dist_center)
            ecc_bonus = 0.0
            if np.any(finite_mask):
                ecc_bonus = -0.015 * float(np.max(dist_center[finite_mask]) / (median_edge + 1e-12))
            source_bonus = -0.4 if cand["source"] == "boundary" else 0.0
            anchor_bonus = 0.0
            if cand.get("source") == "normal_tip_patch":
                anchor_geo_norm = float(cand.get("anchor_geo_norm", 0.0))
                anchor_bonus = -0.50 * float(np.clip(anchor_geo_norm, 0.0, 1.0))
            growth_pen = 0.0
            if cand.get("source") == "normal_tip_patch":
                rg = cand.get("radius_growth", None)
                if rg is not None and np.isfinite(float(rg)):
                    growth_pen = 1.35 * max(0.0, float(rg) - 1.68)
            curvature_bonus = 0.0
            if cand.get("source") == "normal_tip_patch":
                tip_p90 = cand.get("tip_turn_p90", None)
                tip_mean = cand.get("tip_turn_mean", None)
                if tip_p90 is not None and np.isfinite(float(tip_p90)):
                    tip_p90_clip = min(float(tip_p90), 1.15)
                    curvature_bonus += -0.35 * max(0.0, tip_p90_clip - 0.72)
                if tip_mean is not None and np.isfinite(float(tip_mean)):
                    curvature_bonus += -0.15 * max(0.0, float(tip_mean) - 0.36)
            elbow_bonus = 0.0
            if cand.get("source") == "normal_tip_patch":
                elbow_angle = cand.get("elbow_angle", None)
                elbow_axis_ratio = cand.get("elbow_axis_ratio", None)
                if (
                    elbow_angle is not None
                    and elbow_axis_ratio is not None
                    and np.isfinite(float(elbow_angle))
                    and np.isfinite(float(elbow_axis_ratio))
                ):
                    # Favor tip regions with a clear elbow-like turn in one dominant plane.
                    angle_term = np.clip((float(elbow_angle) - 0.95) / 0.80, 0.0, 1.25)
                    axis_term = np.clip((0.72 - float(elbow_axis_ratio)) / 0.72, 0.0, 1.0)
                    elbow_bonus = -0.60 * float(angle_term * axis_term)
            cand["score"] = float(
                size_pen
                + circ_pen
                + cov_pen
                + planarity_pen
                + dist_bonus
                + ecc_bonus
                + source_bonus
                + anchor_bonus
                + growth_pen
                + curvature_bonus
                + elbow_bonus
            )

        # Select num_op loops from clustered candidates using score + diversity.
        selected = []
        selection_mode = "cluster_diverse"
        selection_candidates = list(deduped)
        tip_patch_geo_filter_used = False
        tip_patch_geo_filter_threshold = None
        tip_patch_total = [c for c in deduped if c.get("source") == "normal_tip_patch"]
        tip_patch_geo_vals = np.asarray(
            [float(c.get("anchor_geo_norm", 0.0)) for c in tip_patch_total],
            dtype=np.float64,
        )
        if tip_patch_geo_vals.size >= int(self.num_op + 2):
            sorted_geo = np.sort(tip_patch_geo_vals)[::-1]
            nth_geo = float(sorted_geo[int(self.num_op) - 1])
            # Keep the gate permissive to avoid dropping genuine shorter branches in hard cases.
            geo_gate = float(max(0.45, nth_geo - 0.30))
            filtered = []
            for cand in deduped:
                if cand.get("source") == "normal_tip_patch" and float(cand.get("anchor_geo_norm", 0.0)) < geo_gate:
                    continue
                filtered.append(cand)
            if len(filtered) >= int(self.num_op):
                selection_candidates = filtered
                selection_mode = "cluster_diverse_geo_filtered"
                tip_patch_geo_filter_used = True
                tip_patch_geo_filter_threshold = float(geo_gate)
        self.auto_registration_debug["tip_patch_geo_filter_used"] = bool(tip_patch_geo_filter_used)
        self.auto_registration_debug["tip_patch_geo_filter_threshold"] = (
            None if tip_patch_geo_filter_threshold is None else float(tip_patch_geo_filter_threshold)
        )
        self.auto_registration_debug["candidate_count_after_geo_filter"] = int(len(selection_candidates))

        debug_ranked_candidates = []
        for cand in selection_candidates:
            debug_ranked_candidates.append(
                {
                    "source": str(cand.get("source")),
                    "score": float(cand.get("score", np.inf)),
                    "size": int(np.asarray(cand.get("loop_idx", []), dtype=np.int64).size),
                    "tip_idx": None if cand.get("tip_idx", None) is None else int(cand.get("tip_idx")),
                    "anchor_geo_norm": float(cand.get("anchor_geo_norm", 0.0)),
                    "radius_growth": (
                        None
                        if cand.get("radius_growth", None) is None
                        else float(cand.get("radius_growth"))
                    ),
                    "tip_turn_p90": (
                        None
                        if cand.get("tip_turn_p90", None) is None
                        else float(cand.get("tip_turn_p90"))
                    ),
                    "tip_turn_mean": (
                        None
                        if cand.get("tip_turn_mean", None) is None
                        else float(cand.get("tip_turn_mean"))
                    ),
                    "elbow_angle": (
                        None
                        if cand.get("elbow_angle", None) is None
                        else float(cand.get("elbow_angle"))
                    ),
                    "elbow_axis_ratio": (
                        None
                        if cand.get("elbow_axis_ratio", None) is None
                        else float(cand.get("elbow_axis_ratio"))
                    ),
                    "center": np.asarray(cand.get("center", np.zeros(3)), dtype=np.float64).tolist(),
                }
            )
        debug_ranked_candidates = sorted(debug_ranked_candidates, key=lambda x: float(x["score"]))
        self.auto_registration_debug["candidate_ranking_preview"] = debug_ranked_candidates[: min(12, len(debug_ranked_candidates))]

        if len(selected) == 0:
            remaining = sorted(selection_candidates, key=lambda x: x["score"])
            if len(remaining) > 0:
                selected.append(remaining.pop(0))
            while len(selected) < self.num_op and len(remaining) > 0:
                best_idx, best_obj = None, np.inf
                for idx, cand in enumerate(remaining):
                    sep = min(np.linalg.norm(cand["center"] - s["center"]) for s in selected)
                    sep_gain = 0.06 * (sep / (median_edge + 1e-12))
                    obj = cand["score"] - sep_gain
                    if obj < best_obj:
                        best_obj = obj
                        best_idx = idx
                selected.append(remaining.pop(int(best_idx)))

            # Hard-case cleanup: replace likely aneurysm-apex picks with safer outlet candidates.
            if len(selected) == int(self.num_op):
                swap_used = False
                growth_bad_thr = 1.95
                growth_margin = 0.20
                score_relax = 0.75
                low_curv_guard_for_growth = 0.80
                for si in range(len(selected)):
                    cand_bad = selected[si]
                    if cand_bad.get("source") != "normal_tip_patch":
                        continue
                    rg_bad = cand_bad.get("radius_growth", None)
                    if rg_bad is None or not np.isfinite(float(rg_bad)) or float(rg_bad) < growth_bad_thr:
                        continue
                    curv_bad = cand_bad.get("tip_turn_p90", None)
                    if curv_bad is not None and np.isfinite(float(curv_bad)) and float(curv_bad) > low_curv_guard_for_growth:
                        # Preserve strongly curved transition regions; these are often true outlets.
                        continue
                    others = [selected[j] for j in range(len(selected)) if j != si]
                    best_rep_idx = None
                    best_rep_obj = np.inf
                    for ri, rep in enumerate(remaining):
                        if rep.get("source") != "normal_tip_patch":
                            continue
                        rg_rep = rep.get("radius_growth", None)
                        if rg_rep is None or not np.isfinite(float(rg_rep)):
                            continue
                        if float(rg_rep) > float(rg_bad) - growth_margin:
                            continue
                        if float(rep["score"]) > float(cand_bad["score"]) + score_relax:
                            continue
                        curv_rep = rep.get("tip_turn_p90", None)
                        if (
                            curv_bad is not None
                            and curv_rep is not None
                            and np.isfinite(float(curv_bad))
                            and np.isfinite(float(curv_rep))
                            and float(curv_rep) < float(curv_bad) - 0.12
                        ):
                            continue
                        if len(others) > 0:
                            sep = min(np.linalg.norm(rep["center"] - s["center"]) for s in others)
                            sep_gain = 0.06 * (sep / (median_edge + 1e-12))
                        else:
                            sep_gain = 0.0
                        rep_obj = float(rep["score"] - sep_gain)
                        if rep_obj < best_rep_obj:
                            best_rep_obj = rep_obj
                            best_rep_idx = int(ri)
                    if best_rep_idx is not None:
                        selected[si] = remaining.pop(best_rep_idx)
                        swap_used = True
                if swap_used:
                    selection_mode = f"{selection_mode}+swap_growth"

                # Additional fallback: promote outlet-like candidates with strong tip curvature
                # when all selected candidates are low-curvature in hard cases.
                curvature_swap_used = False
                high_curv_thr = 0.82
                low_curv_thr = 0.68
                growth_guard = 1.90
                curv_score_relax = 0.85
                candidate_pairs = []
                for si, cand_sel in enumerate(selected):
                    if cand_sel.get("source") != "normal_tip_patch":
                        continue
                    curv_sel = cand_sel.get("tip_turn_p90", None)
                    if curv_sel is None or not np.isfinite(float(curv_sel)) or float(curv_sel) > low_curv_thr:
                        continue
                    for ri, rep in enumerate(remaining):
                        if rep.get("source") != "normal_tip_patch":
                            continue
                        curv_rep = rep.get("tip_turn_p90", None)
                        if curv_rep is None or not np.isfinite(float(curv_rep)) or float(curv_rep) < high_curv_thr:
                            continue
                        rg_rep = rep.get("radius_growth", None)
                        if rg_rep is not None and np.isfinite(float(rg_rep)) and float(rg_rep) > growth_guard:
                            continue
                        if float(rep["score"]) > float(cand_sel["score"]) + curv_score_relax:
                            continue
                        curv_gain = float(curv_rep) - float(curv_sel)
                        if curv_gain <= 0.08:
                            continue
                        # Favor curvature gain but avoid sacrificing score too much.
                        gain_obj = curv_gain - 0.30 * max(0.0, float(rep["score"]) - float(cand_sel["score"]))
                        candidate_pairs.append((gain_obj, si, ri))
                if len(candidate_pairs) > 0:
                    candidate_pairs = sorted(candidate_pairs, key=lambda x: float(x[0]), reverse=True)
                    _, best_si, best_ri = candidate_pairs[0]
                    selected[int(best_si)] = remaining.pop(int(best_ri))
                    curvature_swap_used = True
                if curvature_swap_used:
                    selection_mode = f"{selection_mode}+swap_curvature_high"

                # Additional fallback: replace weak low-curvature tiny picks with
                # candidates that have clearly stronger anchor-geodesic support
                # and stronger distal curvature signature.
                anchor_curv_swap_used = False
                bad_anchor_thr = 0.74
                bad_curv_thr = 0.78
                bad_size_thr = max(int(min_loop_vertices) + 6, 30)
                anchor_gain_min = 0.14
                curv_gain_min = 0.12
                size_gain_min = 6
                anchor_curv_score_relax = 0.90
                for si, cand_bad in enumerate(selected):
                    if cand_bad.get("source") != "normal_tip_patch":
                        continue
                    anchor_bad = float(cand_bad.get("anchor_geo_norm", 0.0))
                    curv_bad = cand_bad.get("tip_turn_p90", None)
                    curv_bad_val = None
                    if curv_bad is not None and np.isfinite(float(curv_bad)):
                        curv_bad_val = float(curv_bad)
                    size_bad = int(np.asarray(cand_bad.get("loop_idx", []), dtype=np.int64).size)

                    suspicious = False
                    if anchor_bad < bad_anchor_thr and size_bad <= bad_size_thr:
                        suspicious = True
                    if (
                        curv_bad_val is not None
                        and anchor_bad < bad_anchor_thr
                        and curv_bad_val < bad_curv_thr
                    ):
                        suspicious = True
                    if not suspicious:
                        continue

                    best_rep_idx = None
                    best_rep_obj = np.inf
                    for ri, rep in enumerate(remaining):
                        if rep.get("source") != "normal_tip_patch":
                            continue
                        anchor_rep = float(rep.get("anchor_geo_norm", 0.0))
                        if anchor_rep < anchor_bad + anchor_gain_min:
                            continue
                        curv_rep = rep.get("tip_turn_p90", None)
                        if curv_rep is None or not np.isfinite(float(curv_rep)):
                            continue
                        curv_rep_val = float(curv_rep)
                        if curv_bad_val is None:
                            if curv_rep_val < bad_curv_thr:
                                continue
                        elif curv_rep_val < curv_bad_val + curv_gain_min:
                            continue
                        size_rep = int(np.asarray(rep.get("loop_idx", []), dtype=np.int64).size)
                        if size_rep < size_bad + size_gain_min:
                            continue
                        if float(rep["score"]) > float(cand_bad["score"]) + anchor_curv_score_relax:
                            continue
                        rep_obj = float(rep["score"]) - 0.18 * (anchor_rep - anchor_bad) - 0.22 * max(
                            0.0,
                            curv_rep_val - (curv_bad_val if curv_bad_val is not None else bad_curv_thr),
                        )
                        if rep_obj < best_rep_obj:
                            best_rep_obj = rep_obj
                            best_rep_idx = int(ri)

                    if best_rep_idx is not None:
                        selected[si] = remaining.pop(best_rep_idx)
                        anchor_curv_swap_used = True
                if anchor_curv_swap_used:
                    selection_mode = f"{selection_mode}+swap_anchor_curv"
        self.auto_registration_debug["selection_mode"] = selection_mode

        if len(selected) < self.num_op:
            # If not enough candidates were found, fallback to legacy method.
            self._auto_reg_pending_debug = {
                "fallback_from": "register_openings_auto_normals",
                "fallback_reason": "insufficient_selected_loops",
                "normals_debug_snapshot": dict(self.auto_registration_debug),
            }
            return self.register_openings_auto(min_loop_vertices=min_loop_vertices)

        # Derive centreline endpoints from opening centers.
        opening_centers = np.asarray([cand["center"] for cand in selected], dtype=np.float64)
        endpoint_indices = self._map_points_to_mesh_vertices_unique(opening_centers, k=16)
        if len(endpoint_indices) < self.num_op:
            self._auto_reg_pending_debug = {
                "fallback_from": "register_openings_auto_normals",
                "fallback_reason": "insufficient_endpoint_indices",
                "normals_debug_snapshot": dict(self.auto_registration_debug),
            }
            return self.register_openings_auto(min_loop_vertices=min_loop_vertices)
        endpoint_indices = [int(i) for i in endpoint_indices[: int(self.num_op)]]
        self.auto_registration_debug["opening_center_endpoint_indices_initial"] = endpoint_indices.copy()

        centreline_summary = None
        endpoint_repair_used = False
        endpoint_repair_trials = 0
        try:
            centreline_summary = self._centreline_from_endpoint_indices(
                endpoint_indices=endpoint_indices,
                sort_branches=True,
            )
        except RuntimeError:
            # Recover when one endpoint candidate collapses to a non-terminal/bifurcation region.
            candidate_lists = []
            per_center_limit = max(8, 3 * int(self.num_op))
            query_k = max(12, 5 * int(self.num_op))
            for center in opening_centers:
                try:
                    d, idx = self.mesh_target_trimesh.kdtree.query(center, k=query_k)
                    d = np.asarray(d).reshape(-1).astype(np.float64)
                    idx = np.asarray(idx).reshape(-1).astype(np.int64)
                except Exception:
                    sq = np.sum((verts - center.reshape(1, 3)) ** 2, axis=1)
                    idx = np.argsort(sq).astype(np.int64)
                    d = np.sqrt(np.maximum(sq[idx], 0.0))
                order = np.argsort(d)
                uniq = []
                seen = set()
                for j in order:
                    v = int(idx[int(j)])
                    if v in seen:
                        continue
                    uniq.append(v)
                    seen.add(v)
                    if len(uniq) >= per_center_limit:
                        break
                candidate_lists.append(uniq)
            for i, seed in enumerate(endpoint_indices):
                if i >= len(candidate_lists):
                    break
                if seed in candidate_lists[i]:
                    candidate_lists[i].remove(seed)
                candidate_lists[i].insert(0, int(seed))

            self.auto_registration_debug["opening_center_candidate_sizes"] = [len(c) for c in candidate_lists]
            best_summary = None
            best_endpoint_indices = None
            best_cost = np.inf
            max_trials = 9000
            for combo in itertools.product(*candidate_lists):
                endpoint_repair_trials += 1
                if endpoint_repair_trials > max_trials:
                    break
                if len(set(combo)) < int(self.num_op):
                    continue
                combo_indices = [int(v) for v in combo]
                try:
                    summary_try = self._centreline_from_endpoint_indices(
                        endpoint_indices=combo_indices,
                        sort_branches=True,
                    )
                except RuntimeError:
                    continue
                combo_pts = verts[np.asarray(combo_indices, dtype=np.int64)]
                cost = float(np.sum(np.linalg.norm(combo_pts - opening_centers, axis=1)))
                if cost < best_cost:
                    best_cost = cost
                    best_summary = summary_try
                    best_endpoint_indices = combo_indices
                    if cost <= 2.5 * float(self.num_op) * median_edge:
                        break

            if best_summary is not None:
                endpoint_repair_used = True
                endpoint_indices = [int(v) for v in best_endpoint_indices]
                centreline_summary = best_summary
            else:
                # Last resort: recover centreline from mesh endpoints to avoid hard failure.
                try:
                    centreline_summary = self.extract_centreline_from_mesh(
                        num_endpoints=self.num_op,
                        sort_branches=True,
                    )
                    self.auto_registration_debug["centreline_recovery"] = "mesh_fps_fallback"
                except Exception:
                    self._auto_reg_pending_debug = {
                        "fallback_from": "register_openings_auto_normals",
                        "fallback_reason": "centreline_path_recovery_failed",
                        "normals_debug_snapshot": dict(self.auto_registration_debug),
                    }
                    return self.register_openings_auto(min_loop_vertices=min_loop_vertices)
        self.auto_registration_debug["opening_center_endpoint_repair_used"] = bool(endpoint_repair_used)
        self.auto_registration_debug["opening_center_endpoint_repair_trials"] = int(endpoint_repair_trials)

        # Match selected openings to sorted centreline endpoints.
        endpoint_pts = np.asarray(centreline_summary["endpoints"], dtype=np.float64)
        dist_mat = np.linalg.norm(
            endpoint_pts[:, None, :] - opening_centers[None, :, :],
            axis=-1,
        )
        assign = self._assign_loops_to_branches(dist_mat)
        if assign is None:
            assign = list(range(min(self.num_op, len(selected))))

        paths = centreline_summary["paths"]
        tangents = centreline_summary["tangents"]
        synthetic_cut_refinement_used = []
        final_loop_sources = []
        loop_shape_score_before = []
        loop_shape_score_after = []

        def _loop_shape_objective(loop_idx: np.ndarray, normal: np.ndarray) -> float:
            loop_idx = np.unique(np.asarray(loop_idx, dtype=np.int64).reshape(-1))
            if loop_idx.size < 6:
                return np.inf
            coords = verts[loop_idx]
            centroid = np.mean(coords, axis=0)
            radial = np.linalg.norm(coords - centroid.reshape(1, 3), axis=1)
            radial_cv = float(np.std(radial) / (np.mean(radial) + 1e-6))
            coverage = self._loop_angular_coverage(coords, normal, bins=24)
            planarity = 0.0
            if coords.shape[0] >= 6:
                centered = coords - centroid.reshape(1, 3)
                try:
                    _, svals, _ = np.linalg.svd(centered, full_matrices=False)
                    if svals.shape[0] >= 3:
                        planarity = float(svals[-1] / (svals[0] + 1e-6))
                except np.linalg.LinAlgError:
                    planarity = 0.0
            size_pen = max(0.0, 18.0 - float(loop_idx.size)) * 0.020
            cov_pen = max(0.0, 0.82 - float(coverage)) * 3.5
            return float(2.2 * radial_cv + 1.5 * planarity + cov_pen + size_pen)

        for i in range(self.num_op):
            cand = selected[int(assign[i])]
            loop_idx = np.unique(np.asarray(cand["loop_idx"], dtype=np.int64))
            tangent = np.asarray(tangents[i], dtype=np.float64).reshape(3)
            n_mean = np.asarray(cand["normal"], dtype=np.float64).reshape(3)
            if np.linalg.norm(n_mean) < 1e-12:
                n_mean = tangent.copy()
            if np.linalg.norm(n_mean) < 1e-12:
                n_mean = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            n_mean = n_mean / (np.linalg.norm(n_mean) + 1e-12)
            if np.dot(n_mean, tangent) < 0:
                n_mean = -1.0 * n_mean

            # Optional synthetic-cut refinement: replace irregular loops only when shape quality improves.
            tip_idx = int(paths[i][-1])
            dist_tip, geo_min, geo_max, _ = self._opening_geo_limits(paths[i], tip_idx)
            geo_vals = dist_tip[loop_idx]
            finite_geo = geo_vals[np.isfinite(geo_vals)]
            desired_geo = (
                float(np.median(finite_geo))
                if finite_geo.size > 0
                else float(0.5 * (geo_min + geo_max))
            )
            desired_geo = float(np.clip(desired_geo, geo_min + 0.25 * (geo_max - geo_min), geo_max))

            base_loop_idx = loop_idx.copy()
            base_score = _loop_shape_objective(base_loop_idx, n_mean)
            best_loop_idx = base_loop_idx
            best_score = float(base_score)
            best_refine_tag = "none"

            refine_candidates = []
            section_loop = self._extract_section_loop(
                plane_origin=np.asarray(cand["center"], dtype=np.float64),
                plane_normal=tangent,
                tip_idx=tip_idx,
                geo_min=geo_min,
                geo_max=geo_max,
            )
            if section_loop is not None and section_loop.size >= 6:
                refine_candidates.append(("section", np.asarray(section_loop, dtype=np.int64)))
            sign_loop = self._extract_sign_loop(
                plane_origin=np.asarray(cand["center"], dtype=np.float64),
                plane_normal=tangent,
                tip_idx=tip_idx,
                min_vertices=max(8, int(min_loop_vertices) // 2),
                geo_min=geo_min,
                geo_max=geo_max,
            )
            if sign_loop is not None and sign_loop.size >= 6:
                refine_candidates.append(("sign", np.asarray(sign_loop, dtype=np.int64)))
            geo_loop = self._extract_geodesic_ring_loop(
                tip_idx=tip_idx,
                tangent=tangent,
                dist_tip=dist_tip,
                desired_geo=desired_geo,
                geo_min=geo_min,
                geo_max=geo_max,
                min_vertices=max(8, int(min_loop_vertices) // 2),
            )
            if geo_loop is not None and geo_loop.size >= 6:
                refine_candidates.append(("geodesic_ring", np.asarray(geo_loop, dtype=np.int64)))
            patch_loop = self._extract_normal_patch_loop(
                path=paths[i],
                tangent=tangent,
                min_loop_vertices=max(8, int(min_loop_vertices) // 2),
                normal_dot_min=float(max(0.62, normal_dot_min - 0.06)),
                face_dot_min=float(max(0.82, face_dot_min - 0.06)),
            )
            if patch_loop is not None and patch_loop.size >= 6:
                refine_candidates.append(("normal_patch", np.asarray(patch_loop, dtype=np.int64)))

            for tag, cand_loop in refine_candidates:
                cand_loop = np.unique(np.asarray(cand_loop, dtype=np.int64))
                cand_score = _loop_shape_objective(cand_loop, n_mean)
                if not np.isfinite(cand_score):
                    continue
                # Be more willing to refine tip-patch loops, conservative otherwise.
                min_improve = 0.03 if str(cand.get("source")) == "normal_tip_patch" else 0.10
                if cand_score + min_improve < best_score:
                    best_score = float(cand_score)
                    best_loop_idx = cand_loop
                    best_refine_tag = str(tag)

            loop_idx = best_loop_idx
            refined_used = bool(best_refine_tag != "none")
            final_source = str(cand.get("source", "unknown"))
            if refined_used:
                final_source = f"{final_source}+{best_refine_tag}"

            # Angle-regularized subsampling suppresses jagged star-like polygons on noisy loops.
            regularized_used = False
            if loop_idx.size >= max(16, int(min_loop_vertices)):
                reg_loop_idx = self._regularize_loop_indices_angular(
                    loop_idx=loop_idx,
                    normal=n_mean,
                    bins_min=max(20, int(min_loop_vertices)),
                    bins_max=max(64, 3 * int(min_loop_vertices)),
                    radius_quantile=0.60,
                )
                reg_score = _loop_shape_objective(reg_loop_idx, n_mean)
                if np.isfinite(reg_score):
                    # Keep regularization only when it is meaningfully smoother.
                    reg_improve_needed = 0.03 if str(cand.get("source")) == "normal_tip_patch" else 0.06
                    if reg_score + reg_improve_needed < best_score:
                        loop_idx = np.asarray(reg_loop_idx, dtype=np.int64)
                        best_score = float(reg_score)
                        regularized_used = True
                        final_source = f"{final_source}+angular_regularized"

            loop_idx = self._sort_indices_around_normal(loop_idx, n_mean)
            loop_coords = verts[loop_idx]
            tip = verts[tip_idx]
            if np.dot(n_mean, tip - np.mean(loop_coords, axis=0)) < 0:
                n_mean = -1.0 * n_mean

            self.op_v_indices.append(loop_idx.tolist())
            self.op_v_coords.append(loop_coords)
            if mesh_normals is not None:
                self.op_v_normal.append(mesh_normals[loop_idx])
            else:
                self.op_v_normal.append(np.repeat(n_mean.reshape(1, 3), loop_idx.size, axis=0))
            self.op_n_mean.append(n_mean)
            self.op_tangent.append(np.asarray(tangent, dtype=np.float64))
            self.op_cut_points.append(np.asarray(cand["center"], dtype=np.float64))
            synthetic_cut_refinement_used.append(bool(refined_used or regularized_used))
            final_loop_sources.append(str(final_source))
            loop_shape_score_before.append(float(base_score))
            loop_shape_score_after.append(float(best_score))

        if len(self.op_v_indices) != self.num_op:
            raise RuntimeError(
                f"Normal-based automatic opening registration produced {len(self.op_v_indices)} openings, expected {self.num_op}."
            )
        self.auto_registration_debug.update(
            {
                "selected_count": int(len(selected)),
                "selected_sources": [str(c["source"]) for c in selected],
                "selected_tip_indices": [
                    None if c.get("tip_idx", None) is None else int(c.get("tip_idx"))
                    for c in selected
                ],
                "selected_tip_turn_p90": [
                    None
                    if c.get("tip_turn_p90", None) is None
                    else float(c.get("tip_turn_p90"))
                    for c in selected
                ],
                "selected_elbow_angle": [
                    None
                    if c.get("elbow_angle", None) is None
                    else float(c.get("elbow_angle"))
                    for c in selected
                ],
                "selected_elbow_axis_ratio": [
                    None
                    if c.get("elbow_axis_ratio", None) is None
                    else float(c.get("elbow_axis_ratio"))
                    for c in selected
                ],
                "final_loop_sources": [str(s) for s in final_loop_sources],
                "synthetic_cut_refinement_used": [bool(v) for v in synthetic_cut_refinement_used],
                "loop_shape_score_before": [float(v) for v in loop_shape_score_before],
                "loop_shape_score_after": [float(v) for v in loop_shape_score_after],
                "opening_sizes": [len(v) for v in self.op_v_indices],
                "endpoint_indices": [int(i) for i in centreline_summary.get("endpoint_indices", [])],
            }
        )

        # If this class also tracks differentiable centreline state, keep it in sync.
        if hasattr(self, "num_cep") and int(getattr(self, "num_cep")) == len(centreline_summary["endpoint_indices"]):
            if hasattr(self, "cep_registration"):
                self.cep_registration = [int(i) for i in centreline_summary["endpoint_indices"]]
            if hasattr(self, "centreline_branch_paths"):
                self.centreline_branch_paths = [path.tolist() for path in centreline_summary["paths"]]
            if hasattr(self, "centreline_tangent"):
                self.centreline_tangent = np.asarray(centreline_summary["tangents"], dtype=np.float64)
            if hasattr(self, "centreline_pcd"):
                self.centreline_pcd = torch.tensor(
                    np.concatenate(centreline_summary["centreline_points"], axis=0),
                    dtype=torch.float32,
                )

        return centreline_summary

    def register_openings(self):
        """
        Select nodes on an opening.
        No need to register in a sequence, code will sort it clock-wise.
        No need to register all nodes on an opening, as long as the reconstructed mesh captures the majortiy of the area.
        """
        self._reset_opening_state()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(self.mesh_target.vertices))
        pcd.estimate_normals()
        for idx in range(self.num_op):
            print('Select nodes on opening No. {}'.format(idx))
            v_indices = u_register.pick_points(pcd)
            self.op_v_indices.append(v_indices)
            self.op_v_coords.append(np.asarray(pcd.points)[v_indices])
            self.op_v_normal.append(np.asarray(pcd.normals)[v_indices])
            self.op_n_mean.append(u_register.get_average_cross_product_from_center(np.asarray(pcd.points)[v_indices]))

    def create_opening_meshes(self, viz=False):
        """
        Create opening meshes from registered nodes.
        """
        self.op_rec_v, self.op_rec_f = [], []
        self.op_rec_v_indices_map, self.op_rec_f_map = [], []
        for idx in range(self.num_op):
            points_projected = u_register.pcd_to_approx_plane(self.op_v_coords[idx], self.op_n_mean[idx])
            point_sequence = np.concatenate((np.arange(points_projected.shape[0]), np.array([0])))
            points_projected = np.asarray( list(np.concatenate((points_projected, points_projected[0, :].reshape(-1, 3)))))
            path_topo = trimesh.path.entities.Entity(point_sequence)
            path3D = trimesh.path.path.Path3D([path_topo], vertices=points_projected)
            path2D, to_3D = path3D.to_planar()
            sorted_2D_vertices = u_register.clock_sort_2D_points(np.asarray(path2D.vertices))
            # check points sequence is clock wise
            # warning: do not close the polygon which leads to error
            polygon = shapely.geometry.Polygon(sorted_2D_vertices)
            vertices, faces = trimesh.creation.triangulate_polygon(polygon, triangle_args='p')
            # Normalize triangulation outputs across different engines/versions.
            vertices = np.asarray(vertices)
            faces = np.asarray(faces)
            if faces.size > 0 and np.min(faces) == 1:
                # Older triangulation paths can emit 1-based faces.
                faces = faces - 1
            # Drop duplicated closing vertex if present.
            if vertices.shape[0] > 1 and np.allclose(vertices[0], vertices[-1]):
                vertices = vertices[:-1, :]
            if points_projected.shape[0] > 1 and np.allclose(points_projected[0], points_projected[-1]):
                points_projected = points_projected[:-1, :]
            vertices_3d = u_register.trimesh_points_2d_to_3d(vertices, to_3D)
            # return point sequence in original system
            op_rec_v_indices_map, op_rec_f_map = u_register.get_mapped_sequence_and_faces(self.op_v_indices[idx], points_projected, vertices_3d, faces)
            # trimesh's 2d to 3d is done by sequence, however when feeding vertices to construct path3D or path2D,
            if viz:
                u_register.lazy_viz_mesh(vertices_3d, faces)
            self.op_rec_v.append(vertices_3d)
            self.op_rec_f.append(faces)
            self.op_rec_v_indices_map.append(op_rec_v_indices_map)
            self.op_rec_f_map.append(op_rec_f_map)

    def save_checkpoint_opa(self, chk_path: str):
        # automatic saving
        chk_path = os.path.join(self.root, self.target, 'opa_checkpoint') if chk_path is None else chk_path
        if not os.path.exists(os.path.dirname(chk_path)):
            os.makedirs(os.path.dirname(chk_path))
        chk = {'op_v_indices': self.op_v_indices, 'op_v_coords': self.op_v_coords, 'op_v_normal': self.op_v_normal,
               'op_n_mean': self.op_n_mean,
               'op_rec_v': self.op_rec_v, 'op_rec_f': self.op_rec_f,
               'op_rec_v_indices_map': self.op_rec_v_indices_map, 'op_rec_f_map': self.op_rec_f_map,
               'op_tangent': getattr(self, "op_tangent", []),
               'op_cut_points': getattr(self, "op_cut_points", []),
               'auto_registration_debug': getattr(self, "auto_registration_debug", {})}
        for key in [
            "op_target_rim_v",
            "op_target_rec_v",
            "op_target_rec_f",
            "op_target_plane_center",
            "op_target_plane_normal",
            "op_source_kind",
            "op_source_surface_v",
            "op_source_surface_f",
            "op_target_debug",
        ]:
            if hasattr(self, key):
                chk[key] = getattr(self, key)
        # use self.op_rec_f to offset opening meshes, use self.op_rec_f_map if creating opening meshes from mother mesh
        if not chk_path.endswith('.pkl'):
            chk_path += '.pkl'
        with open(chk_path, 'wb') as f:
            pickle.dump(chk, f)

    def load_checkpoint_opa(
        self,
        chk_path: str,
        redo=False,
        auto=True,
        auto_method: str = "legacy",
        auto_kwargs: Dict[str, object] = None,
    ):
        # automatic loading
        chk_path = os.path.join(self.root, self.target, 'opa_checkpoint') if chk_path is None else \
            chk_path
        if not chk_path.endswith('.pkl'):
            chk_path += '.pkl'
        # register openings, create meshes and save chk if chk doesn't exist
        if not os.path.exists(chk_path) or redo:
            logging.warning('checkpoint does not exist, redo registration.')
            self._reset_opening_state()
            if auto:
                method = str(auto_method).strip().lower()
                kwargs = dict(auto_kwargs or {})
                if method in ("legacy", "auto", "default"):
                    self.register_openings_auto(
                        min_loop_vertices=int(kwargs.get("min_loop_vertices", 24))
                    )
                elif method in ("normals", "auto_normals", "normal"):
                    self.register_openings_auto_normals(
                        min_loop_vertices=int(kwargs.get("min_loop_vertices", 24)),
                        normal_dot_min=float(kwargs.get("normal_dot_min", 0.72)),
                        face_dot_min=float(kwargs.get("face_dot_min", 0.90)),
                    )
                else:
                    raise ValueError(
                        f"Unknown auto_method='{auto_method}'. "
                        "Use one of: legacy, normals."
                    )
            else:
                self.register_openings()
            self.create_opening_meshes()
            self.save_checkpoint_opa(chk_path)
        with open(chk_path, 'rb') as f:
            chk = pickle.load(f)
        for key in chk.keys():
            setattr(self, key, chk[key])
        logging.info('checkpoint has been loaded {}'.format(chk.keys()))
        self.log_register = 'Yes'
        self.log_reconstruct = 'Yes'
        return None

    def return_opening_Meshes_static(self, register_normal=True) -> list:
        # return opening meshes for non-canonical shapes (static)
        opening_Meshes = []
        target_rec_v = getattr(self, "op_target_rec_v", [])
        target_rec_f = getattr(self, "op_target_rec_f", [])
        target_plane_normal = getattr(self, "op_target_plane_normal", [])
        for idx in range(self.num_op):
            verts_np = None
            faces_np = None
            if idx < len(target_rec_v) and idx < len(target_rec_f):
                vv = np.asarray(target_rec_v[idx], dtype=np.float64)
                ff = np.asarray(target_rec_f[idx], dtype=np.int64)
                if vv.ndim == 2 and vv.shape[0] >= 3 and ff.ndim == 2 and ff.shape[0] >= 1:
                    verts_np = vv
                    faces_np = ff
            if verts_np is None:
                verts_np = np.asarray(self.op_rec_v[idx], dtype=np.float64)
                faces_np = np.asarray(self.op_rec_f[idx], dtype=np.int64)
            verts = torch.tensor(verts_np).unsqueeze(0).float()
            faces = torch.tensor(faces_np).unsqueeze(0).long()
            normal_ref = None
            if idx < len(target_plane_normal):
                normal_ref = np.asarray(target_plane_normal[idx], dtype=np.float64).reshape(-1)
            if normal_ref is None or normal_ref.size != 3:
                normal_ref = np.asarray(self.op_n_mean[idx], dtype=np.float64).reshape(-1)
            normals = torch.tensor(np.repeat(normal_ref.reshape(-1, 3), verts.shape[1], axis=0)).unsqueeze(0).float()
            opening_Meshes.append(Meshes(verts=verts, faces=faces, verts_normals=normals if register_normal else None))
        return opening_Meshes

    def return_opening_rim_pointclouds_static(self, prefer_source=True) -> list:
        rims = []
        target_rims = getattr(self, "op_target_rim_v", [])
        for idx in range(self.num_op):
            rim_np = None
            if prefer_source and idx < len(target_rims):
                vv = np.asarray(target_rims[idx], dtype=np.float64)
                if vv.ndim == 2 and vv.shape[0] >= 3 and vv.shape[1] == 3:
                    rim_np = vv
            if rim_np is None and idx < len(self.op_v_coords):
                vv = np.asarray(self.op_v_coords[idx], dtype=np.float64)
                if vv.ndim == 2 and vv.shape[0] >= 3 and vv.shape[1] == 3:
                    rim_np = vv
            if rim_np is None:
                rim_np = np.asarray(self.op_rec_v[idx], dtype=np.float64)
            rims.append(torch.tensor(rim_np).unsqueeze(0).float())
        return rims

    def return_opening_planes_static(self, prefer_source=True) -> list:
        planes = []
        target_centers = getattr(self, "op_target_plane_center", [])
        target_normals = getattr(self, "op_target_plane_normal", [])
        for idx in range(self.num_op):
            center = None
            normal = None
            if prefer_source and idx < len(target_centers) and idx < len(target_normals):
                cc = np.asarray(target_centers[idx], dtype=np.float64).reshape(-1)
                nn = np.asarray(target_normals[idx], dtype=np.float64).reshape(-1)
                if cc.size == 3 and nn.size == 3:
                    center = torch.tensor(cc).float()
                    normal = torch.tensor(nn).float()
            if center is None or normal is None:
                coords = np.asarray(self.op_v_coords[idx], dtype=np.float64)
                center = torch.tensor(coords.mean(axis=0)).float()
                normal = torch.tensor(np.asarray(self.op_n_mean[idx], dtype=np.float64)).float()
            planes.append((center, normal))
        return planes

    def class_normalize(self, norm=10.0):
        # normalize mesh to have max radius of norm
        # this is done by scaling the vertices
        self.mesh_target.vertices = o3d.utility.Vector3dVector(np.asarray(self.mesh_target.vertices) / norm)
        if isinstance(self.mesh_target_trimesh, trimesh.Trimesh):
            self.mesh_target_trimesh.vertices = np.asarray(self.mesh_target_trimesh.vertices) / norm
        self.mesh_target_p3d = o3d_mesh_to_pytorch3d(self.mesh_target)
        for i in range(len(self.op_v_coords)):
            self.op_v_coords[i] /= norm
            self.op_rec_v[i] /= norm
        if hasattr(self, "op_target_rim_v") and isinstance(self.op_target_rim_v, list):
            for i in range(len(self.op_target_rim_v)):
                self.op_target_rim_v[i] = np.asarray(self.op_target_rim_v[i]) / norm
        if hasattr(self, "op_target_rec_v") and isinstance(self.op_target_rec_v, list):
            for i in range(len(self.op_target_rec_v)):
                self.op_target_rec_v[i] = np.asarray(self.op_target_rec_v[i]) / norm
        if hasattr(self, "op_source_surface_v") and isinstance(self.op_source_surface_v, list):
            for i in range(len(self.op_source_surface_v)):
                self.op_source_surface_v[i] = np.asarray(self.op_source_surface_v[i]) / norm
        if hasattr(self, "op_cut_points") and isinstance(self.op_cut_points, list):
            for i in range(len(self.op_cut_points)):
                self.op_cut_points[i] = np.asarray(self.op_cut_points[i]) / norm
        if hasattr(self, "op_target_plane_center") and isinstance(self.op_target_plane_center, list):
            for i in range(len(self.op_target_plane_center)):
                self.op_target_plane_center[i] = np.asarray(self.op_target_plane_center[i]) / norm
        if hasattr(self, "centreline_pcd") and self.centreline_pcd is not None:
            if torch.is_tensor(self.centreline_pcd):
                self.centreline_pcd = self.centreline_pcd / float(norm)
            else:
                self.centreline_pcd = np.asarray(self.centreline_pcd) / float(norm)
        self._mesh_graph_cache = None
        return None

    @staticmethod
    def _normalize_np_vec(vec):
        vec = np.asarray(vec, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm <= 1e-12:
            return None
        return vec / norm

    def _compute_opening_cap_normal(self, idx):
        if idx >= len(self.op_rec_v) or idx >= len(self.op_rec_f):
            return None
        verts = np.asarray(self.op_rec_v[idx], dtype=np.float64)
        faces = np.asarray(self.op_rec_f[idx], dtype=np.int64)
        if verts.ndim != 2 or verts.shape[0] < 3 or faces.ndim != 2 or faces.shape[0] == 0:
            return None
        tri = verts[faces]
        face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        face_normals = face_normals[np.linalg.norm(face_normals, axis=1) > 1e-12]
        if face_normals.shape[0] == 0:
            return None
        return self._normalize_np_vec(face_normals.mean(axis=0))

    def _opening_reference_normal(self, idx, clean_threshold=0.2):
        mesh_verts = np.asarray(self.mesh_target.vertices, dtype=np.float64)
        mesh_center = mesh_verts.mean(axis=0) if mesh_verts.shape[0] > 0 else np.zeros(3, dtype=np.float64)

        opening_coords = None
        if idx < len(self.op_v_coords):
            opening_coords = np.asarray(self.op_v_coords[idx], dtype=np.float64)
        elif idx < len(self.op_rec_v):
            opening_coords = np.asarray(self.op_rec_v[idx], dtype=np.float64)
        if opening_coords is None or opening_coords.ndim != 2 or opening_coords.shape[0] == 0:
            return None

        centroid = opening_coords.mean(axis=0)
        radial = self._normalize_np_vec(centroid - mesh_center)

        if len(self.mesh_target.vertex_normals) == 0:
            self.mesh_target.compute_vertex_normals()
        mesh_normals = np.asarray(self.mesh_target.vertex_normals, dtype=np.float64)
        surface_normal = None
        if idx < len(self.op_v_indices) and mesh_normals.shape[0] > 0:
            indices = np.asarray(self.op_v_indices[idx], dtype=np.int64).reshape(-1)
            valid = indices[(indices >= 0) & (indices < mesh_normals.shape[0])]
            if valid.size > 0:
                surface_normal = self._normalize_np_vec(mesh_normals[valid].mean(axis=0))

        if radial is None:
            return surface_normal
        if surface_normal is None:
            return radial

        if float(np.dot(surface_normal, radial)) < -abs(float(clean_threshold)):
            return radial
        return radial if abs(float(np.dot(surface_normal, radial))) < abs(float(clean_threshold)) else surface_normal

    def _flip_opening_orientation(self, idx):
        if idx < len(self.op_v_normal):
            self.op_v_normal[idx] = -1.0 * np.asarray(self.op_v_normal[idx], dtype=np.float64)
        if idx < len(self.op_n_mean):
            self.op_n_mean[idx] = -1.0 * np.asarray(self.op_n_mean[idx], dtype=np.float64)
        if idx < len(self.op_rec_f):
            faces = np.asarray(self.op_rec_f[idx], dtype=np.int64)
            if faces.ndim == 2 and faces.shape[1] == 3:
                self.op_rec_f[idx] = faces[:, [0, 2, 1]]
        if idx < len(self.op_rec_f_map):
            faces_map = np.asarray(self.op_rec_f_map[idx], dtype=np.int64)
            if faces_map.ndim == 2 and faces_map.shape[1] == 3:
                self.op_rec_f_map[idx] = faces_map[:, [0, 2, 1]]

    def _rebuild_centreline_pcd_from_wave_loops(self):
        if not hasattr(self, "wave_loops") or self.wave_loops is None:
            return
        verts = np.asarray(self.mesh_target.vertices, dtype=np.float64)
        points = []
        for branch in self.wave_loops:
            for loop in branch:
                loop_idx = np.asarray(loop, dtype=np.int64).reshape(-1)
                valid = loop_idx[(loop_idx >= 0) & (loop_idx < verts.shape[0])]
                if valid.size == 0:
                    continue
                points.append(verts[valid].mean(axis=0))
        if not points:
            return
        centreline_np = np.asarray(points, dtype=np.float32)
        if hasattr(self, "centreline_pcd") and torch.is_tensor(self.centreline_pcd):
            self.centreline_pcd = torch.from_numpy(centreline_np)
        else:
            self.centreline_pcd = centreline_np

    def centreline_clean(self, radius=0.0):
        if not hasattr(self, "wave_loops") or self.wave_loops is None:
            return None
        verts = np.asarray(self.mesh_target.vertices, dtype=np.float64)
        if verts.shape[0] == 0:
            return None

        clean_radius = float(radius)
        if clean_radius <= 0.0:
            clean_radius = 0.5 * float(self._get_mesh_graph_cache()["median_edge_length"])
        cleaned_wave_loops = []
        for branch in self.wave_loops:
            cleaned_branch = []
            previous_centroid = None
            for loop in branch:
                loop_idx = np.asarray(loop, dtype=np.int64).reshape(-1)
                loop_idx = loop_idx[(loop_idx >= 0) & (loop_idx < verts.shape[0])]
                loop_idx = np.unique(loop_idx)
                if loop_idx.size == 0:
                    continue
                centroid = verts[loop_idx].mean(axis=0)
                if previous_centroid is None or np.linalg.norm(centroid - previous_centroid) >= clean_radius:
                    cleaned_branch.append(loop_idx.tolist())
                    previous_centroid = centroid
                    continue
                if len(cleaned_branch[-1]) < loop_idx.size:
                    cleaned_branch[-1] = loop_idx.tolist()
                    previous_centroid = centroid
            if not cleaned_branch:
                for loop in branch:
                    loop_idx = np.asarray(loop, dtype=np.int64).reshape(-1)
                    loop_idx = loop_idx[(loop_idx >= 0) & (loop_idx < verts.shape[0])]
                    loop_idx = np.unique(loop_idx)
                    if loop_idx.size > 0:
                        cleaned_branch.append(loop_idx.tolist())
                        break
            cleaned_wave_loops.append(cleaned_branch)
        self.wave_loops = cleaned_wave_loops
        self._rebuild_centreline_pcd_from_wave_loops()
        return None

    def visualize_centreline(self, norm_target):
        # visualize centreline
        # this is not implemented yet
        return None
    
    def sort_opening_normals(self, inspect_true_normal=False, clean_threshold=0.2, bold=False):
        if len(self.op_n_mean) == 0:
            return None
        clean_threshold = float(clean_threshold)
        for idx in range(len(self.op_n_mean)):
            reference_normal = self._opening_reference_normal(idx, clean_threshold=clean_threshold)
            current_normal = self._normalize_np_vec(self.op_n_mean[idx])
            cap_normal = self._compute_opening_cap_normal(idx)

            if reference_normal is None:
                reference_normal = cap_normal
            if reference_normal is None:
                reference_normal = current_normal
            if reference_normal is None:
                continue

            need_flip = False
            if current_normal is not None and float(np.dot(current_normal, reference_normal)) < 0.0:
                need_flip = True
            if not need_flip and cap_normal is not None and float(np.dot(cap_normal, reference_normal)) < 0.0:
                need_flip = True
            if bold and current_normal is not None and float(np.dot(current_normal, reference_normal)) < clean_threshold:
                need_flip = True
            if need_flip:
                self._flip_opening_orientation(idx)
                current_normal = self._normalize_np_vec(self.op_n_mean[idx])

            if current_normal is None:
                self.op_n_mean[idx] = np.asarray(reference_normal, dtype=np.float64)
            if inspect_true_normal:
                print(
                    f"[sort_opening_normals] opening={idx} "
                    f"dot={float(np.dot(self._normalize_np_vec(self.op_n_mean[idx]), reference_normal)):.4f}"
                )
        return None


class RegistrationwOpeningAlignmentwCentreline(RegistrationwOpeningAlignment):
    def __init__(self, args, root, target, num_op=3, suffix=None, num_cep=4, c_suffix="Centerline model.vtk"):
        self.num_cep = num_cep  # number of centreline end points
        self.c_suffix = c_suffix  # suffix of true complex mesh object (only support obj for now)
        self.centreline_pcd = None  # pcd of cecntreline object
        self.cep_registration = None  # registration of centreline end points
        super(RegistrationwOpeningAlignmentwCentreline, self).__init__(args, root, target, num_op, suffix)

    def register_centreline(self):
        """
        Need to pick cep points in a sequence.
        """
        centreline_file = os.path.join(self.root, self.target, self.c_suffix)
        centreline_pcd = load_point_cloud_vtk(centreline_file).cpu()
        self.centreline_pcd = centreline_pcd
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(centreline_pcd.cpu()))
        for idx in range(self.num_cep):
            print("Pick centreline end points for each parent vessel branch.")
            self.cep_registration = torch.Tensor(u_register.pick_points(pcd))

    def save_checkpoint_centreline(self, chk_path: str):
        chk_path = os.path.join(self.root, self.target, 'centreline_checkpoint') if chk_path is None else chk_path
        if not os.path.exists(os.path.dirname(chk_path)):
            os.makedirs(os.path.dirname(chk_path))
        chk = {'centreline_pcd': self.centreline_pcd,
               'cep_registration': self.cep_registration}
        if not chk_path.endswith('.pkl'):
            chk_path += '.pkl'
        with open(chk_path, 'wb') as f:
            pickle.dump(chk, f)

    def load_checkpoint_centreline(self, chk_path: str, redo=False):
        chk_path = os.path.join(self.root, self.target, 'centreline_checkpoint') if chk_path is None else chk_path
        if not chk_path.endswith('.pkl'):
            chk_path += '.pkl'
        if not os.path.exists(chk_path) or redo:
            print('centreline checkpoints does not exist, redo registration.')
            self.register_centreline()
            self.save_checkpoint_centreline(None)
        with open(chk_path, 'rb') as f:
            chk = pickle.load(f)
        for key in chk.keys():
            setattr(self, key, chk[key])
        logging.info('checkpoint of centreline has been loaded {}'.format(chk.keys()))


class RegistrationwOpeningAlignmentwDifferentiableCentreline(RegistrationwOpeningAlignment):
    def __init__(self, args, root, target, num_op=3, num_cep=3, num_waves=5, step_size=2):
        self.num_cep = num_cep  # number of centreline end points
        self.centreline_pcd = None  # pcd of cecntreline object
        self.cep_registration = None  # registration of centreline end points indices
        self.centreline_branch_paths = None
        self.centreline_tangent = None
        self.num_waves = num_waves  # number of waves to cast for each cep point
        self.step_size = step_size  
        self.wave_loops = None  # registration of loops (List[List[]])
        super(RegistrationwOpeningAlignmentwDifferentiableCentreline, self).__init__(args, root, target, num_op, suffix=None)

    def _normalize_cep_registration(self):
        if self.cep_registration is None:
            return
        if isinstance(self.cep_registration, torch.Tensor):
            values = self.cep_registration.detach().cpu().numpy().reshape(-1)
        else:
            values = np.asarray(self.cep_registration).reshape(-1)
        self.cep_registration = [int(v) for v in values]

    def register_centreline_end_points(self, auto=False):
        """
        No need to pick cep points in a sequence.
        """
        if auto:
            summary = None
            cached = getattr(self, "auto_centreline_summary", None)
            if isinstance(cached, dict):
                cached_endpoints = cached.get("endpoint_indices", [])
                if len(cached_endpoints) == int(self.num_cep):
                    summary = cached

            if summary is None:
                op_cut_points = getattr(self, "op_cut_points", None)
                if isinstance(op_cut_points, list) and len(op_cut_points) >= int(self.num_cep):
                    try:
                        centers = np.asarray(op_cut_points[: int(self.num_cep)], dtype=np.float64)
                        endpoint_indices = self._map_points_to_mesh_vertices_unique(centers, k=16)
                        if len(endpoint_indices) == int(self.num_cep):
                            summary = self._centreline_from_endpoint_indices(
                                endpoint_indices=endpoint_indices,
                                sort_branches=True,
                            )
                    except Exception:
                        summary = None

            if summary is None:
                summary = self.extract_centreline_from_mesh(num_endpoints=self.num_cep, sort_branches=True)

            self.cep_registration = [int(i) for i in summary["endpoint_indices"]]
            self.centreline_branch_paths = [path.tolist() for path in summary["paths"]]
            self.centreline_tangent = np.asarray(summary["tangents"], dtype=np.float64)
            self.centreline_pcd = torch.tensor(
                np.concatenate(summary["centreline_points"], axis=0),
                dtype=torch.float32,
            )
            self._normalize_cep_registration()
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(self.mesh_target.vertices))
        pcd.estimate_normals()
        self.cep_registration = u_register.pick_points(pcd)
        self._normalize_cep_registration()

    def save_checkpoint_centreline(self, chk_path: str):
        chk_path = os.path.join(self.root, self.target, 'diff_centreline_checkpoint') if chk_path is None else chk_path
        if not os.path.exists(os.path.dirname(chk_path)):
            os.makedirs(os.path.dirname(chk_path))
        chk = {'diff_cep_registration': self.cep_registration,
               'wave_loops': self.wave_loops}
        if self.centreline_pcd is not None:
            chk['centreline_pcd'] = self.centreline_pcd
        if self.centreline_branch_paths is not None:
            chk['centreline_branch_paths'] = self.centreline_branch_paths
        if self.centreline_tangent is not None:
            chk['centreline_tangent'] = self.centreline_tangent
        if not chk_path.endswith('.pkl'):
            chk_path += '.pkl'
        with open(chk_path, 'wb') as f:
            pickle.dump(chk, f)

    def load_checkpoint_centreline(self, chk_path: str, redo=False, auto=True):
        chk_path = os.path.join(self.root, self.target, 'diff_centreline_checkpoint') if chk_path is None else chk_path
        if not chk_path.endswith('.pkl'):
            chk_path += '.pkl'
        if not os.path.exists(chk_path) or redo:
            logging.warning('checkpoint does not exist, redo registration.')
            if auto:
                self.register_centreline_end_points(auto=True)
            else:
                self.register_centreline_end_points(auto=False)
            self._cast_waves()
            self.save_checkpoint_centreline(chk_path)
        with open(chk_path, 'rb') as f:
            chk = pickle.load(f)
        for key in chk.keys():
            setattr(self, key, chk[key])
        self._normalize_cep_registration()
        print('Differentiable centreline checkpoint has been loaded {}'.format(chk.keys()))

    def _cast_waves(self, random_origin=False, progress=True):
        """
        Cast waves across mesh, refer to:
        https://github.com/navis-org/skeletor.git
        """
        self._normalize_cep_registration()
        if not random_origin:
            origins = self.cep_registration
        else:
            origins = None
        if not isinstance(origins, type(None)):
            if isinstance(origins, int):
                origins = [origins]
            elif not isinstance(origins, (set, list)):
                raise TypeError('`origins` must be vertex ID (int) or list '
                                f'thereof, got "{type(origins)}"')
            origins = np.asarray(origins).astype(int)
        else:
            origins = np.array([])
        # Wave must be a positive integer >= 1
        waves = int(len(origins)) if origins is not None else self.num_waves
        if waves < 1:
            raise ValueError('`waves` must be integer >= 1')
        # Same for step size
        step_size = int(self.step_size)
        if step_size < 1:
            raise ValueError('`step_size` must be integer >= 1')
        mesh = make_trimesh(self.mesh_target_trimesh, validate=False)
        G = ig.Graph(edges=mesh.edges_unique, directed=False)
        # Prepare empty array to fill with centers
        centers = np.full((mesh.vertices.shape[0], 3, waves), fill_value=np.nan)
        radii = np.full((mesh.vertices.shape[0], waves), fill_value=np.nan)
        # Go over each connected component
        with tqdm(desc='Skeletonizing', total=len(G.vs), disable=not progress) as pbar:
            for cc in G.clusters():
                # Make a subgraph for this connected component
                SG = G.subgraph(cc)
                cc = np.array(cc)
                # Select seeds according to the number of waves
                n_waves = min(waves, len(cc))
                pot_seeds = np.arange(len(cc))
                np.random.seed(1985)  # make seeds predictable
                # See if we can use any origins
                if len(origins):
                    # Get those origins in this cc
                    in_cc = np.isin(origins, cc)
                    if any(in_cc):
                        # Map origins into cc
                        cc_map = dict(zip(cc, np.arange(0, len(cc))))
                        seeds = np.array([cc_map[o] for o in origins[in_cc]])
                    else:
                        seeds = np.array([])
                    if len(seeds) < n_waves:
                        remaining_seeds = pot_seeds[~np.isin(pot_seeds, seeds)]
                        seeds = np.append(seeds,
                                          np.random.choice(remaining_seeds,
                                                           size=n_waves - len(seeds),
                                                           replace=False))
                else:
                    seeds = np.random.choice(pot_seeds, size=n_waves, replace=False)
                seeds = seeds.astype(int)
                # Get the distance between the seeds and all other nodes
                dist = np.array(SG.shortest_paths(source=seeds, target=None, mode='all'))
                if step_size > 1:
                    mx = dist.flatten()
                    mx = mx[mx < float('inf')].max()
                    dist = np.digitize(dist, bins=np.arange(0, mx, step_size))
                loops_list = []
                # Cast the desired number of waves
                for w in range(dist.shape[0]):
                    loop_list = []
                    this_wave = dist[w, :]
                    # Collect groups
                    mx = this_wave[this_wave < float('inf')].max()
                    for i in range(0, int(mx) + 1):
                        this_dist = this_wave == i
                        ix = np.where(this_dist)[0]
                        SG2 = SG.subgraph(ix)
                        for cc2 in SG2.clusters():
                            this_verts = cc[ix[cc2]]
                            loop_list.append(this_verts)
                    loops_list.append(loop_list)
                pbar.update(len(cc))
        self.wave_loops = loops_list
        return None


def load_point_cloud_vtk(vtk_file_path):
    # load pcd from vtk file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    point_cloud = reader.GetOutput()
    points = point_cloud.GetPoints()
    num_points = points.GetNumberOfPoints()
    point_cloud_array = np.zeros((num_points, 3))  # Initialize array
    for i in range(num_points):
        point = points.GetPoint(i)
        point_cloud_array[i] = point
    point_cloud_array = torch.Tensor(point_cloud_array)
    return point_cloud_array

def p3d_to_pv(Meshes: Meshes):
    verts = Meshes.verts_packed()
    faces = Meshes.faces_packed()
    verts = verts.detach().cpu().numpy()
    faces = faces.detach().cpu().numpy()
    poly_data = pv.PolyData(verts)
    poly_data.faces = faces
    return poly_data
