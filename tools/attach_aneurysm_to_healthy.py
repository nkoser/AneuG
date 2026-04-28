#!/usr/bin/env python
"""Prototype: cut a jagged ostium hole into a healthy vessel and stitch an aneurysm mesh to it."""
import argparse
import io
import json
import os
import pickle
from collections import defaultdict, deque

import numpy as np
import torch
import trimesh


class _TorchCPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def _load_clean_mesh(path, merge=False):
    mesh = trimesh.load(path, process=False)
    if merge:
        mesh.merge_vertices(digits_vertex=8, merge_tex=True, merge_norm=True)
        if hasattr(mesh, "unique_faces"):
            mesh.update_faces(mesh.unique_faces())
        if hasattr(mesh, "nondegenerate_faces"):
            mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
    return mesh


def _normalize(v):
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return v / (np.linalg.norm(v) + 1e-12)


def _to_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _axis_angle_to_matrix(axis_angle):
    aa = _to_numpy(axis_angle).reshape(3).astype(np.float64)
    angle = float(np.linalg.norm(aa))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = aa / angle
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.array([
        [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
        [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
        [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
    ], dtype=np.float64)


def _load_pickle(path):
    with open(path, "rb") as f:
        return _TorchCPUUnpickler(f).load()


def _apply_h(points, transform):
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def _apply_h_inverse(points, transform):
    points = np.asarray(points, dtype=np.float64)
    inv = np.linalg.inv(transform)
    return _apply_h(points, inv)


def _canonical_norm(canonical_mesh, factor):
    mesh = trimesh.load(canonical_mesh, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return float(np.linalg.norm(np.asarray(mesh.vertices), axis=1).max() * factor)


def _plane_basis(normal):
    normal = _normalize(normal)
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = _normalize(np.cross(normal, helper))
    v = _normalize(np.cross(normal, u))
    return u, v


def _project(points, center, normal):
    u, v = _plane_basis(normal)
    rel = np.asarray(points, dtype=np.float64) - center.reshape(1, 3)
    return rel @ u, rel @ v, rel @ normal.reshape(3)


def _jagged_radius(theta, base_radius, amplitude, harmonics, seed):
    if amplitude <= 0.0:
        return np.full_like(theta, base_radius, dtype=np.float64)
    rng = np.random.default_rng(seed)
    signal = np.zeros_like(theta, dtype=np.float64)
    for k in range(2, int(harmonics) + 2):
        phase = rng.uniform(0.0, 2.0 * np.pi)
        weight = rng.uniform(0.35, 1.0) / k
        signal += weight * np.sin(k * theta + phase)
    signal /= np.max(np.abs(signal)) + 1e-12
    radius = base_radius * (1.0 + amplitude * signal)
    return np.maximum(radius, base_radius * 0.35)


def _boundary_edges(faces):
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ])
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique[counts == 1]


def _boundary_components(edges):
    adj = defaultdict(list)
    for a, b in edges:
        a = int(a)
        b = int(b)
        adj[a].append(b)
        adj[b].append(a)
    components = []
    seen = set()
    for start in adj:
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp = []
        while q:
            node = q.popleft()
            comp.append(node)
            for nb in adj[node]:
                if nb not in seen:
                    seen.add(nb)
                    q.append(nb)
        components.append(np.asarray(comp, dtype=np.int64))
    return components


def _select_loop(mesh, center, min_vertices=16):
    edges = _boundary_edges(np.asarray(mesh.faces, dtype=np.int64))
    if len(edges) == 0:
        raise ValueError("mesh has no boundary after cutting")
    comps = [c for c in _boundary_components(edges) if len(c) >= min_vertices]
    if not comps:
        raise ValueError("no sufficiently large boundary component found")
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    comps.sort(key=lambda c: np.linalg.norm(verts[c].mean(axis=0) - center))
    return comps[0], edges


def _order_loop_by_edges(indices, boundary_edges):
    index_set = set(int(i) for i in indices)
    adj = defaultdict(list)
    for a, b in boundary_edges:
        a = int(a)
        b = int(b)
        if a in index_set and b in index_set:
            adj[a].append(b)
            adj[b].append(a)
    if not adj or any(len(v) != 2 for v in adj.values()):
        return None

    start = min(adj)
    ordered = [start]
    prev = None
    cur = start
    for _ in range(len(adj) + 1):
        nxt_candidates = [n for n in adj[cur] if n != prev]
        if not nxt_candidates:
            return None
        nxt = nxt_candidates[0]
        if nxt == start:
            return np.asarray(ordered, dtype=np.int64)
        ordered.append(nxt)
        prev, cur = cur, nxt
    return None


def _order_loop_by_angle(vertices, indices, center, normal):
    pts = vertices[indices]
    x, y, _ = _project(pts, center, normal)
    angles = np.arctan2(y, x)
    order = np.argsort(angles)
    return indices[order]


def _signed_area_xy(points, center, normal):
    x, y, _ = _project(points, center, normal)
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ensure_same_orientation(vertices, loop_a, loop_b, center, normal):
    area_a = _signed_area_xy(vertices[loop_a], center, normal)
    area_b = _signed_area_xy(vertices[loop_b], center, normal)
    if area_a * area_b < 0:
        return loop_b[::-1]
    return loop_b


def _bridge_loops(loop_a, loop_b):
    n = len(loop_a)
    m = len(loop_b)
    faces = []
    i = 0
    j = 0
    while i < n or j < m:
        next_a = (i + 1) / n if i < n else np.inf
        next_b = (j + 1) / m if j < m else np.inf
        a0 = int(loop_a[i % n])
        b0 = int(loop_b[j % m])
        if next_a <= next_b:
            a1 = int(loop_a[(i + 1) % n])
            if a1 != a0 and b0 != a0 and b0 != a1:
                faces.append([a0, a1, b0])
            i += 1
        else:
            b1 = int(loop_b[(j + 1) % m])
            if b1 != b0 and a0 != b0 and a0 != b1:
                faces.append([a0, b1, b0])
            j += 1
    return np.asarray(faces, dtype=np.int64)


def _rim_from_labels(mesh, labels_path, rim_label):
    labels = np.load(labels_path)
    if len(labels) != len(mesh.vertices):
        raise ValueError(
            f"label count {len(labels)} does not match aneurysm vertices {len(mesh.vertices)}"
        )
    rim = np.where(labels == rim_label)[0]
    if len(rim) < 8:
        raise ValueError(f"found only {len(rim)} rim vertices for label {rim_label}")
    return rim.astype(np.int64)


def _rim_from_boundary(mesh, center):
    loop, _ = _select_loop(mesh, center)
    return loop


def _order_mesh_rim(mesh, rim_indices, center, normal, vertices=None):
    edges = _boundary_edges(np.asarray(mesh.faces, dtype=np.int64))
    ordered = _order_loop_by_edges(rim_indices, edges)
    if ordered is not None:
        return ordered
    if vertices is None:
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
    return _order_loop_by_angle(vertices, rim_indices, center, normal)


def _cut_healthy(healthy, center, normal, radius, radius_scale, slab, jagged_amp, jagged_harmonics, seed):
    verts = np.asarray(healthy.vertices, dtype=np.float64)
    faces = np.asarray(healthy.faces, dtype=np.int64)
    face_centers = verts[faces].mean(axis=1)
    x, y, d = _project(face_centers, center, normal)
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    cut_radius = _jagged_radius(theta, radius * radius_scale, jagged_amp, jagged_harmonics, seed)
    remove = (r <= cut_radius) & (np.abs(d) <= slab)
    cut = trimesh.Trimesh(vertices=verts.copy(), faces=faces[~remove].copy(), process=False)
    cut.remove_unreferenced_vertices()
    return cut, remove


def _snap_aneurysm_to_hole(aneurysm_vertices, aneurysm_rim, hole_vertices, scale=False):
    verts = aneurysm_vertices.copy()
    src = verts[aneurysm_rim]
    dst = hole_vertices
    src_center = src.mean(axis=0)
    dst_center = dst.mean(axis=0)
    if scale:
        src_r = np.linalg.norm(src - src_center, axis=1).mean()
        dst_r = np.linalg.norm(dst - dst_center, axis=1).mean()
        factor = dst_r / (src_r + 1e-12)
        verts = (verts - src_center.reshape(1, 3)) * factor + src_center.reshape(1, 3)
        src_center = verts[aneurysm_rim].mean(axis=0)
    verts = verts + (dst_center - src_center).reshape(1, 3)
    return verts


def _aneurysm_shape_metrics(vertices, rim_indices, center, normal):
    vertices = np.asarray(vertices, dtype=np.float64)
    rim_center = vertices[rim_indices].mean(axis=0)
    rel = vertices - rim_center.reshape(1, 3)
    axial = rel @ normal.reshape(3)
    radial_vec = rel - axial.reshape(-1, 1) * normal.reshape(1, 3)
    radial = np.linalg.norm(radial_vec, axis=1)
    rim_radial = radial[rim_indices]
    return {
        "rim_radius_mean": float(rim_radial.mean()),
        "rim_radius_std": float(rim_radial.std()),
        "sac_axial_min": float(axial.min()),
        "sac_axial_max": float(axial.max()),
        "sac_radial_max": float(radial.max()),
    }


def _scale_aneurysm_shape(vertices, rim_indices, normal, uniform_scale, sac_radial_scale, sac_axial_scale, falloff_power):
    verts = np.asarray(vertices, dtype=np.float64).copy()
    if (
        abs(uniform_scale - 1.0) < 1e-12
        and abs(sac_radial_scale - 1.0) < 1e-12
        and abs(sac_axial_scale - 1.0) < 1e-12
    ):
        return verts

    normal = normal.reshape(3)
    rim_center = verts[rim_indices].mean(axis=0)
    rel = verts - rim_center.reshape(1, 3)
    rel = rel * float(uniform_scale)

    axial = rel @ normal
    radial_vec = rel - axial.reshape(-1, 1) * normal.reshape(1, 3)
    axial_vec = axial.reshape(-1, 1) * normal.reshape(1, 3)

    max_abs_axial = float(np.max(np.abs(axial))) + 1e-12
    falloff = np.clip(np.abs(axial) / max_abs_axial, 0.0, 1.0)
    falloff = falloff ** max(float(falloff_power), 1e-6)
    radial_factor = 1.0 + (float(sac_radial_scale) - 1.0) * falloff

    scaled_rel = radial_vec * radial_factor.reshape(-1, 1) + axial_vec * float(sac_axial_scale)
    return rim_center.reshape(1, 3) + scaled_rel


def _default_paths(args):
    case = args.case
    healthy = args.healthy_mesh
    aneurysm = args.aneurysm_mesh
    labels = args.aneurysm_labels
    centroid = args.ostium_centroid
    normal = args.ostium_normal
    prealign = args.prealign_transform
    ghd_checkpoint = args.ghd_checkpoint
    if case:
        if healthy is None:
            healthy = os.path.join(
                args.healthy_root,
                f"{case}_vessel_submesh_closed",
                f"{case}_vessel_submesh_closed.obj",
            )
        if aneurysm is None:
            aneurysm = os.path.join(args.prepared_root, case, "05_submeshes", "aneurysm_submesh.obj")
        if labels is None and args.aneurysm_mesh is None:
            labels = os.path.join(args.prepared_root, case, "06_submesh_labels", "labels_aneurysm.npy")
        if centroid is None:
            centroid = os.path.join(args.prepared_root, case, "07_other", "centroid_ostium.npy")
        if normal is None:
            normal = os.path.join(args.prepared_root, case, "07_other", "normal_vector.npy")
        if prealign is None:
            prealign = os.path.join(args.aligned_data_root, case, "prealign_transform.npy")
        if ghd_checkpoint is None:
            ghd_checkpoint = os.path.join(
                args.ghd_chk_root,
                case,
                args.ghd_run,
                args.ghd_chk_name,
            )
    return healthy, aneurysm, labels, centroid, normal, prealign, ghd_checkpoint


def _transform_aneurysm_vertices_to_raw(vertices, args, prealign_path, ghd_checkpoint_path):
    vertices = np.asarray(vertices, dtype=np.float64)
    if args.aneurysm_space == "raw":
        return vertices

    if not prealign_path or not os.path.exists(prealign_path):
        raise ValueError("--aneurysm_space requires --prealign_transform or --case")
    prealign = np.load(prealign_path).astype(np.float64)

    if args.aneurysm_space == "aligned":
        return _apply_h_inverse(vertices, prealign)

    if args.aneurysm_space == "ghd_local":
        if not ghd_checkpoint_path or not os.path.exists(ghd_checkpoint_path):
            raise ValueError("--aneurysm_space ghd_local requires --ghd_checkpoint or --case")
        chk = _load_pickle(ghd_checkpoint_path)
        r_mat = _axis_angle_to_matrix(chk["R"])
        s = float(np.abs(_to_numpy(chk["s"]).reshape(-1)[0])) + 1e-12
        t = _to_numpy(chk["T"]).reshape(-1, 3)[0].astype(np.float64)
        canonical_norm = _canonical_norm(args.canonical_mesh, args.canonical_norm_factor)

        target_norm = (vertices @ r_mat.T) * s + t.reshape(1, 3)
        target_aligned = target_norm * canonical_norm
        return _apply_h_inverse(target_aligned, prealign)

    raise ValueError(f"unknown aneurysm_space: {args.aneurysm_space}")


def parse_args():
    p = argparse.ArgumentParser("attach_aneurysm_to_healthy")
    p.add_argument("--case", default=None)
    p.add_argument("--prepared_root", default="/data/prepared_meshes_3")
    p.add_argument("--healthy_root", default="/data/healthy_vessel")
    p.add_argument("--aligned_data_root", default="/data/ghd_prepared_meshes_3_aneurysm_1op_new")
    p.add_argument("--ghd_chk_root", default="/workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999")
    p.add_argument("--ghd_run", default="prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3")
    p.add_argument("--ghd_chk_name", default="ghb_fitting_checkpoint.pkl")
    p.add_argument("--canonical_mesh", default="/workspace/AneuG/checkpoints/canonical_average/part_aligned.obj")
    p.add_argument("--canonical_norm_factor", type=float, default=1.10)
    p.add_argument("--healthy_mesh", default=None)
    p.add_argument("--aneurysm_mesh", default=None)
    p.add_argument("--aneurysm_space", choices=["raw", "aligned", "ghd_local"], default="raw")
    p.add_argument("--aneurysm_labels", default=None)
    p.add_argument("--rim_label", type=int, default=2)
    p.add_argument("--ostium_centroid", default=None)
    p.add_argument("--ostium_normal", default=None)
    p.add_argument("--prealign_transform", default=None)
    p.add_argument("--ghd_checkpoint", default=None)
    p.add_argument("--cut_radius", type=float, default=None)
    p.add_argument("--radius_scale", type=float, default=1.10)
    p.add_argument("--cut_slab", type=float, default=0.06)
    p.add_argument("--jagged_amp", type=float, default=0.12)
    p.add_argument("--jagged_harmonics", type=int, default=7)
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--aneurysm_scale", type=float, default=1.0)
    p.add_argument("--sac_radial_scale", type=float, default=1.0)
    p.add_argument("--sac_axial_scale", type=float, default=1.0)
    p.add_argument("--sac_falloff_power", type=float, default=1.0)
    p.add_argument("--snap_rim", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--scale_rim", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--out_mesh", required=True)
    p.add_argument("--out_cut_mesh", default=None)
    p.add_argument("--out_report", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    healthy_path, aneurysm_path, labels_path, centroid_path, normal_path, prealign_path, ghd_checkpoint_path = _default_paths(args)
    if not all([healthy_path, aneurysm_path, centroid_path, normal_path]):
        raise ValueError("provide --case or explicit mesh/ostium paths")

    center = np.load(centroid_path).astype(np.float64).reshape(3)
    normal = _normalize(np.load(normal_path).astype(np.float64).reshape(3))
    healthy = _load_clean_mesh(healthy_path, merge=True)
    aneurysm = _load_clean_mesh(aneurysm_path, merge=False)
    aneurysm.vertices = _transform_aneurysm_vertices_to_raw(
        np.asarray(aneurysm.vertices, dtype=np.float64),
        args,
        prealign_path,
        ghd_checkpoint_path,
    )

    if labels_path and os.path.exists(labels_path):
        aneurysm_rim = _rim_from_labels(aneurysm, labels_path, args.rim_label)
    else:
        aneurysm_rim = _rim_from_boundary(aneurysm, center)

    aneurysm_vertices = np.asarray(aneurysm.vertices, dtype=np.float64)
    shape_before = _aneurysm_shape_metrics(aneurysm_vertices, aneurysm_rim, center, normal)
    aneurysm_vertices = _scale_aneurysm_shape(
        aneurysm_vertices,
        aneurysm_rim,
        normal,
        args.aneurysm_scale,
        args.sac_radial_scale,
        args.sac_axial_scale,
        args.sac_falloff_power,
    )
    aneurysm.vertices = aneurysm_vertices
    shape_after_scale = _aneurysm_shape_metrics(aneurysm_vertices, aneurysm_rim, center, normal)

    rim_points = aneurysm_vertices[aneurysm_rim]
    x, y, _ = _project(rim_points, center, normal)
    inferred_radius = float(np.sqrt(x * x + y * y).mean())
    cut_radius = float(args.cut_radius) if args.cut_radius is not None else inferred_radius

    cut_healthy, removed = _cut_healthy(
        healthy,
        center,
        normal,
        cut_radius,
        args.radius_scale,
        args.cut_slab,
        args.jagged_amp,
        args.jagged_harmonics,
        args.seed,
    )
    hole_loop, hole_edges = _select_loop(cut_healthy, center)
    hole_loop_ordered = _order_loop_by_edges(hole_loop, hole_edges)
    if hole_loop_ordered is None:
        hole_loop_ordered = _order_loop_by_angle(np.asarray(cut_healthy.vertices), hole_loop, center, normal)
    hole_loop = hole_loop_ordered

    if args.snap_rim:
        aneurysm_vertices = _snap_aneurysm_to_hole(
            aneurysm_vertices,
            aneurysm_rim,
            np.asarray(cut_healthy.vertices)[hole_loop],
            scale=args.scale_rim,
        )
    aneurysm_rim = _order_mesh_rim(aneurysm, aneurysm_rim, center, normal, vertices=aneurysm_vertices)

    base_v = np.asarray(cut_healthy.vertices, dtype=np.float64)
    base_f = np.asarray(cut_healthy.faces, dtype=np.int64)
    aneurysm_offset = len(base_v)
    aneurysm_f = np.asarray(aneurysm.faces, dtype=np.int64) + aneurysm_offset
    aneurysm_loop = aneurysm_rim + aneurysm_offset

    all_vertices = np.vstack([base_v, aneurysm_vertices])
    aneurysm_loop = _ensure_same_orientation(all_vertices, hole_loop, aneurysm_loop, center, normal)
    bridge_f = _bridge_loops(hole_loop, aneurysm_loop)
    all_faces = np.vstack([base_f, aneurysm_f, bridge_f])

    combined = trimesh.Trimesh(vertices=all_vertices, faces=all_faces, process=False)
    combined.remove_unreferenced_vertices()
    os.makedirs(os.path.dirname(args.out_mesh) or ".", exist_ok=True)
    combined.export(args.out_mesh)
    if args.out_cut_mesh:
        os.makedirs(os.path.dirname(args.out_cut_mesh) or ".", exist_ok=True)
        cut_healthy.export(args.out_cut_mesh)

    report = {
        "case": args.case,
        "healthy_mesh": healthy_path,
        "aneurysm_mesh": aneurysm_path,
        "aneurysm_space": args.aneurysm_space,
        "out_mesh": args.out_mesh,
        "healthy_vertices": int(len(healthy.vertices)),
        "healthy_faces": int(len(healthy.faces)),
        "cut_removed_faces": int(removed.sum()),
        "hole_loop_vertices": int(len(hole_loop)),
        "aneurysm_rim_vertices": int(len(aneurysm_rim)),
        "bridge_faces": int(len(bridge_f)),
        "cut_radius": cut_radius,
        "radius_scale": args.radius_scale,
        "cut_slab": args.cut_slab,
        "jagged_amp": args.jagged_amp,
        "aneurysm_scale": args.aneurysm_scale,
        "sac_radial_scale": args.sac_radial_scale,
        "sac_axial_scale": args.sac_axial_scale,
        "sac_falloff_power": args.sac_falloff_power,
        "shape_before": shape_before,
        "shape_after_scale": shape_after_scale,
        "shape_after_snap": _aneurysm_shape_metrics(aneurysm_vertices, aneurysm_rim, center, normal),
        "combined_vertices": int(len(combined.vertices)),
        "combined_faces": int(len(combined.faces)),
        "combined_watertight": bool(combined.is_watertight),
        "combined_boundary_edges": int((_boundary_edges(np.asarray(combined.faces)).shape[0])),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.out_report:
        os.makedirs(os.path.dirname(args.out_report) or ".", exist_ok=True)
        with open(args.out_report, "w") as f:
            json.dump(report, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
