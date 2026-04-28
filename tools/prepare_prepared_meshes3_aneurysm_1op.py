#!/usr/bin/env python3
"""
Build GHD-ready cases (num_op=1) from /data/prepared_meshes_3.

Per case this script:
1) exports aneurysm_submesh.obj -> <target_root>/<case>/part_aligned.obj
2) optional pre-alignment to canonical_average via ostium center/normal (+ optional ICP)
3) builds opening checkpoint from labels_aneurysm == 2 (ostium ring)
4) builds differentiable-centreline checkpoint from one seed near ostium centroid

It also prepares the canonical case from canonical_average.obj by taking its
largest boundary loop as the single opening.
"""

import argparse
import fnmatch
import math
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
import trimesh
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate as shapely_triangulate
from scipy.spatial import Delaunay

# Ensure repository root is importable when the script is called from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ghd.fitting.registration import (  # noqa: E402
    RegistrationwOpeningAlignmentwDifferentiableCentreline,
)


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.size != 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return (v / n).astype(np.float64)


def _sign_nonzero(x: float, eps: float = 1e-12) -> float:
    x = float(x)
    if x > eps:
        return 1.0
    if x < -eps:
        return -1.0
    return 1.0


def _boundary_edges(faces: np.ndarray) -> np.ndarray:
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    edges = np.concatenate(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]),
        axis=0,
    )
    edges = np.sort(edges, axis=1)
    uniq, counts = np.unique(edges, axis=0, return_counts=True)
    return uniq[counts == 1]


def _edge_components(boundary_edges: np.ndarray) -> List[np.ndarray]:
    boundary_edges = np.asarray(boundary_edges, dtype=np.int64).reshape(-1, 2)
    if boundary_edges.size == 0:
        return []
    neighbors: Dict[int, List[int]] = {}
    for a, b in boundary_edges.tolist():
        neighbors.setdefault(int(a), []).append(int(b))
        neighbors.setdefault(int(b), []).append(int(a))
    visited = set()
    comps = []
    for node in neighbors.keys():
        if node in visited:
            continue
        stack = [node]
        comp = []
        visited.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in neighbors.get(cur, []):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        comps.append(np.asarray(comp, dtype=np.int64))
    comps.sort(key=lambda x: int(x.shape[0]), reverse=True)
    return comps


def _plane_basis_from_normal(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normal = _normalize(normal)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, normal))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis_u = _normalize(np.cross(normal, ref))
    axis_v = _normalize(np.cross(normal, axis_u))
    return axis_u, axis_v


def _angle_sort_indices(loop_idx: np.ndarray, verts: np.ndarray, normal: np.ndarray) -> np.ndarray:
    loop_idx = np.unique(np.asarray(loop_idx, dtype=np.int64).reshape(-1))
    if loop_idx.shape[0] < 3:
        return loop_idx
    coords = np.asarray(verts, dtype=np.float64)[loop_idx]
    center = np.mean(coords, axis=0)
    axis_u, axis_v = _plane_basis_from_normal(normal)
    rel = coords - center.reshape(1, 3)
    ang = np.arctan2(rel @ axis_v, rel @ axis_u)
    order = np.argsort(ang)
    return loop_idx[order]


def _order_loop_by_boundary(
    loop_idx: np.ndarray,
    faces: np.ndarray,
    verts: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    loop_idx = np.unique(np.asarray(loop_idx, dtype=np.int64).reshape(-1))
    if loop_idx.shape[0] < 3:
        return loop_idx

    boundary = _boundary_edges(np.asarray(faces, dtype=np.int64).reshape(-1, 3))
    if boundary.shape[0] < 3:
        return _angle_sort_indices(loop_idx, verts, normal)

    loop_set = set(loop_idx.tolist())
    edge_mask = np.array(
        [(int(a) in loop_set and int(b) in loop_set) for a, b in boundary.tolist()],
        dtype=bool,
    )
    ring_edges = boundary[edge_mask]
    if ring_edges.shape[0] < 3:
        return _angle_sort_indices(loop_idx, verts, normal)

    neighbors: Dict[int, List[int]] = {}
    for a, b in ring_edges.tolist():
        aa = int(a)
        bb = int(b)
        neighbors.setdefault(aa, []).append(bb)
        neighbors.setdefault(bb, []).append(aa)
    if len(neighbors) < 3:
        return _angle_sort_indices(loop_idx, verts, normal)

    visited = set()
    components: List[List[int]] = []
    for node in neighbors.keys():
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in neighbors.get(cur, []):
                if nxt not in visited:
                    visited.add(nxt)
                    stack.append(nxt)
        components.append(comp)
    components.sort(key=len, reverse=True)
    comp = components[0]
    comp_set = set(int(v) for v in comp)
    if len(comp_set) < 3:
        return _angle_sort_indices(loop_idx, verts, normal)

    comp_neighbors: Dict[int, List[int]] = {}
    for v in comp_set:
        comp_neighbors[int(v)] = [int(n) for n in neighbors.get(int(v), []) if int(n) in comp_set]

    ends = sorted([v for v in comp_set if len(comp_neighbors[v]) == 1])
    start = ends[0] if len(ends) > 0 else min(comp_set)

    ordered = [int(start)]
    prev = None
    curr = int(start)
    for _ in range(len(comp_set) + 5):
        cand = [n for n in comp_neighbors.get(curr, []) if n != prev]
        if len(cand) == 0:
            break
        next_v = None
        for c in sorted(cand):
            if c not in ordered:
                next_v = int(c)
                break
        if next_v is None:
            if int(start) in cand and len(ordered) >= 3:
                break
            next_v = int(sorted(cand)[0])
            if next_v == int(start):
                break
        ordered.append(next_v)
        prev, curr = curr, next_v
        if curr == int(start):
            break

    ordered_arr = np.asarray(ordered, dtype=np.int64)
    if ordered_arr.shape[0] < 3:
        return _angle_sort_indices(loop_idx, verts, normal)
    if len(set(ordered_arr.tolist())) != ordered_arr.shape[0]:
        return _angle_sort_indices(np.asarray(list(comp_set), dtype=np.int64), verts, normal)
    if ordered_arr.shape[0] != len(comp_set):
        return _angle_sort_indices(np.asarray(list(comp_set), dtype=np.int64), verts, normal)
    return ordered_arr


def _orient_loop_to_normal(loop_idx: np.ndarray, verts: np.ndarray, normal: np.ndarray) -> np.ndarray:
    loop_idx = np.asarray(loop_idx, dtype=np.int64).reshape(-1)
    if loop_idx.shape[0] < 3:
        return loop_idx
    coords = np.asarray(verts, dtype=np.float64)[loop_idx]
    rolled = np.roll(coords, -1, axis=0)
    area_vec = np.sum(np.cross(coords, rolled), axis=0)
    if float(np.dot(area_vec, _normalize(normal))) < 0.0:
        return loop_idx[::-1].copy()
    return loop_idx


def _signed_area_2d(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    if poly.shape[0] < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _point_in_triangle_2d(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    eps: float = 1e-12,
) -> bool:
    p = np.asarray(p, dtype=np.float64).reshape(2)
    a = np.asarray(a, dtype=np.float64).reshape(2)
    b = np.asarray(b, dtype=np.float64).reshape(2)
    c = np.asarray(c, dtype=np.float64).reshape(2)
    v0 = c - a
    v1 = b - a
    v2 = p - a
    den = float(v0[0] * v1[1] - v1[0] * v0[1])
    if abs(den) < eps:
        return False
    u = float((v2[0] * v1[1] - v1[0] * v2[1]) / den)
    v = float((v0[0] * v2[1] - v2[0] * v0[1]) / den)
    w = 1.0 - u - v
    return (u > eps) and (v > eps) and (w > eps)


def _earclip_triangulate_2d(poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    n = int(poly.shape[0])
    if n < 3:
        return np.zeros((0, 3), dtype=np.int64)

    area = _signed_area_2d(poly)
    if abs(area) < 1e-12:
        return np.zeros((0, 3), dtype=np.int64)
    orientation = 1.0 if area > 0.0 else -1.0

    def is_convex(i_prev: int, i_curr: int, i_next: int) -> bool:
        a = poly[i_prev]
        b = poly[i_curr]
        c = poly[i_next]
        cross_z = float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if orientation > 0.0:
            return cross_z > 1e-12
        return cross_z < -1e-12

    vertices = list(range(n))
    faces: List[List[int]] = []
    max_iter = n * n
    iter_count = 0

    while len(vertices) > 3 and iter_count < max_iter:
        iter_count += 1
        ear_found = False
        m = len(vertices)
        for i in range(m):
            i_prev = int(vertices[(i - 1) % m])
            i_curr = int(vertices[i])
            i_next = int(vertices[(i + 1) % m])
            if not is_convex(i_prev, i_curr, i_next):
                continue
            tri_a = poly[i_prev]
            tri_b = poly[i_curr]
            tri_c = poly[i_next]
            contains_point = False
            for j in vertices:
                jj = int(j)
                if jj in (i_prev, i_curr, i_next):
                    continue
                if _point_in_triangle_2d(poly[jj], tri_a, tri_b, tri_c):
                    contains_point = True
                    break
            if contains_point:
                continue
            faces.append([i_prev, i_curr, i_next])
            del vertices[i]
            ear_found = True
            break
        if not ear_found:
            break

    if len(vertices) == 3:
        faces.append([int(vertices[0]), int(vertices[1]), int(vertices[2])])

    faces_arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if faces_arr.shape[0] < (n - 2):
        return np.zeros((0, 3), dtype=np.int64)
    return faces_arr


def _triangulate_polygon_2d_balanced(poly: np.ndarray) -> np.ndarray:
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    n = int(poly.shape[0])
    if n < 3:
        return np.zeros((0, 3), dtype=np.int64)

    # Prefer a more balanced triangulation than plain ear clipping. Shapely's
    # polygon triangulation is robust here and avoids the strong fan patterns
    # that make the opening-cap interpolation visually and numerically awkward.
    try:
        polygon = Polygon(poly)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or float(polygon.area) <= 1e-12:
            raise ValueError("Degenerate 2D polygon.")

        coord_to_idx = {tuple(np.round(p, 12)): int(i) for i, p in enumerate(poly)}
        orientation = _sign_nonzero(_signed_area_2d(poly))
        faces: List[List[int]] = []
        seen = set()

        for tri in shapely_triangulate(polygon):
            rep = tri.representative_point()
            if not polygon.buffer(1e-9).covers(rep):
                continue
            coords = np.asarray(tri.exterior.coords[:-1], dtype=np.float64)
            if coords.shape != (3, 2):
                continue
            idx = []
            ok = True
            for p in coords:
                key = tuple(np.round(p, 12))
                mapped = coord_to_idx.get(key)
                if mapped is None:
                    dist = np.linalg.norm(poly - p.reshape(1, 2), axis=1)
                    mapped = int(np.argmin(dist))
                    if float(dist[mapped]) > 1e-7:
                        ok = False
                        break
                idx.append(int(mapped))
            if not ok or len(set(idx)) != 3:
                continue

            face = np.asarray(idx, dtype=np.int64)
            tri2 = poly[face]
            cross_z = float(
                (tri2[1, 0] - tri2[0, 0]) * (tri2[2, 1] - tri2[0, 1])
                - (tri2[1, 1] - tri2[0, 1]) * (tri2[2, 0] - tri2[0, 0])
            )
            if orientation * cross_z < 0.0:
                face = face[[0, 2, 1]]

            key = tuple(sorted(int(v) for v in face.tolist()))
            if key in seen:
                continue
            seen.add(key)
            faces.append(face.tolist())

        faces_arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        if faces_arr.shape[0] >= (n - 2):
            return faces_arr
    except Exception:
        pass

    return _earclip_triangulate_2d(poly)


def _triangle_quality(tris: np.ndarray) -> np.ndarray:
    tris = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3)
    if tris.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    a = np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1)
    b = np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1)
    c = np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    denom = a * a + b * b + c * c + 1e-12
    return (4.0 * math.sqrt(3.0) * area) / denom


def _remove_duplicate_loop_points(points: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] <= 1:
        return pts.copy()
    keep = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - keep[-1]) > float(tol):
            keep.append(p)
    out = np.asarray(keep, dtype=np.float64)
    if out.shape[0] > 2 and np.linalg.norm(out[0] - out[-1]) <= float(tol):
        out = out[:-1]
    return out


def _resample_closed_curve(points: np.ndarray, num_samples: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 2:
        return pts.copy()
    num_samples = max(3, int(num_samples))
    ring = np.concatenate([pts, pts[:1]], axis=0)
    seg = np.linalg.norm(ring[1:] - ring[:-1], axis=1)
    total = float(seg.sum())
    if total <= 1e-12:
        return np.repeat(pts[:1], num_samples, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t_query = np.linspace(0.0, total, num_samples + 1)[:-1]
    out = []
    for t in t_query:
        idx = int(np.searchsorted(cum, t, side="right") - 1)
        idx = min(max(idx, 0), pts.shape[0] - 1)
        local = float(t - cum[idx])
        length = float(seg[idx]) if idx < seg.shape[0] else 0.0
        alpha = 0.0 if length <= 1e-12 else np.clip(local / length, 0.0, 1.0)
        out.append((1.0 - alpha) * ring[idx] + alpha * ring[idx + 1])
    return np.asarray(out, dtype=np.float64)


def _smooth_closed_curve(points: np.ndarray, window: int = 5) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 5:
        return pts.copy()
    window = int(max(3, min(window, pts.shape[0] - (1 - pts.shape[0] % 2))))
    if window % 2 == 0:
        window += 1
    half = window // 2
    pad = np.concatenate([pts[-half:], pts, pts[:half]], axis=0)
    kernel = np.ones(window, dtype=np.float64) / float(window)
    cols = [np.convolve(pad[:, d], kernel, mode="valid") for d in range(pts.shape[1])]
    return np.stack(cols, axis=1).astype(np.float64)


def _sort_points_around_normal(points: np.ndarray, normal: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 3:
        return pts.copy()
    center = pts.mean(axis=0)
    axis_u, axis_v = _plane_basis_from_normal(normal)
    rel = pts - center.reshape(1, 3)
    ang = np.arctan2(rel @ axis_v, rel @ axis_u)
    order = np.argsort(ang)
    return pts[order]


def _plane_from_points_np(points: np.ndarray, normal_hint: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] < 3:
        center = pts.mean(axis=0) if pts.shape[0] > 0 else np.zeros(3, dtype=np.float64)
        normal = _normalize(normal_hint if normal_hint is not None else np.array([0.0, 0.0, 1.0], dtype=np.float64))
        return center, normal
    center = pts.mean(axis=0)
    x = pts - center.reshape(1, 3)
    try:
        _, _, vh = np.linalg.svd(x, full_matrices=False)
        normal = _normalize(vh[-1])
    except np.linalg.LinAlgError:
        normal = _normalize(normal_hint if normal_hint is not None else np.array([0.0, 0.0, 1.0], dtype=np.float64))
    if normal_hint is not None and float(np.dot(normal, _normalize(normal_hint))) < 0.0:
        normal = -1.0 * normal
    return center, normal


def _largest_polygon(polygon: Polygon) -> Polygon:
    poly = polygon
    if hasattr(poly, "geoms"):
        geoms = [g for g in poly.geoms if isinstance(g, Polygon)]
        if geoms:
            poly = max(geoms, key=lambda g: float(g.area))
    return poly


def _sanitize_planar_loop(loop_2d: np.ndarray) -> Tuple[np.ndarray, str]:
    poly = _remove_duplicate_loop_points(np.asarray(loop_2d, dtype=np.float64))
    if poly.shape[0] < 3:
        return poly, "degenerate"
    stage = "ordered"
    polygon = Polygon(poly)
    if (not polygon.is_valid) or float(polygon.area) <= 1e-12:
        center = poly.mean(axis=0)
        ang = np.arctan2(poly[:, 1] - center[1], poly[:, 0] - center[0])
        poly = poly[np.argsort(ang)]
        poly = _smooth_closed_curve(poly, window=5)
        poly = _remove_duplicate_loop_points(poly)
        polygon = Polygon(poly)
        stage = "angle_sorted"
    if (not polygon.is_valid) or float(polygon.area) <= 1e-12:
        polygon = _largest_polygon(polygon.buffer(0))
        if polygon.is_empty or float(polygon.area) <= 1e-12:
            polygon = Polygon(poly).convex_hull
            stage = "convex_hull"
        else:
            stage = "buffer0"
        poly = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    if _signed_area_2d(poly) < 0.0:
        poly = poly[::-1].copy()
    return poly, stage


def _triangulate_polygon_2d_with_interior(poly: np.ndarray, inner_rings: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    poly, _ = _sanitize_planar_loop(poly)
    if poly.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
    polygon = _largest_polygon(Polygon(poly).buffer(0))
    if polygon.is_empty or float(polygon.area) <= 1e-12:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)
    poly = np.asarray(polygon.exterior.coords[:-1], dtype=np.float64)
    if poly.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)

    center = poly.mean(axis=0)
    points = [poly]
    cover = polygon.buffer(1e-9)
    for ridx in range(max(0, int(inner_rings))):
        scale = 1.0 - 0.55 * float(ridx + 1) / float(max(inner_rings, 1) + 1)
        ring = center.reshape(1, 2) + scale * (poly - center.reshape(1, 2))
        keep = np.array([bool(cover.covers(Point(p))) for p in ring], dtype=bool)
        ring = _remove_duplicate_loop_points(ring[keep])
        if ring.shape[0] >= 3:
            points.append(ring)
    if bool(cover.covers(Point(center))):
        points.append(center.reshape(1, 2))
    all_pts = np.concatenate(points, axis=0)
    rounded = np.round(all_pts, decimals=8)
    _, uniq_idx = np.unique(rounded, axis=0, return_index=True)
    all_pts = all_pts[np.sort(uniq_idx)]
    if all_pts.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0, 3), dtype=np.int64)

    try:
        simplices = Delaunay(all_pts).simplices
    except Exception:
        faces = _triangulate_polygon_2d_balanced(poly)
        return poly.copy(), faces

    orientation = _sign_nonzero(_signed_area_2d(poly))
    faces = []
    seen = set()
    for simp in np.asarray(simplices, dtype=np.int64):
        tri = all_pts[simp]
        tri_poly = Polygon(tri)
        if tri_poly.is_empty or float(tri_poly.area) <= 1e-12:
            continue
        if not cover.covers(tri_poly):
            continue
        face = simp.copy()
        cross_z = float(
            (tri[1, 0] - tri[0, 0]) * (tri[2, 1] - tri[0, 1])
            - (tri[1, 1] - tri[0, 1]) * (tri[2, 0] - tri[0, 0])
        )
        if orientation * cross_z < 0.0:
            face = face[[0, 2, 1]]
        key = tuple(sorted(int(v) for v in face.tolist()))
        if key in seen:
            continue
        seen.add(key)
        faces.append(face.tolist())
    faces_arr = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if faces_arr.shape[0] < 1:
        faces_arr = _triangulate_polygon_2d_balanced(poly)
        return poly.copy(), faces_arr
    return all_pts.astype(np.float64), faces_arr


def _opening_mesh_qc(verts: np.ndarray, faces: np.ndarray, rim_points: np.ndarray, normal: np.ndarray) -> Dict[str, float]:
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    rim_points = np.asarray(rim_points, dtype=np.float64).reshape(-1, 3)
    counts = np.bincount(faces.reshape(-1), minlength=max(int(verts.shape[0]), 1)) if faces.size > 0 else np.zeros((max(int(verts.shape[0]), 1),), dtype=np.int64)
    fan_ratio = float(counts.max()) / max(int(faces.shape[0]), 1) if faces.shape[0] > 0 else 1.0
    tris = verts[faces] if faces.shape[0] > 0 else np.zeros((0, 3, 3), dtype=np.float64)
    tri_quality = _triangle_quality(tris)
    min_tri_quality = float(np.min(tri_quality)) if tri_quality.size > 0 else 0.0
    center, plane_normal = _plane_from_points_np(rim_points, normal_hint=normal)
    rel = rim_points - center.reshape(1, 3)
    planarity_rms = float(np.sqrt(np.mean((rel @ plane_normal) ** 2))) if rim_points.shape[0] > 0 else 0.0
    axis_u, axis_v = _plane_basis_from_normal(plane_normal)
    poly_2d = np.stack([rel @ axis_u, rel @ axis_v], axis=1) if rim_points.shape[0] > 0 else np.zeros((0, 2), dtype=np.float64)
    area_2d = abs(_signed_area_2d(poly_2d))
    rim_radius = math.sqrt(area_2d / math.pi) if area_2d > 1e-12 else 1.0
    planarity_rel = planarity_rms / max(rim_radius, 1e-6)
    ring = np.concatenate([rim_points, rim_points[:1]], axis=0) if rim_points.shape[0] > 0 else np.zeros((0, 3), dtype=np.float64)
    edge_lengths = np.linalg.norm(ring[1:] - ring[:-1], axis=1) if ring.shape[0] > 1 else np.zeros((0,), dtype=np.float64)
    rim_edge_cv = float(edge_lengths.std() / (edge_lengths.mean() + 1e-12)) if edge_lengths.size > 0 else 0.0
    return {
        "fan_ratio": fan_ratio,
        "planarity_rel": planarity_rel,
        "rim_edge_cv": rim_edge_cv,
        "min_tri_quality": min_tri_quality,
    }


def _load_original_ostium_source(
    source_case_dir: Path,
    transform: np.ndarray,
    normal_hint: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:
    mesh_candidates = [
        source_case_dir / "05_submeshes" / "ostium_submesh.obj",
        source_case_dir / "05_submeshes" / "ostium_submesh.ply",
        source_case_dir / "05_submeshes" / "ostium.obj",
        source_case_dir / "05_submeshes" / "ostium_mesh.obj",
    ]
    for mesh_path in mesh_candidates:
        if not mesh_path.is_file():
            continue
        try:
            mesh = trimesh.load(str(mesh_path), process=False)
            verts = _transform_points(np.asarray(mesh.vertices, dtype=np.float64), transform)
            faces = np.asarray(mesh.faces, dtype=np.int64)
            rim_coords = verts
            if faces.ndim == 2 and faces.shape[0] > 0:
                boundary = _boundary_edges(faces)
                comps = _edge_components(boundary)
                if len(comps) > 0 and comps[0].shape[0] >= 3:
                    loop_idx = _order_loop_by_boundary(comps[0], faces, verts, _normalize(normal_hint))
                    loop_idx = _orient_loop_to_normal(loop_idx, verts, _normalize(normal_hint))
                    rim_coords = verts[loop_idx]
            return {
                "kind": "ostium_mesh",
                "rim_coords": np.asarray(rim_coords, dtype=np.float64),
                "surface_v": np.asarray(verts, dtype=np.float64),
                "surface_f": np.asarray(faces, dtype=np.int64),
            }
        except Exception:
            continue

    pcd_path = source_case_dir / "04_subpointclouds" / "subpointcloud_label_2.ply"
    if pcd_path.is_file():
        try:
            pcd = o3d.io.read_point_cloud(str(pcd_path))
            pts = np.asarray(pcd.points, dtype=np.float64)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                pts = _transform_points(pts, transform)
                pts = _sort_points_around_normal(pts, _normalize(normal_hint))
                return {
                    "kind": "ostium_pointcloud",
                    "rim_coords": np.asarray(pts, dtype=np.float64),
                    "surface_v": np.zeros((0, 3), dtype=np.float64),
                    "surface_f": np.zeros((0, 3), dtype=np.int64),
                }
        except Exception:
            pass
    return None


def _build_opening_supervision(
    loop_coords: np.ndarray,
    normal_hint: np.ndarray,
    original_source: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, object]:
    source_kind = "rim_labels"
    rim_coords = np.asarray(loop_coords, dtype=np.float64).reshape(-1, 3)
    source_surface_v = np.zeros((0, 3), dtype=np.float64)
    source_surface_f = np.zeros((0, 3), dtype=np.int64)
    if original_source is not None:
        source_kind = str(original_source.get("kind", source_kind))
        rim_src = np.asarray(original_source.get("rim_coords", rim_coords), dtype=np.float64).reshape(-1, 3)
        if rim_src.shape[0] >= 3:
            rim_coords = rim_src
        source_surface_v = np.asarray(original_source.get("surface_v", source_surface_v), dtype=np.float64).reshape(-1, 3)
        source_surface_f = np.asarray(original_source.get("surface_f", source_surface_f), dtype=np.int64).reshape(-1, 3)

    rim_coords = _remove_duplicate_loop_points(_sort_points_around_normal(rim_coords, normal_hint))
    resample_n = int(np.clip(max(int(rim_coords.shape[0]), 48), 48, 128))
    rim_uniform = _resample_closed_curve(rim_coords, resample_n)
    rim_uniform = _smooth_closed_curve(rim_uniform, window=5)
    plane_center, plane_normal = _plane_from_points_np(rim_uniform, normal_hint=normal_hint)
    axis_u, axis_v = _plane_basis_from_normal(plane_normal)
    rel = rim_uniform - plane_center.reshape(1, 3)
    loop_2d = np.stack([rel @ axis_u, rel @ axis_v], axis=1)
    loop_2d, loop_stage = _sanitize_planar_loop(loop_2d)

    verts_2d, faces = _triangulate_polygon_2d_with_interior(loop_2d, inner_rings=2)
    stage = f"repair_{loop_stage}"
    if faces.shape[0] < 1:
        smooth_2d = _smooth_closed_curve(loop_2d, window=7)
        smooth_2d, _ = _sanitize_planar_loop(smooth_2d)
        verts_2d, faces = _triangulate_polygon_2d_with_interior(smooth_2d, inner_rings=3)
        loop_2d = smooth_2d
        stage = "synthetic_planar_cap"
    if faces.shape[0] < 1:
        faces = _triangulate_polygon_2d_balanced(loop_2d)
        verts_2d = loop_2d.copy()
        stage = "balanced_outer_only"
    if faces.shape[0] < 1:
        raise ValueError("Could not build repaired target opening surface.")

    cap_v = (
        plane_center.reshape(1, 3)
        + verts_2d[:, 0:1] * axis_u.reshape(1, 3)
        + verts_2d[:, 1:2] * axis_v.reshape(1, 3)
    )
    cap_qc = _opening_mesh_qc(cap_v, faces, rim_uniform, plane_normal)

    use_original_surface = False
    src_qc = {}
    if source_surface_v.shape[0] >= 3 and source_surface_f.shape[0] >= 1:
        src_qc = _opening_mesh_qc(source_surface_v, source_surface_f, rim_uniform, plane_normal)
        use_original_surface = bool(
            src_qc["fan_ratio"] <= 0.35 and src_qc["min_tri_quality"] >= 0.08
        )

    target_surface_v = source_surface_v if use_original_surface else cap_v
    target_surface_f = source_surface_f if use_original_surface else faces
    final_stage = "original_surface" if use_original_surface else stage
    return {
        "source_kind": source_kind,
        "rim_points": rim_uniform.astype(np.float64),
        "plane_center": plane_center.astype(np.float64),
        "plane_normal": plane_normal.astype(np.float64),
        "target_surface_v": np.asarray(target_surface_v, dtype=np.float64),
        "target_surface_f": np.asarray(target_surface_f, dtype=np.int64),
        "source_surface_v": source_surface_v,
        "source_surface_f": source_surface_f,
        "debug": {
            "stage": final_stage,
            "cap_qc": cap_qc,
            "source_surface_qc": src_qc,
        },
    }


def _estimate_loop_normal(loop_coords: np.ndarray) -> np.ndarray:
    loop_coords = np.asarray(loop_coords, dtype=np.float64).reshape(-1, 3)
    if loop_coords.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    centered = loop_coords - np.mean(loop_coords, axis=0, keepdims=True)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        n = vh[-1]
    except np.linalg.LinAlgError:
        n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return _normalize(n)


def _loop_radius(loop_coords: np.ndarray, center: np.ndarray) -> float:
    loop_coords = np.asarray(loop_coords, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    if loop_coords.shape[0] < 1:
        return 1.0
    radial = np.linalg.norm(loop_coords - center.reshape(1, 3), axis=1)
    return float(np.median(radial))


def _principal_axis_in_plane(loop_coords: np.ndarray, center: np.ndarray, normal: np.ndarray) -> np.ndarray:
    loop_coords = np.asarray(loop_coords, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(center, dtype=np.float64).reshape(3)
    normal = _normalize(normal)
    if loop_coords.shape[0] < 3:
        ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(ref, normal)) > 0.95:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = ref - np.dot(ref, normal) * normal
        return _normalize(axis)
    x = loop_coords - center.reshape(1, 3)
    x = x - np.outer(np.dot(x, normal), normal)
    cov = x.T @ x
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
    except np.linalg.LinAlgError:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    axis = axis - np.dot(axis, normal) * normal
    return _normalize(axis)


def _rotation_between_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _normalize(a)
    b = _normalize(b)
    v = np.cross(a, b)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0.0:
            return np.eye(3, dtype=np.float64)
        # 180 degrees: choose any axis orthogonal to a
        helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(helper, a)) > 0.95:
            helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = _normalize(np.cross(a, helper))
        return _axis_angle_rotation(axis, math.pi)
    k = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]],
        dtype=np.float64,
    )
    r = np.eye(3, dtype=np.float64) + k + (k @ k) * ((1.0 - c) / (s * s))
    return r


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(axis)
    x, y, z = axis.tolist()
    c = float(math.cos(angle))
    s = float(math.sin(angle))
    one_c = 1.0 - c
    r = np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )
    return r


def _to_homogeneous(r: np.ndarray, t: np.ndarray, s: float = 1.0) -> np.ndarray:
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = float(s) * np.asarray(r, dtype=np.float64).reshape(3, 3)
    m[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return m


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    return (points @ t[:3, :3].T) + t[:3, 3].reshape(1, 3)


def _transform_normal(normal: np.ndarray, transform: np.ndarray) -> np.ndarray:
    r = np.asarray(transform, dtype=np.float64).reshape(4, 4)[:3, :3]
    out = r @ _normalize(normal)
    return _normalize(out)


def _sample_mesh_points(mesh: trimesh.Trimesh, n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.asarray(mesh.vertices, dtype=np.float64)
    try:
        pts, _ = trimesh.sample.sample_surface(mesh, int(n))
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        if pts.shape[0] > 0:
            return pts
    except Exception:
        pass
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.shape[0] <= n:
        return verts
    idx = rng.choice(verts.shape[0], size=int(n), replace=False)
    return verts[idx]


def _icp_refine(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    max_iter: int,
    distance_threshold: float,
) -> Tuple[np.ndarray, float, float]:
    src = o3d.geometry.PointCloud()
    src.points = o3d.utility.Vector3dVector(np.asarray(src_points, dtype=np.float64))
    dst = o3d.geometry.PointCloud()
    dst.points = o3d.utility.Vector3dVector(np.asarray(dst_points, dtype=np.float64))
    result = o3d.pipelines.registration.registration_icp(
        src,
        dst,
        float(distance_threshold),
        np.eye(4, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iter)),
    )
    return np.asarray(result.transformation, dtype=np.float64), float(result.fitness), float(result.inlier_rmse)


def _build_ostium_transform(
    loop_coords_src: np.ndarray,
    center_src: np.ndarray,
    normal_src: np.ndarray,
    canonical_ref: Dict[str, np.ndarray],
    scale_ostium: bool,
) -> Tuple[np.ndarray, Dict[str, float]]:
    center_src = np.asarray(center_src, dtype=np.float64).reshape(3)
    normal_src = _normalize(normal_src)
    center_can = np.asarray(canonical_ref["center"], dtype=np.float64).reshape(3)
    normal_can = _normalize(canonical_ref["normal"])
    axis_can = _normalize(canonical_ref["axis"])
    # Keep source-normal orientation as provided by the case.
    # Flipping here can mirror the sac to the opposite side of the ostium plane.
    normal_dot = float(np.dot(normal_src, normal_can))

    r1 = _rotation_between_vectors(normal_src, normal_can)
    axis_src = _principal_axis_in_plane(loop_coords_src, center_src, normal_src)
    axis_src_r = _normalize(r1 @ axis_src)
    if float(np.dot(axis_src_r, axis_can)) < 0.0:
        axis_src_r = -1.0 * axis_src_r
    ang = math.atan2(
        float(np.dot(normal_can, np.cross(axis_src_r, axis_can))),
        float(np.clip(np.dot(axis_src_r, axis_can), -1.0, 1.0)),
    )
    r2 = _axis_angle_rotation(normal_can, ang)
    r = r2 @ r1

    s = 1.0
    if scale_ostium:
        r_src = _loop_radius(loop_coords_src, center_src)
        r_can = float(canonical_ref["radius"])
        s = r_can / max(r_src, 1e-8)
        s = float(np.clip(s, 0.25, 4.0))

    t = center_can - (s * (r @ center_src))
    transform = _to_homogeneous(r=r, t=t, s=s)
    info = {
        "scale": float(s),
        "normal_dot": float(normal_dot),
        "inplane_angle_deg": float(np.degrees(ang)),
    }
    return transform, info


def _write_alignment_metadata(case_dir: Path, transform: np.ndarray, info: Dict[str, float]) -> None:
    np.save(case_dir / "prealign_transform.npy", np.asarray(transform, dtype=np.float64))
    lines = []
    for k in sorted(info.keys()):
        lines.append(f"{k}: {info[k]}")
    (case_dir / "prealign_info.txt").write_text("\n".join(lines) + "\n")


def _prepare_opening_state(
    reg: RegistrationwOpeningAlignmentwDifferentiableCentreline,
    loop_idx: np.ndarray,
    cut_point: np.ndarray,
    normal_hint: np.ndarray,
    original_source: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    verts = np.asarray(reg.mesh_target.vertices, dtype=np.float64)
    loop_idx = np.unique(np.asarray(loop_idx, dtype=np.int64).reshape(-1))
    if loop_idx.shape[0] < 3:
        raise ValueError(f"Opening loop has too few vertices: {loop_idx.shape[0]}")

    normal = _normalize(normal_hint)
    faces = np.asarray(reg.mesh_target.triangles, dtype=np.int64)
    loop_idx = _order_loop_by_boundary(
        loop_idx=loop_idx,
        faces=faces,
        verts=verts,
        normal=normal,
    )
    loop_idx = _orient_loop_to_normal(loop_idx, verts=verts, normal=normal)
    loop_coords = verts[loop_idx]
    if loop_coords.shape[0] < 3:
        raise ValueError("Opening loop collapsed after sorting.")

    center = np.mean(loop_coords, axis=0)
    axis_u, axis_v = _plane_basis_from_normal(normal)
    rel = loop_coords - center.reshape(1, 3)
    loop_2d = np.stack([rel @ axis_u, rel @ axis_v], axis=1)

    faces_local = _triangulate_polygon_2d_balanced(loop_2d)
    if faces_local.shape[0] < 1:
        faces_local = np.asarray(
            [[0, i, i + 1] for i in range(1, loop_coords.shape[0] - 1)],
            dtype=np.int64,
        )
    if faces_local.shape[0] < 1:
        raise ValueError("Could not triangulate opening loop.")

    reg._reset_opening_state()
    reg.op_v_indices = [loop_idx.tolist()]
    reg.op_v_coords = [loop_coords.copy()]
    reg.op_n_mean = [normal.copy()]
    reg.op_v_normal = [np.repeat(normal.reshape(1, 3), loop_coords.shape[0], axis=0)]
    reg.op_tangent = [normal.copy()]
    reg.op_cut_points = [np.asarray(cut_point, dtype=np.float64).reshape(3)]
    reg.op_rec_v = [loop_coords.copy()]
    reg.op_rec_f = [faces_local.copy()]
    reg.op_rec_v_indices_map = [loop_idx.tolist()]
    reg.op_rec_f_map = [loop_idx[faces_local].astype(np.int64)]
    supervision = _build_opening_supervision(
        loop_coords=loop_coords,
        normal_hint=normal,
        original_source=original_source,
    )
    reg.op_target_rim_v = [np.asarray(supervision["rim_points"], dtype=np.float64)]
    reg.op_target_rec_v = [np.asarray(supervision["target_surface_v"], dtype=np.float64)]
    reg.op_target_rec_f = [np.asarray(supervision["target_surface_f"], dtype=np.int64)]
    reg.op_target_plane_center = [np.asarray(supervision["plane_center"], dtype=np.float64)]
    reg.op_target_plane_normal = [np.asarray(supervision["plane_normal"], dtype=np.float64)]
    reg.op_source_kind = [str(supervision["source_kind"])]
    reg.op_source_surface_v = [np.asarray(supervision["source_surface_v"], dtype=np.float64)]
    reg.op_source_surface_f = [np.asarray(supervision["source_surface_f"], dtype=np.int64)]
    reg.op_target_debug = [dict(supervision["debug"])]


def _save_case_checkpoints(
    reg: RegistrationwOpeningAlignmentwDifferentiableCentreline,
    case_dir: Path,
    opa_name: str,
    cl_name: str,
    cut_point: np.ndarray,
) -> None:
    opa_path = str(case_dir / opa_name)
    cl_path = str(case_dir / cl_name)
    reg.save_checkpoint_opa(opa_path)

    cep_candidates = reg._map_points_to_mesh_vertices_unique(
        np.asarray(cut_point, dtype=np.float64).reshape(1, 3), k=16
    )
    if len(cep_candidates) < 1:
        raise RuntimeError("Could not map ostium centroid to mesh vertices.")
    reg.cep_registration = [int(cep_candidates[0])]
    reg._cast_waves(progress=False)
    reg.save_checkpoint_centreline(cl_path)


def _prepare_canonical(args) -> Dict[str, np.ndarray]:
    canonical_dir = args.canonical_root / args.canonical_name
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_mesh_dst = canonical_dir / "part_aligned.obj"
    if args.overwrite or (not canonical_mesh_dst.exists()):
        shutil.copy2(args.canonical_src, canonical_mesh_dst)

    reg = RegistrationwOpeningAlignmentwDifferentiableCentreline(
        args=SimpleNamespace(device=args.device),
        root=str(args.canonical_root),
        target=args.canonical_name,
        num_op=1,
        num_cep=1,
        step_size=int(args.wave_step_size),
    )
    faces = np.asarray(reg.mesh_target.triangles, dtype=np.int64)
    boundary = _boundary_edges(faces)
    if boundary.shape[0] < 3:
        raise RuntimeError(
            f"Canonical mesh has no usable open boundary ({boundary.shape[0]} edges)."
        )
    comps = _edge_components(boundary)
    if len(comps) < 1:
        raise RuntimeError("Canonical boundary components are empty.")
    loop_idx = comps[0]
    verts = np.asarray(reg.mesh_target.vertices, dtype=np.float64)
    loop_coords = verts[loop_idx]
    cut_point = np.mean(loop_coords, axis=0)
    normal_hint = _estimate_loop_normal(loop_coords)
    _prepare_opening_state(
        reg=reg,
        loop_idx=loop_idx,
        cut_point=cut_point,
        normal_hint=normal_hint,
    )
    _save_case_checkpoints(
        reg=reg,
        case_dir=canonical_dir,
        opa_name=args.opa_name,
        cl_name=args.centreline_name,
        cut_point=cut_point,
    )

    axis = _principal_axis_in_plane(loop_coords, cut_point, normal_hint)
    radius = _loop_radius(loop_coords, cut_point)
    mesh_can = trimesh.load(str(canonical_mesh_dst), process=False)
    verts_can = np.asarray(mesh_can.vertices, dtype=np.float64)
    side_can = float(
        np.dot(
            np.mean(verts_can, axis=0) - np.asarray(cut_point, dtype=np.float64),
            _normalize(normal_hint),
        )
    )
    rng = np.random.default_rng(int(args.seed))
    icp_points = _sample_mesh_points(mesh_can, int(args.prealign_icp_samples), rng=rng)
    return {
        "center": np.asarray(cut_point, dtype=np.float64),
        "normal": np.asarray(normal_hint, dtype=np.float64),
        "axis": np.asarray(axis, dtype=np.float64),
        "radius": np.asarray(radius, dtype=np.float64),
        "icp_points": np.asarray(icp_points, dtype=np.float64),
        "sac_side_sign": float(_sign_nonzero(side_can)),
    }


def _prepare_single_case(
    source_case_dir: Path,
    target_case_dir: Path,
    args,
    canonical_ref: Dict[str, np.ndarray],
) -> Tuple[bool, str]:
    mesh_src = source_case_dir / "05_submeshes" / "aneurysm_submesh.obj"
    labels_src = source_case_dir / "06_submesh_labels" / "labels_aneurysm.npy"
    centroid_src = source_case_dir / "07_other" / "centroid_ostium.npy"
    normal_src = source_case_dir / "07_other" / "normal_vector.npy"

    required = [mesh_src, labels_src]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        return False, f"missing files: {', '.join(missing)}"

    labels = np.asarray(np.load(labels_src)).reshape(-1)
    loop_idx = np.where(labels == 2)[0].astype(np.int64)
    if loop_idx.shape[0] < 3:
        return False, f"ostium ring too small: {loop_idx.shape[0]} vertices"

    mesh = trimesh.load(str(mesh_src), process=False)
    verts_src = np.asarray(mesh.vertices, dtype=np.float64)
    faces_src = np.asarray(mesh.faces, dtype=np.int64)
    if labels.shape[0] != verts_src.shape[0]:
        return False, f"label/vertex mismatch: labels={labels.shape[0]} verts={verts_src.shape[0]}"

    loop_coords_src = verts_src[loop_idx]
    if centroid_src.is_file():
        center_hint = np.asarray(np.load(centroid_src), dtype=np.float64).reshape(-1)
        if center_hint.shape[0] != 3:
            center_hint = np.mean(loop_coords_src, axis=0)
    else:
        center_hint = np.mean(loop_coords_src, axis=0)
    if normal_src.is_file():
        normal_hint = np.asarray(np.load(normal_src), dtype=np.float64).reshape(-1)
        if normal_hint.shape[0] != 3:
            normal_hint = _estimate_loop_normal(loop_coords_src)
    else:
        normal_hint = _estimate_loop_normal(loop_coords_src)
    center_hint = np.asarray(center_hint, dtype=np.float64).reshape(3)
    normal_hint = _normalize(normal_hint)

    transform = np.eye(4, dtype=np.float64)
    align_info: Dict[str, float] = {}
    src_centroid = np.mean(verts_src, axis=0)
    if args.prealign_mode != "none":
        cand = []
        for sign in (1.0, -1.0):
            src_n = _normalize(sign * normal_hint)
            t_i, info_i = _build_ostium_transform(
                loop_coords_src=loop_coords_src,
                center_src=center_hint,
                normal_src=src_n,
                canonical_ref=canonical_ref,
                scale_ostium=bool(args.prealign_scale_ostium),
            )
            centroid_i = _transform_points(src_centroid.reshape(1, 3), t_i)[0]
            cut_i = _transform_points(center_hint.reshape(1, 3), t_i)[0]
            side_i = float(
                np.dot(
                    centroid_i - cut_i,
                    _normalize(canonical_ref["normal"]),
                )
            )
            cand.append(
                {
                    "sign": float(sign),
                    "transform": t_i,
                    "info": info_i,
                    "side": side_i,
                    "side_sign": _sign_nonzero(side_i),
                }
            )

        desired_side_sign = float(canonical_ref.get("sac_side_sign", 1.0))
        matching = [c for c in cand if c["side_sign"] == desired_side_sign]
        if len(matching) == 1:
            chosen = matching[0]
        elif len(matching) >= 2:
            chosen = max(matching, key=lambda c: abs(float(c["side"])))
        else:
            chosen = max(cand, key=lambda c: abs(float(c["side"])))

        transform = np.asarray(chosen["transform"], dtype=np.float64)
        align_info.update({f"ostium_{k}": float(v) for k, v in chosen["info"].items()})
        align_info["normal_sign_choice"] = float(chosen["sign"])
        align_info["sac_side_sign"] = float(chosen["side_sign"])
        align_info["sac_side_abs"] = float(abs(chosen["side"]))

        if bool(args.prealign_icp):
            rng_seed = int(args.seed) + abs(hash(source_case_dir.name)) % 1000000
            rng = np.random.default_rng(rng_seed)
            src_samples = _sample_mesh_points(mesh, int(args.prealign_icp_samples), rng=rng)
            src_samples = _transform_points(src_samples, transform)
            try:
                t_icp, fit, rmse = _icp_refine(
                    src_points=src_samples,
                    dst_points=canonical_ref["icp_points"],
                    max_iter=int(args.prealign_icp_max_iter),
                    distance_threshold=float(args.prealign_icp_distance),
                )
                transform = t_icp @ transform
                align_info["icp_fitness"] = float(fit)
                align_info["icp_rmse"] = float(rmse)
            except Exception as e:
                align_info["icp_failed"] = 1.0
                align_info["icp_error_len"] = float(len(str(e)))

    verts_out = _transform_points(verts_src, transform)
    cut_point = _transform_points(center_hint.reshape(1, 3), transform)[0]
    normal_out = _transform_normal(normal_hint, transform)
    if args.prealign_mode != "none":
        if float(np.dot(normal_out, canonical_ref["normal"])) < 0.0:
            normal_out = -1.0 * normal_out
    mesh_out = trimesh.Trimesh(vertices=verts_out, faces=faces_src, process=False)

    target_case_dir.mkdir(parents=True, exist_ok=True)
    mesh_dst = target_case_dir / "part_aligned.obj"
    mesh_out.export(str(mesh_dst))
    if args.prealign_mode != "none":
        _write_alignment_metadata(target_case_dir, transform=transform, info=align_info)

    reg = RegistrationwOpeningAlignmentwDifferentiableCentreline(
        args=SimpleNamespace(device=args.device),
        root=str(args.target_root),
        target=target_case_dir.name,
        num_op=1,
        num_cep=1,
        step_size=int(args.wave_step_size),
    )
    original_source = _load_original_ostium_source(
        source_case_dir=source_case_dir,
        transform=transform,
        normal_hint=normal_out,
    )
    _prepare_opening_state(
        reg=reg,
        loop_idx=loop_idx,
        cut_point=cut_point,
        normal_hint=normal_out,
        original_source=original_source,
    )
    _save_case_checkpoints(
        reg=reg,
        case_dir=target_case_dir,
        opa_name=args.opa_name,
        cl_name=args.centreline_name,
        cut_point=cut_point,
    )
    return True, "ok"


def parse_args():
    p = argparse.ArgumentParser(
        description="Prepare prepared_meshes_3 aneurysm-only cases for GHD (num_op=1)."
    )
    p.add_argument(
        "--source-root",
        type=Path,
        default=Path("/data/prepared_meshes_3"),
        help="Input root with prepared_meshes_3 cases.",
    )
    p.add_argument(
        "--target-root",
        type=Path,
        default=Path("/data/ghd_prepared_meshes_3_aneurysm_1op"),
        help="Output root for GHD-ready case folders.",
    )
    p.add_argument(
        "--canonical-src",
        type=Path,
        default=Path("/workspace/AneuG/checkpoints/canonical_average.obj"),
        help="Canonical OBJ source file.",
    )
    p.add_argument(
        "--canonical-root",
        type=Path,
        default=Path("/workspace/AneuG/checkpoints"),
        help="Root containing canonical folder for ghd_fitting.",
    )
    p.add_argument(
        "--canonical-name",
        type=str,
        default="canonical_average",
        help="Canonical folder name expected by ghd_fitting.",
    )
    p.add_argument(
        "--case-glob",
        type=str,
        default="*",
        help="Wildcard filter for case names.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of processed cases (0 = all).",
    )
    p.add_argument(
        "--opa-name",
        type=str,
        default="opa_checkpoint_1op",
        help="Checkpoint base name for opening alignment.",
    )
    p.add_argument(
        "--centreline-name",
        type=str,
        default="diff_centreline_checkpoint_1op",
        help="Checkpoint base name for differentiable centreline.",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for registration helper class (cpu recommended here).",
    )
    p.add_argument(
        "--wave-step-size",
        type=int,
        default=2,
        help="Step size used by _cast_waves.",
    )
    p.add_argument(
        "--prealign-mode",
        type=str,
        default="ostium",
        choices=["none", "ostium"],
        help="Initial pre-alignment mode relative to canonical.",
    )
    p.add_argument(
        "--prealign-scale-ostium",
        type=int,
        default=0,
        help="If 1, apply isotropic scale by canonical/case ostium radius ratio.",
    )
    p.add_argument(
        "--prealign-icp",
        type=int,
        default=0,
        help="If 1, run point-to-point ICP refinement after ostium rigid alignment.",
    )
    p.add_argument(
        "--prealign-icp-samples",
        type=int,
        default=3000,
        help="Sample count per mesh for ICP.",
    )
    p.add_argument(
        "--prealign-icp-max-iter",
        type=int,
        default=40,
        help="Maximum ICP iterations.",
    )
    p.add_argument(
        "--prealign-icp-distance",
        type=float,
        default=0.08,
        help="ICP correspondence distance threshold.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260301,
        help="Random seed for deterministic sampling.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing part_aligned/checkpoint files.",
    )
    p.add_argument(
        "--prepare-canonical-only",
        action="store_true",
        help="Only prepare canonical case, skip target cases.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if not args.source_root.is_dir() and not args.prepare_canonical_only:
        raise FileNotFoundError(f"Source root not found: {args.source_root}")
    if not args.canonical_src.is_file():
        raise FileNotFoundError(f"Canonical source OBJ not found: {args.canonical_src}")

    print(f"Preparing canonical case '{args.canonical_name}' ...")
    canonical_ref = _prepare_canonical(args)
    print("Canonical preparation done.")

    if args.prepare_canonical_only:
        return

    args.target_root.mkdir(parents=True, exist_ok=True)
    case_dirs = sorted(
        [
            d
            for d in args.source_root.iterdir()
            if d.is_dir() and fnmatch.fnmatch(d.name, args.case_glob)
        ]
    )
    if args.limit > 0:
        case_dirs = case_dirs[: int(args.limit)]
    print(f"Found {len(case_dirs)} case(s) to process.")

    ok = 0
    fail = 0
    skipped = 0
    for i, src_case in enumerate(case_dirs, 1):
        dst_case = args.target_root / src_case.name
        opa_file = dst_case / (
            args.opa_name if args.opa_name.endswith(".pkl") else f"{args.opa_name}.pkl"
        )
        cl_file = dst_case / (
            args.centreline_name
            if args.centreline_name.endswith(".pkl")
            else f"{args.centreline_name}.pkl"
        )
        mesh_file = dst_case / "part_aligned.obj"
        if (
            (not args.overwrite)
            and mesh_file.exists()
            and opa_file.exists()
            and cl_file.exists()
        ):
            skipped += 1
            print(f"[{i}/{len(case_dirs)}] {src_case.name}: skip (already prepared)")
            continue
        try:
            done, msg = _prepare_single_case(
                source_case_dir=src_case,
                target_case_dir=dst_case,
                args=args,
                canonical_ref=canonical_ref,
            )
            if done:
                ok += 1
                print(f"[{i}/{len(case_dirs)}] {src_case.name}: ok")
            else:
                fail += 1
                print(f"[{i}/{len(case_dirs)}] {src_case.name}: FAIL ({msg})")
        except Exception as e:
            fail += 1
            print(
                f"[{i}/{len(case_dirs)}] {src_case.name}: "
                f"EXCEPTION ({type(e).__name__}: {e})"
            )

    print(
        f"Done. ok={ok} skipped={skipped} failed={fail} "
        f"target_root={args.target_root}"
    )


if __name__ == "__main__":
    main()
