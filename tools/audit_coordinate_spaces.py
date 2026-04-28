#!/usr/bin/env python
"""Audit coordinate spaces for prepared aneurysm, healthy vessel, and GHD alignment."""
import argparse
import io
import json
import os
import pickle

import numpy as np
import torch
import trimesh


class _TorchCPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def _mesh_stats(path, merge=False):
    mesh = trimesh.load(path, process=False)
    if merge:
        mesh.merge_vertices(digits_vertex=8, merge_tex=True, merge_norm=True)
        mesh.remove_unreferenced_vertices()
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    edges = mesh.edges_unique
    counts = np.bincount(mesh.edges_unique_inverse)
    boundary_edges = int((counts == 1).sum())
    return {
        "path": path,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "bounds_min": verts.min(axis=0).tolist(),
        "bounds_max": verts.max(axis=0).tolist(),
        "centroid": verts.mean(axis=0).tolist(),
        "watertight": bool(mesh.is_watertight),
        "boundary_edges": boundary_edges,
    }


def _apply_h(points, transform):
    points = np.asarray(points, dtype=np.float64)
    return points @ transform[:3, :3].T + transform[:3, 3]


def _canonical_norm(canonical_mesh, factor):
    mesh = trimesh.load(canonical_mesh, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return float(np.linalg.norm(np.asarray(mesh.vertices), axis=1).max() * factor)


def _load_pickle(path):
    with open(path, "rb") as f:
        return _TorchCPUUnpickler(f).load()


def parse_args():
    p = argparse.ArgumentParser("audit_coordinate_spaces")
    p.add_argument("--case", required=True)
    p.add_argument("--prepared_root", default="/data/prepared_meshes_3")
    p.add_argument("--healthy_root", default="/data/healthy_vessel")
    p.add_argument("--aligned_root", default="/data/ghd_prepared_meshes_3_aneurysm_1op_new")
    p.add_argument("--ghd_chk_root", default="/workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999")
    p.add_argument("--ghd_run", default="prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3")
    p.add_argument("--ghd_chk_name", default="ghb_fitting_checkpoint.pkl")
    p.add_argument("--canonical_mesh", default="/workspace/AneuG/checkpoints/canonical_average/part_aligned.obj")
    p.add_argument("--canonical_norm_factor", type=float, default=1.10)
    p.add_argument("--out_json", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    case = args.case
    prepared_case = os.path.join(args.prepared_root, case)
    healthy_mesh = os.path.join(
        args.healthy_root,
        f"{case}_vessel_submesh_closed",
        f"{case}_vessel_submesh_closed.obj",
    )
    aligned_case = os.path.join(args.aligned_root, case)
    ghd_chk = os.path.join(args.ghd_chk_root, case, args.ghd_run, args.ghd_chk_name)

    paths = {
        "prepared_full": os.path.join(prepared_case, "01_mesh", "mesh.obj"),
        "prepared_vessel": os.path.join(prepared_case, "05_submeshes", "vessel_submesh.obj"),
        "prepared_aneurysm": os.path.join(prepared_case, "05_submeshes", "aneurysm_submesh.obj"),
        "healthy_vessel_raw": healthy_mesh,
        "healthy_vessel_merged": healthy_mesh,
        "aligned_aneurysm": os.path.join(aligned_case, "part_aligned.obj"),
    }

    report = {"case": case, "meshes": {}}
    for name, path in paths.items():
        if os.path.exists(path):
            report["meshes"][name] = _mesh_stats(path, merge=(name == "healthy_vessel_merged"))

    centroid_path = os.path.join(prepared_case, "07_other", "centroid_ostium.npy")
    normal_path = os.path.join(prepared_case, "07_other", "normal_vector.npy")
    transform_path = os.path.join(aligned_case, "prealign_transform.npy")
    if os.path.exists(centroid_path) and os.path.exists(normal_path):
        centroid = np.load(centroid_path).astype(np.float64).reshape(3)
        normal = np.load(normal_path).astype(np.float64).reshape(3)
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        report["ostium_raw"] = {
            "centroid": centroid.tolist(),
            "normal": normal.tolist(),
        }
        if os.path.exists(transform_path):
            transform = np.load(transform_path).astype(np.float64)
            report["prealign_transform"] = transform.tolist()
            report["ostium_aligned"] = {
                "centroid": _apply_h(centroid[None], transform)[0].tolist(),
                "normal": (normal @ transform[:3, :3].T).tolist(),
            }

    if os.path.exists(ghd_chk):
        chk = _load_pickle(ghd_chk)
        report["ghd_checkpoint"] = {
            "path": ghd_chk,
            "R_axis_angle": np.asarray(chk["R"]).reshape(-1).astype(float).tolist(),
            "s": np.asarray(chk["s"]).reshape(-1).astype(float).tolist(),
            "T": np.asarray(chk["T"]).reshape(-1).astype(float).tolist(),
            "canonical_norm": _canonical_norm(args.canonical_mesh, args.canonical_norm_factor),
        }

    if "prepared_vessel" in report["meshes"] and "healthy_vessel_merged" in report["meshes"]:
        pv = np.array(report["meshes"]["prepared_vessel"]["centroid"])
        hv = np.array(report["meshes"]["healthy_vessel_merged"]["centroid"])
        report["prepared_vs_healthy_centroid_delta"] = (hv - pv).tolist()

    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
