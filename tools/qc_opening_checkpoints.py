#!/usr/bin/env python3
"""
QC scan for opening checkpoints in prepared aneurysm cases.

The script ranks cases by a few geometric warning signals that correlate well
with problematic target openings during fitting previews:
- fan_ratio: one vertex participates in too many opening faces
- planarity_rel: opening rim deviates strongly from a best-fit plane
- rim_edge_cv: rim edge lengths are irregular
 - min_tri_quality: sliver triangles in the opening cap
"""

import argparse
import csv
import fnmatch
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if v.size != 3 or n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n


def _plane_basis_from_normal(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    normal = _normalize(normal)
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, normal))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis_u = _normalize(np.cross(normal, ref))
    axis_v = _normalize(np.cross(normal, axis_u))
    return axis_u, axis_v


def _signed_area_2d(poly: np.ndarray) -> float:
    poly = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _triangle_quality(tris: np.ndarray) -> np.ndarray:
    if tris.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    a = np.linalg.norm(tris[:, 1] - tris[:, 0], axis=1)
    b = np.linalg.norm(tris[:, 2] - tris[:, 1], axis=1)
    c = np.linalg.norm(tris[:, 0] - tris[:, 2], axis=1)
    area = 0.5 * np.linalg.norm(np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]), axis=1)
    denom = a * a + b * b + c * c + 1e-12
    # 1.0 = equilateral, ~0 = sliver/degenerate.
    return (4.0 * math.sqrt(3.0) * area) / denom


def _opening_metrics(verts: np.ndarray, faces: np.ndarray, normal: np.ndarray, rim_points: np.ndarray = None) -> Dict[str, float]:
    verts = np.asarray(verts, dtype=np.float64).reshape(-1, 3)
    faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    valid = np.all((faces >= 0) & (faces < verts.shape[0]), axis=1)
    faces = faces[valid]
    if verts.shape[0] < 3 or faces.shape[0] < 1:
        raise ValueError("Opening mesh is empty or invalid.")
    if rim_points is None:
        rim_points = verts
    rim_points = np.asarray(rim_points, dtype=np.float64).reshape(-1, 3)
    if rim_points.shape[0] < 3:
        rim_points = verts

    counts = np.bincount(faces.reshape(-1), minlength=verts.shape[0])
    fan_ratio = float(counts.max()) / max(int(faces.shape[0]), 1)

    center = rim_points.mean(axis=0)
    X = rim_points - center.reshape(1, 3)
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    plane_normal = _normalize(vh[-1])
    planarity_rms = float(np.sqrt(np.mean((X @ plane_normal) ** 2)))

    axis_u, axis_v = _plane_basis_from_normal(normal)
    poly_2d = np.stack([X @ axis_u, X @ axis_v], axis=1)
    area_2d = abs(_signed_area_2d(poly_2d))
    rim_radius = math.sqrt(area_2d / math.pi) if area_2d > 1e-12 else 1.0
    planarity_rel = planarity_rms / max(rim_radius, 1e-6)

    ring = np.concatenate([rim_points, rim_points[:1]], axis=0)
    rim_edge_lengths = np.linalg.norm(ring[1:] - ring[:-1], axis=1)
    rim_edge_cv = float(rim_edge_lengths.std() / (rim_edge_lengths.mean() + 1e-12))

    tris = verts[faces]
    tri_quality = _triangle_quality(tris)
    min_tri_quality = float(np.min(tri_quality)) if tri_quality.size > 0 else 0.0
    mean_tri_quality = float(np.mean(tri_quality)) if tri_quality.size > 0 else 0.0

    # Higher score = more suspicious.
    score = (
        3.0 * max(fan_ratio - 0.18, 0.0)
        + 4.0 * max(planarity_rel - 0.05, 0.0)
        + 1.5 * max(rim_edge_cv - 0.18, 0.0)
        + 2.0 * max(0.20 - min_tri_quality, 0.0)
    )

    return {
        "fan_ratio": fan_ratio,
        "planarity_rms": planarity_rms,
        "planarity_rel": planarity_rel,
        "rim_edge_cv": rim_edge_cv,
        "min_tri_quality": min_tri_quality,
        "mean_tri_quality": mean_tri_quality,
        "area_2d": area_2d,
        "num_vertices": int(verts.shape[0]),
        "num_faces": int(faces.shape[0]),
        "score": score,
    }


def _iter_checkpoint_paths(root_target: Path, checkpoint_name: str, case_glob: str) -> Iterable[Path]:
    ckpt_file = checkpoint_name if checkpoint_name.endswith(".pkl") else f"{checkpoint_name}.pkl"
    for case_dir in sorted(root_target.iterdir()):
        if not case_dir.is_dir():
            continue
        if case_dir.name == "canonical_average":
            continue
        if case_glob and case_glob != "*" and not fnmatch.fnmatch(case_dir.name, case_glob):
            continue
        ckpt_path = case_dir / ckpt_file
        if ckpt_path.is_file():
            yield ckpt_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rank prepared cases by suspicious opening geometry.")
    p.add_argument("--root-target", required=True, help="Prepared case root, e.g. /data/ghd_prepared_meshes_3_aneurysm_1op_new")
    p.add_argument("--checkpoint-name", default="opa_checkpoint_1op", help="Opening checkpoint basename without or with .pkl")
    p.add_argument("--case-glob", default="*", help="Wildcard filter for case names")
    p.add_argument("--top-k", type=int, default=40, help="How many suspicious cases to print")
    p.add_argument("--fan-threshold", type=float, default=0.20, help="Warn if fan_ratio exceeds this value")
    p.add_argument("--planarity-threshold", type=float, default=0.05, help="Warn if relative planarity exceeds this value")
    p.add_argument("--rim-cv-threshold", type=float, default=0.18, help="Warn if rim edge CV exceeds this value")
    p.add_argument("--tri-quality-threshold", type=float, default=0.20, help="Warn if min triangle quality falls below this value")
    p.add_argument("--min-flags", type=int, default=1, help="Only write bad cases that trip at least this many flags")
    p.add_argument("--min-score", type=float, default=0.0, help="Only write bad cases whose combined suspiciousness score exceeds this value")
    p.add_argument("--csv-out", default="", help="Optional CSV output path")
    p.add_argument("--bad-cases-out", default="", help="Optional text file with one flagged case name per line")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root_target = Path(args.root_target)
    if not root_target.is_dir():
        raise FileNotFoundError(f"Target root not found: {root_target}")

    rows: List[Dict[str, float]] = []
    for ckpt_path in _iter_checkpoint_paths(root_target, args.checkpoint_name, args.case_glob):
        case_name = ckpt_path.parent.name
        with ckpt_path.open("rb") as f:
            chk = pickle.load(f)

        rec_v = chk.get("op_target_rec_v", []) or chk.get("op_rec_v", [])
        rec_f = chk.get("op_target_rec_f", []) or chk.get("op_rec_f", [])
        normals = chk.get("op_target_plane_normal", []) or chk.get("op_n_mean", [])
        rim_v = chk.get("op_target_rim_v", []) or chk.get("op_v_coords", [])
        if not rec_v or not rec_f or not normals:
            continue

        try:
            metrics = _opening_metrics(
                verts=np.asarray(rec_v[0], dtype=np.float64),
                faces=np.asarray(rec_f[0], dtype=np.int64),
                normal=np.asarray(normals[0], dtype=np.float64),
                rim_points=np.asarray(rim_v[0], dtype=np.float64) if rim_v else None,
            )
        except Exception as e:
            print(f"skip {case_name}: {type(e).__name__}: {e}")
            continue

        flags = []
        if metrics["fan_ratio"] > args.fan_threshold:
            flags.append("fan")
        if metrics["planarity_rel"] > args.planarity_threshold:
            flags.append("planar")
        if metrics["rim_edge_cv"] > args.rim_cv_threshold:
            flags.append("rim")
        if metrics["min_tri_quality"] < args.tri_quality_threshold:
            flags.append("tri")
        metrics["case"] = case_name
        metrics["flags"] = ",".join(flags) if flags else "-"
        rows.append(metrics)

    rows.sort(key=lambda x: float(x["score"]), reverse=True)

    print(
        "case, score, flags, fan_ratio, planarity_rel, rim_edge_cv, min_tri_quality, "
        "num_vertices, num_faces"
    )
    for row in rows[: max(int(args.top_k), 0)]:
        print(
            f"{row['case']}, "
            f"{row['score']:.4f}, "
            f"{row['flags']}, "
            f"{row['fan_ratio']:.3f}, "
            f"{row['planarity_rel']:.3f}, "
            f"{row['rim_edge_cv']:.3f}, "
            f"{row['min_tri_quality']:.3f}, "
            f"{int(row['num_vertices'])}, "
            f"{int(row['num_faces'])}"
        )

    if args.csv_out:
        out_path = Path(args.csv_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "case",
            "score",
            "flags",
            "fan_ratio",
            "planarity_rms",
            "planarity_rel",
            "rim_edge_cv",
            "min_tri_quality",
            "mean_tri_quality",
            "area_2d",
            "num_vertices",
            "num_faces",
        ]
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        print(f"\nWrote CSV: {out_path}")

    if args.bad_cases_out:
        out_path = Path(args.bad_cases_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        bad_cases = []
        for row in rows:
            flags_str = str(row.get("flags", "-"))
            flag_count = 0 if flags_str == "-" else len([x for x in flags_str.split(",") if x])
            if flag_count < int(args.min_flags):
                continue
            if float(row.get("score", 0.0)) < float(args.min_score):
                continue
            bad_cases.append(str(row["case"]))
        out_path.write_text("".join(f"{name}\n" for name in bad_cases), encoding="utf-8")
        print(f"Wrote bad-case list: {out_path} ({len(bad_cases)} cases)")


if __name__ == "__main__":
    main()
