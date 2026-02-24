#!/usr/bin/env python3
"""
Batch-render manual-vs-auto registration comparison images per case.

For each case directory in `--case-root`, the script:
1) loads manual checkpoints (`opa_checkpoint.pkl`, `diff_centreline_checkpoint.pkl`)
2) runs automatic opening + centerline registration
3) renders a side-by-side 3D figure (manual | auto)
4) saves one PNG per case into `--output-dir`
"""

from __future__ import annotations

import argparse
import copy
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Ensure repo root (folder containing 'ghd') is importable.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ghd.fitting.registration import RegistrationwOpeningAlignmentwDifferentiableCentreline


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def safe_indices(idx: Sequence[int], n: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return idx
    return idx[(idx >= 0) & (idx < n)]


def opening_surfaces_from_checkpoint(opa_chk: Dict[str, object]) -> List[Tuple[np.ndarray, np.ndarray]]:
    surfaces: List[Tuple[np.ndarray, np.ndarray]] = []
    rec_v = opa_chk.get("op_rec_v", [])
    rec_f = opa_chk.get("op_rec_f", [])
    if not (isinstance(rec_v, list) and isinstance(rec_f, list)):
        return surfaces
    if len(rec_v) != len(rec_f):
        return surfaces
    for v, f in zip(rec_v, rec_f):
        v = np.asarray(v, dtype=np.float64).reshape(-1, 3)
        f = np.asarray(f, dtype=np.int64).reshape(-1, 3)
        if v.shape[0] < 3 or f.shape[0] < 1:
            continue
        valid = np.all((f >= 0) & (f < v.shape[0]), axis=1)
        f = f[valid]
        if f.shape[0] < 1:
            continue
        surfaces.append((v, f))
    return surfaces


def project_vertices_to_mesh(vertices: np.ndarray, mesh_vertices: np.ndarray, mesh_kdtree) -> np.ndarray:
    vertices = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if vertices.shape[0] == 0:
        return vertices
    try:
        _, idx = mesh_kdtree.query(vertices)
        idx = np.asarray(idx, dtype=np.int64).reshape(-1)
        idx = np.clip(idx, 0, mesh_vertices.shape[0] - 1)
        return mesh_vertices[idx]
    except Exception:
        sq = np.sum((vertices[:, None, :] - mesh_vertices[None, :, :]) ** 2, axis=-1)
        idx = np.argmin(sq, axis=1).astype(np.int64)
        return mesh_vertices[idx]


def opening_centers(
    verts: np.ndarray,
    opa_chk: Dict[str, object],
    mode: str = "surfaces",
    mesh_vertices: Optional[np.ndarray] = None,
    mesh_kdtree=None,
) -> np.ndarray:
    centers: List[np.ndarray] = []
    surfaces = opening_surfaces_from_checkpoint(opa_chk)
    if mode != "indices" and len(surfaces) > 0:
        for v, _ in surfaces:
            if mode == "projected_surfaces" and mesh_vertices is not None and mesh_kdtree is not None:
                v_used = project_vertices_to_mesh(v, mesh_vertices, mesh_kdtree)
            else:
                v_used = v
            centers.append(np.mean(v_used, axis=0))
        return np.vstack(centers)

    for idx in opa_chk.get("op_v_indices", []):
        idx = safe_indices(idx, len(verts))
        if idx.size == 0:
            continue
        centers.append(np.mean(verts[idx], axis=0))
    if len(centers) == 0:
        return np.zeros((0, 3), dtype=np.float64)
    return np.vstack(centers)


def get_cep_points(verts: np.ndarray, cl_chk: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray]:
    cep_idx = safe_indices(cl_chk.get("diff_cep_registration", []), len(verts))
    if cep_idx.size == 0:
        return cep_idx, np.zeros((0, 3), dtype=np.float64)
    return cep_idx, verts[cep_idx]


def centerline_polylines_from_paths(verts: np.ndarray, paths: Optional[Sequence[Sequence[int]]]) -> List[np.ndarray]:
    polylines: List[np.ndarray] = []
    if paths is None:
        return polylines
    for path in paths:
        idx = safe_indices(path, len(verts))
        if idx.size >= 2:
            polylines.append(verts[idx])
    return polylines


def centerline_polylines_from_endpoints(
    reg: RegistrationwOpeningAlignmentwDifferentiableCentreline,
    endpoint_idx: Sequence[int],
) -> Tuple[List[np.ndarray], np.ndarray, Optional[int]]:
    verts = np.asarray(reg.mesh_target.vertices)
    endpoint_idx = safe_indices(endpoint_idx, len(verts)).tolist()
    if len(endpoint_idx) < 2:
        return [], np.asarray(endpoint_idx, dtype=np.int64), None

    center_idx = reg._estimate_bifurcation_index(endpoint_idx)
    endpoint_sorted = reg._sort_endpoint_indices(endpoint_idx, center_idx)
    paths = reg._branch_paths_from_endpoints(endpoint_sorted, center_idx)

    polylines: List[np.ndarray] = []
    for path in paths:
        idx = safe_indices(path, len(verts))
        if idx.size >= 2:
            polylines.append(verts[idx])
    return polylines, np.asarray(endpoint_sorted, dtype=np.int64), int(center_idx)


def build_auto_dicts(
    reg_auto: RegistrationwOpeningAlignmentwDifferentiableCentreline,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    auto_opa = {
        "op_v_indices": copy.deepcopy(reg_auto.op_v_indices),
        "op_v_coords": copy.deepcopy(reg_auto.op_v_coords),
        "op_v_normal": copy.deepcopy(reg_auto.op_v_normal),
        "op_n_mean": copy.deepcopy(reg_auto.op_n_mean),
        "op_rec_v": copy.deepcopy(reg_auto.op_rec_v),
        "op_rec_f": copy.deepcopy(reg_auto.op_rec_f),
        "op_rec_v_indices_map": copy.deepcopy(reg_auto.op_rec_v_indices_map),
        "op_rec_f_map": copy.deepcopy(reg_auto.op_rec_f_map),
        "op_tangent": copy.deepcopy(getattr(reg_auto, "op_tangent", [])),
        "op_cut_points": copy.deepcopy(getattr(reg_auto, "op_cut_points", [])),
    }
    auto_cl = {
        "diff_cep_registration": copy.deepcopy(reg_auto.cep_registration),
        "wave_loops": copy.deepcopy(reg_auto.wave_loops),
        "centreline_pcd": copy.deepcopy(getattr(reg_auto, "centreline_pcd", None)),
        "centreline_branch_paths": copy.deepcopy(getattr(reg_auto, "centreline_branch_paths", None)),
        "centreline_tangent": copy.deepcopy(getattr(reg_auto, "centreline_tangent", None)),
    }
    return auto_opa, auto_cl


def compute_auto_registration(
    case_root: Path,
    case_name: str,
    num_openings: int,
    num_cep: int,
    step_size: int,
    min_loop_vertices: int,
    normal_dot_min: float,
    face_dot_min: float,
    device: str,
) -> Tuple[RegistrationwOpeningAlignmentwDifferentiableCentreline, Dict[str, object], Dict[str, object]]:
    args = SimpleNamespace(device=device)
    reg_auto = RegistrationwOpeningAlignmentwDifferentiableCentreline(
        args=args,
        root=str(case_root),
        target=case_name,
        num_op=int(num_openings),
        num_cep=int(num_cep),
        step_size=int(step_size),
    )
    reg_auto.register_openings_auto_normals(
        min_loop_vertices=int(min_loop_vertices),
        normal_dot_min=float(normal_dot_min),
        face_dot_min=float(face_dot_min),
    )
    reg_auto.create_opening_meshes(viz=False)
    reg_auto.register_centreline_end_points(auto=True)
    reg_auto._cast_waves(progress=False)
    auto_opa, auto_cl = build_auto_dicts(reg_auto)
    return reg_auto, auto_opa, auto_cl


def set_equal_axes(ax, verts: np.ndarray) -> None:
    mins = np.min(verts, axis=0)
    maxs = np.max(verts, axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half = 0.55 * span if span > 0 else 1.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def add_mesh(ax, verts: np.ndarray, faces: np.ndarray, color: str = "lightgray", alpha: float = 0.10) -> None:
    tris = verts[faces]
    coll = Poly3DCollection(tris, facecolors=color, edgecolors="none", alpha=alpha)
    ax.add_collection3d(coll)


def add_opening_surfaces(
    ax,
    verts: np.ndarray,
    opa_chk: Dict[str, object],
    color: str,
    alpha: float = 0.42,
    mode: str = "surfaces",
    mesh_vertices: Optional[np.ndarray] = None,
    mesh_kdtree=None,
) -> None:
    surfaces = opening_surfaces_from_checkpoint(opa_chk)
    if mode != "indices" and len(surfaces) > 0:
        for v, f in surfaces:
            if mode == "projected_surfaces" and mesh_vertices is not None and mesh_kdtree is not None:
                v_used = project_vertices_to_mesh(v, mesh_vertices, mesh_kdtree)
            else:
                v_used = v
            tris = v_used[f]
            coll = Poly3DCollection(tris, facecolors=color, edgecolors="none", alpha=alpha)
            ax.add_collection3d(coll)
        return
    for idx in opa_chk.get("op_v_indices", []):
        idx = safe_indices(idx, len(verts))
        if idx.size == 0:
            continue
        xyz = verts[idx]
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=3, c=color, alpha=0.95)


def add_registration_panel(
    ax,
    verts: np.ndarray,
    faces: np.ndarray,
    opa_chk: Dict[str, object],
    cl_chk: Dict[str, object],
    centerline_polylines: Sequence[np.ndarray],
    title: str,
    opening_color: str,
    centerline_color: str,
    endpoint_color: str,
    bifurcation_idx: Optional[int],
    opening_mode: str = "surfaces",
    opening_alpha: float = 0.45,
    mesh_vertices: Optional[np.ndarray] = None,
    mesh_kdtree=None,
) -> None:
    add_mesh(ax, verts, faces, color="lightgray", alpha=0.12)
    add_opening_surfaces(
        ax,
        verts,
        opa_chk,
        color=opening_color,
        alpha=float(opening_alpha),
        mode=opening_mode,
        mesh_vertices=mesh_vertices,
        mesh_kdtree=mesh_kdtree,
    )

    centers = opening_centers(
        verts,
        opa_chk,
        mode=opening_mode,
        mesh_vertices=mesh_vertices,
        mesh_kdtree=mesh_kdtree,
    )
    if centers.shape[0] > 0:
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=20, c=opening_color, depthshade=False)

    for xyz in centerline_polylines:
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=centerline_color, linewidth=2.8)

    cep_idx, cep_pts = get_cep_points(verts, cl_chk)
    if cep_pts.shape[0] > 0:
        ax.scatter(cep_pts[:, 0], cep_pts[:, 1], cep_pts[:, 2], s=40, c=endpoint_color, marker="D", depthshade=False)
        for i, p in zip(cep_idx.tolist(), cep_pts):
            ax.text(p[0], p[1], p[2], str(int(i)), fontsize=8, color=endpoint_color)

    if bifurcation_idx is not None and 0 <= int(bifurcation_idx) < len(verts):
        p = verts[int(bifurcation_idx)]
        ax.scatter([p[0]], [p[1]], [p[2]], s=90, c="black", marker="x", depthshade=False)

    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    set_equal_axes(ax, verts)


def render_case_figure(
    case_name: str,
    reg_auto: RegistrationwOpeningAlignmentwDifferentiableCentreline,
    manual_opa: Dict[str, object],
    manual_cl: Dict[str, object],
    auto_opa: Dict[str, object],
    auto_cl: Dict[str, object],
    out_path: Path,
    manual_opening_mode: str = "indices",
    auto_opening_mode: str = "surfaces",
    manual_opening_alpha: float = 0.28,
    auto_opening_alpha: float = 0.45,
) -> None:
    verts = np.asarray(reg_auto.mesh_target.vertices)
    faces = np.asarray(reg_auto.mesh_target.triangles, dtype=np.int64)
    mesh_tri = reg_auto.mesh_target_trimesh
    mesh_vertices = np.asarray(mesh_tri.vertices, dtype=np.float64)
    mesh_kdtree = getattr(mesh_tri, "kdtree", None)

    manual_centerlines, _, manual_bif_idx = centerline_polylines_from_endpoints(
        reg_auto,
        manual_cl.get("diff_cep_registration", []),
    )
    auto_centerlines = centerline_polylines_from_paths(verts, auto_cl.get("centreline_branch_paths", None))
    if len(auto_centerlines) == 0:
        auto_centerlines, auto_eps_sorted, auto_bif_idx = centerline_polylines_from_endpoints(
            reg_auto,
            auto_cl.get("diff_cep_registration", []),
        )
    else:
        auto_eps_sorted = safe_indices(auto_cl.get("diff_cep_registration", []), len(verts))
        auto_bif_idx = (
            reg_auto._estimate_bifurcation_index(auto_eps_sorted.tolist())
            if auto_eps_sorted.size >= 2
            else None
        )

    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    add_registration_panel(
        ax=ax1,
        verts=verts,
        faces=faces,
        opa_chk=manual_opa,
        cl_chk=manual_cl,
        centerline_polylines=manual_centerlines,
        title=f"manual | {case_name}",
        opening_color="orange",
        centerline_color="crimson",
        endpoint_color="darkred",
        bifurcation_idx=manual_bif_idx,
        opening_mode=manual_opening_mode,
        opening_alpha=float(manual_opening_alpha),
        mesh_vertices=mesh_vertices,
        mesh_kdtree=mesh_kdtree,
    )
    add_registration_panel(
        ax=ax2,
        verts=verts,
        faces=faces,
        opa_chk=auto_opa,
        cl_chk=auto_cl,
        centerline_polylines=auto_centerlines,
        title=f"auto | {case_name}",
        opening_color="deepskyblue",
        centerline_color="cyan",
        endpoint_color="navy",
        bifurcation_idx=auto_bif_idx,
        opening_mode=auto_opening_mode,
        opening_alpha=float(auto_opening_alpha),
        mesh_vertices=mesh_vertices,
        mesh_kdtree=mesh_kdtree,
    )

    fig.suptitle(f"Manual vs Auto Registration | {case_name}", fontsize=14)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def find_cases(case_root: Path, case_glob: str) -> List[Path]:
    case_dirs: List[Path] = []
    for d in sorted(case_root.glob(case_glob)):
        if not d.is_dir():
            continue
        if (d / "opa_checkpoint.pkl").exists() and (d / "diff_centreline_checkpoint.pkl").exists():
            case_dirs.append(d)
    return case_dirs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch render manual-vs-auto registration comparison images.")
    p.add_argument("--case-root", type=Path, default=Path("checkpoints/alignment"))
    p.add_argument("--case-glob", type=str, default="*")
    p.add_argument("--output-dir", type=Path, default=Path("outputs/registration_compare_images"))
    p.add_argument("--num-openings", type=int, default=3)
    p.add_argument("--num-cep", type=int, default=3)
    p.add_argument("--step-size", type=int, default=2)
    p.add_argument("--min-loop-vertices", type=int, default=24)
    p.add_argument("--normal-dot-min", type=float, default=0.72)
    p.add_argument("--face-dot-min", type=float, default=0.90)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-cases", type=int, default=0, help="0 means all cases")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--save-auto-checkpoints", action="store_true")
    p.add_argument("--auto-opa-name", type=str, default="opa_checkpoint_auto.pkl")
    p.add_argument("--auto-cl-name", type=str, default="diff_centreline_checkpoint_auto.pkl")
    p.add_argument(
        "--manual-opening-mode",
        type=str,
        default="indices",
        choices=["surfaces", "indices", "projected_surfaces"],
        help="How to draw manual openings.",
    )
    p.add_argument(
        "--auto-opening-mode",
        type=str,
        default="surfaces",
        choices=["surfaces", "indices", "projected_surfaces"],
        help="How to draw auto openings.",
    )
    p.add_argument("--manual-opening-alpha", type=float, default=0.28)
    p.add_argument("--auto-opening-alpha", type=float, default=0.45)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    case_root = args.case_root.resolve()
    output_dir = args.output_dir.resolve()
    if not case_root.exists():
        raise FileNotFoundError(f"Case root not found: {case_root}")

    cases = find_cases(case_root, args.case_glob)
    if args.max_cases > 0:
        cases = cases[: int(args.max_cases)]
    if len(cases) == 0:
        print(f"No matching cases with manual checkpoints found in: {case_root}")
        return 0

    print(f"Found {len(cases)} cases.")
    ok = 0
    failed = 0
    for i, case_dir in enumerate(cases, 1):
        case_name = case_dir.name
        out_path = output_dir / f"{case_name}_manual_vs_auto.png"
        if out_path.exists() and not args.overwrite:
            print(f"[{i}/{len(cases)}] skip existing: {out_path.name}")
            continue

        try:
            manual_opa = load_pickle(case_dir / "opa_checkpoint.pkl")
            manual_cl = load_pickle(case_dir / "diff_centreline_checkpoint.pkl")

            reg_auto, auto_opa, auto_cl = compute_auto_registration(
                case_root=case_root,
                case_name=case_name,
                num_openings=args.num_openings,
                num_cep=args.num_cep,
                step_size=args.step_size,
                min_loop_vertices=args.min_loop_vertices,
                normal_dot_min=args.normal_dot_min,
                face_dot_min=args.face_dot_min,
                device=args.device,
            )

            if args.save_auto_checkpoints:
                with open(case_dir / args.auto_opa_name, "wb") as f:
                    pickle.dump(auto_opa, f)
                with open(case_dir / args.auto_cl_name, "wb") as f:
                    pickle.dump(auto_cl, f)

            render_case_figure(
                case_name=case_name,
                reg_auto=reg_auto,
                manual_opa=manual_opa,
                manual_cl=manual_cl,
                auto_opa=auto_opa,
                auto_cl=auto_cl,
                out_path=out_path,
                manual_opening_mode=args.manual_opening_mode,
                auto_opening_mode=args.auto_opening_mode,
                manual_opening_alpha=args.manual_opening_alpha,
                auto_opening_alpha=args.auto_opening_alpha,
            )
            ok += 1
            dbg = getattr(reg_auto, "auto_registration_debug", {})
            refine_used = dbg.get("synthetic_cut_refinement_used")
            refine_count = None
            if isinstance(refine_used, (list, tuple)):
                refine_count = int(sum(bool(v) for v in refine_used))
            print(
                f"[{i}/{len(cases)}] ok: {case_name} -> {out_path.name} | "
                f"method={dbg.get('method')} mode={dbg.get('selection_mode')} "
                f"sources={dbg.get('selected_sources')} final_sources={dbg.get('final_loop_sources')} "
                f"synthetic_refine={refine_count} fallback={dbg.get('fallback_reason')}"
            )
        except Exception as e:
            failed += 1
            print(f"[{i}/{len(cases)}] FAILED: {case_name} | {type(e).__name__}: {e}")

    print(f"Done. success={ok} failed={failed} output={output_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
