import argparse
import copy
from ghd.fitting.fitter import fit_ghd
import os
import logging
import pickle
import fnmatch
from ghd.fitting.registration import RegistrationwOpeningAlignmentwDifferentiableCentreline
import torch
import random
import yaml
import numpy as np
from typing import List, Optional, Sequence
from types import SimpleNamespace

# conf
DEFAULTS = {
    "register": True,
    "chk_num": 4,  # number of checkpoints during fitting
    "epochs": 15000,
    "device": "cuda:0",
    "root_template": "./checkpoints/alignment_male_medium",
    "root_target": "./checkpoints/alignment_male_medium",
    "name_canonical": "canonical_typeB",
    "name_target": "AN213_full_clean",
    "case_glob": "*",
    "viz_freq": 200,
    "chk_freq": None,
    "log_freq": 100,
    "lr": 0.00075,
    "lr_scheduler": "step",  # step | plateau | none
    "step_size": 2500,
    "gamma": 0.75,
    "plateau_factor": 0.5,
    "plateau_patience": 300,
    "plateau_threshold": 1e-4,
    "plateau_cooldown": 100,
    "min_lr": 1e-6,
    "early_stopping": 0,
    "early_stopping_patience": 1200,
    "early_stopping_min_delta": 1e-5,
    "early_stopping_min_epochs": 2000,
    "num_op": 3,
    "num_Basis": 13 ** 2,
    "mix_lap_weights": [1.0, 0.1, 0.1],
    "sample_num": int(2.5e5),
    "op_sample_num": int(1e3),
    "op_clean_threshold": 0.2,
    "op_bold": 0,
    "save_root": "./checkpoints/ghb_fitting_male_medium",
    "meta": "cut_446_decreasing_centrelineloss_10",
    "num_sp": 2,
    "do_dpi": 4,
    "do_style": "number_control_v2",
    "do_loss_type": "dice_loss_attention",
    "use_do_dropper": 0,
    "attention_max_w": 3.0,
    "attention_smooth": 0.02,
    "do_number": 25000,
    "weighter_style": "strategy_v1_linear",
    "mesh_filename": "part_aligned.obj",
    "redo_registration": 0,
    "auto_opening_method": "normals",  # normals | legacy
    "auto_min_loop_vertices": 24,
    "auto_normal_dot_min": 0.72,
    "auto_face_dot_min": 0.90,
    "opa_checkpoint_name": "opa_checkpoint",
    "centreline_checkpoint_name": "diff_centreline_checkpoint",
    "render_auto_registration": 0,
    "render_auto_registration_only": 0,
    "render_auto_registration_overwrite": 0,
    "render_auto_registration_dir": "./checkpoints/auto_registration_qc",
    "loss_weighting": {
        "loss_do": 1.0,
        "loss_p0": 1.0 * 1,
        "loss_n1": 0.8 * 1,
        "loss_laplacian": 0.1,
        "loss_edge": 0.1,
        "loss_consistency": 0.1,
        "loss_rigid": 100.0,
        "loss_openings_p": 5,
        "loss_openings_n": 0.1,
        "loss_diff_centreline": 10.0,
    },
}


def deep_merge(base, override):
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path):
    if not path:
        return {}
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data or {}


def compute_defaults(cfg):
    cfg = copy.deepcopy(cfg)
    if cfg.get("chk_freq") is None:
        cfg["chk_freq"] = round(cfg["epochs"] / cfg["chk_num"])
    return cfg


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _safe_indices(idx: Sequence[int], n: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    if idx.size == 0:
        return idx
    return idx[(idx >= 0) & (idx < int(n))]


def _opening_surfaces_from_checkpoint(opa_chk: dict):
    surfaces = []
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


def _get_cep_points(verts: np.ndarray, cl_chk: dict):
    cep_idx = _safe_indices(cl_chk.get("diff_cep_registration", []), len(verts))
    if cep_idx.size == 0:
        return cep_idx, np.zeros((0, 3), dtype=np.float64)
    return cep_idx, verts[cep_idx]


def _centerline_polylines_from_paths(verts: np.ndarray, paths: Optional[Sequence[Sequence[int]]]):
    polylines = []
    if paths is None:
        return polylines
    for path in paths:
        idx = _safe_indices(path, len(verts))
        if idx.size >= 2:
            polylines.append(verts[idx])
    return polylines


def _centerline_polylines_from_endpoints(
    reg: RegistrationwOpeningAlignmentwDifferentiableCentreline,
    endpoint_idx: Sequence[int],
):
    verts = np.asarray(reg.mesh_target.vertices)
    endpoint_idx = _safe_indices(endpoint_idx, len(verts)).tolist()
    if len(endpoint_idx) < 2:
        return [], None
    center_idx = reg._estimate_bifurcation_index(endpoint_idx)
    endpoint_sorted = reg._sort_endpoint_indices(endpoint_idx, center_idx)
    paths = reg._branch_paths_from_endpoints(endpoint_sorted, center_idx)
    polylines = []
    for path in paths:
        idx = _safe_indices(path, len(verts))
        if idx.size >= 2:
            polylines.append(verts[idx])
    return polylines, int(center_idx)


def _set_equal_axes(ax, verts: np.ndarray) -> None:
    mins = np.min(verts, axis=0)
    maxs = np.max(verts, axis=0)
    center = 0.5 * (mins + maxs)
    span = float(np.max(maxs - mins))
    half = 0.55 * span if span > 0 else 1.0
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(center[2] - half, center[2] + half)
    ax.set_box_aspect((1.0, 1.0, 1.0))


def _render_auto_registration_case(
    args,
    case_name: str,
    auto_opa_path: str,
    auto_cl_path: str,
    out_path: str,
) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except Exception as e:
        print(f"Skipping render for {case_name}: matplotlib unavailable ({e})")
        return False

    auto_opa = _load_pickle(auto_opa_path)
    auto_cl = _load_pickle(auto_cl_path)

    reg_ctx = RegistrationwOpeningAlignmentwDifferentiableCentreline(
        args=SimpleNamespace(device="cpu"),
        root=args.root_target,
        target=case_name,
        num_op=int(args.num_op),
        num_cep=int(args.num_op),
        step_size=2,
    )
    verts = np.asarray(reg_ctx.mesh_target.vertices)
    faces = np.asarray(reg_ctx.mesh_target.triangles, dtype=np.int64)

    auto_centerlines = _centerline_polylines_from_paths(verts, auto_cl.get("centreline_branch_paths", None))
    if len(auto_centerlines) == 0:
        auto_centerlines, auto_bif_idx = _centerline_polylines_from_endpoints(
            reg_ctx,
            auto_cl.get("diff_cep_registration", []),
        )
    else:
        auto_eps_sorted = _safe_indices(auto_cl.get("diff_cep_registration", []), len(verts))
        auto_bif_idx = (
            reg_ctx._estimate_bifurcation_index(auto_eps_sorted.tolist())
            if auto_eps_sorted.size >= 2
            else None
        )

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    tris = verts[faces]
    coll = Poly3DCollection(tris, facecolors="lightgray", edgecolors="none", alpha=0.12)
    ax.add_collection3d(coll)

    for v, f in _opening_surfaces_from_checkpoint(auto_opa):
        tris = v[f]
        coll = Poly3DCollection(tris, facecolors="deepskyblue", edgecolors="none", alpha=0.45)
        ax.add_collection3d(coll)

    for xyz in auto_centerlines:
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="cyan", linewidth=2.8)

    cep_idx, cep_pts = _get_cep_points(verts, auto_cl)
    if cep_pts.shape[0] > 0:
        ax.scatter(cep_pts[:, 0], cep_pts[:, 1], cep_pts[:, 2], s=40, c="navy", marker="D", depthshade=False)
        for i, p in zip(cep_idx.tolist(), cep_pts):
            ax.text(p[0], p[1], p[2], str(int(i)), fontsize=8, color="navy")
    if auto_bif_idx is not None and 0 <= int(auto_bif_idx) < len(verts):
        p = verts[int(auto_bif_idx)]
        ax.scatter([p[0]], [p[1]], [p[2]], s=90, c="black", marker="x", depthshade=False)

    ax.set_title(f"auto | {case_name}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    _set_equal_axes(ax, verts)

    fig.suptitle(f"Auto Registration | {case_name}", fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return True


def render_auto_registration_qc(args, labels: List[str]) -> None:
    out_dir = os.path.abspath(str(getattr(args, "render_auto_registration_dir", "./checkpoints/auto_registration_qc")))
    overwrite = bool(getattr(args, "render_auto_registration_overwrite", 0))
    ck_opa = str(getattr(args, "opa_checkpoint_name", "opa_checkpoint"))
    ck_cl = str(getattr(args, "centreline_checkpoint_name", "diff_centreline_checkpoint"))
    ck_opa_file = ck_opa if ck_opa.endswith(".pkl") else f"{ck_opa}.pkl"
    ck_cl_file = ck_cl if ck_cl.endswith(".pkl") else f"{ck_cl}.pkl"

    ok = 0
    skipped = 0
    failed = 0
    for i, label in enumerate(labels, 1):
        case_dir = os.path.join(args.root_target, label)
        auto_opa_path = os.path.join(case_dir, ck_opa_file)
        auto_cl_path = os.path.join(case_dir, ck_cl_file)
        out_path = os.path.join(out_dir, f"{label}_auto_only.png")

        if not os.path.exists(auto_opa_path) or not os.path.exists(auto_cl_path):
            skipped += 1
            print(f"[render {i}/{len(labels)}] skip {label}: missing auto checkpoints")
            continue
        if os.path.exists(out_path) and not overwrite:
            skipped += 1
            print(f"[render {i}/{len(labels)}] skip existing: {os.path.basename(out_path)}")
            continue

        try:
            done = _render_auto_registration_case(
                args=args,
                case_name=label,
                auto_opa_path=auto_opa_path,
                auto_cl_path=auto_cl_path,
                out_path=out_path,
            )
            if done:
                ok += 1
                print(f"[render {i}/{len(labels)}] ok: {label} -> {out_path}")
            else:
                failed += 1
                print(f"[render {i}/{len(labels)}] FAILED: {label}")
        except Exception as e:
            failed += 1
            print(f"[render {i}/{len(labels)}] FAILED: {label} | {type(e).__name__}: {e}")
    print(f"Auto-registration render done. ok={ok} skipped={skipped} failed={failed} output={out_dir}")


parser = argparse.ArgumentParser("ghb_fitting_oa")
parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
parser.add_argument("--register", type=int, default=1)
parser.add_argument("--chk_num", type=int, default=DEFAULTS["chk_num"])
parser.add_argument('--device', type=str, default=DEFAULTS["device"]) # device to use for fitting
parser.add_argument('--root_template', type=str, default=DEFAULTS["root_template"]) # root directory for template meshes, should contain a subdirectory named as args.name_canonical
parser.add_argument('--root_target', type=str, default=DEFAULTS["root_target"]) # root directory for target meshes, should contain subdirectories named as args.name_target, each of which contains the registered mesh and the cleaned centreline for the corresponding case. The registered mesh should be named "mesh_target_p3d.pt", and the cleaned centreline should be named "centreline_clean.pt". If registration is not performed, these files will be generated during registration.
parser.add_argument('--name_canonical', type=str, default=DEFAULTS["name_canonical"]) # name of the canonical template mesh, should be a subdirectory of args.root_template, and should contain a file named "mesh_template_p3d.pt" for the template mesh. If registration is not performed, this file will be generated during registration.
parser.add_argument('--name_target', type=str, default=DEFAULTS["name_target"]) # name of the target mesh, should be a subdirectory of args.root_target, and should contain the registered mesh and the cleaned centreline for the corresponding case if registration is performed. If registration is not performed, these files will be generated during registration.
parser.add_argument('--case_glob', type=str, default=DEFAULTS["case_glob"]) # wildcard filter for case folders (e.g. p429*)
parser.add_argument('--viz_freq', type=int, default=DEFAULTS["viz_freq"]) # frequency for visualization during fitting, in number of epochs
parser.add_argument('--chk_freq', type=int, default=None) # frequency for saving checkpoints during fitting, in number of epochs. Should be set such that epochs / chk_freq = chk_num, i.e., the total number of checkpoints saved during fitting is equal to chk_num.
parser.add_argument('--log_freq', type=int, default=DEFAULTS["log_freq"]) # frequency for logging during fitting, in number of epochs
parser.add_argument('--lr', type=float, default=DEFAULTS["lr"]) # learning rate for fitting, can be adjusted based on the scale of the meshes and the number of epochs. A smaller learning rate may be needed for larger meshes or more epochs to ensure stable convergence.
parser.add_argument('--lr_scheduler', type=str, default=DEFAULTS["lr_scheduler"]) # step | plateau | none
parser.add_argument('--step_size', type=int, default=DEFAULTS["step_size"]) # StepLR step size
parser.add_argument('--gamma', type=float, default=DEFAULTS["gamma"]) # StepLR gamma
parser.add_argument('--plateau_factor', type=float, default=DEFAULTS["plateau_factor"]) # ReduceLROnPlateau factor
parser.add_argument('--plateau_patience', type=int, default=DEFAULTS["plateau_patience"]) # ReduceLROnPlateau patience (epochs)
parser.add_argument('--plateau_threshold', type=float, default=DEFAULTS["plateau_threshold"]) # ReduceLROnPlateau threshold
parser.add_argument('--plateau_cooldown', type=int, default=DEFAULTS["plateau_cooldown"]) # ReduceLROnPlateau cooldown (epochs)
parser.add_argument('--min_lr', type=float, default=DEFAULTS["min_lr"]) # minimum learning rate
parser.add_argument('--early_stopping', type=int, default=DEFAULTS["early_stopping"]) # 1 enables early stopping
parser.add_argument('--early_stopping_patience', type=int, default=DEFAULTS["early_stopping_patience"]) # epochs with no sufficient improvement
parser.add_argument('--early_stopping_min_delta', type=float, default=DEFAULTS["early_stopping_min_delta"]) # minimal improvement to reset patience
parser.add_argument('--early_stopping_min_epochs', type=int, default=DEFAULTS["early_stopping_min_epochs"]) # warmup epochs before early stopping can trigger
parser.add_argument('--num_op', type=int, default=DEFAULTS["num_op"]) # number of operators to use for fitting, can be adjusted based on the complexity of the meshes and the desired level of detail in the fitting. More operators may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the loss weights.
parser.add_argument('--num_Basis', type=int, default=DEFAULTS["num_Basis"]) # number of basis functions to use for fitting, can be adjusted based on the desired level of detail in the fitting.
parser.add_argument('--mix_lap_weights', type=list, default=DEFAULTS["mix_lap_weights"]) # weights for the mixed Laplacian regularization terms, can be adjusted to balance the smoothness and fidelity of the fitting.
parser.add_argument('--sample_num', type=int, default=DEFAULTS["sample_num"]) # number of samples to use for fitting, can be adjusted based on the scale of the meshes and the desired level of detail in the fitting.
parser.add_argument('--op_sample_num', type=int, default=DEFAULTS["op_sample_num"]) # number of operator samples to use for fitting, can be adjusted based on the complexity of the meshes and the desired level of detail in the fitting.
parser.add_argument('--op_clean_threshold', type=float, default=DEFAULTS["op_clean_threshold"]) # threshold for cleaning operator samples, can be adjusted based on the desired level of detail in the fitting.
parser.add_argument('--op_bold', type=int, default=DEFAULTS["op_bold"])  # confidence that trimesh offers uniform mesh normal directions
parser.add_argument('--save_root', type=str, default=DEFAULTS["save_root"]) # root directory for saving fitting results, should contain subdirectories named as args.name_target, each of which contains the fitting results for the corresponding case. The fitting results will be saved in a subdirectory named as args.meta under each target subdirectory, and the checkpoint for each case will be saved as "ghd_fitting_checkpoint.pkl" under the corresponding meta subdirectory.
parser.add_argument('--meta', type=str, default=DEFAULTS["meta"]) # meta name for saving fitting results, can be adjusted to distinguish different fitting settings. The fitting results will be saved in a subdirectory named as args.meta under each target subdirectory.
parser.add_argument('--epochs', type=int, default=DEFAULTS["epochs"]) # number of epochs for fitting, can be adjusted based on the desired level of detail in the fitting and the computational resources available. More epochs may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the learning rate and loss weights.
parser.add_argument('--num_sp', type=int, default=DEFAULTS["num_sp"]) # number of spectral components to use for fitting, can be adjusted based on the desired level of detail in the fitting. More spectral components may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the loss weights.
parser.add_argument('--do_dpi', type=int, default=DEFAULTS["do_dpi"]) # number of iterations for the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the computational resources available. More iterations may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the learning rate and loss weights.
parser.add_argument('--do_style', type=str, default=DEFAULTS["do_style"]) # style for the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. Different styles may use different loss functions and optimization strategies for the registration, and may require different tuning of the learning rate and loss weights.
parser.add_argument('--do_loss_type', type=str, default=DEFAULTS["do_loss_type"]) # loss type for the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. Different loss types may have different sensitivities to outliers and may require different tuning of the learning rate and loss weights.
parser.add_argument('--use_do_dropper', type=int, default=DEFAULTS["use_do_dropper"]) # whether to use a dropper for the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. Using a dropper may help to improve the robustness of the registration by reducing the influence of outliers, but may also require more careful tuning of the learning rate and loss weights.
parser.add_argument('--attention_max_w', type=float, default=DEFAULTS["attention_max_w"]) # maximum weight for the attention mechanism in the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. A larger maximum weight may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the learning rate and loss weights.
parser.add_argument('--attention_smooth', type=float, default=DEFAULTS["attention_smooth"]) # smoothing factor for the attention mechanism in the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. A larger smoothing factor may help to improve the robustness of the registration by reducing the influence of outliers, but may also require more careful tuning of the learning rate and loss weights.
parser.add_argument('--do_number', type=int, default=DEFAULTS["do_number"]) # number of points to use for the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the computational resources available. More points may allow for a better fit but may also increase the risk of overfitting and require more careful tuning of the learning rate and loss weights.
parser.add_argument('--weighter_style', type=str, default=DEFAULTS["weighter_style"]) # style for the weighter in the differentiable point cloud registration, can be adjusted based on the desired level of detail in the fitting and the characteristics of the meshes. Different styles may use different weighting strategies and may require different tuning of the learning rate and loss weights.
parser.add_argument('--mesh_filename', type=str, default=DEFAULTS["mesh_filename"]) # mesh filename inside each target subfolder (e.g., part_aligned.obj). If not found, falls back to <root>/<label>.obj
parser.add_argument('--redo_registration', type=int, default=DEFAULTS["redo_registration"]) # force regeneration of opening/centreline checkpoints
parser.add_argument('--auto_opening_method', type=str, default=DEFAULTS["auto_opening_method"]) # normals | legacy
parser.add_argument('--auto_min_loop_vertices', type=int, default=DEFAULTS["auto_min_loop_vertices"]) # min vertices for automatic opening loops
parser.add_argument('--auto_normal_dot_min', type=float, default=DEFAULTS["auto_normal_dot_min"]) # normal threshold for normals-based auto registration
parser.add_argument('--auto_face_dot_min', type=float, default=DEFAULTS["auto_face_dot_min"]) # face threshold for normals-based auto registration
parser.add_argument('--opa_checkpoint_name', type=str, default=DEFAULTS["opa_checkpoint_name"]) # base filename for opening checkpoint (without .pkl)
parser.add_argument('--centreline_checkpoint_name', type=str, default=DEFAULTS["centreline_checkpoint_name"]) # base filename for diff centreline checkpoint (without .pkl)
parser.add_argument('--render_auto_registration', type=int, default=DEFAULTS["render_auto_registration"]) # render auto registration checkpoints to png
parser.add_argument('--render_auto_registration_only', type=int, default=DEFAULTS["render_auto_registration_only"]) # stop after auto registration rendering
parser.add_argument('--render_auto_registration_overwrite', type=int, default=DEFAULTS["render_auto_registration_overwrite"]) # overwrite existing render pngs
parser.add_argument('--render_auto_registration_dir', type=str, default=DEFAULTS["render_auto_registration_dir"]) # output dir for auto registration render pngs

pre_args, _ = parser.parse_known_args()
config_path = pre_args.config
if config_path is None:
    # Use repository-local config by default when available.
    inferred_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ghd_fitting_config.yaml")
    if os.path.isfile(inferred_config):
        config_path = inferred_config
config_data = load_config(config_path)
merged = compute_defaults(deep_merge(DEFAULTS, config_data))
parser.set_defaults(**{k: merged[k] for k in vars(pre_args).keys() if k in merged})
args = parser.parse_args()

register = bool(args.register)
chk_num = int(args.chk_num)
if args.chk_freq is None:
    args.chk_freq = round(args.epochs / chk_num)

loss_weighting = copy.deepcopy(merged.get("loss_weighting", DEFAULTS["loss_weighting"]))

if not os.path.isdir(args.root_target):
    raise FileNotFoundError(
        f"Target root directory not found: '{args.root_target}'. "
        f"Pass --root_target or --config pointing to a valid path."
    )

label_list = [label for label in os.listdir(args.root_target) if
              os.path.isdir(os.path.join(args.root_target, label)) and label != args.name_canonical]
case_glob = str(getattr(args, "case_glob", "*"))
if case_glob and case_glob != "*":
    label_list = [label for label in label_list if fnmatch.fnmatch(label, case_glob)]
exclude_list = []
random.shuffle(label_list)
mesh_filename = getattr(args, "mesh_filename", None)
if mesh_filename:
    filtered = []
    for label in label_list:
        mesh_path = os.path.join(args.root_target, label, mesh_filename)
        if os.path.isfile(mesh_path):
            filtered.append(label)
        else:
            print(f"Skipping case {label}: missing {mesh_filename}")
    label_list = filtered

# perform registration for opa classes and differentiable centrelines
if register:
    def _checkpoint_paths(case_root, case_name, ckpt_name):
        ckpt_name = str(ckpt_name)
        ckpt_file = ckpt_name if ckpt_name.endswith(".pkl") else f"{ckpt_name}.pkl"
        return os.path.join(case_root, case_name, ckpt_name), os.path.join(case_root, case_name, ckpt_file)

    auto_opening_method = str(getattr(args, "auto_opening_method", "normals"))
    auto_kwargs = {
        "min_loop_vertices": int(getattr(args, "auto_min_loop_vertices", 24)),
        "normal_dot_min": float(getattr(args, "auto_normal_dot_min", 0.72)),
        "face_dot_min": float(getattr(args, "auto_face_dot_min", 0.90)),
    }
    redo_registration = bool(getattr(args, "redo_registration", 0))
    for label in label_list:
        args.name_target = label
        opa_chk_path, opa_chk_file = _checkpoint_paths(
            args.root_target,
            args.name_target,
            getattr(args, "opa_checkpoint_name", "opa_checkpoint"),
        )
        cl_chk_path, cl_chk_file = _checkpoint_paths(
            args.root_target,
            args.name_target,
            getattr(args, "centreline_checkpoint_name", "diff_centreline_checkpoint"),
        )
        has_opa = os.path.exists(opa_chk_file)
        has_cl = os.path.exists(cl_chk_file)
        if has_opa and has_cl and not redo_registration:
            print("Registration for case {} has been found, skipping".format(label))
        else:
            if redo_registration:
                print("Registration for case {} will be recomputed (--redo_registration=1)".format(label))
            else:
                print("Registration for case {} not found".format(label))
            target = RegistrationwOpeningAlignmentwDifferentiableCentreline(args, args.root_target, args.name_target)
            redo_opa = bool(redo_registration or (not has_opa))
            # Keep centreline consistent with refreshed openings.
            redo_cl = bool(redo_registration or (not has_cl) or redo_opa)
            target.load_checkpoint_opa(
                opa_chk_path,
                redo=redo_opa,
                auto=True,
                auto_method=auto_opening_method,
                auto_kwargs=auto_kwargs,
            )
            target.load_checkpoint_centreline(cl_chk_path, redo=redo_cl, auto=True)
            norm_target = torch.max(torch.norm(getattr(target, "mesh_target_p3d").verts_packed(), dim=-1)).detach().item()
            target.class_normalize(norm=norm_target)
            target.centreline_clean(radius=0.5 / norm_target)
            target.visualize_centreline(norm_target)

if bool(getattr(args, "render_auto_registration", 0)):
    render_auto_registration_qc(args, label_list)
    if bool(getattr(args, "render_auto_registration_only", 0)):
        print("Stopping after auto-registration rendering (--render_auto_registration_only=1).")
        raise SystemExit(0)

# perform ghd fitting
for label in label_list:
    args.name_target = label
    print('Now performing ghd fitting for case {}'.format(label))
    case_log_root = os.path.join(args.save_root, args.name_target, args.meta)
    done_markers = [
        os.path.join(case_log_root, "ghb_fitting_checkpoint.pkl"),
        os.path.join(case_log_root, "ghd_fitting_checkpoint.pkl"),  # backward compatibility
    ]
    if not any(os.path.exists(path) for path in done_markers):
        copied_loss_weighting = loss_weighting.copy()
        fit_ghd(args, copied_loss_weighting, hard_normalize=True, keep_size=True)
    else:
        print("Skipping ghd fitting for case {}".format(label))
