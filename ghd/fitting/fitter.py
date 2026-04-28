import pytorch3d.io
import pytorch3d
import torch
from pytorch3d.io import load_objs_as_meshes, save_obj
import os
import sys
import pickle
import re
import numpy as np
from ghd.base.mesh_geometry import MeshThickness
import torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ghd.fitting.registration import RegistrationwOpeningAlignmentwDifferentiableCentreline
from ghd.base.graph_harmonic_deformation import (
    Graph_Harmonic_Deform_opening_alignment_dynamic,
    Graph_Harmonic_Deform,
)
from ghd.losses import (
    Mesh_loss_opening_alignment,
    Mesh_loss_differentiable_occupancy,
    Mesh_loss_do_differentiable_centreline,
    Mesh_loss,
)
from torch.utils.tensorboard import SummaryWriter
from ghd.fitting.logger import log_dict_printer
from ghd.fitting.logger import viz_fitting_static, viz_fitting_debug
from ghd.fitting.logger import update_and_plot_loss_history
from ghd.fitting.weighter import base_loss_weighter
from ghd.fitting.dropper import Do_Dropper
from ghd.base.mesh_geometry3 import Winding_Occupancy
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _find_resume_checkpoint(log_path, chk_freq):
    """
    Find latest checkpoint and infer epoch.
    Supports numbered files (ghb_fitting_checkpoint_<k>.pkl) as well as
    the final unnamed checkpoints (ghb_fitting_checkpoint.pkl,
    ghd_fitting_checkpoint.pkl) which store the epoch inside.
    """
    if not os.path.isdir(log_path):
        return None, None
    pattern = re.compile(r"^ghb_fitting_checkpoint_(\d+)\.pkl$")
    # (epoch, ckpt_path) – we always compare by actual epoch
    candidates = []
    for filename in os.listdir(log_path):
        m = pattern.match(filename)
        if m is None:
            continue
        step_idx = int(m.group(1))
        ckpt_path = os.path.join(log_path, filename)
        epoch = step_idx * chk_freq
        candidates.append((epoch, ckpt_path))
    # Also consider the final (unnamed) checkpoints – they may be newer.
    for fname in ("ghb_fitting_checkpoint.pkl", "ghd_fitting_checkpoint.pkl"):
        fpath = os.path.join(log_path, fname)
        if not os.path.exists(fpath):
            continue
        try:
            with open(fpath, "rb") as f:
                chk = pickle.load(f)
            ep = chk.get("epoch")
            if ep is not None:
                candidates.append((int(ep), fpath))
        except Exception:
            pass
    if not candidates:
        return None, None
    epoch, ckpt_path = max(candidates, key=lambda x: x[0])
    return ckpt_path, epoch


def _load_fitter_checkpoint(
    canonical_fitter,
    ckpt_path,
    device,
    fallback_epoch=None,
    optimizer=None,
    scheduler=None,
):
    with open(ckpt_path, "rb") as f:
        chk = pickle.load(f)
    with torch.no_grad():
        if "GHD_coefficient" in chk:
            canonical_fitter.deformation_param.data.copy_(chk["GHD_coefficient"].to(device))
        if "R" in chk:
            canonical_fitter.R.data.copy_(chk["R"].to(device))
        if "s" in chk:
            canonical_fitter.s.data.copy_(chk["s"].to(device))
        if "T" in chk:
            canonical_fitter.T.data.copy_(chk["T"].to(device))
    if optimizer is not None and "optimizer_state" in chk:
        optimizer.load_state_dict(chk["optimizer_state"])
        _optimizer_state_to_device(optimizer, device)
    if scheduler is not None and "scheduler_state" in chk and chk["scheduler_state"] is not None:
        scheduler.load_state_dict(chk["scheduler_state"])
    epoch = chk.get("epoch", fallback_epoch)
    return int(epoch) if epoch is not None else None


def _optimizer_state_to_device(optimizer, device):
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def _reapply_resume_hyperparameters(args, optimizer, scheduler, scheduler_type):
    if optimizer is None:
        return

    target_lr = float(getattr(args, "lr", optimizer.param_groups[0].get("lr", 0.0)))
    current_lrs = []
    for group in optimizer.param_groups:
        group["lr"] = target_lr
        if "initial_lr" in group:
            group["initial_lr"] = target_lr
        current_lrs.append(float(group["lr"]))

    if scheduler is None:
        return

    if scheduler_type == "step":
        scheduler.step_size = int(getattr(args, "step_size", getattr(scheduler, "step_size", 1)))
        scheduler.gamma = float(getattr(args, "gamma", getattr(scheduler, "gamma", 1.0)))
        scheduler.base_lrs = list(current_lrs)
        scheduler._last_lr = list(current_lrs)
    elif scheduler_type == "plateau":
        scheduler.factor = float(getattr(args, "plateau_factor", getattr(scheduler, "factor", 0.5)))
        scheduler.patience = int(getattr(args, "plateau_patience", getattr(scheduler, "patience", 10)))
        scheduler.threshold = float(getattr(args, "plateau_threshold", getattr(scheduler, "threshold", 1e-4)))
        scheduler.cooldown = int(getattr(args, "plateau_cooldown", getattr(scheduler, "cooldown", 0)))
        min_lr = float(getattr(args, "min_lr", 0.0))
        scheduler.min_lrs = [min_lr for _ in optimizer.param_groups]
        scheduler._last_lr = list(current_lrs)


def _loss_value_to_float(loss):
    if isinstance(loss, (list, tuple)):
        return float(sum(_loss_value_to_float(item) for item in loss))
    if torch.is_tensor(loss):
        value = torch.nan_to_num(loss.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        if value.numel() == 0:
            return 0.0
        if value.numel() == 1:
            return float(value.cpu().item())
        return float(value.reshape(-1).sum().cpu().item())
    try:
        return float(loss)
    except Exception:
        return 0.0


def _evaluate_prefit_guard(args, canonical_fitter, mesh_losser):
    status = "disabled"
    metrics = {}
    case_scale_overrides = {}
    warmup_epochs_override = None
    if not bool(int(getattr(args, "prefit_guard_enabled", 0))):
        return {
            "status": status,
            "metrics": metrics,
            "case_scale_overrides": case_scale_overrides,
            "warmup_epochs_override": warmup_epochs_override,
            "flags": [],
        }

    tracked_keys = [
        "loss_openings_p",
        "loss_openings_surface_p",
        "loss_openings_plane",
        "loss_diff_centreline",
    ]
    device = torch.device(args.device)
    dummy_points = torch.zeros((1, 3), device=device, dtype=torch.float32)
    dummy_gt = torch.zeros((1,), device=device, dtype=torch.float32)
    dummy_index = torch.zeros((1,), device=device, dtype=torch.long)
    with torch.no_grad():
        warped_mesh, warped_openings = canonical_fitter.forward_with_opening_alignment()
        loss_dict = mesh_losser.forward_do_dcforward_opa_do(
            warped_mesh,
            warped_openings,
            {key: 1.0 for key in tracked_keys},
            dummy_points,
            dummy_gt,
            dummy_index,
        )
    metrics = {key: _loss_value_to_float(loss_dict.get(key, 0.0)) for key in tracked_keys}

    moderate_flags = []
    severe_flags = []
    threshold_specs = [
        ("loss_openings_p", float(getattr(args, "prefit_opening_p_threshold", 0.14)), float(getattr(args, "prefit_opening_p_severe_threshold", 0.28))),
        ("loss_openings_surface_p", float(getattr(args, "prefit_opening_surface_p_threshold", 0.012)), float(getattr(args, "prefit_opening_surface_p_severe_threshold", 0.03))),
        ("loss_openings_plane", float(getattr(args, "prefit_opening_plane_threshold", 0.045)), float(getattr(args, "prefit_opening_plane_severe_threshold", 0.09))),
        ("loss_diff_centreline", float(getattr(args, "prefit_centreline_threshold", 0.18)), float(getattr(args, "prefit_centreline_severe_threshold", 0.30))),
    ]
    for key, threshold, severe_threshold in threshold_specs:
        value = float(metrics.get(key, 0.0))
        if value > threshold:
            moderate_flags.append(key)
        if value > severe_threshold:
            severe_flags.append(key)

    base_warmup = int(getattr(args, "opening_warmup_epochs", 0) or 0)
    base_warmup = base_warmup if base_warmup > 0 else max(120, int(0.12 * max(int(getattr(args, "epochs", 1)), 1)))
    if severe_flags or len(moderate_flags) >= 2:
        status = "severe"
        case_scale_overrides = {
            "loss_openings_p": 0.15,
            "loss_openings_surface_p": 0.15,
            "loss_openings_n": 0.20,
            "loss_openings_plane": 0.05,
            "loss_diff_centreline": 0.10,
        }
        warmup_epochs_override = max(base_warmup, int(base_warmup * 2.0))
    elif len(moderate_flags) >= 1:
        status = "moderate"
        case_scale_overrides = {
            "loss_openings_p": 0.35,
            "loss_openings_surface_p": 0.35,
            "loss_openings_n": 0.40,
            "loss_openings_plane": 0.20,
            "loss_diff_centreline": 0.25,
        }
        warmup_epochs_override = max(base_warmup, int(base_warmup * 1.5))
    else:
        status = "ok"

    return {
        "status": status,
        "metrics": metrics,
        "case_scale_overrides": case_scale_overrides,
        "warmup_epochs_override": warmup_epochs_override,
        "flags": severe_flags if severe_flags else moderate_flags,
    }


def _collect_n_ring_vertex_indices(faces, seed_indices, n_rings):
    """
    Given a face tensor [F, 3] and a set of seed vertex indices, expand the
    seed set by *n_rings* hops along mesh edges.  Returns a sorted 1-D
    LongTensor of unique vertex indices.
    """
    faces_np = faces.detach().cpu().numpy()
    # Build adjacency
    from collections import defaultdict
    adj = defaultdict(set)
    for f in faces_np:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        adj[a].update([b, c])
        adj[b].update([a, c])
        adj[c].update([a, b])
    current = set(int(i) for i in seed_indices)
    for _ in range(n_rings):
        frontier = set()
        for v in current:
            frontier.update(adj.get(v, set()))
        current = current | frontier
    return torch.tensor(sorted(current), dtype=torch.long)


def _rim_refine(
    args,
    canonical_fitter,
    mesh_losser,
    log_path,
    writer,
):
    """
    Stage 2: Local vertex refinement around each opening rim.

    After the spectral GHD fitting, this pass creates a free vertex-offset
    parameter for vertices in the N-ring neighbourhood of each opening cap
    and optimises only those offsets.  This bypasses the spectral low-pass
    limit and allows sharp, local adjustment at the ostium.
    """
    device = torch.device(args.device)
    refine_epochs = int(getattr(args, "rim_refine_epochs", 800))
    refine_lr = float(getattr(args, "rim_refine_lr", 0.0006))
    n_rings = int(getattr(args, "rim_refine_rings", 3))
    lap_w = float(getattr(args, "rim_refine_laplacian_weight", 0.15))
    edge_w = float(getattr(args, "rim_refine_edge_weight", 0.10))
    op_p_w = float(getattr(args, "rim_refine_opening_p_weight", 14.0))
    op_plane_w = float(getattr(args, "rim_refine_opening_plane_weight", 10.0))
    surface_p_w = float(getattr(args, "rim_refine_surface_p_weight", 3.0))
    grad_clip = float(getattr(args, "rim_refine_grad_clip_norm", 0.5))

    from pytorch3d.loss import (
        chamfer_distance,
        mesh_laplacian_smoothing,
        mesh_edge_loss,
    )

    # 1) Collect rim-neighbourhood vertex indices across all openings
    all_rim_idx = set()
    for idx in range(canonical_fitter.num_op):
        cap_idx = canonical_fitter.op_rec_v_indices_map[idx]
        all_rim_idx.update(int(i) for i in cap_idx)
    faces = canonical_fitter.base_shape.faces_packed()
    rim_verts_idx = _collect_n_ring_vertex_indices(
        faces, list(all_rim_idx), n_rings
    ).to(device)
    n_rim = rim_verts_idx.shape[0]
    n_total = canonical_fitter.base_shape.verts_packed().shape[0]
    print(
        f"[RimRefine] Refining {n_rim}/{n_total} vertices "
        f"({n_rings} rings, {refine_epochs} epochs, lr={refine_lr})"
    )

    # 2) Get the current warped mesh (frozen GHD params) and compute base verts
    with torch.no_grad():
        warped_mesh_frozen, _ = canonical_fitter.forward_with_opening_alignment()
        base_verts = warped_mesh_frozen.verts_packed().clone()  # [V, 3]

    # 3) Create free offset parameter (only for rim vertices)
    rim_offsets = torch.nn.Parameter(
        torch.zeros(n_rim, 3, device=device, dtype=torch.float32)
    )
    optimizer_refine = torch.optim.Adam([rim_offsets], lr=refine_lr)
    scheduler_refine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_refine, T_max=refine_epochs, eta_min=refine_lr * 0.05
    )

    # 4) Precompute reference edge lengths for the full mesh
    edges_packed = canonical_fitter.base_shape.edges_packed()
    ref_edge_len = torch.norm(
        base_verts[edges_packed[:, 0]] - base_verts[edges_packed[:, 1]], dim=-1
    ).mean().detach()

    # 5) Build opening target references (reuse from mesh_losser)
    target_openings = getattr(mesh_losser, "target_openings", [])
    target_rims = getattr(mesh_losser, "target_opening_rims", [])
    target_planes = getattr(mesh_losser, "target_opening_planes", [])
    sample_num = int(getattr(args, "sample_num", 260000))
    op_sample_num = int(getattr(args, "op_sample_num", 4000))

    # Target surface for global surface chamfer
    target_mesh = getattr(mesh_losser, "target_mesh").to(device)

    # 6) Refinement loop
    epoch_iter_r = range(refine_epochs)
    if tqdm is not None:
        epoch_iter_r = tqdm(epoch_iter_r, desc="RimRefine", dynamic_ncols=True)
    for r_epoch in epoch_iter_r:
        # Build refined mesh: base_verts + scatter offsets into rim positions
        full_offsets = torch.zeros_like(base_verts)
        full_offsets[rim_verts_idx] = rim_offsets
        refined_verts = base_verts + full_offsets
        refined_mesh = canonical_fitter.base_shape.update_padded(
            refined_verts.unsqueeze(0)
        )

        # Build refined opening meshes
        refined_openings = []
        for oidx in range(canonical_fitter.num_op):
            cap_idx = canonical_fitter.op_rec_v_indices_map[oidx]
            cap_verts = refined_verts[cap_idx].unsqueeze(0)
            cap_faces = torch.tensor(
                canonical_fitter.op_rec_f[oidx],
                dtype=torch.int64, device=device
            ).unsqueeze(0)
            refined_openings.append(Meshes(verts=cap_verts, faces=cap_faces))

        loss_total = torch.zeros(1, device=device)

        # a) Opening boundary chamfer (strongest signal)
        for oidx in range(min(len(refined_openings), len(target_rims))):
            rim_pts_warped = mesh_losser._sample_opening_boundary_points(
                refined_openings[oidx], op_sample_num
            )
            rim_pts_target = mesh_losser._sample_point_cloud(
                target_rims[oidx], op_sample_num
            )
            loss_rim_p, _ = chamfer_distance(
                rim_pts_warped, rim_pts_target,
                x_normals=None, y_normals=None,
            )
            if torch.isfinite(loss_rim_p):
                loss_total = loss_total + op_p_w * loss_rim_p

            # Opening cap surface chamfer
            if surface_p_w > 0 and oidx < len(target_openings):
                cap_pts_w = mesh_losser._safe_sample_points_from_meshes(
                    refined_openings[oidx], op_sample_num, return_normals=False
                )
                cap_pts_t = mesh_losser._safe_sample_points_from_meshes(
                    target_openings[oidx].to(device), op_sample_num,
                    return_normals=False,
                )
                loss_cap_p, _ = chamfer_distance(
                    cap_pts_w, cap_pts_t,
                    x_normals=None, y_normals=None,
                )
                if torch.isfinite(loss_cap_p):
                    loss_total = loss_total + surface_p_w * loss_cap_p

            # Plane alignment
            if op_plane_w > 0 and oidx < len(target_planes):
                loss_plane = mesh_losser._opening_planarity_loss(
                    refined_openings[oidx],
                    target_plane=target_planes[oidx],
                )
                if torch.isfinite(loss_plane):
                    loss_total = loss_total + op_plane_w * loss_plane

        # b) Local Laplacian smoothing on refined mesh
        loss_lap = mesh_laplacian_smoothing(refined_mesh, method="cot")
        if torch.isfinite(loss_lap):
            loss_total = loss_total + lap_w * loss_lap

        # c) Edge length preservation
        loss_edge = mesh_edge_loss(refined_mesh, ref_edge_len)
        if torch.isfinite(loss_edge):
            loss_total = loss_total + edge_w * loss_edge

        # Optimise
        optimizer_refine.zero_grad()
        loss_total.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_([rim_offsets], max_norm=grad_clip)
        optimizer_refine.step()
        scheduler_refine.step()

        loss_val = loss_total.detach().cpu().item()
        if writer is not None:
            writer.add_scalar("RimRefine/total_loss", loss_val, r_epoch)
        if tqdm is not None and hasattr(epoch_iter_r, "set_postfix"):
            epoch_iter_r.set_postfix({"loss": f"{loss_val:.5f}"})
        if r_epoch % 100 == 0:
            print(f"[RimRefine] epoch {r_epoch}/{refine_epochs}  loss={loss_val:.6f}")

    # 7) Bake the refined offsets back into the canonical_fitter base shape
    with torch.no_grad():
        full_offsets_final = torch.zeros_like(base_verts)
        full_offsets_final[rim_verts_idx] = rim_offsets.detach()
        refined_verts_final = base_verts + full_offsets_final
        canonical_fitter.base_shape = canonical_fitter.base_shape.update_padded(
            refined_verts_final.unsqueeze(0)
        )
        # Update opening meshes too
        for oidx in range(canonical_fitter.num_op):
            cap_idx = canonical_fitter.op_rec_v_indices_map[oidx]
            cap_verts = refined_verts_final[cap_idx].unsqueeze(0)
            cap_faces = torch.tensor(
                canonical_fitter.op_rec_f[oidx],
                dtype=torch.int64, device=device
            ).unsqueeze(0)
            canonical_fitter.open_Meshes[oidx] = Meshes(
                verts=cap_verts, faces=cap_faces
            )

    print(
        f"[RimRefine] Done. Max offset magnitude: "
        f"{rim_offsets.detach().abs().max().item():.6f}"
    )
    return rim_offsets.detach().cpu(), rim_verts_idx.cpu()


def fit_ghd(args, loss_weighting, hard_normalize=True, keep_size=True, canonical_chk=None):
    # intialize registration
    canonical, target = initailize_registration(args, hard_normalize=hard_normalize, keep_size=keep_size)

    # create graph fitter and losser. Pass canonical_chk into the constructor so the
    # expensive eigsh is skipped when the cache already exists (huge CPU win when
    # running many cases in parallel).
    eigen_chk_arg = canonical_chk if (canonical_chk is not None and os.path.exists(canonical_chk)) else None
    print(f"[fit_ghd] canonical_chk={canonical_chk!r} exists={os.path.exists(canonical_chk) if canonical_chk else 'n/a'} -> eigen_chk_arg={eigen_chk_arg!r}", flush=True)
    canonical_fitter = Graph_Harmonic_Deform_opening_alignment_dynamic(args, canonical, eigen_chk=eigen_chk_arg)
    if canonical_chk is not None and not os.path.exists(canonical_chk):
        chk = {'GBH_eigval': getattr(canonical_fitter, "GBH_eigval").detach().cpu(),
               'GBH_eigvec': getattr(canonical_fitter, "GBH_eigvec").detach().cpu()}
        with open(canonical_chk, 'wb') as f:
            pickle.dump(chk, f)

    mesh_losser = Mesh_loss_do_differentiable_centreline(args, canonical, target)

    query_points, do_gt = mesh_losser.get_static_mask_and_gt(style=args.do_style)
    if args.do_loss_type == "dice_loss_attention":
        print('using attention dice loss, calculating attention weight map now')
        mesh_losser.get_weights_attention(query_points, min_w=1.0, max_w=args.attention_max_w, smooth=args.attention_smooth, inspect=False)
    query_points, do_gt = query_points.to(torch.device(args.device)), do_gt.to(torch.device(args.device))

    # thickness loss
    thinknesser = MeshThickness(r=0.2, num_bundle_filtered=100, innerp_threshold=0.6, num_sel=25)

    # training manager
    log_path = os.path.join(args.save_root, args.name_target, args.meta)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)
    optimizer = torch.optim.AdamW([canonical_fitter.deformation_param, canonical_fitter.s, canonical_fitter.T, canonical_fitter.R],
                                  lr=args.lr)
    scheduler_type = str(getattr(args, "lr_scheduler", "step")).lower()
    if scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(getattr(args, "step_size", 2500)),
            gamma=float(getattr(args, "gamma", 0.75)),
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(getattr(args, "plateau_factor", 0.5)),
            patience=int(getattr(args, "plateau_patience", 300)),
            threshold=float(getattr(args, "plateau_threshold", 1e-4)),
            cooldown=int(getattr(args, "plateau_cooldown", 100)),
            min_lr=float(getattr(args, "min_lr", 1e-6)),
        )
    elif scheduler_type == "none":
        scheduler = None
    else:
        print(f"Unknown lr_scheduler='{scheduler_type}', falling back to StepLR.")
        scheduler_type = "step"
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(getattr(args, "step_size", 2500)),
            gamma=float(getattr(args, "gamma", 0.75)),
        )
    print(f"Using LR scheduler: {scheduler_type}")
    loss_history_weighted = {}
    loss_history_raw = {}
    start_epoch = 0
    early_stopping_enabled = bool(getattr(args, "early_stopping", 0))
    early_stopping_patience = int(getattr(args, "early_stopping_patience", 1200))
    early_stopping_min_delta = float(getattr(args, "early_stopping_min_delta", 1e-5))
    early_stopping_min_epochs = int(getattr(args, "early_stopping_min_epochs", 2000))
    nonfinite_guard = bool(getattr(args, "nonfinite_guard", 1))
    best_total_loss = float("inf")
    no_improve_epochs = 0
    last_epoch = start_epoch - 1
    if early_stopping_enabled:
        print(
            "Early stopping enabled: "
            f"patience={early_stopping_patience}, "
            f"min_delta={early_stopping_min_delta}, "
            f"min_epochs={early_stopping_min_epochs}"
        )

    resume_ckpt, fallback_epoch = _find_resume_checkpoint(log_path, args.chk_freq)
    if resume_ckpt is not None:
        resumed_epoch = _load_fitter_checkpoint(
            canonical_fitter=canonical_fitter,
            ckpt_path=resume_ckpt,
            device=torch.device(args.device),
            fallback_epoch=fallback_epoch,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        if resumed_epoch is not None and bool(int(getattr(args, "resume_override_hparams", 1))):
            _reapply_resume_hyperparameters(args, optimizer, scheduler, scheduler_type)
            print(
                "Reapplied resume hyperparameters from config: "
                f"lr={float(optimizer.param_groups[0]['lr']):.6f}, "
                f"scheduler={scheduler_type}"
            )
        if resumed_epoch is not None and resumed_epoch < args.epochs - 1:
            start_epoch = resumed_epoch + 1
            print(f"Resuming fitting from epoch {start_epoch} (loaded: {resume_ckpt})")
        elif resumed_epoch is not None:
            print(f"Resume checkpoint found at epoch {resumed_epoch}, but target epochs={args.epochs}.")
    last_epoch = start_epoch - 1

    prefit_guard = _evaluate_prefit_guard(args, canonical_fitter, mesh_losser)
    if prefit_guard["status"] != "disabled":
        metric_message = ", ".join(
            f"{key}={float(value):.4f}" for key, value in prefit_guard["metrics"].items()
        )
        print(f"[PrefitGuard] {args.name_target}: {metric_message}")
        if prefit_guard["status"] != "ok":
            print(
                f"[PrefitGuard] Suspicious initialization detected for {args.name_target}: "
                f"status={prefit_guard['status']}, flags={prefit_guard['flags']}. "
                "Applying slower warmup for opening and centreline losses."
            )

    loss_weighter = base_loss_weighter(
        args,
        glo_loss_weighting=loss_weighting,
        style=args.weighter_style,
        case_scale_overrides=prefit_guard.get("case_scale_overrides", {}),
        warmup_epochs_override=prefit_guard.get("warmup_epochs_override", None),
        fit_start_epoch=start_epoch,
    )
    do_dropper = Do_Dropper(
        args,
        getattr(mesh_losser, "weights_attention"),
        drop_num=25,
        drop_rate=0.75,
    )
    use_dropper = getattr(args, "use_do_dropper", 0)
    print("using do dropper") if use_dropper == 1 else print("using static do")
    query_points_update, do_gt_update = query_points, do_gt
    static_do_index = torch.arange(query_points.shape[0], device=query_points.device)

    # main_loop
    epoch_iter = range(start_epoch, args.epochs)
    if tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc=f"Fitting {args.name_target}", dynamic_ncols=True)
    for epoch in epoch_iter:
        last_epoch = epoch
        warped_mesh, warped_openings = canonical_fitter.forward_with_opening_alignment()
        loc_loss_weighting = loss_weighter.easy_weighting(epoch)  # update loss weighting
        if use_dropper == 1:
            do_index, update_do = do_dropper.forward(epoch)
        else:
            do_index, update_do = static_do_index, False
        if update_do:
            query_points_update, do_gt_update = query_points[do_index].clone(), do_gt[do_index].clone()  # update query points and do gt
        loss_dict = mesh_losser.forward_do_dcforward_opa_do(warped_mesh, warped_openings, loc_loss_weighting,
                                                            query_points_update, do_gt_update, do_index)
        # thickness loss
        if "loss_thickness" in loss_weighting:
            thickness_dict, thickness, _, sign = thinknesser.forward(warped_mesh)
            mask_thickness = torch.where(thickness.abs() > 0.1, torch.zeros_like(thickness), torch.ones_like(thickness))
            signed = torch.sign(sign)
            loss_thickness = (F.relu(0.04 - thickness * signed) + F.relu(0.01 - thickness_dict * signed))*mask_thickness
            sign_barrier = sign.abs().clamp_min(5e-2)
            loss_thickness = loss_thickness.mean() + (1e-4 / (sign_barrier ** 2 + 1e-6) * mask_thickness).mean()
            loss_dict["loss_thickness"] = loss_thickness

        total_loss = torch.zeros(1, device=torch.device(args.device))
        log_dict = {'epoch': epoch}
        log_dict_raw = {'epoch': epoch}
        should_log = (epoch % args.log_freq == 0) or (epoch == args.epochs - 1)
        for term, loss in loss_dict.items():
            if term not in ['loss_openings_p', 'loss_openings_n']:
                raw_item = loss.detach().cpu().item()
                weighted_item = (loss * loc_loss_weighting[term]).detach().cpu().item()
                total_loss += loss * loc_loss_weighting[term]
                log_dict_raw[term] = raw_item
                log_dict[term] = weighted_item
                if should_log:
                    writer.add_scalar('TrainRaw/' + term, raw_item, epoch)
                    writer.add_scalar('TrainWeighted/' + term, weighted_item, epoch)
            else:
                loss_openings = torch.sum(torch.stack(loss), dim=0)
                raw_item = loss_openings.detach().cpu().item()
                weighted_item = (loss_openings * loc_loss_weighting[term]).detach().cpu().item()
                total_loss += loss_openings * loc_loss_weighting[term]
                log_dict_raw[term] = raw_item
                log_dict[term] = weighted_item
                if should_log:
                    writer.add_scalar('TrainRaw/' + term, raw_item, epoch)
                    writer.add_scalar('TrainWeighted/' + term, weighted_item, epoch)
        total_loss_item = total_loss.detach().cpu().item()
        if not np.isfinite(total_loss_item):
            if nonfinite_guard:
                print(
                    f"[Fitting] Non-finite total_loss at epoch {epoch}. "
                    "Skipping optimizer step for this epoch."
                )
                optimizer.zero_grad(set_to_none=True)
                continue
            raise FloatingPointError(
                f"[Fitting] Non-finite total_loss at epoch {epoch} and nonfinite_guard=0."
            )
        if should_log:
            writer.add_scalar('TrainWeighted/total_loss', total_loss_item, epoch)
        current_lr = float(optimizer.param_groups[0]["lr"])
        if should_log:
            writer.add_scalar('Train/lr', current_lr, epoch)
        log_dict['total_loss'] = total_loss_item
        log_dict['lr'] = current_lr
        loss_history_weighted = update_and_plot_loss_history(
            loss_history=loss_history_weighted,
            log_dict=log_dict,
            log_path=log_path,
            epoch=epoch,
            plot_every=args.log_freq,
            filename="loss_components_weighted.png",
            title="Fitting Loss Components (Weighted)",
            ylabel="Weighted Loss",
        )
        loss_history_raw = update_and_plot_loss_history(
            loss_history=loss_history_raw,
            log_dict=log_dict_raw,
            log_path=log_path,
            epoch=epoch,
            plot_every=args.log_freq,
            filename="loss_components_raw.png",
            title="Fitting Loss Components (Raw)",
            ylabel="Raw Loss",
        )
        if epoch % args.log_freq == 0:
            print("Raw losses:")
            log_dict_printer(log_dict_raw)
            print("Weighted losses:")
            log_dict_printer(log_dict)
        if epoch % (4 * args.log_freq) == 0:
            print(args.name_target)

        # logging
        viz_freq = int(getattr(args, "viz_freq", 0))
        if viz_freq > 0 and (epoch % max(1, viz_freq) == 0 or epoch == args.epochs - 1):
            viz_fitting_static(
                epoch,
                log_path,
                warped_mesh,
                getattr(mesh_losser, "target_mesh"),
                args,
                target_openings=getattr(mesh_losser, "target_openings", None),
                warped_openings=warped_openings,
                warped_opening_points=getattr(mesh_losser, "preview_opening_points_warped", None),
                warped_opening_normal_points=getattr(mesh_losser, "preview_opening_normal_points_warped", None),
                warped_opening_normals=getattr(mesh_losser, "preview_opening_normals_warped", None),
                target_opening_normal_points=getattr(mesh_losser, "preview_opening_normal_points_target", None),
                target_opening_normals=getattr(mesh_losser, "preview_opening_normals_target", None),
            )

        # gradient descent
        optimizer.zero_grad()
        total_loss.backward()
        grad_clip_norm = float(getattr(args, "grad_clip_norm", 0.0))
        if grad_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(
                [canonical_fitter.deformation_param, canonical_fitter.s, canonical_fitter.T, canonical_fitter.R],
                max_norm=grad_clip_norm,
            )
        optimizer.step()
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(total_loss_item)
            else:
                scheduler.step()

        if early_stopping_enabled:
            improved = (best_total_loss - total_loss_item) > early_stopping_min_delta
            if improved:
                best_total_loss = total_loss_item
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1
            if epoch >= early_stopping_min_epochs and no_improve_epochs >= early_stopping_patience:
                print(
                    "Early stopping triggered at epoch "
                    f"{epoch}: best_total_loss={best_total_loss:.6f}, "
                    f"current_total_loss={total_loss_item:.6f}, "
                    f"no_improve_epochs={no_improve_epochs}"
                )
                break
        if tqdm is not None and hasattr(epoch_iter, "set_postfix"):
            postfix = {"total": f"{total_loss_item:.4f}"}
            # Stable, readable ordering for core losses in the progress bar.
            preferred_keys = [
                "loss_do",
                "loss_p0",
                "loss_n1",
                "loss_laplacian",
                "loss_edge",
                "loss_consistency",
                "loss_rigid",
                "loss_openings_p",
                "loss_openings_surface_p",
                "loss_openings_n",
                "loss_openings_plane",
                "loss_openings_rim_curvature",
                "loss_openings_centroid_axis",
                "loss_diff_centreline",
                "loss_thickness",
            ]
            for key in preferred_keys:
                if key in log_dict:
                    postfix[key] = f"{float(log_dict[key]):.4f}"
            # Include any additional keys that are not in the preferred set.
            for key, value in log_dict.items():
                if key in ("epoch", "total_loss") or key in preferred_keys:
                    continue
                postfix[key] = f"{float(value):.4f}"
            epoch_iter.set_postfix(postfix)

        # saving chk
        if epoch % args.chk_freq == 0 and epoch != 0:
            chk_path = os.path.join(log_path, "ghb_fitting_checkpoint_" + str(round(epoch / args.chk_freq)) + ".pkl")
            chk = {'R': getattr(canonical_fitter, "R").detach().cpu(),
                   's': getattr(canonical_fitter, 's').detach().cpu().abs(),
                   'T': getattr(canonical_fitter, 'T').detach().cpu(),
                   'GHD_coefficient': getattr(canonical_fitter, 'deformation_param').detach().cpu(),
                   'optimizer_state': optimizer.state_dict(),
                   'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
                   'epoch': epoch}
            with open(chk_path, 'wb') as f:
                pickle.dump(chk, f)
            print('GHB fitting results have been saved to {}'.format(chk_path))

    # --- Stage 2: Local rim vertex refinement ---
    rim_refine_enabled = bool(int(getattr(args, "rim_refine_enabled", 0)))
    if rim_refine_enabled and bool(int(getattr(args, "rim_refine_on_suspicious_only", 0))):
        rim_refine_enabled = prefit_guard.get("status") in ("moderate", "severe")
    rim_offsets_cpu = None
    rim_verts_idx_cpu = None
    if rim_refine_enabled:
        print("\n" + "=" * 60)
        print("Stage 2: Local rim vertex refinement")
        print("=" * 60)
        rim_offsets_cpu, rim_verts_idx_cpu = _rim_refine(
            args=args,
            canonical_fitter=canonical_fitter,
            mesh_losser=mesh_losser,
            log_path=log_path,
            writer=writer,
        )
        # Save a post-rim-refine preview so the visual result is visible.
        # NOTE: We use the baked base_shape directly (not forward_with_opening_alignment)
        # because after baking, base_shape already contains the fully warped + refined verts.
        # Calling forward again would double-apply the GHD deformation.
        try:
            with torch.no_grad():
                warped_post = canonical_fitter.base_shape
                openings_post = [
                    canonical_fitter.open_Meshes[oidx]
                    for oidx in range(canonical_fitter.num_op)
                ]
            viz_fitting_static(
                last_epoch + 1,          # use epoch+1 to distinguish from pre-refine
                log_path,
                warped_post,
                getattr(mesh_losser, "target_mesh"),
                args,
                target_openings=getattr(mesh_losser, "target_openings", None),
                warped_openings=openings_post,
                warped_opening_points=None,
                warped_opening_normal_points=None,
                warped_opening_normals=None,
                target_opening_normal_points=None,
                target_opening_normals=None,
            )
            print(f"[RimRefine] Post-refine preview saved (epoch {last_epoch + 1})")
        except Exception as e:
            print(f"[RimRefine] Could not save post-refine preview: {e}")

    # saving
    chk_path = os.path.join(log_path, "ghb_fitting_checkpoint.pkl")
    chk_path_alias = os.path.join(log_path, "ghd_fitting_checkpoint.pkl")
    chk = {'R': getattr(canonical_fitter, "R").detach().cpu(),
           's': getattr(canonical_fitter, 's').detach().cpu().abs(),
           'T': getattr(canonical_fitter, 'T').detach().cpu(),
           'GHD_coefficient': getattr(canonical_fitter, 'deformation_param').detach().cpu(),
           'optimizer_state': optimizer.state_dict(),
           'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
           'epoch': last_epoch}
    if rim_offsets_cpu is not None:
        chk['rim_refine_offsets'] = rim_offsets_cpu
        chk['rim_refine_verts_idx'] = rim_verts_idx_cpu
    with open(chk_path, 'wb') as f:
        pickle.dump(chk, f)
    with open(chk_path_alias, 'wb') as f:
        pickle.dump(chk, f)
    print('GHB fitting results have been saved to {}'.format(chk_path))

def initailize_registration(args, hard_normalize=True, keep_size=True):
    print("Bold opening normal sorting = {}".format(True if args.op_bold == 1 else False))
    def _checkpoint_path(case_root, case_name, ckpt_name):
        ckpt_name = str(ckpt_name)
        return os.path.join(case_root, case_name, ckpt_name)

    auto_opening_method = str(getattr(args, "auto_opening_method", "normals"))
    auto_kwargs = {
        "min_loop_vertices": int(getattr(args, "auto_min_loop_vertices", 24)),
        "normal_dot_min": float(getattr(args, "auto_normal_dot_min", 0.72)),
        "face_dot_min": float(getattr(args, "auto_face_dot_min", 0.90)),
    }
    opa_ckpt_name = getattr(args, "opa_checkpoint_name", "opa_checkpoint")
    cl_ckpt_name = getattr(args, "centreline_checkpoint_name", "diff_centreline_checkpoint")

    canonical_opa_chk = _checkpoint_path(args.root_template, args.name_canonical, opa_ckpt_name)
    canonical_cl_chk = _checkpoint_path(args.root_template, args.name_canonical, cl_ckpt_name)
    target_opa_chk = _checkpoint_path(args.root_target, args.name_target, opa_ckpt_name)
    target_cl_chk = _checkpoint_path(args.root_target, args.name_target, cl_ckpt_name)

    canonical = RegistrationwOpeningAlignmentwDifferentiableCentreline(
        args,
        args.root_template,
        args.name_canonical,
        num_op=int(args.num_op),
        num_cep=int(args.num_op),
    )
    canonical.load_checkpoint_opa(canonical_opa_chk, auto_method=auto_opening_method, auto_kwargs=auto_kwargs)
    canonical.sort_opening_normals(inspect_true_normal=False, clean_threshold=0.2, bold=True if args.op_bold == 1 else False)
    canonical.load_checkpoint_centreline(canonical_cl_chk, redo=False)
    norm_canonical = torch.max(torch.norm(getattr(canonical, "mesh_target_p3d").verts_packed(), dim=-1)).detach().item() * 1.10 if hard_normalize else 10.0
    if keep_size:
        keep_size_factor = float(getattr(args, "keep_size_factor", 1.0))
        keep_size_factor = keep_size_factor if keep_size_factor > 0 else 1.0
        norm_canonical = keep_size_factor * norm_canonical
        print(
            "keeping shared normalization scale, "
            f"canonical factor={keep_size_factor:.4f}"
        )
    canonical.class_normalize(norm=norm_canonical)
    canonical.centreline_clean(radius=0.5 / norm_canonical)

    target = RegistrationwOpeningAlignmentwDifferentiableCentreline(
        args,
        args.root_target,
        args.name_target,
        num_op=int(args.num_op),
        num_cep=int(args.num_op),
    )
    target.load_checkpoint_opa(target_opa_chk, auto_method=auto_opening_method, auto_kwargs=auto_kwargs)
    target.sort_opening_normals(inspect_true_normal=False, clean_threshold=0.2, bold=True if args.op_bold == 1 else False)
    target.load_checkpoint_centreline(target_cl_chk, redo=False)
    norm_target = torch.max(torch.norm(getattr(target, "mesh_target_p3d").verts_packed(),
                                       dim=-1)).detach().item() * 1.10 if hard_normalize else 7.5
    norm_target = norm_canonical if keep_size else norm_target
    target.class_normalize(norm=norm_target)
    target.centreline_clean(radius=0.5 / norm_target)
    print("canonical and target Meshes have been normalized using radius={} and {}".format(norm_canonical, norm_target))
    return canonical, target

def Mesh_normalize(mesh: Meshes, extra_factor=0.1):
    norm = torch.max(torch.norm(mesh.verts_packed(), dim=-1)).detach().item() * (1+extra_factor)
    original_mesh_verts = mesh.verts_padded().float()
    updated_mesh_verts = original_mesh_verts / norm
    normalized_mesh = mesh.update_padded(updated_mesh_verts)
    return normalized_mesh
