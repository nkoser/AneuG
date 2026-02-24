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
    Find latest intermediate checkpoint and infer epoch.
    Supports files like: ghb_fitting_checkpoint_<k>.pkl
    """
    if not os.path.isdir(log_path):
        return None, None
    pattern = re.compile(r"^ghb_fitting_checkpoint_(\d+)\.pkl$")
    candidates = []
    for filename in os.listdir(log_path):
        m = pattern.match(filename)
        if m is None:
            continue
        step_idx = int(m.group(1))
        ckpt_path = os.path.join(log_path, filename)
        candidates.append((step_idx, ckpt_path))
    if not candidates:
        return None, None
    step_idx, ckpt_path = max(candidates, key=lambda x: x[0])
    # This is how checkpoints are currently named in this project.
    epoch = step_idx * chk_freq
    return ckpt_path, epoch


def _load_fitter_checkpoint(canonical_fitter, ckpt_path, device, fallback_epoch=None):
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
    epoch = chk.get("epoch", fallback_epoch)
    return int(epoch) if epoch is not None else None


def fit_ghd(args, loss_weighting, hard_normalize=True, keep_size=True, canonical_chk=None):
    # intialize registration
    canonical, target = initailize_registration(args, hard_normalize=hard_normalize, keep_size=keep_size)

    # create graph fitter and losser
    canonical_fitter = Graph_Harmonic_Deform_opening_alignment_dynamic(args, canonical)
    if canonical_chk is not None:
        if not os.path.exists(canonical_chk):
            chk = {'GBH_eigval': getattr(canonical_fitter, "GBH_eigvec").detach().cpu(),
                'GBH_eigvec': getattr(canonical_fitter, "GBH_eigvec").detach().cpu()}
            with open(canonical_chk, 'wb') as f:
                    pickle.dump(chk, f)
        else:
            with open(canonical_chk, 'rb') as f:
                chk = pickle.load(f)
            for key_ in chk.keys():
                setattr(canonical_fitter, key_, chk[key_].to(torch.device(args.device)))

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
    writer = SummaryWriter(log_path)
    loss_weighter = base_loss_weighter(args, glo_loss_weighting=loss_weighting, style=args.weighter_style)
    do_dropper = Do_Dropper(
        args,
        getattr(mesh_losser, "weights_attention"),
        drop_num=25,
        drop_rate=0.75,
    )
    use_dropper = getattr(args, "use_do_dropper", 0)
    print("using do dropper") if use_dropper == 1 else print("using static do")
    query_points_update, do_gt_update = query_points, do_gt
    loss_history_weighted = {}
    loss_history_raw = {}
    start_epoch = 0
    early_stopping_enabled = bool(getattr(args, "early_stopping", 0))
    early_stopping_patience = int(getattr(args, "early_stopping_patience", 1200))
    early_stopping_min_delta = float(getattr(args, "early_stopping_min_delta", 1e-5))
    early_stopping_min_epochs = int(getattr(args, "early_stopping_min_epochs", 2000))
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
        )
        if resumed_epoch is not None and resumed_epoch < args.epochs - 1:
            start_epoch = resumed_epoch + 1
            print(f"Resuming fitting from epoch {start_epoch} (loaded: {resume_ckpt})")
        elif resumed_epoch is not None:
            print(f"Resume checkpoint found at epoch {resumed_epoch}, but target epochs={args.epochs}.")
    last_epoch = start_epoch - 1

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
            do_index, update_do = do_dropper.forward(epoch)
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
            loss_thickness = loss_thickness.mean() + (1e-4 / (sign ** 2 + 1e-6) * mask_thickness).mean()
            loss_dict["loss_thickness"] = loss_thickness

        total_loss = torch.zeros(1, device=torch.device(args.device))
        log_dict = {'epoch': epoch}
        log_dict_raw = {'epoch': epoch}
        for term, loss in loss_dict.items():
            if term not in ['loss_openings_p', 'loss_openings_n']:
                raw_item = loss.detach().cpu().item()
                weighted_item = (loss * loc_loss_weighting[term]).detach().cpu().item()
                total_loss += loss * loc_loss_weighting[term]
                writer.add_scalar('TrainRaw/' + term, raw_item, epoch)
                writer.add_scalar('TrainWeighted/' + term, weighted_item, epoch)
                log_dict_raw[term] = raw_item
                log_dict[term] = weighted_item
            else:
                loss_openings = torch.sum(torch.stack(loss), dim=0)
                raw_item = loss_openings.detach().cpu().item()
                weighted_item = (loss_openings * loc_loss_weighting[term]).detach().cpu().item()
                total_loss += loss_openings * loc_loss_weighting[term]
                writer.add_scalar('TrainRaw/' + term, raw_item, epoch)
                writer.add_scalar('TrainWeighted/' + term, weighted_item, epoch)
                log_dict_raw[term] = raw_item
                log_dict[term] = weighted_item
        total_loss_item = total_loss.detach().cpu().item()
        if not np.isfinite(total_loss_item):
            print(
                f"[Fitting] Non-finite total_loss at epoch {epoch}. "
                "Skipping optimizer step for this epoch."
            )
            optimizer.zero_grad(set_to_none=True)
            continue
        writer.add_scalar('TrainWeighted/total_loss', total_loss_item, epoch)
        current_lr = float(optimizer.param_groups[0]["lr"])
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
        viz_fitting_static(epoch, log_path, warped_mesh, getattr(mesh_losser, "target_mesh"), args)

        # gradient descent
        optimizer.zero_grad()
        total_loss.backward()
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
                "loss_openings_n",
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
                   'epoch': epoch}
            with open(chk_path, 'wb') as f:
                pickle.dump(chk, f)
            print('GHB fitting results have been saved to {}'.format(chk_path))

    # saving
    chk_path = os.path.join(log_path, "ghb_fitting_checkpoint.pkl")
    chk_path_alias = os.path.join(log_path, "ghd_fitting_checkpoint.pkl")
    chk = {'R': getattr(canonical_fitter, "R").detach().cpu(),
           's': getattr(canonical_fitter, 's').detach().cpu().abs(),
           'T': getattr(canonical_fitter, 'T').detach().cpu(),
           'GHD_coefficient': getattr(canonical_fitter, 'deformation_param').detach().cpu(),
           'epoch': last_epoch}
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

    canonical = RegistrationwOpeningAlignmentwDifferentiableCentreline(args, args.root_template, args.name_canonical)
    canonical.load_checkpoint_opa(canonical_opa_chk, auto_method=auto_opening_method, auto_kwargs=auto_kwargs)
    canonical.sort_opening_normals(inspect_true_normal=False, clean_threshold=0.2, bold=True if args.op_bold == 1 else False)
    canonical.load_checkpoint_centreline(canonical_cl_chk, redo=False)
    norm_canonical = torch.max(torch.norm(getattr(canonical, "mesh_target_p3d").verts_packed(), dim=-1)).detach().item() * 1.10 if hard_normalize else 10.0
    if keep_size:
        norm_canonical = 2.50 * norm_canonical
        print("keeping same size ratio, which means canonical is normalized using 2.50 * radius")
    canonical.class_normalize(norm=norm_canonical)
    canonical.centreline_clean(radius=0.5 / norm_canonical)

    target = RegistrationwOpeningAlignmentwDifferentiableCentreline(args, args.root_target, args.name_target)
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
