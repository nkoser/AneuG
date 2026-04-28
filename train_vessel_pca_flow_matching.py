#!/usr/bin/env python
"""Train conditional flow matching in a PCA-compressed GHD token space."""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.vae_datasets_vessel import VesselAwareGHDDataset
from models.vessel_aware_flow_matching import VesselAwareFlowMatching
from models.vessel_conditioner import OstiumConditioner
from train_vessel_flow_matching import TeeStream, collate_fn


def parse_args():
    p = argparse.ArgumentParser("train_vessel_pca_flow_matching")
    p.add_argument("--ghd_chk_root", default="/workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999")
    p.add_argument("--ghd_run", default="prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3")
    p.add_argument("--ghd_chk_name", default="ghb_fitting_checkpoint.pkl")
    p.add_argument("--data_root", default="/data/prepared_meshes_3")
    p.add_argument("--aligned_data_root", default="/data/ghd_prepared_meshes_3_aneurysm_1op_new")
    p.add_argument("--canonical_mesh", default="/workspace/AneuG/checkpoints/canonical_average/part_aligned.obj")
    p.add_argument("--condition_space", choices=["raw", "ghd_local"], default="ghd_local")
    p.add_argument("--canonical_norm_factor", type=float, default=1.10)
    p.add_argument("--cases_file", default=None)
    p.add_argument("--train_cases_file", default=None)
    p.add_argument("--val_cases_file", default=None)
    p.add_argument("--val_fraction", type=float, default=0.15)
    p.add_argument("--split_seed", type=int, default=123)

    p.add_argument("--save_root", default="./checkpoints/vessel_aware_pca_flow_matching")
    p.add_argument("--meta", default="v11_pca_flow_ghdlocal")
    p.add_argument("--pca_dim", type=int, default=32)
    p.add_argument("--whiten_pca", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--num_vessel_pts", type=int, default=256)
    p.add_argument("--vessel_cond_dim", type=int, default=32)
    p.add_argument("--vessel_feat_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--time_dim", type=int, default=64)
    p.add_argument("--flow_blocks", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.02)

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--lr_step", type=int, default=1500)
    p.add_argument("--lr_gamma", type=float, default=0.5)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--noise_scale", type=float, default=1.0)
    p.add_argument("--t_min", type=float, default=0.0)
    p.add_argument("--t_max", type=float, default=1.0)
    p.add_argument("--condition_dropout", type=float, default=0.1)
    p.add_argument("--w_velocity", type=float, default=1.0)
    p.add_argument("--w_endpoint", type=float, default=0.25)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_freq", type=int, default=10)
    p.add_argument("--val_freq", type=int, default=50)
    p.add_argument("--save_freq", type=int, default=1000)
    p.add_argument("--log_file", default=None)
    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_case_file(path):
    with open(path, "r") as f:
        if path.endswith(".json"):
            return json.load(f)
        return [line.strip() for line in f if line.strip()]


def load_cases(args):
    if args.cases_file:
        return load_case_file(args.cases_file)
    return [
        case for case in os.listdir(args.ghd_chk_root)
        if os.path.isdir(os.path.join(args.ghd_chk_root, case))
    ]


def split_cases(case_names, val_fraction, split_seed):
    case_names = list(case_names)
    if len(case_names) < 2 or val_fraction <= 0.0:
        return case_names, []
    rng = random.Random(split_seed)
    shuffled = case_names[:]
    rng.shuffle(shuffled)
    n_val = int(round(len(shuffled) * val_fraction))
    n_val = max(1, min(n_val, len(shuffled) - 1))
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def write_case_list(path, cases):
    with open(path, "w") as f:
        json.dump(list(cases), f, indent=2)


def make_dataset(args, cases, normalize=True):
    return VesselAwareGHDDataset(
        ghd_chk_root=args.ghd_chk_root,
        ghd_run=args.ghd_run,
        ghd_chk_name=args.ghd_chk_name,
        data_root=args.data_root,
        cases=cases,
        num_vessel_pts=args.num_vessel_pts,
        condition_space=args.condition_space,
        aligned_data_root=args.aligned_data_root,
        canonical_mesh=args.canonical_mesh,
        canonical_norm_factor=args.canonical_norm_factor,
        normalize=normalize,
    )


def copy_normalization_stats(src, dst):
    for name in (
        "ghd_mean",
        "ghd_std",
        "ostium_mean",
        "ostium_std",
        "vessel_center",
        "vessel_scale",
    ):
        value = getattr(src, name)
        setattr(dst, name, value.clone() if torch.is_tensor(value) else value)


def dataset_ghd_matrix(dataset):
    return torch.stack([dataset[i]["ghd"] for i in range(len(dataset))], dim=0).float()


def fit_pca(x, pca_dim, whiten=True):
    mean = x.mean(dim=0, keepdim=True)
    centered = x - mean
    _, svals, vh = torch.linalg.svd(centered, full_matrices=False)
    max_dim = min(int(pca_dim), vh.shape[0])
    components = vh[:max_dim].contiguous()
    scores = centered @ components.t()
    score_scale = scores.std(dim=0, keepdim=True) + 1e-6
    if not whiten:
        score_scale = torch.ones_like(score_scale)
    variance = svals.pow(2) / max(x.shape[0] - 1, 1)
    explained = variance / (variance.sum() + 1e-12)
    return {
        "pca_mean": mean.contiguous(),
        "pca_components": components,
        "pca_score_scale": score_scale.contiguous(),
        "pca_explained_variance": explained[:max_dim].contiguous(),
        "pca_total_explained": float(explained[:max_dim].sum().item()),
        "pca_dim_actual": max_dim,
    }


def encode_pca(x, pca):
    return ((x - pca["pca_mean"].to(x.device)) @ pca["pca_components"].to(x.device).t()) / pca["pca_score_scale"].to(x.device)


def decode_pca(z, pca):
    return (z * pca["pca_score_scale"].to(z.device)) @ pca["pca_components"].to(z.device) + pca["pca_mean"].to(z.device)


def maybe_drop_condition(cond: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0:
        return cond
    mask = (torch.rand(cond.shape[0], 1, device=cond.device) >= p).to(cond.dtype)
    return cond * mask


def flow_loss_for_batch(model, conditioner, batch, pca, device, args, condition_dropout=0.0):
    x_ghd = batch["ghd"].to(device)
    x1 = encode_pca(x_ghd, pca)
    ostium = batch["ostium_params"].to(device)
    vessel = batch["vessel_pts"].to(device)
    cond = maybe_drop_condition(conditioner(vessel, ostium), condition_dropout)

    x0 = torch.randn_like(x1) * args.noise_scale
    t = torch.empty(x1.shape[0], 1, device=device).uniform_(args.t_min, args.t_max)
    x_t = (1.0 - t) * x0 + t * x1
    target_v = x1 - x0

    pred_v = model(x_t, t, cond)
    velocity_loss = F.mse_loss(pred_v, target_v)
    endpoint_pred = x_t + (1.0 - t) * pred_v
    endpoint_loss = F.mse_loss(endpoint_pred, x1)
    loss = args.w_velocity * velocity_loss + args.w_endpoint * endpoint_loss
    return loss, velocity_loss, endpoint_loss


def evaluate_val_loss(model, conditioner, loader, pca, device, args):
    model.eval()
    conditioner.eval()
    totals = []
    vel_losses = []
    endpoint_losses = []
    pca_recon = []
    with torch.no_grad():
        for batch in loader:
            loss, velocity_loss, endpoint_loss = flow_loss_for_batch(
                model,
                conditioner,
                batch,
                pca,
                device,
                args,
                condition_dropout=0.0,
            )
            x = batch["ghd"].to(device)
            x_proj = decode_pca(encode_pca(x, pca), pca)
            pca_recon.append(float(torch.sqrt(torch.mean((x_proj - x) ** 2, dim=1) + 1e-8).mean().cpu()))
            totals.append(float(loss.detach().cpu()))
            vel_losses.append(float(velocity_loss.detach().cpu()))
            endpoint_losses.append(float(endpoint_loss.detach().cpu()))
    if not totals:
        return None
    return {
        "val_total": float(np.mean(totals)),
        "val_velocity": float(np.mean(vel_losses)),
        "val_endpoint": float(np.mean(endpoint_losses)),
        "val_pca_recon_rmse": float(np.mean(pca_recon)),
    }


def save_checkpoint(path, epoch, args, model, conditioner, optimizer, train_dataset, pca, val_dataset=None):
    torch.save({
        "flow_model": model.state_dict(),
        "conditioner": conditioner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": vars(args),
        "ghd_mean": train_dataset.ghd_mean,
        "ghd_std": train_dataset.ghd_std,
        "ostium_mean": train_dataset.ostium_mean,
        "ostium_std": train_dataset.ostium_std,
        "vessel_center": train_dataset.vessel_center,
        "vessel_scale": train_dataset.vessel_scale,
        "pca_mean": pca["pca_mean"].cpu(),
        "pca_components": pca["pca_components"].cpu(),
        "pca_score_scale": pca["pca_score_scale"].cpu(),
        "pca_explained_variance": pca["pca_explained_variance"].cpu(),
        "pca_total_explained": pca["pca_total_explained"],
        "pca_dim_actual": pca["pca_dim_actual"],
        "case_names": train_dataset.case_names,
        "train_case_names": train_dataset.case_names,
        "val_case_names": val_dataset.case_names if val_dataset is not None else [],
    }, path)


def main():
    args = parse_args()
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        log_f = open(args.log_file, "a", buffering=1)
        sys.stdout = TeeStream(sys.stdout, log_f)
        sys.stderr = TeeStream(sys.stderr, log_f)

    set_seed(args.seed)
    device = torch.device(args.device)
    run_dir = os.path.join(args.save_root, args.meta)
    os.makedirs(run_dir, exist_ok=True)

    if args.train_cases_file and args.val_cases_file:
        train_cases = load_case_file(args.train_cases_file)
        val_cases = load_case_file(args.val_cases_file)
        valid_cases = sorted(set(train_cases + val_cases))
        print(f"Loaded explicit split: {len(train_cases)} train, {len(val_cases)} val")
    else:
        cases = load_cases(args)
        print(f"Found {len(cases)} GHD checkpoint directories")
        print("Scanning valid finished cases ...")
        scan_dataset = make_dataset(args, cases, normalize=False)
        valid_cases = list(scan_dataset.case_names)
        train_cases, val_cases = split_cases(valid_cases, args.val_fraction, args.split_seed)

    print("Building train dataset ...")
    train_dataset = make_dataset(args, train_cases, normalize=True)
    val_dataset = None
    if val_cases:
        print("Building val dataset ...")
        val_dataset = make_dataset(args, val_cases, normalize=True)
        copy_normalization_stats(train_dataset, val_dataset)

    write_case_list(os.path.join(run_dir, "case_names.json"), train_dataset.case_names)
    write_case_list(os.path.join(run_dir, "cases_train.json"), train_dataset.case_names)
    write_case_list(os.path.join(run_dir, "cases_val.json"), val_dataset.case_names if val_dataset else [])
    write_case_list(os.path.join(run_dir, "cases_all_valid.json"), valid_cases)

    train_x = dataset_ghd_matrix(train_dataset)
    pca = fit_pca(train_x, args.pca_dim, whiten=args.whiten_pca)
    pca["pca_mean"] = pca["pca_mean"].to(device)
    pca["pca_components"] = pca["pca_components"].to(device)
    pca["pca_score_scale"] = pca["pca_score_scale"].to(device)
    print(
        f"PCA dim {pca['pca_dim_actual']} / {train_x.shape[1]} | "
        f"explained variance {pca['pca_total_explained']:.4f} | "
        f"whiten={args.whiten_pca}"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = None
    if val_dataset is not None and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            drop_last=False,
        )

    conditioner = OstiumConditioner(
        vessel_feat_dim=args.vessel_feat_dim,
        ostium_plane_dim=8,
        ostium_feat_dim=16,
        cond_out_dim=args.vessel_cond_dim,
    ).to(device)
    model = VesselAwareFlowMatching(
        input_dim=pca["pca_dim_actual"],
        hidden_dim=args.hidden_dim,
        cond_dim=args.vessel_cond_dim,
        time_dim=args.time_dim,
        blocks=args.flow_blocks,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(conditioner.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step,
        gamma=args.lr_gamma,
    )

    print("\n" + "=" * 60)
    print(
        f"Starting PCA-flow matching: {args.epochs} epochs, "
        f"{len(train_dataset)} train / {len(val_dataset) if val_dataset else 0} val samples"
    )
    print(f"  pca_dim {pca['pca_dim_actual']} | hidden {args.hidden_dim} | blocks {args.flow_blocks}")
    print(f"  condition_space {args.condition_space} | vessel pts {args.num_vessel_pts}")
    print("=" * 60 + "\n")

    best_val = None
    for epoch in range(args.epochs + 1):
        model.train()
        conditioner.train()
        totals = []
        vel_losses = []
        endpoint_losses = []
        grad_norm = 0.0

        for batch in train_loader:
            loss, velocity_loss, endpoint_loss = flow_loss_for_batch(
                model,
                conditioner,
                batch,
                pca,
                device,
                args,
                condition_dropout=args.condition_dropout,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(conditioner.parameters()),
                    args.grad_clip,
                )
            optimizer.step()
            totals.append(float(loss.detach().cpu()))
            vel_losses.append(float(velocity_loss.detach().cpu()))
            endpoint_losses.append(float(endpoint_loss.detach().cpu()))

        val_metrics = None
        if val_loader is not None and (epoch % args.val_freq == 0 or epoch == args.epochs):
            val_metrics = evaluate_val_loss(model, conditioner, val_loader, pca, device, args)

        if epoch % args.log_freq == 0 or val_metrics is not None:
            log_dict = {
                "epoch": epoch,
                "total": float(np.mean(totals)),
                "velocity": float(np.mean(vel_losses)),
                "endpoint": float(np.mean(endpoint_losses)),
                "lr": optimizer.param_groups[0]["lr"],
                "grad_norm": float(grad_norm),
            }
            if val_metrics is not None:
                log_dict.update(val_metrics)
            print(log_dict)

        if val_metrics is not None and (best_val is None or val_metrics["val_total"] < best_val):
            best_val = val_metrics["val_total"]
            save_path = os.path.join(run_dir, "models_best_val.pth")
            save_checkpoint(save_path, epoch, args, model, conditioner, optimizer, train_dataset, pca, val_dataset)
            print(f"Saved best-val checkpoint -> {save_path} (val_total={best_val:.6f})")

        if (epoch % args.save_freq == 0 and epoch > 0) or epoch == args.epochs:
            save_path = os.path.join(run_dir, f"models_epoch_{epoch}.pth")
            save_checkpoint(save_path, epoch, args, model, conditioner, optimizer, train_dataset, pca, val_dataset)
            print(f"Saved checkpoint -> {save_path}")

        scheduler.step()

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
