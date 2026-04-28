#!/usr/bin/env python
"""Train vessel-aware CVAE variants on GHD coefficients with a fixed train/val split."""
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from first_stage_vessel_aware import KL_divergence_terms, collate_fn
from models.vae_datasets_vessel import VesselAwareGHDDataset
from models.vessel_aware_cvae import VesselAwareCVAE as VesselAwareCVAEV2
from models.vessel_aware_cvae_v8_resnet import VesselAwareCVAEV8ResNet
from models.vessel_conditioner import OstiumConditioner


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args():
    p = argparse.ArgumentParser("train_vessel_cvae_coeff_trainval")
    p.add_argument("--ghd_chk_root", default="/workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999")
    p.add_argument("--ghd_run", default="prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3")
    p.add_argument("--ghd_chk_name", default="ghb_fitting_checkpoint.pkl")
    p.add_argument("--data_root", default="/data/prepared_meshes_3")
    p.add_argument("--aligned_data_root", default="/data/ghd_prepared_meshes_3_aneurysm_1op_new")
    p.add_argument("--canonical_mesh", default="/workspace/AneuG/checkpoints/canonical_average/part_aligned.obj")
    p.add_argument("--condition_space", choices=["raw", "ghd_local"], default="ghd_local")
    p.add_argument("--canonical_norm_factor", type=float, default=1.10)
    p.add_argument("--train_cases_file", required=True)
    p.add_argument("--val_cases_file", required=True)

    p.add_argument("--save_root", default="./checkpoints/vessel_aware_cvae")
    p.add_argument("--meta", default="cvae_coeff_trainval")
    p.add_argument("--model_type", choices=["v2", "v8_resnet"], default="v2")
    p.add_argument("--hidden_dim", type=int, default=192)
    p.add_argument("--latent_dim", type=int, default=32)
    p.add_argument("--vessel_cond_dim", type=int, default=32)
    p.add_argument("--vessel_feat_dim", type=int, default=64)
    p.add_argument("--num_vessel_pts", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.08)
    p.add_argument("--encoder_blocks", type=int, default=3)
    p.add_argument("--decoder_blocks", type=int, default=6)
    p.add_argument("--use_conditional_prior", action="store_true")

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--lr_step", type=int, default=300)
    p.add_argument("--lr_gamma", type=float, default=0.5)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--condition_dropout", type=float, default=0.15)
    p.add_argument("--uncond_warmup", type=int, default=0,
                   help="For first N epochs, drop condition entirely (decoder learns p(x) before p(x|c))."
                        " After N epochs, condition is gradually re-introduced via condition_dropout.")
    p.add_argument("--ghd_noise_std", type=float, default=0.0,
                   help="Std of Gaussian noise added to GHD encoder input during training")
    p.add_argument("--ghd_mask_prob", type=float, default=0.0,
                   help="Probability of masking each GHD input dimension during training")
    p.add_argument("--w_mse", type=float, default=1.0)
    p.add_argument("--w_kl", type=float, default=0.003)
    p.add_argument("--ae_warmup", type=int, default=0)
    p.add_argument("--kl_warmup", type=int, default=200)
    p.add_argument("--kl_schedule", choices=["linear", "cyclic"], default="linear",
                   help="KL weight schedule type")
    p.add_argument("--kl_cycle_len", type=int, default=100,
                   help="Cycle length in epochs for --kl_schedule cyclic")
    p.add_argument("--kl_cycle_ratio", type=float, default=0.5,
                   help="Fraction of each cycle used for linear KL ramp (0,1]")
    p.add_argument("--kl_cycle_gamma", type=float, default=1.0,
                   help="Optional per-cycle amplitude decay (<=1 shrinks later cycles)")
    p.add_argument("--kl_cap", type=float, default=20.0)
    p.add_argument("--free_bits", type=float, default=0.0)
    p.add_argument("--early_stop_patience", type=int, default=8,
                   help="Number of validation checks without improvement before stopping (<=0 disables)")
    p.add_argument("--early_stop_min_delta", type=float, default=1e-4,
                   help="Minimum val_total improvement required to reset early stopping")
    p.add_argument("--pca_dim", type=int, default=0,
                   help="If >0, fit PCA on train (normalized) GHDs and project to this many components."
                        " Loss is computed in PCA space; basis is saved in checkpoint for eval.")
    p.add_argument("--pca_var", type=float, default=0.0,
                   help="Alternative to --pca_dim: keep enough components to explain this fraction of variance.")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_freq", type=int, default=10)
    p.add_argument("--val_freq", type=int, default=25)
    p.add_argument("--save_freq", type=int, default=1000)
    p.add_argument("--log_file", default=None)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_cases_file(path):
    with open(path, "r") as f:
        if path.endswith(".json"):
            return json.load(f)
        return [line.strip() for line in f if line.strip()]


def make_dataset(args, cases):
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
        normalize=True,
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


def fit_and_apply_pca(train_dataset, val_dataset, pca_dim, pca_var):
    """Fit PCA on train (normalized) GHD coeffs and rewrite both datasets in-place.

    After this call:
      - dataset.ghd_list contains K-dim PCA scores (zero-mean, unit-variance per component is NOT enforced;
        we keep them as raw projections of the unit-variance normalized features so MSE in PCA space lower-bounds
        MSE in normalized GHD space).
      - dataset.ghd_mean / ghd_std are reset to 0 / 1 of shape [1, K] so __getitem__ no-ops on normalize.
      - Returns (basis [K,D], explained_variance_ratio [K], orig_ghd_mean, orig_ghd_std).
    """
    # Build [N, D] of normalized train ghds (mirrors what __getitem__ would yield).
    mean = train_dataset.ghd_mean  # [1, D]
    std = train_dataset.ghd_std    # [1, D]
    norm_train = torch.stack([
        ((g.view(1, -1) - mean) / std).view(-1) for g in train_dataset.ghd_list
    ], dim=0)  # [N, D]
    N, D = norm_train.shape
    # Center already (column means ~ 0 by construction). SVD decomposition.
    centered = norm_train - norm_train.mean(dim=0, keepdim=True)
    # full_matrices=False is faster; we only need top-K rows of Vt.
    U, S, Vt = torch.linalg.svd(centered, full_matrices=False)  # Vt: [min(N,D), D]
    var = (S ** 2) / max(N - 1, 1)
    var_total = var.sum().clamp_min(1e-12)
    var_ratio = var / var_total
    if pca_var > 0.0:
        cum = torch.cumsum(var_ratio, dim=0)
        K = int(torch.searchsorted(cum, torch.tensor(pca_var)).item()) + 1
        K = min(max(K, 1), Vt.shape[0])
    else:
        K = min(max(int(pca_dim), 1), Vt.shape[0])
    basis = Vt[:K].contiguous()  # [K, D]
    explained = float(var_ratio[:K].sum().item())
    print(f"[PCA] fit: D={D} -> K={K}, explained variance = {explained:.4f}")

    def _project(ds, ds_mean, ds_std):
        new_ghd_list = []
        for g in ds.ghd_list:
            x_norm = ((g.view(1, -1) - ds_mean) / ds_std).view(-1)
            score = (basis @ x_norm)  # [K]
            new_ghd_list.append(score)
        ds.ghd_list = new_ghd_list
        # Save the original normalization for downstream inverse mapping.
        ds.orig_ghd_mean = ds_mean.clone()
        ds.orig_ghd_std = ds_std.clone()
        ds.pca_basis = basis.clone()
        ds.ghd_mean = torch.zeros(1, K)
        ds.ghd_std = torch.ones(1, K)

    orig_mean = mean.clone()
    orig_std = std.clone()
    _project(train_dataset, orig_mean, orig_std)
    _project(val_dataset, orig_mean, orig_std)
    return basis, var_ratio[:K], orig_mean, orig_std


def maybe_drop_condition(cond, p):
    if p <= 0.0:
        return cond
    mask = (torch.rand(cond.shape[0], 1, device=cond.device) >= p).to(cond.dtype)
    return cond * mask


def maybe_corrupt_ghd_input(ghd, noise_std, mask_prob):
    ghd_in = ghd
    if mask_prob > 0.0:
        keep = (torch.rand_like(ghd_in) >= mask_prob).to(ghd_in.dtype)
        ghd_in = ghd_in * keep
    if noise_std > 0.0:
        ghd_in = ghd_in + noise_std * torch.randn_like(ghd_in)
    return ghd_in


def build_generator(args, input_dim, device):
    common = dict(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        vessel_cond_dim=args.vessel_cond_dim,
        extra_cond_dim=0,
        dropout=args.dropout,
    )
    if args.model_type == "v8_resnet":
        return VesselAwareCVAEV8ResNet(
            **common,
            encoder_blocks=args.encoder_blocks,
            decoder_blocks=args.decoder_blocks,
        ).to(device)
    return VesselAwareCVAEV2(**common).to(device)


def kl_weight(args, epoch):
    if epoch < args.ae_warmup:
        return 0.0

    if args.kl_schedule == "cyclic":
        epoch_eff = epoch - args.ae_warmup
        if args.kl_cycle_len <= 0:
            return args.w_kl
        cycle_len = args.kl_cycle_len
        cycle_pos = epoch_eff % cycle_len
        ramp_ratio = min(max(args.kl_cycle_ratio, 1e-6), 1.0)
        ramp_len = max(1, int(cycle_len * ramp_ratio))
        cycle_idx = epoch_eff // cycle_len
        cycle_scale = args.kl_cycle_gamma ** cycle_idx
        if cycle_pos < ramp_len:
            phase = cycle_pos / ramp_len
        else:
            phase = 1.0
        return args.w_kl * phase * cycle_scale

    if args.kl_warmup > 0 and epoch < args.ae_warmup + args.kl_warmup:
        return args.w_kl * ((epoch - args.ae_warmup) / args.kl_warmup)
    return args.w_kl


def loss_for_batch(generator, conditioner, batch, args, device, epoch, train):
    ghd = batch["ghd"].to(device)
    ostium = batch["ostium_params"].to(device)
    vessel = batch["vessel_pts"].to(device)
    cond = conditioner(vessel, ostium)
    if train:
        if args.uncond_warmup > 0 and epoch < args.uncond_warmup:
            # Force decoder to learn p(x) by zeroing all condition signals.
            cond = torch.zeros_like(cond)
        else:
            cond = maybe_drop_condition(cond, args.condition_dropout)
        ghd_in = maybe_corrupt_ghd_input(ghd, args.ghd_noise_std, args.ghd_mask_prob)
    else:
        ghd_in = ghd

    deterministic = epoch < args.ae_warmup
    recon, mu, logvar = generator(ghd_in, cond, deterministic=deterministic)

    prior_mu, prior_logvar = None, None
    if args.use_conditional_prior:
        if not hasattr(generator, "prior"):
            raise ValueError("--use_conditional_prior requires a model with prior(cond)")
        prior_mu, prior_logvar = generator.prior(cond)

    mse = F.mse_loss(recon, ghd)
    kl_raw, kl_train = KL_divergence_terms(
        mu,
        logvar,
        prior_mu=prior_mu,
        prior_logvar=prior_logvar,
        free_bits=args.free_bits,
    )
    kl_capped = torch.clamp(kl_train, max=args.kl_cap)
    kl_w = kl_weight(args, epoch)
    loss = args.w_mse * mse + kl_w * kl_capped
    return loss, mse, kl_raw, kl_train, kl_capped, kl_w


def run_epoch(generator, conditioner, loader, optimizer, args, device, epoch, train):
    generator.train(train)
    conditioner.train(train)
    totals, mses, kl_raws, kl_trains = [], [], [], []
    last_kl_w = 0.0
    for batch in loader:
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            loss, mse, kl_raw, kl_train, _kl_capped, kl_w = loss_for_batch(
                generator,
                conditioner,
                batch,
                args,
                device,
                epoch,
                train=train,
            )
            if train:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(generator.parameters()) + list(conditioner.parameters()),
                        args.grad_clip,
                    )
                optimizer.step()
        totals.append(float(loss.detach().cpu()))
        mses.append(float(mse.detach().cpu()))
        kl_raws.append(float(kl_raw.detach().cpu()))
        kl_trains.append(float(kl_train.detach().cpu()))
        last_kl_w = float(kl_w)
    prefix = "train" if train else "val"
    return {
        f"{prefix}_total": float(np.mean(totals)),
        f"{prefix}_mse": float(np.mean(mses)),
        f"{prefix}_kl_raw": float(np.mean(kl_raws)),
        f"{prefix}_kl_train": float(np.mean(kl_trains)),
        "kl_w": last_kl_w,
    }


def save_checkpoint(path, epoch, args, generator, conditioner, optimizer, train_dataset, val_dataset):
    payload = {
        "generator": generator.state_dict(),
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
        "case_names": train_dataset.case_names,
        "train_case_names": train_dataset.case_names,
        "val_case_names": val_dataset.case_names,
    }
    if hasattr(train_dataset, "pca_basis"):
        payload["pca_basis"] = train_dataset.pca_basis
        payload["orig_ghd_mean"] = train_dataset.orig_ghd_mean
        payload["orig_ghd_std"] = train_dataset.orig_ghd_std
    torch.save(payload, path)


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

    train_cases = load_cases_file(args.train_cases_file)
    val_cases = load_cases_file(args.val_cases_file)

    print(f"Building train dataset ({len(train_cases)} cases requested) ...")
    train_dataset = make_dataset(args, train_cases)
    print(f"Building val dataset ({len(val_cases)} cases requested) ...")
    val_dataset = make_dataset(args, val_cases)
    copy_normalization_stats(train_dataset, val_dataset)

    if args.pca_dim > 0 or args.pca_var > 0.0:
        fit_and_apply_pca(train_dataset, val_dataset, args.pca_dim, args.pca_var)

    with open(os.path.join(run_dir, "cases_train.json"), "w") as f:
        json.dump(train_dataset.case_names, f, indent=2)
    with open(os.path.join(run_dir, "cases_val.json"), "w") as f:
        json.dump(val_dataset.case_names, f, indent=2)
    with open(os.path.join(run_dir, "case_names.json"), "w") as f:
        json.dump(train_dataset.case_names, f, indent=2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        drop_last=False,
    )

    input_dim = train_dataset.get_dim()
    conditioner = OstiumConditioner(
        vessel_feat_dim=args.vessel_feat_dim,
        ostium_plane_dim=8,
        ostium_feat_dim=16,
        cond_out_dim=args.vessel_cond_dim,
    ).to(device)
    generator = build_generator(args, input_dim, device)
    optimizer = torch.optim.AdamW(
        list(generator.parameters()) + list(conditioner.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step,
        gamma=args.lr_gamma,
    )

    print("\n" + "=" * 60)
    print(f"Starting coeff-only CVAE {args.model_type}: {args.epochs} epochs")
    print(f"  train {len(train_dataset)} | val {len(val_dataset)} | input_dim {input_dim}")
    print(f"  hidden {args.hidden_dim} | latent {args.latent_dim} | cond {args.vessel_cond_dim}")
    print(
        f"  ae_warmup {args.ae_warmup} | kl_warmup {args.kl_warmup} "
        f"| w_kl {args.w_kl} | kl_schedule {args.kl_schedule}"
    )
    if args.kl_schedule == "cyclic":
        print(
            f"  kl_cycle_len {args.kl_cycle_len} | kl_cycle_ratio {args.kl_cycle_ratio} "
            f"| kl_cycle_gamma {args.kl_cycle_gamma}"
        )
    print(f"  batch {args.batch_size} | lr {args.lr} | wd {args.weight_decay} | early_stop {args.early_stop_patience}")
    print("=" * 60 + "\n")

    best_val = None
    best_epoch = None
    bad_val_checks = 0
    best_train_total = None
    stop_epoch = None
    stop_reason = "completed"
    for epoch in range(args.epochs + 1):
        train_metrics = run_epoch(
            generator,
            conditioner,
            train_loader,
            optimizer,
            args,
            device,
            epoch,
            train=True,
        )

        val_metrics = None
        if epoch % args.val_freq == 0 or epoch == args.epochs:
            val_metrics = run_epoch(
                generator,
                conditioner,
                val_loader,
                optimizer,
                args,
                device,
                epoch,
                train=False,
            )

        if epoch % args.log_freq == 0 or val_metrics is not None:
            log = {"epoch": epoch, **train_metrics}
            if val_metrics is not None:
                log.update(val_metrics)
            print(log)

        if val_metrics is not None:
            current_val = val_metrics["val_total"]
            improved = best_val is None or current_val < (best_val - args.early_stop_min_delta)
            if improved:
                best_val = current_val
                best_epoch = epoch
                bad_val_checks = 0
                best_train_total = train_metrics["train_total"]
                save_path = os.path.join(run_dir, "models_best_val.pth")
                save_checkpoint(save_path, epoch, args, generator, conditioner, optimizer, train_dataset, val_dataset)
                print(f"Saved best-val checkpoint -> {save_path} (val_total={best_val:.6f})")
            elif args.early_stop_patience > 0:
                bad_val_checks += 1
                print(
                    f"No val improvement for {bad_val_checks} check(s) "
                    f"(best epoch {best_epoch}, best val_total={best_val:.6f})"
                )

        if (epoch % args.save_freq == 0 and epoch > 0) or epoch == args.epochs:
            save_path = os.path.join(run_dir, f"models_epoch_{epoch}.pth")
            save_checkpoint(save_path, epoch, args, generator, conditioner, optimizer, train_dataset, val_dataset)
            print(f"Saved checkpoint -> {save_path}")

        if (
            val_metrics is not None
            and args.early_stop_patience > 0
            and bad_val_checks >= args.early_stop_patience
        ):
            stop_epoch = epoch
            stop_reason = "early_stop"
            print(
                f"Early stopping at epoch {epoch} after {bad_val_checks} stale validation checks "
                f"(best epoch {best_epoch}, best val_total={best_val:.6f})"
            )
            break

        scheduler.step()

    if stop_epoch is None:
        stop_epoch = epoch

    summary = {
        "meta": args.meta,
        "model_type": args.model_type,
        "train_cases": len(train_dataset),
        "val_cases": len(val_dataset),
        "stop_epoch": stop_epoch,
        "stop_reason": stop_reason,
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "best_train_total": best_train_total,
        "best_gap": None if (best_val is None or best_train_total is None) else (best_val - best_train_total),
        "kl_schedule": args.kl_schedule,
        "w_kl": args.w_kl,
        "kl_warmup": args.kl_warmup,
        "kl_cycle_len": args.kl_cycle_len,
        "kl_cycle_ratio": args.kl_cycle_ratio,
        "kl_cycle_gamma": args.kl_cycle_gamma,
        "free_bits": args.free_bits,
        "condition_dropout": args.condition_dropout,
        "ghd_noise_std": args.ghd_noise_std,
        "ghd_mask_prob": args.ghd_mask_prob,
        "hidden_dim": args.hidden_dim,
        "latent_dim": args.latent_dim,
        "dropout": args.dropout,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    }
    summary_path = os.path.join(run_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary -> {summary_path}")

    print("\nTraining finished.")


if __name__ == "__main__":
    main()
