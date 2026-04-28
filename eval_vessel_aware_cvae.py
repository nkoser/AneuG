#!/usr/bin/env python
"""Evaluate vessel-aware CVAE checkpoints on GHD reconstruction and sampling."""
import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from first_stage_vessel_aware import collate_fn
from models.vae_datasets_vessel import VesselAwareGHDDataset
from models.vessel_aware_cvae import VesselAwareCVAE as VesselAwareCVAEV2
from models.vessel_aware_cvae_v8_resnet import VesselAwareCVAEV8ResNet
from models.vessel_conditioner import OstiumConditioner


DEFAULT_GHD_ROOT = (
    "/workspace/AneuG/checkpoints/"
    "ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999"
)
DEFAULT_GHD_RUN = "prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3"


def parse_args():
    parser = argparse.ArgumentParser("eval_vessel_aware_cvae")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--ghd_chk_root", default=None)
    parser.add_argument("--ghd_run", default=None)
    parser.add_argument("--ghd_chk_name", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--aligned_data_root", default=None)
    parser.add_argument("--canonical_mesh", default=None)
    parser.add_argument("--condition_space", choices=["raw", "ghd_local"], default=None)
    parser.add_argument("--canonical_norm_factor", type=float, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch_size", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--mmd_sigma", type=float, default=1.0,
                        help="RBF bandwidth for MMD between samples and GT (in normalized GHD units).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases_file", default=None)
    parser.add_argument("--cases_from_checkpoint", action="store_true")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--out_csv", default=None)
    return parser.parse_args()


def _ckpt_args(ckpt):
    args = ckpt.get("args", {})
    if args is None:
        return {}
    if isinstance(args, dict):
        return args
    return vars(args)


def _get_config(args, saved_args, name, default):
    override = getattr(args, name, None)
    if override is not None:
        return override
    return saved_args.get(name, default)


def _build_model(saved_args, input_dim, device):
    model_type = saved_args.get("model_type", "v2")
    common = dict(
        input_dim=input_dim,
        hidden_dim=int(saved_args.get("hidden_dim", 256)),
        latent_dim=int(saved_args.get("latent_dim", 64)),
        vessel_cond_dim=int(saved_args.get("vessel_cond_dim", 32)),
        extra_cond_dim=0,
        dropout=float(saved_args.get("dropout", 0.02)),
    )
    if model_type == "v8_resnet":
        return VesselAwareCVAEV8ResNet(
            **common,
            encoder_blocks=int(saved_args.get("encoder_blocks", 3)),
            decoder_blocks=int(saved_args.get("decoder_blocks", 6)),
        ).to(device)
    return VesselAwareCVAEV2(**common).to(device)


def _copy_stats_from_checkpoint(dataset, ckpt):
    for name in (
        "ghd_mean",
        "ghd_std",
        "ostium_mean",
        "ostium_std",
        "vessel_center",
        "vessel_scale",
    ):
        if name in ckpt:
            setattr(dataset, name, ckpt[name].cpu())


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_cases_file(path):
    with open(path, "r") as f:
        if path.endswith(".json"):
            return json.load(f)
        return [line.strip() for line in f if line.strip()]


def _pairwise_sample_rmse(samples):
    # samples: [S, B, D] -> pairwise RMSE per case, averaged over pairs.
    num_samples = samples.shape[0]
    if num_samples < 2:
        return torch.zeros(samples.shape[1], device=samples.device)
    rmses = []
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            rmses.append(torch.sqrt(torch.mean((samples[i] - samples[j]) ** 2, dim=1) + 1e-8))
    return torch.stack(rmses, dim=0).mean(dim=0)


def _rbf_mmd2(x, y, sigma):
    # x: [N, D], y: [M, D] -- unbiased MMD^2 with RBF kernel.
    def _pdist2(a, b):
        return (a * a).sum(-1, keepdim=True) + (b * b).sum(-1).unsqueeze(0) - 2 * a @ b.t()
    gamma = 1.0 / (2.0 * sigma * sigma)
    Kxx = torch.exp(-gamma * _pdist2(x, x))
    Kyy = torch.exp(-gamma * _pdist2(y, y))
    Kxy = torch.exp(-gamma * _pdist2(x, y))
    n, m = x.shape[0], y.shape[0]
    if n > 1:
        Kxx = (Kxx.sum() - Kxx.diag().sum()) / (n * (n - 1))
    else:
        Kxx = Kxx.mean()
    if m > 1:
        Kyy = (Kyy.sum() - Kyy.diag().sum()) / (m * (m - 1))
    else:
        Kyy = Kyy.mean()
    return (Kxx + Kyy - 2.0 * Kxy.mean()).item()


def main():
    args = parse_args()
    _set_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved_args = _ckpt_args(ckpt)

    ghd_chk_root = _get_config(args, saved_args, "ghd_chk_root", DEFAULT_GHD_ROOT)
    ghd_run = _get_config(args, saved_args, "ghd_run", DEFAULT_GHD_RUN)
    ghd_chk_name = _get_config(args, saved_args, "ghd_chk_name", "ghb_fitting_checkpoint.pkl")
    data_root = _get_config(args, saved_args, "data_root", "/data/prepared_meshes_3")
    condition_space = _get_config(args, saved_args, "condition_space", "raw")
    aligned_data_root = _get_config(
        args,
        saved_args,
        "aligned_data_root",
        "/data/ghd_prepared_meshes_3_aneurysm_1op_new",
    )
    canonical_mesh = _get_config(
        args,
        saved_args,
        "canonical_mesh",
        "/workspace/AneuG/checkpoints/canonical_average/part_aligned.obj",
    )
    canonical_norm_factor = float(_get_config(args, saved_args, "canonical_norm_factor", 1.10))

    if args.cases_file:
        cases = _load_cases_file(args.cases_file)
    elif args.cases_from_checkpoint and "case_names" in ckpt:
        cases = list(ckpt["case_names"])
    else:
        cases = [
            case for case in os.listdir(ghd_chk_root)
            if os.path.isdir(os.path.join(ghd_chk_root, case))
        ]
    dataset = VesselAwareGHDDataset(
        ghd_chk_root=ghd_chk_root,
        ghd_run=ghd_run,
        ghd_chk_name=ghd_chk_name,
        data_root=data_root,
        cases=cases,
        num_vessel_pts=int(saved_args.get("num_vessel_pts", 256)),
        condition_space=condition_space,
        aligned_data_root=aligned_data_root,
        canonical_mesh=canonical_mesh,
        canonical_norm_factor=canonical_norm_factor,
        normalize=True,
    )
    _copy_stats_from_checkpoint(dataset, ckpt)
    pca_basis = ckpt.get("pca_basis", None)  # [K, D]
    orig_ghd_list = None  # normalized 432-D ghd list before PCA projection
    if pca_basis is not None:
        pca_basis = pca_basis.to(device)
        orig_mean = ckpt["orig_ghd_mean"]
        orig_std = ckpt["orig_ghd_std"]
        # Keep originals (normalized) for downstream comparison.
        orig_ghd_list = [
            ((g.view(1, -1) - orig_mean) / orig_std).view(-1)
            for g in dataset.ghd_list
        ]
        # Replace dataset.ghd_list with PCA scores so model receives K-dim input matching training.
        new_ghd_list = [pca_basis.cpu() @ x for x in orig_ghd_list]
        dataset.ghd_list = new_ghd_list
        dataset.ghd_mean = torch.zeros(1, pca_basis.shape[0])
        dataset.ghd_std = torch.ones(1, pca_basis.shape[0])
        print(f"[eval] PCA active: K={pca_basis.shape[0]}, D={pca_basis.shape[1]}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    conditioner = OstiumConditioner(
        vessel_feat_dim=int(saved_args.get("vessel_feat_dim", 64)),
        ostium_plane_dim=8,
        ostium_feat_dim=16,
        cond_out_dim=int(saved_args.get("vessel_cond_dim", 32)),
    ).to(device)
    model = _build_model(saved_args, dataset.get_dim(), device)
    conditioner.load_state_dict(ckpt["conditioner"])
    model.load_state_dict(ckpt["generator"])
    conditioner.eval()
    model.eval()

    rows = []
    offset = 0
    latent_dim = int(saved_args.get("latent_dim", 64))
    use_cond_prior = bool(saved_args.get("use_conditional_prior", False)) and hasattr(model, "prior")
    all_gt = []
    all_first_sample = []  # one sample per case for population MMD
    with torch.no_grad():
        for batch in loader:
            x = batch["ghd"].to(device)
            ostium = batch["ostium_params"].to(device)
            vessel = batch["vessel_pts"].to(device)
            cond = conditioner(vessel, ostium)

            recon, mu, logvar = model(x, cond, deterministic=True)
            if pca_basis is not None:
                # Lift to original 432-D normalized space for fair RMSE.
                B = x.shape[0]
                x_orig = torch.stack(
                    [orig_ghd_list[offset + i].to(device) for i in range(B)],
                    dim=0,
                )
                recon_orig = recon @ pca_basis  # [B, K] @ [K, D] = [B, D]
                recon_rmse = torch.sqrt(F.mse_loss(recon_orig, x_orig, reduction="none").mean(dim=1) + 1e-8)
            else:
                recon_rmse = torch.sqrt(F.mse_loss(recon, x, reduction="none").mean(dim=1) + 1e-8)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)

            if use_cond_prior:
                p_mu, p_logvar = model.prior(cond)
                p_std = torch.exp(0.5 * p_logvar)
            else:
                p_mu = torch.zeros(x.shape[0], latent_dim, device=device)
                p_std = torch.ones(x.shape[0], latent_dim, device=device)

            sample_list = []
            for _ in range(args.num_samples):
                eps = torch.randn(x.shape[0], latent_dim, device=device)
                z = p_mu + p_std * eps
                sample_list.append(model.decode(z, cond))
            samples = torch.stack(sample_list, dim=0)
            if pca_basis is not None:
                # samples: [S, B, K] -> [S, B, D]; recon: [B, K] -> [B, D]
                S, B, _ = samples.shape
                samples_orig = samples.reshape(S * B, -1) @ pca_basis
                samples_orig = samples_orig.reshape(S, B, -1)
                recon_orig_for_diff = recon @ pca_basis  # [B, D]
                x_for_diff = x_orig  # already [B, D] from above
                samples_for_diff = samples_orig
            else:
                samples_for_diff = samples
                recon_orig_for_diff = recon
                x_for_diff = x
            sample_pair_rmse = _pairwise_sample_rmse(samples_for_diff)
            per_sample_rmse = torch.sqrt(torch.mean((samples_for_diff - x_for_diff.unsqueeze(0)) ** 2, dim=2) + 1e-8)  # [S, B]
            sample_to_gt = per_sample_rmse.mean(dim=0)
            best_of_k = per_sample_rmse.min(dim=0).values
            sample_to_recon = torch.sqrt(torch.mean((samples_for_diff - recon_orig_for_diff.unsqueeze(0)) ** 2, dim=2) + 1e-8).mean(dim=0)

            all_gt.append(x_for_diff.cpu())
            all_first_sample.append(samples_for_diff[0].cpu())

            for local_idx in range(x.shape[0]):
                case_name = dataset.case_names[offset + local_idx]
                rows.append({
                    "case": case_name,
                    "recon_rmse": float(recon_rmse[local_idx].cpu()),
                    "kl": float(kl[local_idx].cpu()),
                    "sample_pair_rmse": float(sample_pair_rmse[local_idx].cpu()),
                    "sample_to_recon_rmse": float(sample_to_recon[local_idx].cpu()),
                    "sample_to_gt_rmse": float(sample_to_gt[local_idx].cpu()),
                    "best_of_k_rmse": float(best_of_k[local_idx].cpu()),
                })
            offset += x.shape[0]

    summary = {
        "ckpt": args.ckpt,
        "model_type": saved_args.get("model_type", "v2"),
        "condition_space": condition_space,
        "epoch": int(ckpt.get("epoch", -1)),
        "num_cases": len(rows),
        "num_samples": args.num_samples,
        "pca_active": pca_basis is not None,
        "metric_space": "normalized_ghd_via_pca_lift" if pca_basis is not None else "normalized_ghd",
    }
    metric_names = [key for key in rows[0].keys() if key != "case"]
    for name in metric_names:
        vals = np.array([row[name] for row in rows], dtype=np.float64)
        summary[f"{name}_mean"] = float(vals.mean())
        summary[f"{name}_std"] = float(vals.std())

    if all_gt:
        gt_pop = torch.cat(all_gt, dim=0)
        sm_pop = torch.cat(all_first_sample, dim=0)
        summary["mmd_rbf"] = _rbf_mmd2(sm_pop, gt_pop, args.mmd_sigma)
        summary["mmd_sigma"] = args.mmd_sigma

    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"summary": summary, "cases": rows}, f, indent=2, sort_keys=True)
    if args.out_csv:
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
