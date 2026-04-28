#!/usr/bin/env python
"""Ensemble evaluation of multiple CVAE checkpoints.

For every val case, draws K samples from each of the M ensemble members and
reports best-of-(K*M) RMSE versus GT, along with per-member metrics.
Inputs may be a mix of PCA and non-PCA checkpoints — metrics are reported in
the original normalized-GHD space.
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from first_stage_vessel_aware import collate_fn
from models.vae_datasets_vessel import VesselAwareGHDDataset
from models.vessel_aware_cvae import VesselAwareCVAE as VesselAwareCVAEV2
from models.vessel_aware_cvae_v8_resnet import VesselAwareCVAEV8ResNet
from models.vessel_conditioner import OstiumConditioner


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--cases_file", required=True)
    p.add_argument("--num_samples_per_member", type=int, default=32)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", default=None)
    return p.parse_args()


def _ckpt_args(ckpt):
    a = ckpt.get("args", {})
    return a if isinstance(a, dict) else vars(a) if a else {}


def build_member(ckpt, device):
    sa = _ckpt_args(ckpt)
    pca_basis = ckpt.get("pca_basis", None)
    if pca_basis is not None:
        input_dim = pca_basis.shape[0]
    else:
        input_dim = ckpt["ghd_mean"].shape[1]
    common = dict(
        input_dim=input_dim,
        hidden_dim=int(sa.get("hidden_dim", 256)),
        latent_dim=int(sa.get("latent_dim", 64)),
        vessel_cond_dim=int(sa.get("vessel_cond_dim", 32)),
        extra_cond_dim=0,
        dropout=float(sa.get("dropout", 0.02)),
    )
    if sa.get("model_type") == "v8_resnet":
        model = VesselAwareCVAEV8ResNet(
            **common,
            encoder_blocks=int(sa.get("encoder_blocks", 3)),
            decoder_blocks=int(sa.get("decoder_blocks", 6)),
        )
    else:
        model = VesselAwareCVAEV2(**common)
    model.load_state_dict(ckpt["generator"])
    model.to(device).eval()
    cond = OstiumConditioner(
        vessel_feat_dim=int(sa.get("vessel_feat_dim", 64)),
        ostium_plane_dim=8,
        ostium_feat_dim=16,
        cond_out_dim=int(sa.get("vessel_cond_dim", 32)),
    )
    cond.load_state_dict(ckpt["conditioner"])
    cond.to(device).eval()
    return model, cond, sa, pca_basis.to(device) if pca_basis is not None else None


def main():
    args = parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    cases = json.load(open(args.cases_file))
    # Use first ckpt's metadata for dataset config + normalization stats.
    first = torch.load(args.ckpts[0], map_location="cpu", weights_only=False)
    sa = _ckpt_args(first)
    # If first ckpt is PCA-trained, ghd_mean/std are 0/1 over K dims; use orig stats for dataset.
    if "orig_ghd_mean" in first:
        ds_ghd_mean = first["orig_ghd_mean"]
        ds_ghd_std = first["orig_ghd_std"]
    else:
        ds_ghd_mean = first["ghd_mean"]
        ds_ghd_std = first["ghd_std"]
    dataset = VesselAwareGHDDataset(
        ghd_chk_root=sa.get("ghd_chk_root"),
        ghd_run=sa.get("ghd_run"),
        ghd_chk_name=sa.get("ghd_chk_name"),
        data_root=sa.get("data_root"),
        cases=cases,
        num_vessel_pts=int(sa.get("num_vessel_pts", 256)),
        condition_space=sa.get("condition_space", "ghd_local"),
        aligned_data_root=sa.get("aligned_data_root"),
        canonical_mesh=sa.get("canonical_mesh"),
        canonical_norm_factor=float(sa.get("canonical_norm_factor", 1.10)),
        normalize=True,
    )
    for k in ("ghd_mean", "ghd_std", "ostium_mean", "ostium_std", "vessel_center", "vessel_scale"):
        if k in first:
            setattr(dataset, k, first[k].cpu())
    # Override ghd_mean/std with original stats so __getitem__ produces 432-D normalized GHDs.
    dataset.ghd_mean = ds_ghd_mean.cpu()
    dataset.ghd_std = ds_ghd_std.cpu()

    # Pre-compute normalized GTs in original GHD space.
    orig_gt = []
    mean_t = ds_ghd_mean
    std_t = ds_ghd_std
    for g in dataset.ghd_list:
        orig_gt.append(((g.view(1, -1) - mean_t) / std_t).view(-1))
    orig_gt = torch.stack(orig_gt, dim=0).to(device)  # [N, D]

    members = []
    for path in args.ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m, c, msa, pb = build_member(ck, device)
        # PCA basis differs per checkpoint potentially; if PCA, use that ckpt's orig_mean/std.
        if pb is not None:
            om = ck["orig_ghd_mean"].to(device)
            os_ = ck["orig_ghd_std"].to(device)
        else:
            om = mean_t.to(device); os_ = std_t.to(device)
        members.append({
            "name": os.path.basename(os.path.dirname(path)),
            "model": m, "cond": c, "args": msa, "pca": pb,
            "orig_mean": om, "orig_std": os_,
            "latent": int(msa.get("latent_dim", 64)),
            "use_cp": bool(msa.get("use_conditional_prior", False)) and hasattr(m, "prior"),
        })

    loader = DataLoader(dataset, batch_size=200, shuffle=False, collate_fn=collate_fn)

    per_member_b = {m["name"]: [] for m in members}
    per_member_g = {m["name"]: [] for m in members}
    ensemble_b = []
    ensemble_g = []
    K = args.num_samples_per_member

    with torch.no_grad():
        for batch in loader:
            ostium = batch["ostium_params"].to(device)
            vessel = batch["vessel_pts"].to(device)
            x_orig = orig_gt  # [N, D]; only one batch given B=200 and val<200

            all_samples_orig = []  # list of [K, B, D]
            for m in members:
                cond = m["cond"](vessel, ostium)
                if m["use_cp"]:
                    pmu, plv = m["model"].prior(cond)
                    pstd = torch.exp(0.5 * plv)
                else:
                    pmu = torch.zeros(cond.shape[0], m["latent"], device=device)
                    pstd = torch.ones_like(pmu)
                samples = []
                for _ in range(K):
                    eps = torch.randn(cond.shape[0], m["latent"], device=device)
                    z = pmu + pstd * eps
                    samples.append(m["model"].decode(z, cond))
                S = torch.stack(samples, dim=0)  # [K, B, D_member]
                if m["pca"] is not None:
                    S = S @ m["pca"]  # lift to original-D
                # Member space: normalized GHD with member's own mean/std; convert if differs.
                # If member's orig stats != global stats, re-normalize: S_global = (S * mstd + mmean - gmean) / gstd
                if not torch.allclose(m["orig_mean"], mean_t.to(device)) or not torch.allclose(m["orig_std"], std_t.to(device)):
                    S = (S * m["orig_std"] + m["orig_mean"] - mean_t.to(device)) / std_t.to(device)
                all_samples_orig.append(S)

                # Per-member metrics
                per_rmse = torch.sqrt(((S - x_orig.unsqueeze(0)) ** 2).mean(-1) + 1e-8)  # [K, B]
                per_member_b[m["name"]].append(per_rmse.min(0).values.cpu())
                per_member_g[m["name"]].append(per_rmse.mean(0).cpu())

            big = torch.cat(all_samples_orig, dim=0)  # [K*M, B, D]
            ens_rmse = torch.sqrt(((big - x_orig.unsqueeze(0)) ** 2).mean(-1) + 1e-8)
            ensemble_b.append(ens_rmse.min(0).values.cpu())
            ensemble_g.append(ens_rmse.mean(0).cpu())

    print("\n=== Per-member (K=%d each) ===" % K)
    for m in members:
        b = torch.cat(per_member_b[m["name"]]).mean().item()
        g = torch.cat(per_member_g[m["name"]]).mean().item()
        print(f"  {m['name']:55s}  best_of_K={b:.4f}  s_to_gt={g:.4f}")

    eb = torch.cat(ensemble_b).mean().item()
    eg = torch.cat(ensemble_g).mean().item()
    M = len(members)
    print(f"\n=== Ensemble (M={M}, total samples per case = {K*M}) ===")
    print(f"  best_of_KM = {eb:.4f}")
    print(f"  s_to_gt    = {eg:.4f}")

    if args.out_json:
        out = {
            "members": [{
                "name": m["name"],
                "best_of_K": torch.cat(per_member_b[m["name"]]).mean().item(),
                "s_to_gt": torch.cat(per_member_g[m["name"]]).mean().item(),
            } for m in members],
            "ensemble": {
                "M": M, "K": K,
                "best_of_KM": eb, "s_to_gt": eg,
            },
            "cases_file": args.cases_file,
        }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        json.dump(out, open(args.out_json, "w"), indent=2)


if __name__ == "__main__":
    main()
