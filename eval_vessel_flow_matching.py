#!/usr/bin/env python
"""Evaluate vessel-aware conditional flow-matching checkpoints."""
import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.vae_datasets_vessel import VesselAwareGHDDataset
from models.vessel_aware_flow_matching import VesselAwareFlowMatching
from models.vessel_conditioner import OstiumConditioner
from train_vessel_flow_matching import collate_fn


DEFAULT_GHD_ROOT = (
    "/workspace/AneuG/checkpoints/"
    "ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999"
)
DEFAULT_GHD_RUN = "prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3"


def parse_args():
    p = argparse.ArgumentParser("eval_vessel_flow_matching")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--ghd_chk_root", default=None)
    p.add_argument("--ghd_run", default=None)
    p.add_argument("--ghd_chk_name", default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--aligned_data_root", default=None)
    p.add_argument("--canonical_mesh", default=None)
    p.add_argument("--condition_space", choices=["raw", "ghd_local"], default=None)
    p.add_argument("--canonical_norm_factor", type=float, default=None)
    p.add_argument("--cases_file", default=None)
    p.add_argument("--eval_all_cases", action="store_true")
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=200)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--num_samples", type=int, default=16)
    p.add_argument("--num_steps", type=int, default=32)
    p.add_argument("--sampler", choices=["euler", "heun"], default="heun")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out_json", default=None)
    p.add_argument("--out_csv", default=None)
    return p.parse_args()


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


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_cases_file(path):
    with open(path, "r") as f:
        if path.endswith(".json"):
            return json.load(f)
        return [line.strip() for line in f if line.strip()]


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


def _pairwise_sample_rmse(samples):
    num_samples = samples.shape[0]
    if num_samples < 2:
        return torch.zeros(samples.shape[1], device=samples.device)
    rmses = []
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            rmses.append(torch.sqrt(torch.mean((samples[i] - samples[j]) ** 2, dim=1) + 1e-8))
    return torch.stack(rmses, dim=0).mean(dim=0)


def _build_model(saved_args, input_dim, device):
    return VesselAwareFlowMatching(
        input_dim=input_dim,
        hidden_dim=int(saved_args.get("hidden_dim", 512)),
        cond_dim=int(saved_args.get("vessel_cond_dim", 32)),
        time_dim=int(saved_args.get("time_dim", 64)),
        blocks=int(saved_args.get("flow_blocks", 8)),
        dropout=float(saved_args.get("dropout", 0.02)),
    ).to(device)


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
    condition_space = _get_config(args, saved_args, "condition_space", "ghd_local")
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
    elif not args.eval_all_cases and "case_names" in ckpt:
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
    model.load_state_dict(ckpt["flow_model"])
    conditioner.eval()
    model.eval()

    rows = []
    offset = 0
    with torch.no_grad():
        for batch in loader:
            x = batch["ghd"].to(device)
            ostium = batch["ostium_params"].to(device)
            vessel = batch["vessel_pts"].to(device)
            cond = conditioner(vessel, ostium)

            zero_sample = model.sample(
                cond,
                num_steps=args.num_steps,
                temperature=args.temperature,
                method=args.sampler,
                x0=torch.zeros_like(x),
            )
            zero_to_gt = torch.sqrt(torch.mean((zero_sample - x) ** 2, dim=1) + 1e-8)

            cond_rep = cond.repeat_interleave(args.num_samples, dim=0)
            samples_flat = model.sample(
                cond_rep,
                num_steps=args.num_steps,
                temperature=args.temperature,
                method=args.sampler,
            )
            samples = samples_flat.view(x.shape[0], args.num_samples, -1).transpose(0, 1).contiguous()
            sample_pair = _pairwise_sample_rmse(samples)
            sample_to_gt = torch.sqrt(torch.mean((samples - x.unsqueeze(0)) ** 2, dim=2) + 1e-8).mean(dim=0)
            sample_to_zero = torch.sqrt(torch.mean((samples - zero_sample.unsqueeze(0)) ** 2, dim=2) + 1e-8).mean(dim=0)

            for local_idx in range(x.shape[0]):
                rows.append({
                    "case": dataset.case_names[offset + local_idx],
                    "zero_to_gt_rmse": float(zero_to_gt[local_idx].cpu()),
                    "sample_pair_rmse": float(sample_pair[local_idx].cpu()),
                    "sample_to_gt_rmse": float(sample_to_gt[local_idx].cpu()),
                    "sample_to_zero_rmse": float(sample_to_zero[local_idx].cpu()),
                })
            offset += x.shape[0]

    summary = {
        "ckpt": args.ckpt,
        "model_type": "flow_matching",
        "condition_space": condition_space,
        "epoch": int(ckpt.get("epoch", -1)),
        "num_cases": len(rows),
        "num_samples": args.num_samples,
        "num_steps": args.num_steps,
        "sampler": args.sampler,
        "temperature": args.temperature,
    }
    metric_names = [key for key in rows[0].keys() if key != "case"]
    for name in metric_names:
        vals = np.array([row[name] for row in rows], dtype=np.float64)
        summary[f"{name}_mean"] = float(vals.mean())
        summary[f"{name}_std"] = float(vals.std())

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
