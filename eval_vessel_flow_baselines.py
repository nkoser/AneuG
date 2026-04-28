#!/usr/bin/env python
"""Evaluate non-generative baselines for vessel-aware GHD prediction."""
import argparse
import csv
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.vae_datasets_vessel import VesselAwareGHDDataset
from models.vessel_conditioner import OstiumConditioner
from train_vessel_flow_matching import collate_fn


DEFAULT_GHD_ROOT = (
    "/workspace/AneuG/checkpoints/"
    "ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999"
)
DEFAULT_GHD_RUN = "prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3"


def parse_args():
    p = argparse.ArgumentParser("eval_vessel_flow_baselines")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--train_cases_file", default=None)
    p.add_argument("--eval_cases_file", default=None)
    p.add_argument("--ghd_chk_root", default=None)
    p.add_argument("--ghd_run", default=None)
    p.add_argument("--ghd_chk_name", default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--aligned_data_root", default=None)
    p.add_argument("--canonical_mesh", default=None)
    p.add_argument("--condition_space", choices=["raw", "ghd_local"], default=None)
    p.add_argument("--canonical_norm_factor", type=float, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude_same_case", action="store_true")
    p.add_argument("--out_json", default=None)
    p.add_argument("--out_csv", default=None)
    return p.parse_args()


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _load_cases_file(path):
    with open(path, "r") as f:
        if path.endswith(".json"):
            return json.load(f)
        return [line.strip() for line in f if line.strip()]


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


def _build_dataset(args, saved_args, ckpt, cases):
    dataset = VesselAwareGHDDataset(
        ghd_chk_root=_get_config(args, saved_args, "ghd_chk_root", DEFAULT_GHD_ROOT),
        ghd_run=_get_config(args, saved_args, "ghd_run", DEFAULT_GHD_RUN),
        ghd_chk_name=_get_config(args, saved_args, "ghd_chk_name", "ghb_fitting_checkpoint.pkl"),
        data_root=_get_config(args, saved_args, "data_root", "/data/prepared_meshes_3"),
        cases=cases,
        num_vessel_pts=int(saved_args.get("num_vessel_pts", 256)),
        condition_space=_get_config(args, saved_args, "condition_space", "ghd_local"),
        aligned_data_root=_get_config(
            args,
            saved_args,
            "aligned_data_root",
            "/data/ghd_prepared_meshes_3_aneurysm_1op_new",
        ),
        canonical_mesh=_get_config(
            args,
            saved_args,
            "canonical_mesh",
            "/workspace/AneuG/checkpoints/canonical_average/part_aligned.obj",
        ),
        canonical_norm_factor=float(_get_config(args, saved_args, "canonical_norm_factor", 1.10)),
        normalize=True,
    )
    _copy_stats_from_checkpoint(dataset, ckpt)
    return dataset


def _collect(dataset, batch_size, num_workers, device, conditioner=None):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    xs = []
    ostiums = []
    vessels = []
    learned_conds = []

    with torch.no_grad():
        for batch in loader:
            x = batch["ghd"].to(device)
            ostium = batch["ostium_params"].to(device)
            vessel = batch["vessel_pts"].to(device)
            xs.append(x.cpu())
            ostiums.append(ostium.cpu())
            vessels.append(vessel.cpu())
            if conditioner is not None:
                learned_conds.append(conditioner(vessel, ostium).cpu())

    result = {
        "x": torch.cat(xs, dim=0),
        "ostium": torch.cat(ostiums, dim=0),
        "vessel": torch.cat(vessels, dim=0),
    }
    if learned_conds:
        result["learned_cond"] = torch.cat(learned_conds, dim=0)
    return result


def _vessel_summary(vessel):
    mean = vessel.mean(dim=1)
    std = vessel.std(dim=1)
    minv = vessel.min(dim=1).values
    maxv = vessel.max(dim=1).values
    return torch.cat([mean, std, minv, maxv], dim=1)


def _geom_summary_features(data):
    return torch.cat([data["ostium"], _vessel_summary(data["vessel"])], dim=1)


def _pairwise_rmse(a, b):
    return torch.cdist(a, b) / np.sqrt(a.shape[1])


def _nearest_prediction(query_feat, train_feat, train_x, query_cases=None, train_cases=None, exclude_same_case=False):
    distances = torch.cdist(query_feat, train_feat)
    if exclude_same_case and query_cases is not None and train_cases is not None:
        for i, case in enumerate(query_cases):
            for j, train_case in enumerate(train_cases):
                if case == train_case:
                    distances[i, j] = float("inf")
    nn_dist, nn_idx = distances.min(dim=1)
    pred = train_x[nn_idx]
    return pred, nn_idx, nn_dist


def _rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=1) + 1e-8)


def _add_metric(row, name, values, idx):
    row[name] = float(values[idx].cpu())


def _summarize(rows):
    summary = {}
    metric_names = [key for key in rows[0].keys() if key not in ("case", "nn_case_by_ghd", "nn_case_by_ostium", "nn_case_by_geom", "nn_case_by_learned_cond")]
    for name in metric_names:
        vals = np.array([row[name] for row in rows], dtype=np.float64)
        summary[f"{name}_mean"] = float(vals.mean())
        summary[f"{name}_std"] = float(vals.std())
        summary[f"{name}_median"] = float(np.median(vals))
    return summary


def main():
    args = parse_args()
    _set_seed(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved_args = _ckpt_args(ckpt)

    if args.train_cases_file:
        train_cases = _load_cases_file(args.train_cases_file)
    else:
        train_cases = ckpt.get("train_case_names") or ckpt.get("case_names")
    if args.eval_cases_file:
        eval_cases = _load_cases_file(args.eval_cases_file)
    else:
        eval_cases = ckpt.get("val_case_names") or ckpt.get("case_names")

    if not train_cases:
        raise ValueError("No train cases found. Pass --train_cases_file or use a train/val checkpoint.")
    if not eval_cases:
        raise ValueError("No eval cases found. Pass --eval_cases_file or use a train/val checkpoint.")

    train_dataset = _build_dataset(args, saved_args, ckpt, train_cases)
    eval_dataset = _build_dataset(args, saved_args, ckpt, eval_cases)

    conditioner = OstiumConditioner(
        vessel_feat_dim=int(saved_args.get("vessel_feat_dim", 64)),
        ostium_plane_dim=8,
        ostium_feat_dim=16,
        cond_out_dim=int(saved_args.get("vessel_cond_dim", 32)),
    ).to(device)
    conditioner.load_state_dict(ckpt["conditioner"])
    conditioner.eval()

    train = _collect(train_dataset, args.batch_size, args.num_workers, device, conditioner=conditioner)
    eval_data = _collect(eval_dataset, args.batch_size, args.num_workers, device, conditioner=conditioner)

    train_x = train["x"]
    eval_x = eval_data["x"]
    mean_pred = train_x.mean(dim=0, keepdim=True).expand_as(eval_x)
    zero_pred = torch.zeros_like(eval_x)

    nearest_kwargs = {
        "query_cases": eval_dataset.case_names,
        "train_cases": train_dataset.case_names,
        "exclude_same_case": args.exclude_same_case,
    }
    ghd_pred, ghd_idx, ghd_nn_dist = _nearest_prediction(eval_x, train_x, train_x, **nearest_kwargs)
    ostium_pred, ostium_idx, ostium_nn_dist = _nearest_prediction(
        eval_data["ostium"], train["ostium"], train_x, **nearest_kwargs
    )
    geom_pred, geom_idx, geom_nn_dist = _nearest_prediction(
        _geom_summary_features(eval_data), _geom_summary_features(train), train_x, **nearest_kwargs
    )
    learned_pred, learned_idx, learned_nn_dist = _nearest_prediction(
        eval_data["learned_cond"], train["learned_cond"], train_x, **nearest_kwargs
    )

    metrics = {
        "zero_rmse": _rmse(zero_pred, eval_x),
        "train_mean_rmse": _rmse(mean_pred, eval_x),
        "oracle_ghd_nn_rmse": _rmse(ghd_pred, eval_x),
        "ostium_nn_rmse": _rmse(ostium_pred, eval_x),
        "geom_summary_nn_rmse": _rmse(geom_pred, eval_x),
        "learned_cond_nn_rmse": _rmse(learned_pred, eval_x),
        "oracle_ghd_nn_feature_dist": ghd_nn_dist,
        "ostium_nn_feature_dist": ostium_nn_dist,
        "geom_summary_nn_feature_dist": geom_nn_dist,
        "learned_cond_nn_feature_dist": learned_nn_dist,
    }

    rows = []
    for i, case in enumerate(eval_dataset.case_names):
        row = {"case": case}
        for name, values in metrics.items():
            _add_metric(row, name, values, i)
        row["nn_case_by_ghd"] = train_dataset.case_names[int(ghd_idx[i])]
        row["nn_case_by_ostium"] = train_dataset.case_names[int(ostium_idx[i])]
        row["nn_case_by_geom"] = train_dataset.case_names[int(geom_idx[i])]
        row["nn_case_by_learned_cond"] = train_dataset.case_names[int(learned_idx[i])]
        rows.append(row)

    summary = {
        "ckpt": args.ckpt,
        "condition_space": _get_config(args, saved_args, "condition_space", "ghd_local"),
        "epoch": int(ckpt.get("epoch", -1)),
        "num_train_cases": len(train_dataset),
        "num_eval_cases": len(eval_dataset),
        "exclude_same_case": bool(args.exclude_same_case),
    }
    summary.update(_summarize(rows))

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
