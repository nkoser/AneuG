#!/usr/bin/env python
"""Build a CVAE train/val split from finished per-case GHD checkpoints.

The vessel-aware CVAE dataset only checks that a checkpoint file exists.  This
helper is stricter: it keeps only cases whose final GHD checkpoint reached a
minimum epoch and whose conditioning files are present.
"""
import argparse
import io
import json
import os
import pickle
import random
from pathlib import Path

import torch


class TorchCPUUnpickler(pickle.Unpickler):
    """Load pickles that may contain CUDA tensors onto CPU."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)
        return super().find_class(module, name)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return TorchCPUUnpickler(f).load()


def has_condition_files(data_root, aligned_root, case):
    required = [
        Path(data_root) / case / "01_mesh" / "mesh.obj",
        Path(data_root) / case / "02_labels" / "labels.npy",
        Path(data_root) / case / "05_submeshes" / "vessel_submesh.obj",
        Path(data_root) / case / "07_other" / "centroid_ostium.npy",
        Path(data_root) / case / "07_other" / "normal_vector.npy",
        Path(aligned_root) / case / "prealign_transform.npy",
    ]
    return all(path.exists() for path in required)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ghd-root", required=True)
    p.add_argument("--ghd-run", default="prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3")
    p.add_argument("--ghd-chk-name", default="ghb_fitting_checkpoint.pkl")
    p.add_argument("--data-root", default="/data/prepared_meshes_3")
    p.add_argument("--aligned-root", default="/data/ghd_prepared_meshes_3_aneurysm_1op_new")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-epoch", type=int, default=3999)
    p.add_argument("--val-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.ghd_root)
    if not root.is_dir():
        raise FileNotFoundError(f"GHD root not found: {root}")

    kept = []
    skipped = {
        "missing_checkpoint": [],
        "low_epoch": [],
        "bad_checkpoint": [],
        "nonfinite_ghd": [],
        "missing_condition_files": [],
    }

    for case_dir in sorted(root.iterdir()):
        if not case_dir.is_dir():
            continue
        case = case_dir.name
        chk_path = case_dir / args.ghd_run / args.ghd_chk_name
        if not chk_path.exists():
            skipped["missing_checkpoint"].append(case)
            continue
        try:
            chk = load_checkpoint(chk_path)
        except Exception:
            skipped["bad_checkpoint"].append(case)
            continue
        epoch = int(chk.get("epoch", -1))
        if epoch < args.min_epoch:
            skipped["low_epoch"].append({"case": case, "epoch": epoch})
            continue
        ghd = chk.get("GHD_coefficient", None)
        if ghd is None or torch.isnan(ghd).any() or torch.isinf(ghd).any():
            skipped["nonfinite_ghd"].append(case)
            continue
        if not has_condition_files(args.data_root, args.aligned_root, case):
            skipped["missing_condition_files"].append(case)
            continue
        kept.append(case)

    rng = random.Random(args.seed)
    shuffled = kept[:]
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * args.val_frac)) if shuffled else 0
    val_cases = sorted(shuffled[:val_count])
    train_cases = sorted(shuffled[val_count:])
    all_cases = sorted(kept)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("cases_all.json", all_cases),
        ("cases_train.json", train_cases),
        ("cases_val.json", val_cases),
    ):
        with (out_dir / name).open("w") as f:
            json.dump(data, f, indent=2)

    summary = {
        "ghd_root": str(root),
        "ghd_run": args.ghd_run,
        "ghd_chk_name": args.ghd_chk_name,
        "min_epoch": args.min_epoch,
        "seed": args.seed,
        "val_frac": args.val_frac,
        "num_all": len(all_cases),
        "num_train": len(train_cases),
        "num_val": len(val_cases),
        "skipped_counts": {key: len(value) for key, value in skipped.items()},
        "skipped": skipped,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
