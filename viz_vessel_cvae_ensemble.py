#!/usr/bin/env python
"""Render an ensemble vessel-aware CVAE: GT | Recon (per member 0) | K samples.

Usage:
  python viz_vessel_cvae_ensemble.py \\
      --ckpts checkpoints/vessel_aware_cvae/cvae_Fpca95_seed{1..5}_*/models_best_val.pth \\
      --cases_file checkpoints/vessel_aware_cvae/splits_only3999_20260427_223850/cases_val.json \\
      --num_cases 4 --k_per_member 1 \\
      --out_dir checkpoints/vessel_aware_cvae/viz_ensemble_Fpca95
"""
import argparse, json, os, pathlib, pickle, sys
import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, MeshRasterizer, MeshRenderer, PointLights,
    RasterizationSettings, SoftPhongShader, TexturesVertex, look_at_view_transform,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import axis_angle_to_matrix
from ghd.base.graph_harmonic_deformation import Graph_Harmonic_Deform

from models.vae_datasets_vessel import VesselAwareGHDDataset
from models.vessel_aware_cvae_ensemble import VesselAwareCVAEEnsemble


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpts", nargs="+", required=True)
    p.add_argument("--cases_file", required=True)
    p.add_argument("--num_cases", type=int, default=4)
    p.add_argument("--k_per_member", type=int, default=1)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--canonical_mesh", default="checkpoints/canonical_average/part_aligned.obj")
    p.add_argument("--eigen_chk", default="checkpoints/canonical_average/eigen_chk_144.pkl")
    p.add_argument("--opa_chk", default="checkpoints/canonical_average/opa_checkpoint_1op.pkl")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    obj_dir = out_dir / "objs"; obj_dir.mkdir(exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.RandomState(args.seed)

    # Ensemble + dataset (raw, no normalization).
    ens = VesselAwareCVAEEnsemble(args.ckpts, device=str(device))
    sa = ens.members[0]["args"]
    cases_all = json.load(open(args.cases_file))
    pick_idx = rng.choice(len(cases_all), size=min(args.num_cases, len(cases_all)), replace=False)
    cases = [cases_all[i] for i in pick_idx]
    ds = VesselAwareGHDDataset(
        ghd_chk_root=sa["ghd_chk_root"], ghd_run=sa["ghd_run"], ghd_chk_name=sa["ghd_chk_name"],
        data_root=sa["data_root"], cases=cases,
        num_vessel_pts=int(sa.get("num_vessel_pts", 256)),
        condition_space=sa.get("condition_space", "ghd_local"),
        aligned_data_root=sa.get("aligned_data_root"),
        canonical_mesh=sa.get("canonical_mesh"),
        canonical_norm_factor=float(sa.get("canonical_norm_factor", 1.10)),
        normalize=False,
    )
    print(f"Loaded {len(ds)} cases for viz")

    # GHD reconstructor.
    canonical_Meshes = load_objs_as_meshes([args.canonical_mesh]).to(device)
    norm = torch.max(torch.norm(canonical_Meshes.verts_packed(), dim=-1)).item() * 1.10
    canonical_Meshes = canonical_Meshes.update_padded(canonical_Meshes.verts_padded() / norm)
    ghd_mod = Graph_Harmonic_Deform(canonical_Meshes, num_Basis=144, eigen_chk=args.eigen_chk)
    eigvec = ghd_mod.GBH_eigvec.float().to(device)
    num_Basis = eigvec.shape[1]
    faces_np = canonical_Meshes.faces_padded().squeeze(0).cpu().numpy()
    verts_base = canonical_Meshes.verts_padded().to(device)
    with open(args.opa_chk, "rb") as f:
        opa = pickle.load(f)
    ring_idx = opa["op_v_indices"][0]

    ghd_mean = ens.ghd_mean  # [1, 432] on device
    ghd_std = ens.ghd_std

    def reconstruct_verts(coeff_norm, R=None, s=None, T=None):
        """coeff_norm: [432] in normalized GHD space -> verts [V, 3] numpy."""
        c = coeff_norm.to(device) * ghd_std.squeeze(0) + ghd_mean.squeeze(0)
        c = c.reshape(1, num_Basis, 3)
        offset = torch.einsum("vm,bmc->bvc", eigvec, c)
        v = verts_base + offset
        if R is not None:
            R_mat = axis_angle_to_matrix(R.to(device))
            v = (v @ R_mat.transpose(-1, -2)) * s.to(device).unsqueeze(-1) + T.to(device).unsqueeze(1)
        return v.squeeze(0).cpu().numpy()

    def load_rst(case_name):
        chk_path = os.path.join(sa["ghd_chk_root"], case_name, sa["ghd_run"], sa["ghd_chk_name"])
        with open(chk_path, "rb") as f:
            chk = pickle.load(f)
        return chk["R"], chk["s"].abs(), chk["T"]

    # Build batch and sample.
    vp = torch.stack([ds[i]["vessel_pts"] for i in range(len(ds))]).to(device)
    op = torch.stack([ds[i]["ostium_params"] for i in range(len(ds))]).to(device)
    samples_norm = ens.sample(vp, op, k_per_member=args.k_per_member)  # [M*K, B, 432]

    # GT in normalized space (each ds[i]['ghd'] is raw, so normalize with ens stats).
    gt_norm = torch.stack([ds[i]["ghd"] for i in range(len(ds))]).to(device)
    gt_norm = (gt_norm - ghd_mean) / ghd_std  # broadcast

    # Best-of-(M*K) per case + per-case best sample index for galleries.
    rmse = torch.sqrt(((samples_norm - gt_norm.unsqueeze(0)) ** 2).mean(-1) + 1e-8)  # [S, B]
    best_idx = rmse.argmin(0)
    best_rmse = rmse.min(0).values
    print("Per-case best RMSE:", [f"{r:.3f}" for r in best_rmse.cpu().tolist()])

    # Pick: GT, best-of-S, 3 random samples from S.
    n_samples = samples_norm.shape[0]
    titles = ["GT", "Best of S"] + [f"Sample {i+1}" for i in range(min(3, n_samples))]

    rows = []
    for b, case_name in enumerate(ds.case_names):
        R_c, s_c, T_c = load_rst(case_name)
        gt_v = reconstruct_verts(gt_norm[b], R_c, s_c, T_c)
        best_v = reconstruct_verts(samples_norm[best_idx[b], b], R_c, s_c, T_c)
        rand_idx = rng.choice(n_samples, size=min(3, n_samples), replace=False)
        smp_vs = [reconstruct_verts(samples_norm[i, b], R_c, s_c, T_c) for i in rand_idx]
        rows.append((case_name, gt_v, best_v, smp_vs))
        # save OBJs
        for label, vv in [("gt", gt_v), ("best", best_v)] + [(f"s{i+1}", v) for i, v in enumerate(smp_vs)]:
            trimesh.Trimesh(vertices=vv, faces=faces_np, process=False).export(str(obj_dir / f"{case_name}_{label}.obj"))
    print(f"Saved OBJs to {obj_dir}")

    # Render gallery.
    render_device = torch.device("cpu")
    faces_t = torch.from_numpy(faces_np).long()

    def render(verts_np, elev=20, azim=135, img_size=384):
        v = torch.from_numpy(verts_np).float().unsqueeze(0)
        f = faces_t.unsqueeze(0)
        center = v.mean(dim=1, keepdim=True)
        max_extent = (v - center).abs().max().item()
        dist = max_extent * 4.5
        colors = torch.tensor([0.55, 0.75, 0.95]).expand_as(v).clone()
        colors[0, ring_idx] = torch.tensor([0.95, 0.2, 0.2])
        textures = TexturesVertex(verts_features=colors)
        mesh = Meshes(verts=v, faces=f, textures=textures).to(render_device)
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=center.squeeze(0))
        cameras = FoVPerspectiveCameras(device=render_device, R=R, T=T)
        rs = RasterizationSettings(image_size=img_size, blur_radius=0.0, faces_per_pixel=1)
        lights = PointLights(device=render_device, location=[[2.0, 2.0, 2.0]])
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=rs),
            shader=SoftPhongShader(device=render_device, cameras=cameras, lights=lights),
        )
        img = renderer(mesh)[0, ..., :3].clamp(0, 1).numpy()
        return (img * 255).astype(np.uint8)

    n_cols = len(titles)
    for view_name, elev, azim in [("v1", 20, 135), ("v2", -15, 45)]:
        fig, axes = plt.subplots(len(rows), n_cols, figsize=(n_cols * 3, len(rows) * 3))
        if len(rows) == 1:
            axes = axes[None, :]
        for r, (case_name, gt_v, best_v, smp_vs) in enumerate(rows):
            verts_seq = [gt_v, best_v] + smp_vs
            for c in range(n_cols):
                ax = axes[r][c]
                if c < len(verts_seq):
                    img = render(verts_seq[c], elev=elev, azim=azim)
                    ax.imshow(img)
                ax.set_title(f"{case_name[:18]}\n{titles[c]}" if c == 0 else titles[c], fontsize=9)
                ax.axis("off")
        plt.suptitle(f"Ensemble {len(ens)} members × {args.k_per_member} samples ({view_name})", fontsize=12, y=1.0)
        plt.tight_layout()
        out_png = out_dir / f"ensemble_gallery_{view_name}.png"
        fig.savefig(str(out_png), dpi=130, bbox_inches="tight")
        plt.close()
        print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
