#!/usr/bin/env python
"""v5 visualization in CANONICAL frame (no R,s,T applied) — matches fitting preview."""
import torch, numpy as np, pickle, trimesh, pathlib, matplotlib, sys, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform, FoVPerspectiveCameras,
    RasterizationSettings, MeshRasterizer,
    SoftPhongShader, MeshRenderer, PointLights,
    TexturesVertex,
)
from ghd.base.graph_harmonic_deformation import Graph_Harmonic_Deform

META   = "v5_cap_v6rr_v3_eigen_fixed"
CKPT   = f"checkpoints/vessel_aware_cvae/{META}/models_epoch_10000.pth"
VIZ    = pathlib.Path(f"checkpoints/vessel_aware_cvae/{META}/viz"); VIZ.mkdir(exist_ok=True, parents=True)
CAN    = "checkpoints/canonical_average/part_aligned.obj"
OPA    = "checkpoints/canonical_average/opa_checkpoint_1op.pkl"
EIGEN  = "checkpoints/canonical_average/eigen_chk_144.pkl"
DATA   = "/data/prepared_meshes_3"
GHD_CHK_ROOT = "/workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_roundrobin_v3"
GHD_RUN      = "prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3"
device = torch.device("cpu")

with open(OPA, "rb") as f: opa = pickle.load(f)
ring_idx = opa["op_v_indices"][0]

canonical_Meshes = load_objs_as_meshes([CAN]).to(device)
norm = torch.max(torch.norm(canonical_Meshes.verts_packed(), dim=-1)).item() * 1.10
canonical_Meshes = canonical_Meshes.update_padded(canonical_Meshes.verts_padded() / norm)
ghd_mod = Graph_Harmonic_Deform(canonical_Meshes, num_Basis=144, eigen_chk=EIGEN)
eigvec = ghd_mod.GBH_eigvec.float()
num_Basis = eigvec.shape[1]
faces_np = canonical_Meshes.faces_padded().squeeze(0).cpu().numpy()
verts_base = canonical_Meshes.verts_padded()

def reconstruct_verts(coeff_flat, mean, std):
    """Canonical frame only — NO R,s,T."""
    c = coeff_flat * std + mean
    c = c.reshape(1, num_Basis, 3)
    offset = torch.einsum('vm,bmc->bvc', eigvec, c)
    return (verts_base + offset).squeeze(0).numpy()

from models.vae_datasets_vessel import VesselAwareGHDDataset
cases = [c for c in os.listdir(GHD_CHK_ROOT) if os.path.isdir(os.path.join(GHD_CHK_ROOT, c))]
dataset = VesselAwareGHDDataset(
    ghd_chk_root=GHD_CHK_ROOT, ghd_run=GHD_RUN,
    ghd_chk_name="ghb_fitting_checkpoint.pkl",
    data_root=DATA, cases=cases, normalize=True, withscale=False, num_vessel_pts=256,
)
ghd_mean, ghd_std = dataset.get_mean_std()

from models.vessel_conditioner import OstiumConditioner
from models.vessel_aware_cvae import VesselAwareCVAE
cond_net = OstiumConditioner(vessel_feat_dim=64, ostium_plane_dim=8, ostium_feat_dim=16, cond_out_dim=32).to(device)
cvae = VesselAwareCVAE(input_dim=dataset.get_dim(), hidden_dim=256, latent_dim=64,
                        vessel_cond_dim=32, extra_cond_dim=0, dropout=0.02).to(device)
ckpt = torch.load(CKPT, map_location="cpu")
cond_net.load_state_dict(ckpt["conditioner"]); cond_net.eval()
cvae.load_state_dict(ckpt["generator"]); cvae.eval()
print(f"Models loaded, dataset: {len(dataset)} cases")

rng = np.random.RandomState(42)
idxs = rng.choice(len(dataset), size=4, replace=False)

rows = []
for idx in idxs:
    case_name = dataset.case_names[idx]
    item = dataset[idx]
    x   = item['ghd'].unsqueeze(0).to(device)
    osp = item['ostium_params'].unsqueeze(0).to(device)
    vpt = item['vessel_pts'].unsqueeze(0).to(device)
    with torch.no_grad():
        cond = cond_net(vpt, osp)
        recon, mu, logvar = cvae(x, cond, deterministic=True)
        gt_v    = reconstruct_verts(x, ghd_mean, ghd_std)
        recon_v = reconstruct_verts(recon, ghd_mean, ghd_std)
        samples_v = []
        for _ in range(2):
            z = torch.randn(1, 64)
            sam = cvae.decode(z, cond)
            samples_v.append(reconstruct_verts(sam, ghd_mean, ghd_std))
    rows.append((case_name, gt_v, recon_v, samples_v[0], samples_v[1]))
    print(f"  {case_name}: done")

faces_t = torch.from_numpy(faces_np).long()
render_device = torch.device("cpu")

def render_mesh(verts_np, elev=20, azim=135, img_size=512):
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
    raster = RasterizationSettings(image_size=img_size, blur_radius=0.0, faces_per_pixel=1, cull_backfaces=False)
    lights = PointLights(device=render_device, location=[[2.0, 2.0, 2.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
        shader=SoftPhongShader(device=render_device, cameras=cameras, lights=lights),
    )
    img = renderer(mesh)[0, ..., :3].clamp(0, 1).numpy()
    return (img * 255).astype(np.uint8)

titles = ["GT", "Recon", "Sample 1", "Sample 2"]
# canonical frame: ostium near origin, dome along +Z. Use slight elevation from above-side.
for view_name, elev, azim in [("canon_side", 15, 35), ("canon_front", 5, 0)]:
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for r, (case_name, gt_v, recon_v, s1, s2) in enumerate(rows):
        for c, (label, verts) in enumerate(zip(titles, [gt_v, recon_v, s1, s2])):
            img = render_mesh(verts, elev=elev, azim=azim)
            ax = axes[r][c]; ax.imshow(img)
            ax.set_title(f"{case_name[:25]}\n{label}" if c == 0 else label, fontsize=9)
            ax.axis("off")
    plt.suptitle(f"v5 CVAE (canonical frame) — 10k ep ({view_name})", fontsize=14, y=1.01)
    plt.tight_layout()
    out = str(VIZ / f"v5_results_canonical_{view_name}.png")
    fig.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
    print("Gallery saved:", out)
