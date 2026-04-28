#!/usr/bin/env python
"""Generate visual results from the v5 vessel-aware CVAE checkpoint.

v5 = trained on the v3 GHD round-robin tokens with FIXED eigenvectors
     (eigen_chk_144.pkl). The Graph_Harmonic_Deform here MUST load the same
     eigen_chk, otherwise the basis would differ and decoding would be garbage.
"""
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
from pytorch3d.transforms import axis_angle_to_matrix
from ghd.base.graph_harmonic_deformation import Graph_Harmonic_Deform

# ── 1. Paths ──────────────────────────────────────────────
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

# ── 2. Canonical mesh + ring ─────────────────────────────
with open(OPA, "rb") as f: opa = pickle.load(f)
ring_idx = opa["op_v_indices"][0]

# ── 3. LightGHDReconstruct (eigen_chk MUST match the fitting run) ──
canonical_Meshes = load_objs_as_meshes([CAN]).to(device)
norm = torch.max(torch.norm(canonical_Meshes.verts_packed(), dim=-1)).item() * 1.10
canonical_Meshes = canonical_Meshes.update_padded(canonical_Meshes.verts_padded() / norm)
ghd_mod = Graph_Harmonic_Deform(canonical_Meshes, num_Basis=144, eigen_chk=EIGEN)
eigvec = ghd_mod.GBH_eigvec.float()  # [V, num_Basis]
num_Basis = eigvec.shape[1]
faces_pt3d = canonical_Meshes.faces_padded()
faces_np = faces_pt3d.squeeze(0).cpu().numpy()
verts_base = canonical_Meshes.verts_padded()

def reconstruct_verts(coeff_flat, mean, std, R=None, s=None, T=None):
    c = coeff_flat * std + mean
    c = c.reshape(1, num_Basis, 3)
    offset = torch.einsum('vm,bmc->bvc', eigvec, c)
    v = verts_base + offset
    if R is not None and s is not None and T is not None:
        R_mat = axis_angle_to_matrix(R)
        v = (v @ R_mat.transpose(-1, -2)) * s.unsqueeze(-1) + T.unsqueeze(1)
    return v.squeeze(0).numpy()

# ── 4. Dataset ───────────────────────────────────────────
from models.vae_datasets_vessel import VesselAwareGHDDataset
cases = [c for c in os.listdir(GHD_CHK_ROOT) if os.path.isdir(os.path.join(GHD_CHK_ROOT, c))]
dataset = VesselAwareGHDDataset(
    ghd_chk_root=GHD_CHK_ROOT,
    ghd_run=GHD_RUN,
    ghd_chk_name="ghb_fitting_checkpoint.pkl",
    data_root=DATA, cases=cases, normalize=True, withscale=False, num_vessel_pts=256,
)
ghd_mean, ghd_std = dataset.get_mean_std()

# ── 5. Models ────────────────────────────────────────────
from models.vessel_conditioner import OstiumConditioner
from models.vessel_aware_cvae import VesselAwareCVAE
cond_net = OstiumConditioner(
    vessel_feat_dim=64, ostium_plane_dim=8, ostium_feat_dim=16, cond_out_dim=32,
).to(device)
input_dim = dataset.get_dim()
cvae = VesselAwareCVAE(
    input_dim=input_dim, hidden_dim=256, latent_dim=64,
    vessel_cond_dim=32, extra_cond_dim=0, dropout=0.02,
).to(device)
ckpt = torch.load(CKPT, map_location="cpu")
cond_net.load_state_dict(ckpt["conditioner"]); cond_net.eval()
cvae.load_state_dict(ckpt["generator"]); cvae.eval()
print(f"Models loaded, dataset: {len(dataset)} cases")

# ── 6. Pick test cases ───────────────────────────────────
rng = np.random.RandomState(42)
idxs = rng.choice(len(dataset), size=4, replace=False)

def load_rst(case_name):
    chk_path = os.path.join(GHD_CHK_ROOT, case_name, GHD_RUN, "ghb_fitting_checkpoint.pkl")
    with open(chk_path, 'rb') as f:
        chk = pickle.load(f)
    return chk['R'], chk['s'].abs(), chk['T']

rows = []
for idx in idxs:
    case_name = dataset.case_names[idx]
    item = dataset[idx]
    x   = item['ghd'].unsqueeze(0).to(device)
    osp = item['ostium_params'].unsqueeze(0).to(device)
    vpt = item['vessel_pts'].unsqueeze(0).to(device)
    R_case, s_case, T_case = load_rst(case_name)
    with torch.no_grad():
        cond = cond_net(vpt, osp)
        recon, mu, logvar = cvae(x, cond, deterministic=True)
        gt_v    = reconstruct_verts(x,     ghd_mean, ghd_std, R_case, s_case, T_case)
        recon_v = reconstruct_verts(recon, ghd_mean, ghd_std, R_case, s_case, T_case)
        samples_v = []
        for _ in range(2):
            z = torch.randn(1, 64)
            sam = cvae.decode(z, cond)
            samples_v.append(reconstruct_verts(sam, ghd_mean, ghd_std, R_case, s_case, T_case))
    rows.append((case_name, gt_v, recon_v, samples_v[0], samples_v[1]))
    print(f"  {case_name}: done")

# ── 7. Save OBJs ─────────────────────────────────────────
for case_name, gt_v, recon_v, s1, s2 in rows:
    for label, verts in [("gt", gt_v), ("recon", recon_v), ("sample1", s1), ("sample2", s2)]:
        m = trimesh.Trimesh(vertices=verts, faces=faces_np, process=False)
        m.export(str(VIZ / f"{case_name}_{label}.obj"))
print("OBJs saved to", VIZ)

# ── 8. Render ────────────────────────────────────────────
render_device = torch.device("cpu")
faces_t = torch.from_numpy(faces_np).long()

def render_mesh(verts_np, elev=25, azim=135, img_size=512):
    v = torch.from_numpy(verts_np).float().unsqueeze(0)
    f = faces_t.unsqueeze(0)
    center = v.mean(dim=1, keepdim=True)
    v_centered = v - center
    max_extent = v_centered.abs().max().item()
    dist = max_extent * 4.5
    colors = torch.tensor([0.55, 0.75, 0.95]).expand_as(v)
    ring_mask = torch.zeros(v.shape[1], dtype=torch.bool)
    ring_mask[ring_idx] = True
    colors_mod = colors.clone()
    colors_mod[0, ring_mask] = torch.tensor([0.95, 0.2, 0.2])
    textures = TexturesVertex(verts_features=colors_mod)
    mesh = Meshes(verts=v, faces=f, textures=textures).to(render_device)
    at = center.squeeze(0)
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=at)
    cameras = FoVPerspectiveCameras(device=render_device, R=R, T=T)
    raster_settings = RasterizationSettings(
        image_size=img_size, blur_radius=0.0, faces_per_pixel=1, cull_backfaces=False,
    )
    lights = PointLights(device=render_device, location=[[2.0, 2.0, 2.0]])
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=render_device, cameras=cameras, lights=lights),
    )
    img = renderer(mesh)
    img = img[0, ..., :3].clamp(0, 1).numpy()
    return (img * 255).astype(np.uint8)

titles = ["GT", "Recon", "Sample 1", "Sample 2"]
for view_name, elev, azim in [("view1", 20, 135), ("view2", -15, 45)]:
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    for r, (case_name, gt_v, recon_v, s1, s2) in enumerate(rows):
        for c, (label, verts) in enumerate(zip(titles, [gt_v, recon_v, s1, s2])):
            img = render_mesh(verts, elev=elev, azim=azim)
            ax = axes[r][c]
            ax.imshow(img)
            ax.set_title(f"{case_name[:25]}\n{label}" if c == 0 else label, fontsize=9)
            ax.axis("off")
    plt.suptitle(f"v5 Vessel-Aware CVAE (cap_v6_rr_v3, eigen_fixed) — 10k ep ({view_name})",
                 fontsize=14, y=1.01)
    plt.tight_layout()
    out = str(VIZ / f"v5_results_gallery_{view_name}.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print("Gallery saved:", out)
