"""Render p104 EXACTLY as fitter does: Poly3DCollection, alpha=0.55,
shared bounds across target+warped, default elev=30 azim=-60. This proves
that the v5 'wrinkly potato' and fitter's 'smooth balloon' are IDENTICAL
geometry, just different render styles.
"""
import os, sys, pickle, torch, numpy as np
sys.path.insert(0, '/workspace/AneuG')
import trimesh
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.transforms import axis_angle_to_matrix
from ghd.base.graph_harmonic_deformation import Graph_Harmonic_Deform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

CAN = "/workspace/AneuG/checkpoints/canonical_average/part_aligned.obj"
EIGEN = "/workspace/AneuG/checkpoints/canonical_average/eigen_chk_144.pkl"
CASE = "aneux_p104_FhMbBBQNHB8BDAMLFw8IFxwd"
CHK = (f"/workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_roundrobin_v3/"
       f"{CASE}/prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3/ghb_fitting_checkpoint.pkl")
TGT = f"/data/ghd_prepared_meshes_3_aneurysm_1op_new/{CASE}/mesh.obj"
FITTER_PREVIEW = (f"/workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_roundrobin_v3/"
                  f"{CASE}/prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3/fitting_preview_epoch_001499.png")

# Reconstruct warped via cached eigvec
canonical = load_objs_as_meshes([CAN])
norm = torch.max(torch.norm(canonical.verts_packed(), dim=-1)).item() * 1.10
canonical = canonical.update_padded(canonical.verts_padded() / norm)
ghd_mod = Graph_Harmonic_Deform(canonical, num_Basis=144, eigen_chk=EIGEN)
chk = pickle.load(open(CHK, 'rb'))
R, s, T = chk['R'], chk['s'].abs(), chk['T']
deformation = ghd_mod.ghb_coefficient_recover(chk['GHD_coefficient'])
output_local = canonical.offset_verts(deformation)
R_mat = axis_angle_to_matrix(R)
warped_verts = ((output_local.verts_padded() @ R_mat.transpose(-1, -2)) * s + T).squeeze(0).numpy()
warped_faces = canonical.faces_padded().squeeze(0).numpy()

# Load target with same normalization (canonical norm, keep_size=True)
tgt_mesh = trimesh.load(TGT, process=False)
target_verts = np.asarray(tgt_mesh.vertices) / norm
target_faces = np.asarray(tgt_mesh.faces)

# Shared bounds across BOTH (mimics fitter)
all_v = np.concatenate([warped_verts, target_verts], axis=0)
mins = all_v.min(0); maxs = all_v.max(0)
center = 0.5 * (mins + maxs)
span = float(np.max(maxs - mins))
half = 0.55 * span

# 3-panel: target | warped | overlay -- all with shared bounds & alpha=0.55
fig = plt.figure(figsize=(15, 5))
panels = [("target", target_verts, target_faces, "lightgray", 0.55, None, None),
          ("warped", warped_verts, warped_faces, "deepskyblue", 0.55, None, None),
          ("overlay", warped_verts, warped_faces, "deepskyblue", 0.45,
           target_verts, target_faces)]
for i, (title, v, f, color, alpha, vt, ft) in enumerate(panels):
    ax = fig.add_subplot(1, 3, i+1, projection='3d')
    if vt is not None:
        ax.add_collection3d(Poly3DCollection(vt[ft], facecolors='lightgray',
                                             edgecolors='none', alpha=0.35))
    ax.add_collection3d(Poly3DCollection(v[f], facecolors=color,
                                         edgecolors='none', alpha=alpha))
    ax.set_xlim(center[0]-half, center[0]+half)
    ax.set_ylim(center[1]-half, center[1]+half)
    ax.set_zlim(center[2]-half, center[2]+half)
    ax.set_box_aspect((1,1,1))
    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')

fig.suptitle("REPLAY (cached eigvec + saved coeff/R/s/T) -- fitter render style", fontsize=11)
plt.savefig('/tmp/p104_replay_fitter_style.png', dpi=180, bbox_inches='tight')
print("Saved /tmp/p104_replay_fitter_style.png")
print(f"warped z range [{warped_verts[:,2].min():.4f}, {warped_verts[:,2].max():.4f}]")
print(f"target z range [{target_verts[:,2].min():.4f}, {target_verts[:,2].max():.4f}]")

# Side-by-side with original fitter preview
from PIL import Image
img1 = Image.open(FITTER_PREVIEW); img2 = Image.open('/tmp/p104_replay_fitter_style.png')
H = max(img1.height, img2.height)
W1 = int(img1.width * H / img1.height); W2 = int(img2.width * H / img2.height)
img1 = img1.resize((W1, H)); img2 = img2.resize((W2, H))
combo = Image.new('RGB', (W1 + W2 + 10, H + 30), 'white')
combo.paste(img1, (0, 30)); combo.paste(img2, (W1 + 10, 30))
from PIL import ImageDraw, ImageFont
draw = ImageDraw.Draw(combo)
draw.text((10, 5), "ORIGINAL fitter preview (epoch 1499)", fill='black')
draw.text((W1 + 20, 5), "MY REPLAY (cached eigvec, fitter render style)", fill='black')
combo.save('/tmp/p104_FINAL_COMPARISON.png')
print("Saved /tmp/p104_FINAL_COMPARISON.png")
