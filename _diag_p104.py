"""
Diagnostic: does viz_v5's reconstruct path give the SAME mesh as
the fitter's Graph_Harmonic_Deform_opening_alignment_dynamic?
"""
import os, sys, pickle, torch, numpy as np
sys.path.insert(0, '/workspace/AneuG')
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.transforms import axis_angle_to_matrix
from ghd.base.graph_harmonic_deformation import (
    Graph_Harmonic_Deform,
    Graph_Harmonic_Deform_opening_alignment_dynamic,
)

device = torch.device('cpu')
CAN = "/workspace/AneuG/checkpoints/canonical_average/part_aligned.obj"
EIGEN = "/workspace/AneuG/checkpoints/canonical_average/eigen_chk_144.pkl"
CHK = ("/workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_roundrobin_v3/"
       "aneux_p104_FhMbBBQNHB8BDAMLFw8IFxwd/"
       "prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3/"
       "ghb_fitting_checkpoint.pkl")

# ----- Path A: viz_v5's reconstruct -----
canonical = load_objs_as_meshes([CAN])
norm = torch.max(torch.norm(canonical.verts_packed(), dim=-1)).item() * 1.10
print(f"norm = {norm:.6f}")
canonical = canonical.update_padded(canonical.verts_padded() / norm)
ghd_mod = Graph_Harmonic_Deform(canonical, num_Basis=144, eigen_chk=EIGEN)
eigvec = ghd_mod.GBH_eigvec.float()
verts_base = canonical.verts_padded()  # [1,V,3]

chk = pickle.load(open(CHK, 'rb'))
R, s, T = chk['R'], chk['s'].abs(), chk['T']
coeff = chk['GHD_coefficient']  # [144, 3]

c = coeff.unsqueeze(0)  # [1, 144, 3]
offset = torch.einsum('vm,bmc->bvc', eigvec, c)
v_A = verts_base + offset  # [1, V, 3]
R_mat = axis_angle_to_matrix(R)
v_A_xform = (v_A @ R_mat.transpose(-1, -2)) * s.unsqueeze(-1) + T.unsqueeze(1)
print(f"Path A (viz_v5): verts shape {v_A_xform.shape}, "
      f"z range [{v_A_xform[...,2].min():.4f}, {v_A_xform[...,2].max():.4f}]")

# ----- Path B: would need full registration class. Skip. Just run forward of base eigvec
# matching ghb_coefficient_recover -> base.offset_verts -> R/s/T -----
deformation = ghd_mod.ghb_coefficient_recover(coeff)  # [V, 3]
output_shape = canonical.offset_verts(deformation)
v_B_local = output_shape.verts_padded()
v_B = (v_B_local @ R_mat.transpose(-1, -2)) * s + T
print(f"Path B (Pytorch3d): verts shape {v_B.shape}, "
      f"z range [{v_B[...,2].min():.4f}, {v_B[...,2].max():.4f}]")

diff = (v_A_xform - v_B).abs()
print(f"max abs diff A vs B: {diff.max().item():.6e}")
print(f"mean abs diff A vs B: {diff.mean().item():.6e}")

# ----- Render Path A using fitter's _draw_mesh style for direct comparison -----
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

verts_np = v_A_xform.squeeze(0).numpy()
faces_np = canonical.faces_padded().squeeze(0).numpy()
tris = verts_np[faces_np]

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d')
pc = Poly3DCollection(tris, facecolors=(0.0, 0.6, 1.0, 0.65), edgecolors='none')
ax.add_collection3d(pc)
mn = verts_np.min(0); mx = verts_np.max(0)
c = (mn + mx) / 2
r = (mx - mn).max() / 2
ax.set_xlim(c[0]-r, c[0]+r); ax.set_ylim(c[1]-r, c[1]+r); ax.set_zlim(c[2]-r, c[2]+r)
ax.set_box_aspect((1,1,1))
ax.set_title(f"Path A reconstruction (cached eigvec) | s={s.item():.3f}")
plt.savefig('/tmp/p104_pathA.png', dpi=180, bbox_inches='tight')
print("Saved /tmp/p104_pathA.png")
