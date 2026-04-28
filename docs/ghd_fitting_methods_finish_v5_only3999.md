# Methods: GHD Fitting Setup and Loss Formulation

This document describes the Graph Harmonic Deformation (GHD) fitting used to generate the checkpoints in:

```text
/workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999
```

The run follows the configuration in `config/ghd_fitting_config_cap_v6_finish_v5.yaml`, with the output root redirected to the `only3999` checkpoint directory. The text below is written as a paper-style Methods description, with additional implementation-level details for reproducibility.

## Experimental Setup

Each aneurysm case is fitted independently. The canonical template is `canonical_average`, and each target case is a prepared aneurysm mesh with one ostium opening:

```text
canonical mesh:
  /workspace/AneuG/checkpoints/canonical_average.obj

target mesh:
  /data/ghd_prepared_meshes_3_aneurysm_1op_new/{case}/part_aligned.obj
```

The fitting script does not recompute opening or centreline registration for this run:

```text
register = 0
redo_registration = 0
num_op = 1
```

Instead, it loads precomputed supervision checkpoints for each target case:

```text
opa_checkpoint_1op.pkl
diff_centreline_checkpoint_1op.pkl
```

The output path for each fitted case is:

```text
{save_root}/{case}/prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3/
```

where:

```text
save_root = /workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999
```

## Mesh Normalization

The canonical and target meshes are normalized before fitting. Since `keep_size = 0`, the target is not forced to share the canonical scale. Instead, each mesh is normalized by its own radius:

```text
r = 1.10 * max_i ||v_i||
v_norm = v / r
```

Opening vertices, reconstructed opening-cap vertices, target rim vertices, target opening surfaces, and target opening-plane centers are scaled consistently with the corresponding mesh. The centreline wave loops are then cleaned with a radius-dependent threshold, reducing overly dense or redundant loop samples.

## Deformation Parameterization

The deformation is represented with a spectral graph basis on the canonical mesh. A mixed graph Laplacian is constructed from three operators:

```text
L_mix = 1.0 * L_cotan + 0.1 * L_normalized + 0.1 * L_standard
```

The first `144` eigenvectors of this operator are used as the GHD basis:

```text
num_Basis = 144
```

For efficiency and reproducibility, the eigenvectors are loaded from:

```text
/workspace/AneuG/checkpoints/canonical_average/eigen_chk_144.pkl
```

Let:

```text
V0 in R^{N x 3}      canonical vertices
E  in R^{N x 144}    graph harmonic basis
C  in R^{144 x 3}    learnable GHD coefficients
```

The spectral deformation is:

```text
Delta V = E C
```

The deformed mesh is then globally transformed:

```text
V_fit = (V0 + Delta V) R^T |s| + T
```

where `R` is obtained from an axis-angle vector, `s` is a learnable scalar, and `T` is a translation. The optimized parameter set is:

```text
theta = {C, R, s, T}
```

The same spectral displacement and global transform are applied to the reconstructed opening mesh. This is important because the surface, ostium, and centreline losses all supervise the same deformed shape in the same coordinate frame.

## Training Objective

For each case, the optimization minimizes a weighted sum of losses:

```text
L_total(theta, t) =
    w_do(t)          L_do
  + w_p0(t)          L_p0
  + w_n1(t)          L_n1
  + w_lap(t)         L_laplacian
  + w_edge(t)        L_edge
  + w_cons(t)        L_consistency
  + w_rigid(t)       L_rigid
  + w_rim(t)         L_openings_p
  + w_osurf(t)       L_openings_surface_p
  + w_onorm(t)       L_openings_n
  + w_plane(t)       L_openings_plane
  + w_curv(t)        L_openings_rim_curvature
  + w_centroid(t)    L_openings_centroid_axis
  + w_cl(t)          L_diff_centreline
  + w_thick(t)       L_thickness
```

All raw losses are computed first. The scalar total loss is then assembled by multiplying each raw loss by its epoch-dependent weight. For list-valued opening losses, the list is summed over openings before weighting; in this run there is only one opening, so this is equivalent to a single scalar term.

## Surface Chamfer and Normal Losses

The global surface data term is computed from PyTorch3D surface samples. At every epoch, points and normals are sampled from the deformed source mesh and the target mesh:

```text
sample_num = 140000
```

The point loss is a bidirectional Chamfer distance:

```text
L_p0 = CD(P_fit, P_target)
```

The normal term is the normal component returned by the same Chamfer computation:

```text
L_n1 = normal_consistency(P_fit, P_target)
```

Sampling is stochastic across epochs because `sample_points_from_meshes` is called during each forward pass. If sampling fails because a mesh becomes locally degenerate, the code falls back to vertex sampling so that training can continue.

### Decapped Chamfer

The prepared meshes contain cap faces at the ostium. These faces are not used for the global Chamfer supervision:

```text
decap_chamfer = 1
decap_chamfer_rings = 1
```

The decap mask is built from the registered opening vertices. First, all opening-rim vertices are collected. Because `decap_chamfer_rings = 1`, this set is expanded by one mesh-adjacency ring. A face is removed from the global Chamfer mesh if all three of its vertices belong to this expanded opening set.

Thus:

```text
L_p0 and L_n1       computed on decapped meshes
regularizers       computed on full meshes
```

This avoids conflicting gradients near the ostium: the global surface Chamfer should fit the aneurysm body, while the ostium geometry is controlled by the dedicated opening losses.

## Differentiable Occupancy Loss

The occupancy term encourages volumetric agreement between fitted and target shape. Query points are static for each case:

```text
do_style = number_control_v2
do_number = 12000
```

The code uses `12000` inside points and `12000` outside points, so each case has:

```text
Q = 24000 query points
```

If `{target_case}/do_points.pt` exists, these points are loaded. Otherwise, the code samples points from the joint bounding box of the canonical and target meshes. Target occupancy is estimated with winding occupancy. Points with target occupancy above `0.95` are considered inside; points below `0.05` are considered outside.

During point generation, the sampler first collects an expanded candidate set:

```text
expand_ratio = 2
candidate inside points  = 2 * 12000
candidate outside points = 2 * 12000
```

The final inside points are taken from the inside candidate set. The final outside points are selected using an inverse-square field-strength score relative to the union of canonical and target vertices, which biases outside supervision toward the surface region.

For the fitted mesh, occupancy at query point `x_j` is computed with differentiable winding occupancy:

```text
p_j = sigmoid(100 * (W(V_fit, x_j) - 0.5))
```

The target label is binary:

```text
y_j in {0, 1}
```

The loss is an attention-weighted binary Dice loss:

```text
Dice_w =
  (2 * sum_j a_j p_j y_j + 1)
  /
  (sum_j a_j p_j + sum_j a_j y_j + 1 + 1e-5)

L_do = 1 - Dice_w
```

The attention weight `a_j` is computed from inverse-squared distances to target mesh vertices and linearly scaled to:

```text
1.0 <= a_j <= 3.0
```

This increases the importance of query points near the target surface. The differentiable occupancy dropper is disabled in this run:

```text
use_do_dropper = 0
```

so the same static set of query points is used at every epoch.

## Opening and Ostium Losses

The run is configured for a single ostium:

```text
num_op = 1
opening_match_mode = index
opening_loss_mode = rim_ordered
opening_normal_bidirectional = 1
op_sample_num = 1200
```

The target opening supervision is loaded from the opening checkpoint. When available, the code uses repaired target opening data:

```text
target cap surface:  op_target_rec_v, op_target_rec_f
target rim:          op_target_rim_v
target plane:        op_target_plane_center, op_target_plane_normal
```

If those fields are not available, it falls back to the reconstructed opening geometry stored in the registration checkpoint.

### Ordered Cyclic Rim Loss

The primary ostium boundary loss is stored under:

```text
loss_openings_p
```

Although the name is historical, in this configuration it is not a random boundary Chamfer. Because `opening_loss_mode = rim_ordered`, the loss is an ordered cyclic rim correspondence loss.

The deformed opening boundary is extracted from the opening-cap mesh. The target rim is loaded as a point set. Both loops are ordered in a best-fit plane by polar angle. They are then resampled as closed curves. With `op_sample_num = 1200`, the ordered rim loss uses:

```text
ordered rim samples = max(64, min(160, op_sample_num / 12))
                    = 100
```

Let `r_i` be the deformed rim samples and `q_i` the target rim samples after resampling. The loss searches over cyclic shifts and also allows a flipped traversal direction:

```text
L_openings_p =
  min_{shift, flip} (1 / N) sum_i ||r_i - q_{pi(i)}||_2^2
```

This makes the loss invariant to the arbitrary start vertex of the rim loop and robust to clockwise/counter-clockwise orientation differences.

### Opening Cap Surface Loss

The opening surface term compares the deformed reconstructed cap surface with the target opening surface:

```text
L_openings_surface_p = CD(S_open_fit, S_open_target)
```

This term uses surface sampling with:

```text
op_sample_num = 1200
```

It complements the ordered rim term: the rim term supervises the ostium boundary, while the cap-surface term regularizes the filled opening patch.

### Opening Normal Loss

The opening normal term aligns the fitted opening plane normal with the target plane normal:

```text
L_openings_n = 1 - |n_fit dot n_target|
```

The absolute value is used because `opening_normal_bidirectional = 1`; therefore, equivalent plane normals with opposite sign are not penalized.

This term has a very small base weight (`0.01`) because the plane and centroid-axis losses already provide stronger orientation signals.

### Opening Plane Loss

The planarity term estimates a best-fit plane for the deformed opening vertices. It penalizes both deviation from the fitted plane and deviation from the target plane:

```text
L_openings_plane =
  0.5 * [
    mean_v dist(v_open, P_self)^2
    +
    mean_v dist(v_open, P_target)^2
  ]
```

This is the strongest explicitly weighted opening term in the base configuration. It forces the reconstructed ostium to remain planar while also anchoring it to the target ostium plane.

### Rim Curvature Loss

The rim curvature term compares local shape variation along the ostium boundary. After ordering and resampling both rims, a turn-angle curvature proxy is computed:

```text
e_i     = normalize(r_{i+1} - r_i)
kappa_i = 1 - dot(e_{i-1}, e_i)
```

The fitted and target curvature profiles are compared with the same cyclic shift and flip invariance used for the ordered rim coordinates:

```text
L_openings_rim_curvature =
  min_{shift, flip} mean_i ||kappa_fit_i - kappa_target_{pi(i)}||_2^2
```

For `op_sample_num = 1200`, this term uses:

```text
curvature samples = max(48, min(128, op_sample_num / 16))
                  = 75
```

This helps preserve local ostium shape, for example ovality, bends, or local kinks in the rim.

### Centroid-Axis Loss

The centroid-axis loss is a robust low-frequency opening term:

```text
L_openings_centroid_axis =
  ||c_fit - c_target||_2^2
  +
  0.5 * (1 - |n_fit dot n_target|)
```

Unlike most opening terms, this loss is not warmed up in the dynamic weighting schedule; it is active at full base weight from the beginning. This is intentional: it gives a stable gross-alignment signal even when the detailed rim correspondence is still poor.

## Differentiable Centreline Loss

The target centreline supervision is derived from `wave_loops` in `diff_centreline_checkpoint_1op.pkl`. For each wave loop, the vertices in that loop are averaged to form one centreline point. The same operation is applied to the deformed canonical mesh, making the centreline representation differentiable with respect to the fitted vertices.

Let:

```text
C_fit     fitted centreline point cloud
C_target  target centreline point cloud
```

The loss is:

```text
L_diff_centreline = CD(C_fit, C_target)
```

This term encourages agreement of the parent-vessel axis and the aneurysm-to-vessel transition geometry, complementing surface and ostium losses.

## Mesh Regularization

Four geometric regularizers are applied to the full deformed mesh.

### Laplacian Smoothing

The Laplacian term uses PyTorch3D cotangent mesh Laplacian smoothing:

```text
L_laplacian = mesh_laplacian_smoothing(V_fit, method = "cot")
```

It suppresses local high-frequency artifacts introduced during fitting.

### Edge Length Loss

The edge loss penalizes deviation from the canonical mean edge length:

```text
L_edge = mesh_edge_loss(V_fit, mean_edge_length_canonical)
```

This helps avoid local stretching and collapsing.

### Normal Consistency

The normal consistency term penalizes inconsistent neighboring face normals:

```text
L_consistency = mesh_normal_consistency(V_fit)
```

It improves surface smoothness and discourages fold-like artifacts.

### Local Rigidity Loss

The local rigidity term is an as-rigid-as-possible prior. For each vertex neighborhood, the code solves a local orthogonal Procrustes problem between canonical and deformed neighbor offsets. It penalizes the residual after the best local rotation:

```text
L_rigid =
  mean_i sum_{j in N(i)} w_ij
    ||R_i (v_j^0 - v_i^0) - (v_j - v_i)||_2
```

where `R_i` is the best local rotation and `w_ij` are cotangent-derived weights. This discourages non-isometric distortion while still allowing the global aneurysm geometry to change.

## Thickness Loss

A thickness barrier is added with a low weight. Thickness is estimated by comparing each vertex against candidate opposing faces selected by a combination of normal alignment and Euclidean distance:

```text
MeshThickness(
  r = 0.2,
  num_bundle_filtered = 100,
  innerp_threshold = 0.6,
  num_sel = 25
)
```

The implemented barrier is:

```text
mask = 1 if |thickness| <= 0.1 else 0
signed = sign(sign_estimate)

L_thickness =
  mean [
    ReLU(0.04 - thickness * signed)
    +
    ReLU(0.01 - point_face_distance * signed)
  ] * mask
  +
  mean [
    1e-4 / (clamp(|sign_estimate|, min=5e-2)^2 + 1e-6)
  ] * mask
```

This term is not intended to drive the fit directly. It acts as a stabilizer against very thin, inverted, or ambiguous local configurations.

## Base Loss Weights

The base weights before dynamic scheduling are:

```text
loss_do                         1.25
loss_p0                         1.50
loss_n1                         0.80
loss_laplacian                  0.12
loss_edge                       0.12
loss_consistency                0.12
loss_rigid                     12.00
loss_openings_p                 8.00
loss_openings_surface_p         2.00
loss_openings_n                 0.01
loss_openings_plane            20.00
loss_openings_rim_curvature     0.60
loss_openings_centroid_axis     6.00
loss_diff_centreline            1.00
loss_thickness                  0.05
```

The large `loss_rigid` weight is strongest early in fitting, before being decayed. Among opening terms, the strongest weights are assigned to plane alignment, ordered rim correspondence, and centroid-axis alignment.

## Dynamic Weighting Schedule

The run uses:

```text
weighter_style = strategy_v2_robust_opening
```

Let:

```text
p(t) = clamp(t / (epochs - 1), 0, 1)
```

where `epochs = 4000`. The shape regularizers decay linearly:

```text
w_rigid(t)       = w_rigid(0)       * [(1 - p(t)) + 0.20 * p(t)]
w_laplacian(t)   = w_laplacian(0)   * [(1 - p(t)) + 0.35 * p(t)]
w_edge(t)        = w_edge(0)        * [(1 - p(t)) + 0.35 * p(t)]
w_consistency(t) = w_consistency(0) * [(1 - p(t)) + 0.35 * p(t)]
```

Thus at the end of a full 4000-epoch run:

```text
loss_rigid       12.00 -> 2.40
loss_laplacian    0.12 -> 0.042
loss_edge         0.12 -> 0.042
loss_consistency  0.12 -> 0.042
```

Opening and centreline terms are warmed up from reduced initial scales. The warmup is evaluated relative to the current fit start, so resumed cases ramp from the resume epoch:

```text
q(t) = clamp((t - fit_start_epoch) / warmup_epochs, 0, 1)
scale(t) = start_scale * (1 - q(t)) + 1.0 * q(t)
```

For non-suspicious cases, the base warmup is:

```text
warmup_epochs = 500
```

and the start scales are:

```text
loss_openings_p              0.45
loss_openings_surface_p      0.45
loss_openings_n              0.55
loss_openings_plane          0.18
loss_openings_rim_curvature  0.30
loss_diff_centreline         0.22
```

Therefore, in a non-suspicious case, the initial effective opening/centreline weights are:

```text
loss_openings_p              8.00 * 0.45 = 3.60
loss_openings_surface_p      2.00 * 0.45 = 0.90
loss_openings_n              0.01 * 0.55 = 0.0055
loss_openings_plane         20.00 * 0.18 = 3.60
loss_openings_rim_curvature  0.60 * 0.30 = 0.18
loss_diff_centreline         1.00 * 0.22 = 0.22
```

The centroid-axis term is not in the warmup list and is therefore active at full weight from the first epoch:

```text
loss_openings_centroid_axis = 6.00
```

## PrefitGuard

Before optimization, a diagnostic forward pass evaluates the initial fit using four losses:

```text
loss_openings_p
loss_openings_surface_p
loss_openings_plane
loss_diff_centreline
```

The thresholds are:

```text
term                         moderate threshold   severe threshold
loss_openings_p              0.140                0.280
loss_openings_surface_p      0.012                0.030
loss_openings_plane          0.045                0.090
loss_diff_centreline         0.180                0.300
```

A case is flagged as severe if any severe threshold is exceeded, or if two or more moderate flags are present. A case is flagged as moderate if exactly one moderate threshold is exceeded.

For moderate cases:

```text
warmup_epochs = max(500, 1.5 * 500) = 750
loss_openings_p          start scale = 0.35
loss_openings_surface_p  start scale = 0.35
loss_openings_n          start scale = 0.40
loss_openings_plane      start scale = 0.20
loss_diff_centreline     start scale = 0.25
```

For severe cases:

```text
warmup_epochs = max(500, 2.0 * 500) = 1000
loss_openings_p          start scale = 0.15
loss_openings_surface_p  start scale = 0.15
loss_openings_n          start scale = 0.20
loss_openings_plane      start scale = 0.05
loss_diff_centreline     start scale = 0.10
```

The rim-curvature term keeps its default start scale of `0.30`, but uses the longer warmup if the case is moderate or severe. The centroid-axis term remains full strength even in suspicious cases.

## Optimization Protocol

The model is optimized with AdamW:

```text
optimizer      AdamW
learning rate  0.0025
scheduler      StepLR
step_size      1800
gamma          0.8
grad clipping  1.0
epochs         4000
target epoch   3999
```

The trainable parameters are:

```text
GHD coefficients C
rotation R
scale s
translation T
```

Each epoch performs:

```text
1. Forward GHD deformation and global transform.
2. Forward transformed opening mesh.
3. Compute epoch-specific loss weights.
4. Compute raw loss dictionary.
5. Add thickness loss if enabled.
6. Multiply raw losses by current weights.
7. Sum to total weighted loss.
8. Backpropagate.
9. Clip gradients to norm 1.0.
10. AdamW optimizer step.
11. StepLR scheduler step.
12. Log raw and weighted losses.
```

There is no minibatching over meshes: one optimization problem is solved per target case. Stochasticity comes from surface and opening point sampling during the loss computation. Occupancy query points are static for the case.

The StepLR schedule gives:

```text
epochs 0-1799      lr = 0.0025
epochs 1800-3599   lr = 0.0020
epochs 3600-3999   lr = 0.0016
```

Early stopping is enabled:

```text
early_stopping = 1
patience       = 120 epochs
min_delta      = 0.0002
min_epochs     = 1200
```

After epoch `1200`, training stops if the weighted total loss does not improve by at least `0.0002` for `120` consecutive epochs. If a non-finite total loss occurs, the optimizer step for that epoch is skipped.

## Parallel Execution

The configuration permits multiple cases to be fitted in parallel:

```text
parallel_cases = 3
parallel_devices = cuda:0
```

The script launches one subprocess per case. Each subprocess receives:

```text
--register 0
--parallel_cases 1
--case_glob {case}
--name_target {case}
--device cuda:0
```

Thus, each worker solves one full-case fitting problem. Because all workers share the same GPU in this configuration, this setting trades memory/runtime efficiency for concurrent case scheduling.

## Checkpointing and Resume Behavior

With:

```text
chk_num = 4
epochs = 4000
```

the checkpoint frequency is:

```text
chk_freq = round(4000 / 4) = 1000
```

Numbered checkpoints are written at epochs `1000`, `2000`, and `3000`:

```text
ghb_fitting_checkpoint_1.pkl
ghb_fitting_checkpoint_2.pkl
ghb_fitting_checkpoint_3.pkl
```

The final checkpoint is written at the last reached epoch, either `3999` or the early-stopping epoch:

```text
ghb_fitting_checkpoint.pkl
ghd_fitting_checkpoint.pkl
```

Both final files contain the same core fields:

```text
R
s
T
GHD_coefficient
optimizer_state
scheduler_state
epoch
```

Before fitting a case, the script checks whether a final checkpoint already exists and whether its stored `epoch` is at least `3999`. If so, the case is skipped. Otherwise, fitting resumes from the newest available checkpoint. Optimizer and scheduler states are loaded, but the current configuration hyperparameters are reapplied, including the learning rate and scheduler settings.

## Optional Stage-2 Rim Refinement

After the spectral GHD fit, an optional local refinement stage can be applied:

```text
rim_refine_enabled = 1
rim_refine_on_suspicious_only = 1
```

Thus, Stage 2 runs only for cases flagged as moderate or severe by the PrefitGuard. This stage freezes the fitted GHD parameters and optimizes free vertex offsets in a local neighborhood around the opening rim.

The refined vertex set is:

```text
all vertices within 2 mesh rings of the opening cap/rim
```

The optimization variables are local offsets:

```text
delta_rim in R^{N_rim x 3}
```

The base vertices are the already fitted GHD result. During refinement:

```text
V_refined = V_fit + scatter(delta_rim)
```

The Stage-2 objective is:

```text
L_stage2 =
  12.0 * L_rim_boundary_chamfer
  + 8.0 * L_opening_plane
  + 2.0 * L_opening_surface
  + 0.12 * L_laplacian
  + 0.08 * L_edge
```

Unlike the main fitting stage, the Stage-2 rim boundary term uses sampled boundary Chamfer rather than the ordered cyclic rim loss. The refinement optimizer is:

```text
optimizer           Adam
learning rate       0.0007
epochs              160
scheduler           CosineAnnealingLR
eta_min             0.0007 * 0.05
grad clipping       0.5
rim_refine_rings    2
```

After optimization, the refined local offsets are baked into the fitted mesh. If Stage 2 was run, the final checkpoint additionally contains:

```text
rim_refine_offsets
rim_refine_verts_idx
```

The `GHD_coefficient` remains the compact spectral token. The optional rim offsets are supplemental information needed to reproduce the exact post-refinement mesh.

## Logged Outputs

For each fitted case, the output folder contains:

```text
ghb_fitting_checkpoint_*.pkl
ghb_fitting_checkpoint.pkl
ghd_fitting_checkpoint.pkl
loss_components_raw.png
loss_components_weighted.png
events.out.tfevents.*
fitting_preview_epoch_*.png
```

Raw and weighted losses are logged separately. Preview renderings are saved every:

```text
viz_freq = 500 epochs
```

Scalar logging and loss-plot updates occur every:

```text
log_freq = 400 epochs
```

The final `GHD_coefficient` is used downstream as the compact shape representation for first-stage and vessel-aware VAE/CVAE training.

## Code References

The procedure is implemented in:

```text
ghd_fitting.py
ghd/fitting/fitter.py
ghd/fitting/weighter.py
ghd/fitting/registration.py
ghd/base/graph_harmonic_deformation.py
ghd/base/mesh_geometry.py
ghd/base/mesh_geometry3.py
ghd/losses/mesh_loss.py
ghd/losses/meshloss.py
ghd/losses/meshloss_do.py
ghd/losses/meshloss_dc.py
ghd/losses/diceloss.py
config/ghd_fitting_config_cap_v6_finish_v5.yaml
```
