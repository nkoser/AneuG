# Aneurysm-to-Healthy Attachment Pipeline

Goal: attach a generated open aneurysm mesh back onto the healthy vessel tree in
the same raw/prepared coordinate system used by `prepared_meshes_3`.

## Coordinate Spaces

Use raw/prepared space as the final attachment space. The healthy vessel meshes
and the original prepared vessel/aneurysm meshes live in this space.

Generated GHD meshes are expected in `ghd_local` space:

```text
target_norm = (p_ghd_local @ R.T) * s + T
target_aligned = target_norm * canonical_norm
target_raw = inverse(prealign_transform)(target_aligned)
```

The attachment tool supports:

```text
--aneurysm_space raw
--aneurysm_space aligned
--aneurysm_space ghd_local
```

## Attachment Steps

1. Load and clean the closed healthy vessel tree.
2. Load the open aneurysm mesh.
3. Transform the aneurysm into raw/prepared space if needed.
4. Cut a jagged ostium hole into the healthy vessel using center, normal, radius,
   slab width, and jaggedness.
5. Extract and order the healthy cut boundary loop from real boundary edges.
6. Extract and order the aneurysm rim from labels or from its open boundary.
7. Optionally snap the aneurysm rim center to the hole center.
8. Bridge both loops with triangles.
9. Export the combined mesh and report watertightness/boundary edges.

The prototype exposes scale controls because a watertight attachment can still
look anatomically wrong if the generated sac is too large:

```text
--aneurysm_scale       uniform scale around the aneurysm rim center
--sac_axial_scale      scale dome height along the ostium normal
--sac_radial_scale     scale dome width away from the rim plane
--sac_falloff_power    how quickly radial sac scaling starts away from the rim
```

## Current Prototype

Tool:

```bash
python tools/attach_aneurysm_to_healthy.py --case aneux_C0075 \
  --jagged_amp 0.16 --radius_scale 1.10 --cut_slab 0.06 \
  --out_mesh /workspace/AneuG/checkpoints/attach_aneurysm/aneux_C0075/aneux_C0075_attached_jagged.obj
```

Compact C0075 test output:

```bash
python tools/attach_aneurysm_to_healthy.py --case aneux_C0075 \
  --aneurysm_scale 0.70 --sac_axial_scale 0.70 --sac_radial_scale 0.85 \
  --jagged_amp 0.16 --radius_scale 1.10 --cut_slab 0.045 \
  --out_mesh /workspace/AneuG/checkpoints/attach_aneurysm/aneux_C0075/aneux_C0075_attached_small.obj
```

For generated GHD-local meshes:

```bash
python tools/attach_aneurysm_to_healthy.py --case aneux_C0075 \
  --aneurysm_mesh /path/to/generated_open_aneurysm.obj \
  --aneurysm_space ghd_local \
  --out_mesh /path/to/attached.obj
```

## UI Direction

Build the UI around the cut, not around the stitching.

Minimum useful UI:

- 3D vessel viewer.
- click-to-place ostium center on the healthy vessel.
- normal direction from local surface normal, with optional flip.
- radius brush/slider.
- jaggedness slider.
- cut slab/depth slider.
- preview cut loop before accepting.
- attach/export button.

The jagged boundary should be generated as a noisy radial contour in the ostium
tangent plane, then projected/selected on the vessel surface. This gives a
controlled irregular edge while preserving a single clean boundary loop for
bridging.
