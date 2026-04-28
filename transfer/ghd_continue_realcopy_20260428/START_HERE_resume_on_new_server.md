# GHD Fitting auf anderem Server fortsetzen

Dieser Ordner ist ein Real-Copy-Bundle. Symlinks wurden beim Kopieren aufgelöst.

Bundle:

```text
/workspace/AneuG/transfer/ghd_continue_realcopy_20260428
```

## 1. Diese Ordner brauchst du

Wenn du die Ordner einzeln herunterlädst, lade genau diese vier:

```text
checkpoints/canonical_average
checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999
config
data/ghd_prepared_meshes_3_aneurysm_1op_new
```

Auf dem neuen Server sollten sie danach hier liegen:

```text
/workspace/AneuG/checkpoints/canonical_average
/workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999
/workspace/AneuG/config
/data/ghd_prepared_meshes_3_aneurysm_1op_new
```

Wichtig: Der Code-Repo `/workspace/AneuG` muss auf dem Zielserver auch vorhanden sein.

## 2. Warum diese Ordner nötig sind

`checkpoints/canonical_average` enthält die Canonical-Mesh-Dateien und die feste GHD-Basis:

```text
part_aligned.obj
opa_checkpoint_1op.pkl
diff_centreline_checkpoint_1op.pkl
eigen_chk_144.pkl
```

`checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999` enthält die Resume-Checkpoints:

```text
{case}/prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3/ghb_fitting_checkpoint*.pkl
```

`data/ghd_prepared_meshes_3_aneurysm_1op_new` enthält die vorbereiteten Target-Cases:

```text
part_aligned.obj
opa_checkpoint_1op.pkl
diff_centreline_checkpoint_1op.pkl
do_points.pt
prealign_transform.npy
```

## 3. Nach dem Hochladen kurz prüfen

Auf dem Zielserver:

```bash
find /workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999 -type l | wc -l
find /workspace/AneuG/checkpoints/canonical_average -type l | wc -l
find /data/ghd_prepared_meshes_3_aneurysm_1op_new -type l | wc -l
```

Alle drei Werte sollten `0` sein.

Dann prüfen:

```bash
ls /workspace/AneuG/checkpoints/canonical_average/eigen_chk_144.pkl
ls /workspace/AneuG/config/ghd_fitting_config_cap_v6_finish_v5.yaml
ls /data/ghd_prepared_meshes_3_aneurysm_1op_new/aneux_C0075/part_aligned.obj
```

## 4. Fitting fortsetzen

Starte im Repo:

```bash
cd /workspace/AneuG
```

Dann:

```bash
conda run --no-capture-output -n unified_env python ghd_fitting.py \
  --config /workspace/AneuG/config/ghd_fitting_config_cap_v6_finish_v5.yaml \
  --save_root /workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999 \
  --epochs 4000 \
  --parallel_cases 3 \
  --parallel_devices cuda:0
```

Falls der neue Server mehrere GPUs hat, z.B. vier Stück:

```bash
conda run --no-capture-output -n unified_env python ghd_fitting.py \
  --config /workspace/AneuG/config/ghd_fitting_config_cap_v6_finish_v5.yaml \
  --save_root /workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999 \
  --epochs 4000 \
  --parallel_cases 4 \
  --parallel_devices cuda:0,cuda:1,cuda:2,cuda:3
```

## 5. Was passiert beim Start

Das Script schaut pro Case in:

```text
/workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999/{case}/prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3/
```

Wenn ein Case schon bis Epoch `3999` fertig ist, wird er übersprungen.

Wenn ein Case nur bis z.B. `1201`, `2500` oder `3637` gekommen ist, wird er aus dem letzten `ghb_fitting_checkpoint*.pkl` weitergeführt.

## 6. Nach dem Lauf prüfen

```bash
find /workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999 \
  -path '*/prepared3_aneurysm_1op_quality_cap_v6_roundrobin_v3/fitting_preview_epoch_003999.png' | wc -l
```

Ziel: alle Cases, die gefittet werden sollen, sollten danach eine `fitting_preview_epoch_003999.png` haben.

Zusätzlich:

```bash
find /workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999 \
  -maxdepth 4 -type f -name 'ghb_fitting_checkpoint.pkl' | wc -l
```

## 7. Wenn `/data` auf dem Zielserver nicht passt

Am schnellsten ist es, den Ordner wirklich nach `/data` zu legen:

```text
/data/ghd_prepared_meshes_3_aneurysm_1op_new
```

Wenn das nicht geht, kann man beim Start `--root_target` überschreiben:

```bash
--root_target /anderer/pfad/ghd_prepared_meshes_3_aneurysm_1op_new
```

Dann müssen aber alle anderen Pfade weiterhin stimmen.
