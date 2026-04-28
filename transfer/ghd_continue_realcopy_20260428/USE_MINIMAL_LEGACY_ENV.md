# Minimal Legacy Environment statt vollem environment.yml

Das volle `environment.yml` enthaelt viele alte Notebook-/Visualisierungs-Pakete.
Auf dem glibc-2.27-Server ist es an `embreex==2.17.7.post5` gescheitert.
Dieses Paket ist fuer das GHD-Fitting nicht noetig.

Nutze stattdessen die kleine Datei:

```text
/workspace/AneuG/transfer/ghd_continue_realcopy_20260428/environment_ghd_legacy_minimal.yml
```

## 1. Kaputten Versuch entfernen

Falls `gnn2` halb angelegt wurde:

```bash
conda env remove -n gnn2
```

## 2. Minimal-Env anlegen

```bash
cd /workspace/AneuG
conda env create -f /workspace/AneuG/transfer/ghd_continue_realcopy_20260428/environment_ghd_legacy_minimal.yml
conda activate ghd_legacy_minimal
```

## 3. Import-Test

```bash
python - <<'PY'
import torch
import pytorch3d
import torch_scatter
import torch_geometric
import trimesh
import open3d
import pyvista
import igraph
import shapely
from ghd.fitting.fitter import fit_ghd
print("imports ok")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
```

## 4. Fitting starten

```bash
cd /workspace/AneuG

conda run --no-capture-output -n ghd_legacy_minimal python ghd_fitting.py \
  --config /workspace/AneuG/config/ghd_fitting_config_cap_v6_finish_v5.yaml \
  --save_root /workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999 \
  --epochs 4000 \
  --parallel_cases 3 \
  --parallel_devices cuda:0
```

## Hinweis

Ich habe im Code `point_cloud_utils` und `skeletor` fuer diesen Resume-Pfad
optional gemacht. Wenn du auf dem anderen Server eine aeltere Repo-Kopie hast,
uebernimm bitte diese zwei Dateien aus der aktuellen Version:

```text
utils/utils.py
ghd/fitting/registration.py
```

Oder direkt im Repo auf dem Zielserver:

```bash
cd /workspace/AneuG
python /workspace/AneuG/transfer/ghd_continue_realcopy_20260428/patch_legacy_optional_imports.py
```

Wenn dein Repo dort unter `/home/sukin707/Aneug` liegt, entsprechend:

```bash
cd /home/sukin707/Aneug
python /home/sukin707/Aneug/transfer/ghd_continue_realcopy_20260428/patch_legacy_optional_imports.py
```
