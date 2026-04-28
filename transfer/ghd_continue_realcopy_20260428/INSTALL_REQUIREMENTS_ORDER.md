# Requirements und Installationsreihenfolge

Diese Liste ist auf die aktuell funktionierende Umgebung `unified_env` abgestimmt.

Getestete Hauptversionen:

```text
Python       3.10.16
Torch        2.8.0+cu128
Torchvision  0.23.0+cu128
PyTorch3D    0.7.9
PyG          2.7.0
CUDA wheels  cu128
```

## 0. Voraussetzung

Der Zielserver braucht einen NVIDIA-Treiber, der CUDA-12.x Runtime-Wheels laufen lassen kann. Wenn `nvidia-smi` sehr alt ist, lieber erst den Treiber prüfen.

## 1. Conda Environment anlegen

```bash
conda create -n unified_env python=3.10 -y
conda activate unified_env
python -m pip install --upgrade pip setuptools wheel
```

## 2. Basis- und Geometriepakete

Diese Pakete zuerst installieren. Sie sind weniger heikel als Torch/PyG:

```bash
python -m pip install -r /workspace/AneuG/transfer/ghd_continue_realcopy_20260428/requirements_main_packages.txt
```

Falls `open3d` oder `vtk` per pip Probleme machen, stattdessen diese beiden über conda-forge installieren:

```bash
conda install -c conda-forge open3d=0.18.0 vtk=9.2.6 pyvista=0.47.0 -y
```

Danach den pip-Befehl für `requirements_main_packages.txt` nochmal ausführen.

## Hinweis fuer NVIDIA 525 / CUDA 12.0 Server

Wenn der Zielserver nur NVIDIA-Treiber 525 und CUDA 12.0 meldet, nimm zuerst die
konservativere CUDA-12.6-Variante:

```bash
python -m pip install -r /workspace/AneuG/transfer/ghd_continue_realcopy_20260428/requirements_torch_cuda126.txt
python -m pip install -r /workspace/AneuG/transfer/ghd_continue_realcopy_20260428/requirements_pyg_torch28_cuda126.txt
```

Die PyTorch-Datei nutzt bewusst `torch==2.8.0` ohne `+cu126`; der CUDA-Index
waehlt dann das passende CUDA-Wheel.

Siehe auch:

```text
/workspace/AneuG/transfer/ghd_continue_realcopy_20260428/CUDA_525_DRIVER_NOTE.md
```

## 3. PyTorch CUDA 12.8 installieren

```bash
python -m pip install -r /workspace/AneuG/transfer/ghd_continue_realcopy_20260428/requirements_torch_cuda128.txt
```

Auch hier steht in der Requirements-Datei bewusst `torch==2.8.0` ohne
`+cu128`; der Index bestimmt das CUDA-Wheel.

Schnelltest:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
PY
```

`torch.cuda.is_available()` sollte auf dem GPU-Server `True` sein.

## 4. PyTorch Geometric Extensions installieren

Wichtig: Diese Pakete muessen zu Torch `2.8.0+cu128` passen.

```bash
python -m pip install -r /workspace/AneuG/transfer/ghd_continue_realcopy_20260428/requirements_pyg_torch28_cuda128.txt
```

Schnelltest:

```bash
python - <<'PY'
import torch_geometric
import torch_scatter
import torch_sparse
import torch_cluster
print(torch_geometric.__version__)
print(torch_scatter.__version__)
print(torch_sparse.__version__)
print(torch_cluster.__version__)
PY
```

## 5. PyTorch3D pruefen

PyTorch3D immer nach PyTorch installieren:

```bash
python -m pip install -r /workspace/AneuG/transfer/ghd_continue_realcopy_20260428/requirements_pytorch3d.txt
```

Schnelltest:

```bash
python - <<'PY'
import pytorch3d
from pytorch3d.structures import Meshes
from pytorch3d.loss import chamfer_distance
print(pytorch3d.__version__)
PY
```

## 6. Repo-Pfad

Im Repo selbst ist kein `pip install -e .` noetig. Starte die Scripts aus dem Repo-Root:

```bash
cd /workspace/AneuG
```

Wenn du aus einem anderen Working Directory startest, setze:

```bash
export PYTHONPATH=/workspace/AneuG:$PYTHONPATH
```

## 7. Kompletttest fuer das GHD-Fitting

```bash
cd /workspace/AneuG
conda activate unified_env

python - <<'PY'
import torch
import pytorch3d
import torch_geometric
import torch_scatter
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

Wenn das laeuft, kannst du das Fitting fortsetzen:

```bash
conda run --no-capture-output -n unified_env python ghd_fitting.py \
  --config /workspace/AneuG/config/ghd_fitting_config_cap_v6_finish_v5.yaml \
  --save_root /workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999 \
  --epochs 4000 \
  --parallel_cases 3 \
  --parallel_devices cuda:0
```

Bei mehreren GPUs:

```bash
conda run --no-capture-output -n unified_env python ghd_fitting.py \
  --config /workspace/AneuG/config/ghd_fitting_config_cap_v6_finish_v5.yaml \
  --save_root /workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999 \
  --epochs 4000 \
  --parallel_cases 4 \
  --parallel_devices cuda:0,cuda:1,cuda:2,cuda:3
```

## 8. Typische Stolperstellen

Wenn `torch_scatter` oder `torch_sparse` import-fehlschlagen, passt meistens der PyG-Wheel nicht zur Torch/CUDA-Version. Dann Torch-Version pruefen und exakt den passenden `data.pyg.org` Link verwenden.

Wenn `pytorch3d` import-fehlschlaegt, erst Torch/CUDA pruefen. PyTorch3D muss nach Torch installiert werden.

Wenn `open3d` oder `vtk` Probleme machen, diese Pakete ueber `conda-forge` installieren.

Wenn Matplotlib wegen `/root/.config/matplotlib` meckert, ist das nicht kritisch. Optional:

```bash
export MPLCONFIGDIR=/tmp/matplotlib
```
