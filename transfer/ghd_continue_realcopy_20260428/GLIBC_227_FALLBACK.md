# Fallback fuer Server mit glibc 2.27

Deine Serverdaten:

```text
Python 3.10.18
Linux x86_64
glibc 2.27
NVIDIA driver 525
CUDA reported 12.0
```

Wenn `pip` bei `torch==2.8.0` sagt:

```text
from versions: none
No matching distribution found
```

dann ist das sehr wahrscheinlich kein CUDA-Problem, sondern ein Wheel-Kompatibilitaetsproblem
mit dem alten System-`glibc`. Moderne PyTorch-Linux-Wheels werden von `pip` dann
gar nicht erst als kompatibel angezeigt.

## Beste Loesung

Nutze auf dem Server ein neueres Userland:

- Docker / Apptainer / Singularity mit Ubuntu 20.04 oder 22.04
- oder OS-Upgrade auf Ubuntu 20.04+

Dann kannst du den Torch-2.8-Stack weiter nutzen.

## Praktischer Fallback ohne OS-Upgrade

Nutze den alten CUDA-11.8/Torch-2.1-Stack aus dem vorhandenen `environment.yml`.
Der passt besser zu glibc 2.27.

Im Repo:

```bash
cd /workspace/AneuG
conda env create -f environment.yml
conda activate gnn2
```

Dann Import-Test:

```bash
python - <<'PY'
import torch
import pytorch3d
import torch_scatter
import torch_geometric
import trimesh
import open3d
from ghd.fitting.fitter import fit_ghd
print("imports ok")
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
PY
```

Wenn das funktioniert, kannst du das Fitting mit dieser Umgebung starten:

```bash
cd /workspace/AneuG

conda run --no-capture-output -n gnn2 python ghd_fitting.py \
  --config /workspace/AneuG/config/ghd_fitting_config_cap_v6_finish_v5.yaml \
  --save_root /workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999 \
  --epochs 4000 \
  --parallel_cases 3 \
  --parallel_devices cuda:0
```

## Wenn `environment.yml` zu schwer ist

Manueller Minimalversuch:

```bash
conda create -n ghd_legacy python=3.8 -y
conda activate ghd_legacy

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
conda install pytorch3d=0.7.8 -c pytorch3d -c conda-forge -y
conda install pytorch-scatter=2.1.2 -c pyg -y

python -m pip install \
  numpy scipy matplotlib PyYAML tqdm trimesh scikit-learn pandas einops \
  igraph networkx shapely point-cloud-utils skeletor pyvista open3d
```

Danach den gleichen Import-Test laufen lassen.

## Einschätzung

Fuer das reine Resume sollten die Checkpoints auch mit einem etwas aelteren
Torch-Stack ladbar sein, weil sie im Wesentlichen Tensoren, Optimizer-State und
Scheduler-State enthalten. Der sicherste Weg bleibt aber: neueres Userland und
Torch 2.8.
