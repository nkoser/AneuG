# CUDA 12.0 / NVIDIA 525 Driver Setup

Der Zielserver hat laut Angabe CUDA 12.0 und NVIDIA-Treiber 525.

Wichtig: Bei PyTorch-Pip-Wheels ist nicht die lokal installierte CUDA-Toolkit-Version
entscheidend, sondern der NVIDIA-Treiber. Die Wheels bringen ihre eigene CUDA-Runtime
mit.

NVIDIA beschreibt fuer CUDA 12.x Minor-Version-Kompatibilitaet:

```text
CUDA 12.x applications: minimum driver >= 525
```

Deshalb ist ein 525er Treiber grundsaetzlich innerhalb der CUDA-12-Familie nutzbar.
Trotzdem ist `cu128` auf so einem alten Treiber etwas sportlich. Fuer den anderen
Server ist diese Reihenfolge am sinnvollsten:

## Empfohlener Versuch: Torch 2.8 mit CUDA 12.6 Wheels

Damit bleibt die Torch-Major/Minor-Version identisch zur aktuellen Umgebung
(`torch==2.8.0`), aber die CUDA-Wheels sind konservativer als `cu128`.

```bash
conda create -n unified_env python=3.10 -y
conda activate unified_env
python -m pip install --upgrade pip setuptools wheel

python -m pip install -r /workspace/AneuG/transfer/ghd_continue_realcopy_20260428/requirements_main_packages.txt
python -m pip install -r /workspace/AneuG/transfer/ghd_continue_realcopy_20260428/requirements_torch_cuda126.txt
python -m pip install -r /workspace/AneuG/transfer/ghd_continue_realcopy_20260428/requirements_pyg_torch28_cuda126.txt
python -m pip install -r /workspace/AneuG/transfer/ghd_continue_realcopy_20260428/requirements_pytorch3d.txt
```

Test:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(1024, 1024, device="cuda")
    print((x @ x).mean().item())
PY
```

Wenn dieser Test laeuft, nimm diese Umgebung fuer das Resume.

## Falls CUDA 12.6 Wheels nicht laufen

Dann gibt es zwei saubere Optionen:

1. NVIDIA-Treiber upgraden, ideal fuer `cu128`.
2. Auf eine aeltere Torch/CUDA-Kombination wechseln, z.B. CUDA 11.8 Stack aus
   `environment.yml`.

Option 2 kann funktionieren, ist aber fuer das direkte Resume weniger schoen,
weil die Checkpoints in der aktuellen Umgebung mit Torch 2.8 geschrieben wurden.
Darum erst `torch==2.8.0+cu126` versuchen.

## Danach Fitting starten

```bash
cd /workspace/AneuG

conda run --no-capture-output -n unified_env python ghd_fitting.py \
  --config /workspace/AneuG/config/ghd_fitting_config_cap_v6_finish_v5.yaml \
  --save_root /workspace/AneuG/checkpoints/ghd_fitting_prepared3_aneurysm_1op_cap_v6_finish_v5_only3999 \
  --epochs 4000 \
  --parallel_cases 3 \
  --parallel_devices cuda:0
```
