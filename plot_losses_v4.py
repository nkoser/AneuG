#!/usr/bin/env python
"""Plot training loss curves from v4 training log."""
import json, re, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOG = "checkpoints/vessel_aware_cvae/v4_cap_v6rrf_train.log"
OUT = pathlib.Path("checkpoints/vessel_aware_cvae/v4_cap_v6rrf/viz")
OUT.mkdir(exist_ok=True, parents=True)

# Parse log lines like {'epoch': 0, 'total': 2.84, ...}
records = []
with open(LOG) as f:
    for line in f:
        line = line.strip()
        if line.startswith("{") and "'epoch'" in line:
            # Convert single-quoted Python dict to valid JSON
            js = line.replace("'", '"')
            try:
                records.append(json.loads(js))
            except json.JSONDecodeError:
                pass

if not records:
    print("No log entries found!")
    exit(1)

epochs = np.array([r['epoch'] for r in records])
print(f"Parsed {len(records)} log entries, epochs {epochs[0]}–{epochs[-1]}")

# ── 1. Main losses (total, mse, kl) ──
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Total loss
ax = axes[0, 0]
ax.plot(epochs, [r['total'] for r in records], 'b-', linewidth=0.8)
ax.set_title('Total Loss', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# MSE loss
ax = axes[0, 1]
ax.plot(epochs, [r['mse'] for r in records], 'r-', linewidth=0.8)
ax.set_title('MSE (GHD coefficients)', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# KL
ax = axes[0, 2]
ax.plot(epochs, [r['kl_raw'] for r in records], 'g-', linewidth=0.8, label='KL raw')
ax.plot(epochs, [r['kl_w'] for r in records], 'k--', linewidth=0.8, label='KL weight')
ax.set_title('KL Divergence', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Value')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Vertex + Normal
ax = axes[1, 0]
ax.plot(epochs, [r['vert'] for r in records], 'b-', linewidth=0.8, label='Vertex MSE')
ax.plot(epochs, [r['norm'] for r in records], 'orange', linewidth=0.8, label='Normal MSE')
ax.set_title('Mesh-Space Losses', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Geo losses (plane, penetration, ring_match)
ax = axes[1, 1]
ax.plot(epochs, [r['plane'] for r in records], 'm-', linewidth=0.8, label='Plane')
ax.plot(epochs, [r['penetration'] for r in records], 'c-', linewidth=0.8, label='Penetration')
ax.plot(epochs, [r['ring_match'] for r in records], 'brown', linewidth=0.8, label='Ring Match')
ax.axvline(x=1000, color='gray', linestyle=':', alpha=0.7, label='Geo phase-in')
ax.set_title('Vessel-Aware Losses', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.set_yscale('log')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Regularisation (laplacian, consistency)
ax = axes[1, 2]
ax.plot(epochs, [r['lap'] for r in records], 'purple', linewidth=0.8, label='Laplacian')
ax.plot(epochs, [r['consistency'] for r in records], 'teal', linewidth=0.8, label='Consistency')
ax.set_title('Regularisation Losses', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

plt.suptitle('v4 Vessel-Aware CVAE Training (cap_v6_rrf, 92 samples, 10k epochs)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
out_path = str(OUT / "loss_curves.png")
fig.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Loss curves saved: {out_path}")

# ── 2. Zoomed-in view of last 5000 epochs ──
mask = epochs >= 5000
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))

ax = axes2[0]
ax.plot(epochs[mask], [r['mse'] for r, m in zip(records, mask) if m], 'r-', linewidth=0.8)
ax.set_title('MSE (last 5k epochs)', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.grid(True, alpha=0.3)

ax = axes2[1]
ax.plot(epochs[mask], [r['plane'] for r, m in zip(records, mask) if m], 'm-', linewidth=0.8, label='Plane')
ax.plot(epochs[mask], [r['penetration'] for r, m in zip(records, mask) if m], 'c-', linewidth=0.8, label='Penetration')
ax.set_title('Plane + Penetration (last 5k)', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax = axes2[2]
ax.plot(epochs[mask], [r['kl_raw'] for r, m in zip(records, mask) if m], 'g-', linewidth=0.8)
ax.set_title('KL raw (last 5k)', fontsize=12, fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Value')
ax.grid(True, alpha=0.3)

plt.suptitle('v4 Training — Last 5000 Epochs Detail', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
out2 = str(OUT / "loss_curves_zoomed.png")
fig2.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Zoomed loss curves saved: {out2}")
