"""Ensemble inference helper for vessel-aware CVAE checkpoints.

Loads N CVAE checkpoints (mix of PCA / non-PCA, v2 / v8_resnet, conditional /
unconditional prior) and exposes a single ``sample`` API that returns decoded
GHD coefficients in the *original normalized GHD space* (i.e. after PCA lift
if applicable but before the dataset's ``de_normalize_ghd``).

Typical usage in a generation pipeline:

    ens = VesselAwareCVAEEnsemble(["seed1.pth", "seed2.pth", ...], device="cuda")
    # vessel_pts: [B, N_vessel, 3], ostium_params: [B, 8]
    samples_norm = ens.sample(vessel_pts, ostium_params, k_per_member=8)
    # samples_norm: [M*K, B, D_norm]
    samples_ghd = samples_norm * ens.ghd_std.to(samples_norm.device) + ens.ghd_mean.to(samples_norm.device)

    # Or pick best-of-(M*K) against a target GT:
    best = ens.best_of(samples_norm, gt_norm)  # [B, D_norm]
"""
from __future__ import annotations

from typing import List, Optional

import torch

from models.vessel_aware_cvae import VesselAwareCVAE as VesselAwareCVAEV2
from models.vessel_aware_cvae_v8_resnet import VesselAwareCVAEV8ResNet
from models.vessel_conditioner import OstiumConditioner


def _ckpt_args(ckpt):
    a = ckpt.get("args", {})
    return a if isinstance(a, dict) else (vars(a) if a else {})


def _build(ckpt, device):
    sa = _ckpt_args(ckpt)
    pca = ckpt.get("pca_basis", None)
    input_dim = pca.shape[0] if pca is not None else ckpt["ghd_mean"].shape[1]
    common = dict(
        input_dim=input_dim,
        hidden_dim=int(sa.get("hidden_dim", 256)),
        latent_dim=int(sa.get("latent_dim", 64)),
        vessel_cond_dim=int(sa.get("vessel_cond_dim", 32)),
        extra_cond_dim=0,
        dropout=float(sa.get("dropout", 0.02)),
    )
    if sa.get("model_type") == "v8_resnet":
        m = VesselAwareCVAEV8ResNet(
            **common,
            encoder_blocks=int(sa.get("encoder_blocks", 3)),
            decoder_blocks=int(sa.get("decoder_blocks", 6)),
        )
    else:
        m = VesselAwareCVAEV2(**common)
    m.load_state_dict(ckpt["generator"])
    m.to(device).eval()
    c = OstiumConditioner(
        vessel_feat_dim=int(sa.get("vessel_feat_dim", 64)),
        ostium_plane_dim=8,
        ostium_feat_dim=16,
        cond_out_dim=int(sa.get("vessel_cond_dim", 32)),
    )
    c.load_state_dict(ckpt["conditioner"])
    c.to(device).eval()
    return m, c, sa, pca


class VesselAwareCVAEEnsemble:
    def __init__(self, ckpt_paths: List[str], device: str = "cuda"):
        if not ckpt_paths:
            raise ValueError("Need at least one checkpoint")
        self.device = torch.device(device)
        self.members = []
        first = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
        # Use first ckpt's dataset normalization stats as the canonical GHD frame.
        if "orig_ghd_mean" in first:
            self.ghd_mean = first["orig_ghd_mean"].to(self.device)
            self.ghd_std = first["orig_ghd_std"].to(self.device)
        else:
            self.ghd_mean = first["ghd_mean"].to(self.device)
            self.ghd_std = first["ghd_std"].to(self.device)
        self.ostium_mean = first["ostium_mean"].to(self.device)
        self.ostium_std = first["ostium_std"].to(self.device)
        self.vessel_center = first["vessel_center"].to(self.device)
        self.vessel_scale = first["vessel_scale"].to(self.device)

        for path in ckpt_paths:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            model, cond, sa, pca = _build(ck, self.device)
            if "orig_ghd_mean" in ck:
                m_mean = ck["orig_ghd_mean"].to(self.device)
                m_std = ck["orig_ghd_std"].to(self.device)
            else:
                m_mean = ck["ghd_mean"].to(self.device)
                m_std = ck["ghd_std"].to(self.device)
            self.members.append({
                "path": path,
                "model": model,
                "cond": cond,
                "args": sa,
                "pca": pca.to(self.device) if pca is not None else None,
                "m_mean": m_mean,
                "m_std": m_std,
                "latent": int(sa.get("latent_dim", 64)),
                "use_cp": bool(sa.get("use_conditional_prior", False)) and hasattr(model, "prior"),
            })

    def __len__(self):
        return len(self.members)

    @torch.no_grad()
    def sample(
        self,
        vessel_pts: torch.Tensor,
        ostium_params: torch.Tensor,
        k_per_member: int = 8,
        normalize_inputs: bool = True,
    ) -> torch.Tensor:
        """Return [M*K, B, D_norm] samples in the canonical normalized-GHD space.

        Inputs are expected in raw (un-normalized) form by default; pass
        ``normalize_inputs=False`` if they are already normalized.
        """
        vp = vessel_pts.to(self.device)
        op = ostium_params.to(self.device)
        if normalize_inputs:
            vp = (vp - self.vessel_center) / self.vessel_scale
            op = (op - self.ostium_mean) / self.ostium_std
        outs = []
        for m in self.members:
            cond = m["cond"](vp, op)
            B = cond.shape[0]
            if m["use_cp"]:
                pmu, plv = m["model"].prior(cond)
                pstd = torch.exp(0.5 * plv)
            else:
                pmu = torch.zeros(B, m["latent"], device=self.device)
                pstd = torch.ones_like(pmu)
            for _ in range(k_per_member):
                z = pmu + pstd * torch.randn_like(pmu)
                s = m["model"].decode(z, cond)  # [B, D_member]
                if m["pca"] is not None:
                    s = s @ m["pca"]  # lift to [B, D_full]
                # Re-map to canonical normalization frame if member differs.
                if not torch.equal(m["m_mean"], self.ghd_mean) or not torch.equal(m["m_std"], self.ghd_std):
                    s = (s * m["m_std"] + m["m_mean"] - self.ghd_mean) / self.ghd_std
                outs.append(s)
        return torch.stack(outs, dim=0)  # [M*K, B, D_norm]

    @torch.no_grad()
    def sample_ghd(self, vessel_pts, ostium_params, k_per_member=8, normalize_inputs=True):
        """Same as ``sample`` but returns un-normalized GHD coefficients."""
        s = self.sample(vessel_pts, ostium_params, k_per_member, normalize_inputs)
        return s * self.ghd_std + self.ghd_mean

    @staticmethod
    @torch.no_grad()
    def best_of(samples: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Pick best-of-S sample per case against GT. samples: [S, B, D], gt: [B, D]."""
        rmse = torch.sqrt(((samples - gt.unsqueeze(0)) ** 2).mean(-1) + 1e-8)  # [S, B]
        idx = rmse.argmin(0)  # [B]
        return samples[idx, torch.arange(samples.shape[1], device=samples.device)]
