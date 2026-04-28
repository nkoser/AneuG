"""
Vessel-Aware Conditional GHD VAE.

The decoder receives:
  z          [B, latent_dim]   — latent code
  cond       [B, cond_dim]     — combined condition = vessel_cond ⊕ optional morpho markers

The OstiumConditioner (vessel_conditioner.py) produces the vessel part of the
condition.  This module concatenates it with any extra scalar conditions
(morphological markers) and feeds it into the decoder.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    """Self-contained residual block (avoids importing from vae_models which has
    a heavy transitive import chain)."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        out += identity
        return self.relu(out)


class VesselAwareCVAE(nn.Module):
    """
    Conditional VAE for GHD coefficients with vessel/ostium conditioning.

    Architecture:
        Encoder:  GHD → MLP+ResBlocks → mu, logvar
        Decoder:  (z ⊕ vessel_cond) → MLP+ResBlocks → GHD_recon

    The vessel_cond comes from OstiumConditioner (external).
    """

    def __init__(
        self,
        input_dim: int,         # GHD flat dim (e.g. 588)
        hidden_dim: int = 256,
        latent_dim: int = 64,
        vessel_cond_dim: int = 32,   # output of OstiumConditioner
        extra_cond_dim: int = 0,     # optional morpho markers etc.
        dropout: float = 0.02,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cond_dim = vessel_cond_dim + extra_cond_dim

        # ---------- Encoder ----------
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_res1 = _ResidualBlock(hidden_dim, hidden_dim)
        self.enc_drop1 = nn.Dropout(dropout)
        self.enc_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_res2 = _ResidualBlock(hidden_dim, hidden_dim)
        self.enc_drop2 = nn.Dropout(dropout)

        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ---------- Decoder ----------
        self.dec_fc1 = nn.Linear(latent_dim + self.cond_dim, hidden_dim)
        self.dec_res1 = _ResidualBlock(hidden_dim, hidden_dim)
        self.dec_drop1 = nn.Dropout(dropout)
        self.dec_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dec_res2 = _ResidualBlock(hidden_dim, hidden_dim)
        self.dec_drop2 = nn.Dropout(dropout)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

    # ---- helpers ---- #
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---- forward ---- #
    def encode(self, x: torch.Tensor):
        """x: [B, input_dim] → mu, logvar: [B, latent_dim]"""
        h = F.relu(self.enc_fc1(x))
        h = self.enc_drop1(self.enc_res1(h))
        h = F.relu(self.enc_fc2(h))
        h = self.enc_drop2(self.enc_res2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-6.0, 2.0)  # prevent log-variance explosion
        return mu, logvar

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        z:    [B, latent_dim]
        cond: [B, cond_dim]   (vessel_cond ⊕ extra_cond)
        returns: [B, input_dim]
        """
        h = torch.cat([z, cond], dim=-1)
        h = F.relu(self.dec_fc1(h))
        h = self.dec_drop1(self.dec_res1(h))
        h = F.relu(self.dec_fc2(h))
        h = self.dec_drop2(self.dec_res2(h))
        return self.dec_out(h)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                deterministic: bool = False):
        """
        x:    [B, input_dim]   normalised GHD coefficients
        cond: [B, cond_dim]    condition vector
        deterministic: if True, use z=mu (AE mode, no sampling)
        returns: recon, mu, logvar
        """
        mu, logvar = self.encode(x)
        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar

    def forward_encoder(self, x: torch.Tensor, deterministic: bool = False):
        mu, logvar = self.encode(x)
        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        return z, mu, logvar
