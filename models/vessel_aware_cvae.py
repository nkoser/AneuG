"""
Vessel-Aware Conditional GHD VAE  -- v2.

Changes from v1:
  1. **Conditional Encoder**: receives (GHD + cond) so the latent space is
     vessel-aware from the start.
  2. **Multi-layer condition injection** in the decoder (cond injected at every
     residual block, not just at input).
  3. **logvar clamping** to prevent variance explosion.
  4. **Deterministic mode** (z=mu) for AE warmup.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ResidualBlock(nn.Module):
    """Self-contained residual block."""
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return F.relu(out + identity)


class _CondResidualBlock(nn.Module):
    """Residual block with condition injection.
    Projects (h || cond) -> h through a bottleneck before the residual.
    """
    def __init__(self, dim, cond_dim):
        super().__init__()
        self.proj = nn.Linear(dim + cond_dim, dim)
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x, cond):
        h = F.relu(self.proj(torch.cat([x, cond], dim=-1)))
        identity = h
        out = F.relu(self.bn1(self.fc1(h)))
        out = self.bn2(self.fc2(out))
        return F.relu(out + identity)


class VesselAwareCVAE(nn.Module):
    """
    Conditional VAE for GHD coefficients with vessel/ostium conditioning (v2).

    Architecture:
        Encoder:  (GHD + cond) -> MLP+ResBlocks -> mu, logvar
        Decoder:  (z + cond) -> CondResBlock -> CondResBlock -> GHD_recon
    """

    def __init__(
        self,
        input_dim: int,             # GHD flat dim (e.g. 588)
        hidden_dim: int = 256,
        latent_dim: int = 64,
        vessel_cond_dim: int = 32,  # output of OstiumConditioner
        extra_cond_dim: int = 0,
        dropout: float = 0.02,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cond_dim = vessel_cond_dim + extra_cond_dim

        # ---------- Conditional Encoder ----------
        enc_input_dim = input_dim + self.cond_dim   # GHD + cond
        self.enc_fc1 = nn.Linear(enc_input_dim, hidden_dim)
        self.enc_res1 = _ResidualBlock(hidden_dim)
        self.enc_drop1 = nn.Dropout(dropout)
        self.enc_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.enc_res2 = _ResidualBlock(hidden_dim)
        self.enc_drop2 = nn.Dropout(dropout)

        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ---------- Decoder with multi-layer condition injection ----------
        self.dec_fc1 = nn.Linear(latent_dim + self.cond_dim, hidden_dim)
        self.dec_cres1 = _CondResidualBlock(hidden_dim, self.cond_dim)
        self.dec_drop1 = nn.Dropout(dropout)
        self.dec_cres2 = _CondResidualBlock(hidden_dim, self.cond_dim)
        self.dec_drop2 = nn.Dropout(dropout)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

    # ---- helpers ---- #
    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # ---- forward ---- #
    def encode(self, x: torch.Tensor, cond: torch.Tensor):
        """
        x:    [B, input_dim]
        cond: [B, cond_dim]
        Returns mu, logvar: [B, latent_dim]
        """
        h = torch.cat([x, cond], dim=-1)
        h = F.relu(self.enc_fc1(h))
        h = self.enc_drop1(self.enc_res1(h))
        h = F.relu(self.enc_fc2(h))
        h = self.enc_drop2(self.enc_res2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-6.0, 2.0)
        return mu, logvar

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        z:    [B, latent_dim]
        cond: [B, cond_dim]
        returns: [B, input_dim]
        """
        h = torch.cat([z, cond], dim=-1)
        h = F.relu(self.dec_fc1(h))
        h = self.dec_drop1(self.dec_cres1(h, cond))
        h = self.dec_drop2(self.dec_cres2(h, cond))
        return self.dec_out(h)

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                deterministic: bool = False):
        """
        x:    [B, input_dim]   normalised GHD coefficients
        cond: [B, cond_dim]    condition vector
        deterministic: if True, use z=mu (AE mode, no sampling)
        returns: recon, mu, logvar
        """
        mu, logvar = self.encode(x, cond)
        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar

    def forward_encoder(self, x: torch.Tensor, cond: torch.Tensor,
                        deterministic: bool = False):
        mu, logvar = self.encode(x, cond)
        if deterministic:
            z = mu
        else:
            z = self.reparameterize(mu, logvar)
        return z, mu, logvar
