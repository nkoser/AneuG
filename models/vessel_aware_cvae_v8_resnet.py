"""
Vessel-aware CVAE v8 with a deeper FiLM/ResNet decoder.

The main change versus the current v2 model is the decoder: latent z and the
vessel condition are injected into every residual block through FiLM-style
scale/shift modulation and a learned gate. This keeps the latent code active
throughout the decoder instead of only concatenating it at the input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _EncoderResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(F.silu(self.norm1(x)))
        h = self.dropout(h)
        h = self.fc2(F.silu(self.norm2(h)))
        return x + h


class _FiLMResBlock(nn.Module):
    def __init__(self, dim: int, style_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.film1 = nn.Linear(style_dim, dim * 2)
        self.fc1 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.film2 = nn.Linear(style_dim, dim * 2)
        self.fc2 = nn.Linear(dim, dim)
        self.gate = nn.Linear(style_dim, dim)
        self.dropout = nn.Dropout(dropout)

        # Start each residual branch gently; the input/output projections still
        # learn immediately, while deep blocks do not destabilise epoch 0.
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @staticmethod
    def _apply_film(x: torch.Tensor, film: torch.Tensor) -> torch.Tensor:
        gamma, beta = film.chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = self._apply_film(h, self.film1(style))
        h = self.fc1(F.silu(h))
        h = self.dropout(h)
        h = self.norm2(h)
        h = self._apply_film(h, self.film2(style))
        h = self.fc2(F.silu(h))
        gate = torch.sigmoid(self.gate(style))
        return x + gate * h


class VesselAwareCVAEV8ResNet(nn.Module):
    """
    Conditional VAE for GHD coefficients with a deeper latent-aware decoder.

    Architecture:
        Encoder: (GHD + cond) -> MLP + residual blocks -> mu, logvar
        Decoder: (z + cond) -> MLP -> repeated FiLM residual blocks -> GHD
        Optional prior: cond -> p(z | cond)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 384,
        latent_dim: int = 64,
        vessel_cond_dim: int = 32,
        extra_cond_dim: int = 0,
        dropout: float = 0.02,
        encoder_blocks: int = 3,
        decoder_blocks: int = 6,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cond_dim = vessel_cond_dim + extra_cond_dim

        enc_input_dim = input_dim + self.cond_dim
        self.enc_in = nn.Linear(enc_input_dim, hidden_dim)
        self.enc_blocks = nn.ModuleList(
            [_EncoderResBlock(hidden_dim, dropout) for _ in range(encoder_blocks)]
        )
        self.enc_norm = nn.LayerNorm(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        style_dim = latent_dim + self.cond_dim
        self.dec_in = nn.Linear(style_dim, hidden_dim)
        self.dec_blocks = nn.ModuleList(
            [_FiLMResBlock(hidden_dim, style_dim, dropout) for _ in range(decoder_blocks)]
        )
        self.dec_norm = nn.LayerNorm(hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

        self.prior_net = nn.Sequential(
            nn.Linear(self.cond_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.prior_mu = nn.Linear(hidden_dim, latent_dim)
        self.prior_logvar = nn.Linear(hidden_dim, latent_dim)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def prior(self, cond: torch.Tensor):
        h = self.prior_net(cond)
        mu = self.prior_mu(h)
        logvar = self.prior_logvar(h).clamp(-6.0, 2.0)
        return mu, logvar

    def encode(self, x: torch.Tensor, cond: torch.Tensor):
        h = torch.cat([x, cond], dim=-1)
        h = F.silu(self.enc_in(h))
        for block in self.enc_blocks:
            h = block(h)
        h = F.silu(self.enc_norm(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h).clamp(-6.0, 2.0)
        return mu, logvar

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        style = torch.cat([z, cond], dim=-1)
        h = F.silu(self.dec_in(style))
        for block in self.dec_blocks:
            h = block(h, style)
        h = F.silu(self.dec_norm(h))
        return self.dec_out(h)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, deterministic: bool = False):
        mu, logvar = self.encode(x, cond)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        recon = self.decode(z, cond)
        return recon, mu, logvar

    def forward_encoder(self, x: torch.Tensor, cond: torch.Tensor, deterministic: bool = False):
        mu, logvar = self.encode(x, cond)
        z = mu if deterministic else self.reparameterize(mu, logvar)
        return z, mu, logvar
