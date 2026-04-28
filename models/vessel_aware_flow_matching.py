"""Conditional flow-matching model for vessel-aware GHD token generation."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.view(-1, 1)
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype)
            * (-math.log(10000.0) / max(half - 1, 1))
        )
        args = t * freqs.view(1, -1)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMVelocityBlock(nn.Module):
    def __init__(self, hidden_dim: int, style_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.film1 = nn.Linear(style_dim, hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.film2 = nn.Linear(style_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate = nn.Linear(style_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    @staticmethod
    def _film(x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        gamma, beta = params.chunk(2, dim=-1)
        return x * (1.0 + gamma) + beta

    def forward(self, x: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        h = self._film(self.norm1(x), self.film1(style))
        h = self.fc1(F.silu(h))
        h = self.dropout(h)
        h = self._film(self.norm2(h), self.film2(style))
        h = self.fc2(F.silu(h))
        return x + torch.sigmoid(self.gate(style)) * h


class VesselAwareFlowMatching(nn.Module):
    """
    Rectified-flow / conditional-flow-matching velocity network.

    Training target:
        x0 ~ N(0, I)
        x1 = normalized GHD token
        x_t = (1 - t) x0 + t x1
        v* = x1 - x0
        model predicts v_theta(x_t, t, condition)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        cond_dim: int = 32,
        time_dim: int = 64,
        blocks: int = 8,
        dropout: float = 0.02,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.cond_dim = cond_dim
        self.time_dim = time_dim

        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.style = nn.Sequential(
            nn.Linear(cond_dim + time_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.x_in = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [FiLMVelocityBlock(hidden_dim, hidden_dim, dropout) for _ in range(blocks)]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        t_emb = self.time_embed(t)
        style = self.style(torch.cat([cond, t_emb], dim=-1))
        h = F.silu(self.x_in(x_t))
        for block in self.blocks:
            h = block(h, style)
        return self.out(F.silu(self.norm(h)))

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        num_steps: int = 32,
        temperature: float = 1.0,
        method: str = "heun",
        x0: torch.Tensor = None,
    ) -> torch.Tensor:
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if method not in ("euler", "heun"):
            raise ValueError("method must be 'euler' or 'heun'")

        batch = cond.shape[0]
        if x0 is None:
            x = torch.randn(batch, self.input_dim, device=cond.device, dtype=cond.dtype) * temperature
        else:
            x = x0.to(device=cond.device, dtype=cond.dtype)

        dt = 1.0 / float(num_steps)
        for step in range(num_steps):
            t0 = torch.full((batch, 1), step * dt, device=cond.device, dtype=cond.dtype)
            v0 = self.forward(x, t0, cond)
            if method == "euler" or step == num_steps - 1:
                x = x + dt * v0
            else:
                x_euler = x + dt * v0
                t1 = torch.full((batch, 1), (step + 1) * dt, device=cond.device, dtype=cond.dtype)
                v1 = self.forward(x_euler, t1, cond)
                x = x + 0.5 * dt * (v0 + v1)
        return x
