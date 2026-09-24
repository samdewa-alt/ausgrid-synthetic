from __future__ import annotations

import torch
from torch import nn


class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim: int = 16, hidden: int = 128, cond_dim: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(96 + cond_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU())
        self.posterior = nn.Linear(hidden, 2 * latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim + cond_dim, hidden), nn.ReLU(),
                                     nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 96))

    def decode(self, z: torch.Tensor, cond: torch.Tensor, night: torch.Tensor | None = None) -> torch.Tensor:
        output = torch.nn.functional.softplus(self.decoder(torch.cat([z, cond], dim=1))).reshape(-1, 2, 48)
        if night is not None:
            solar = output[:, 1, :] * (~night).float()
            output = torch.stack((output[:, 0, :], solar), dim=1)
        return output

    def forward(self, x: torch.Tensor, cond: torch.Tensor, night: torch.Tensor | None = None):
        hidden = self.encoder(torch.cat([x.flatten(1), cond], dim=1))
        mean, logvar = self.posterior(hidden).chunk(2, dim=1)
        logvar = logvar.clamp(-10, 10)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return self.decode(z, cond, night), mean, logvar
