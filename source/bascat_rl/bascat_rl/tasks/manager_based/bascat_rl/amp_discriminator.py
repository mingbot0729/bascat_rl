# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AMP discriminator network: takes AMP state (e.g. 22-dim) and outputs probability 'real'."""

import torch
import torch.nn as nn

from .mdp.amp_observations import AMP_OBS_DIM


class AMPDiscriminator(nn.Module):
    """MLP discriminator for AMP. Input: (N, amp_obs_dim), Output: (N, 1) logits or probs."""

    def __init__(
        self,
        input_dim: int = AMP_OBS_DIM,
        hidden_dims: tuple = (256, 256),
        use_sigmoid: bool = True,
    ):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ELU(inplace=True)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        if self.use_sigmoid:
            return torch.sigmoid(out)
        return out


def update_discriminator(
    D: nn.Module,
    expert_states: torch.Tensor,
    policy_states: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 1,
    batch_size: int = 256,
) -> float:
    """Train D for a few steps: real = expert_states (label 1), fake = policy_states (label 0). Returns mean loss."""
    D.train()
    expert_states = expert_states.to(device)
    policy_states = policy_states.to(device)
    n_expert = expert_states.shape[0]
    n_policy = policy_states.shape[0]
    total_loss = 0.0
    n_batches = 0
    for _ in range(num_epochs):
        perm_e = torch.randperm(n_expert, device=device)
        perm_p = torch.randperm(n_policy, device=device)
        for start in range(0, min(n_expert, n_policy), batch_size):
            end = start + batch_size
            real = expert_states[perm_e[start % n_expert : (start % n_expert) + batch_size]]
            fake = policy_states[perm_p[start % n_policy : (start % n_policy) + batch_size]]
            if real.shape[0] == 0 or fake.shape[0] == 0:
                break
            # Pad to same size
            m = min(real.shape[0], fake.shape[0])
            real, fake = real[:m], fake[:m]
            optimizer.zero_grad()
            out_real = D(real)
            out_fake = D(fake)
            loss_real = nn.functional.binary_cross_entropy(out_real, torch.ones_like(out_real, device=device))
            loss_fake = nn.functional.binary_cross_entropy(out_fake, torch.zeros_like(out_fake, device=device))
            loss = (loss_real + loss_fake) / 2
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)
