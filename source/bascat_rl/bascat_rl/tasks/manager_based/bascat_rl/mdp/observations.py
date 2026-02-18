"""Custom observation functions for bipedal locomotion.

Provides gait phase and gait command observations that the policy
uses to know where in the stepping cycle it should be.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def get_gait_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the current gait phase as [sin(phase), cos(phase)].

    The phase is computed from the episode step counter and gait frequency.
    Using sin/cos ensures continuity (no discontinuity at phase wrap-around).

    Returns:
        torch.Tensor: Shape (num_envs, 2) -- [sin(2*pi*phase), cos(2*pi*phase)]
    """
    if not hasattr(env, "episode_length_buf"):
        return torch.zeros(env.num_envs, 2, device=env.device)

    # Get gait frequency from the gait command
    command_term = env.command_manager.get_term("gait_command")
    frequencies = command_term.command[:, 0]  # Hz

    # Compute phase: (step_count * dt * frequency) mod 1.0
    gait_indices = torch.remainder(
        env.episode_length_buf * env.step_dt * frequencies, 1.0
    )
    gait_indices = gait_indices.unsqueeze(-1)

    sin_phase = torch.sin(2 * torch.pi * gait_indices)
    cos_phase = torch.cos(2 * torch.pi * gait_indices)

    return torch.cat([sin_phase, cos_phase], dim=-1)


def get_gait_command(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Get the current gait command parameters as observation.

    Returns:
        torch.Tensor: Shape (num_envs, 3) -- [frequency, phase_offset, contact_duration]
    """
    return env.command_manager.get_command(command_name)
