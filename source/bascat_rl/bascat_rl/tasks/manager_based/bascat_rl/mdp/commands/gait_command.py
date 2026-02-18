"""Gait command generator for bipedal locomotion.

Generates gait frequency, phase offset, and contact duration commands
that define the desired stepping pattern. From Tron bipedal example.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from .commands_cfg import UniformGaitCommandCfg


class GaitCommand(CommandTerm):
    """Command generator that generates gait frequency, phase offset and contact duration.

    The command is a 3D vector: [frequency, phase_offset, contact_duration]
    - frequency: stepping frequency in Hz (e.g. 2.0 = 2 steps per second)
    - phase_offset: offset between left and right feet (0.5 = alternating)
    - contact_duration: fraction of cycle that foot is on ground (0.5 = 50% stance)
    """

    cfg: UniformGaitCommandCfg

    def __init__(self, cfg: UniformGaitCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        # Command buffer: [frequency, phase_offset, contact_duration]
        self.gait_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics = {}

    def __str__(self) -> str:
        msg = "GaitCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    @property
    def command(self) -> torch.Tensor:
        """The gait command. Shape is (num_envs, 3)."""
        return self.gait_command

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        """Resample gait parameters for specified environments."""
        r = torch.empty(len(env_ids), device=self.device)
        self.gait_command[env_ids, 0] = r.uniform_(*self.cfg.ranges.frequencies)
        self.gait_command[env_ids, 1] = r.uniform_(*self.cfg.ranges.offsets)
        self.gait_command[env_ids, 2] = r.uniform_(*self.cfg.ranges.durations)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass

    def _debug_vis_callback(self, event):
        pass
