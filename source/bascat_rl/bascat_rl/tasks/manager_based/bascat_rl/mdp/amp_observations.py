# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""AMP observation: state vector for the discriminator (must match reference .npz format)."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# AMP state dimension for Ostrich (1 + 3 + 6 + 3 + 3 + 6)
AMP_OBS_DIM = 22


def amp_state_ostrich(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """AMP state for Ostrich: [root_height(1), proj_gravity(3), joint_pos(6), root_lin_vel(3), root_ang_vel(3), joint_vel(6)] = 22 dims.
    Use the same order when saving reference motion to .npz.
    """
    asset = env.scene[asset_cfg.name]
    n = asset.data.root_pos_w.shape[0]
    h = asset.data.root_pos_w[:, 2:3]       # (N, 1)
    # Projected gravity in body frame (same as policy obs)
    g_world = torch.zeros(n, 3, device=asset.data.root_quat_w.device)
    g_world[:, 2] = -1.0
    g = quat_apply_inverse(asset.data.root_quat_w, g_world)  # (N, 3)
    q = asset.data.joint_pos                # (N, 6)
    v_lin = asset.data.root_lin_vel_w       # (N, 3)
    v_ang = asset.data.root_ang_vel_w       # (N, 3)
    qd = asset.data.joint_vel               # (N, 6)
    return torch.cat([h, g, q, v_lin, v_ang, qd], dim=1)  # (N, 22)
