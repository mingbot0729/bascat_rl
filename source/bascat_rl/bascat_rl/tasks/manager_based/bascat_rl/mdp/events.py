# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom event terms for domain randomization."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

logger = logging.getLogger(__name__)


def apply_external_force_torque_stochastic(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    force_range: dict[str, tuple[float, float]],
    torque_range: dict[str, tuple[float, float]],
    probability: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Apply random external forces and torques with a given probability each step.

    Unlike push_by_setting_velocity (which teleports velocity), this applies actual
    physics forces that the robot must react to naturally. Each step, each env has
    a `probability` chance of receiving a random force/torque impulse.

    From Tron bipedal locomotion example.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    # Clear existing external forces
    asset._external_force_b *= 0
    asset._external_torque_b *= 0

    # Resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # Stochastic mask: only apply to a random subset of envs
    random_values = torch.rand(env_ids.shape, device=env_ids.device)
    mask = random_values < probability
    masked_env_ids = env_ids[mask]

    if len(masked_env_ids) == 0:
        return

    # Resolve number of bodies
    num_bodies = len(asset_cfg.body_ids) if isinstance(asset_cfg.body_ids, list) else asset.num_bodies

    # Sample random forces and torques
    size = (len(masked_env_ids), num_bodies, 3)
    force_range_list = [force_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    force_range_t = torch.tensor(force_range_list, device=asset.device)
    forces = math_utils.sample_uniform(force_range_t[:, 0], force_range_t[:, 1], size, asset.device)

    torque_range_list = [torque_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z"]]
    torque_range_t = torch.tensor(torque_range_list, device=asset.device)
    torques = math_utils.sample_uniform(torque_range_t[:, 0], torque_range_t[:, 1], size, asset.device)

    # Apply forces (only takes effect when asset.write_data_to_sim() is called)
    asset.set_external_force_and_torque(forces, torques, env_ids=masked_env_ids, body_ids=asset_cfg.body_ids)


def randomize_terrain_friction(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
):
    """Randomize the terrain/floor physics material friction at startup.

    Samples static and dynamic friction from the given ranges and applies them to the
    terrain's physics material. The terrain is shared across envs, so one sample is
    applied to the whole floor. Uses the terrain's rigid body view if available.
    """
    terrain = env.scene.terrain
    if not hasattr(terrain, "root_physx_view") or terrain.root_physx_view is None:
        logger.debug(
            "randomize_terrain_friction: terrain has no root_physx_view, skipping floor friction randomization"
        )
        return
    try:
        # Sample one set of friction values (floor is shared)
        static = math_utils.sample_uniform(
            static_friction_range[0], static_friction_range[1], (1, 1), device="cpu"
        ).item()
        dynamic = math_utils.sample_uniform(
            dynamic_friction_range[0], dynamic_friction_range[1], (1, 1), device="cpu"
        ).item()
        dynamic = min(dynamic, static)  # PhysX: dynamic <= static
        # Get material buffer: (num_instances, num_shapes, 3) for [static_friction, dynamic_friction, restitution]
        materials = terrain.root_physx_view.get_material_properties()
        if materials is None or materials.numel() == 0:
            logger.debug("randomize_terrain_friction: no material buffer, skipping")
            return
        # Apply same friction to all terrain shapes (one shared floor)
        materials[:, :, 0] = static
        materials[:, :, 1] = dynamic
        # restitution unchanged
        all_ids = torch.arange(env.scene.num_envs, device="cpu") if env_ids is None else env_ids.cpu()
        terrain.root_physx_view.set_material_properties(materials, all_ids)
        logger.info(
            "randomize_terrain_friction: set floor friction static=%.2f dynamic=%.2f", static, dynamic
        )
    except Exception as e:
        logger.warning("randomize_terrain_friction failed (floor friction not randomized): %s", e)
