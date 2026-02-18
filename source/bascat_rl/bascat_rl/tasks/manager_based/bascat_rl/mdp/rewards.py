"""Reward functions for the Ostrich biped locomotion task.

Adapted from the Tron point-foot example. Only contains rewards actually
used by the environment config.
"""

from __future__ import annotations

import torch
from torch import distributions
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def joint_powers_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint powers on the articulation using L1-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.abs(torch.mul(asset.data.applied_torque, asset.data.joint_vel)), dim=1)


def no_fly(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 1.0,
) -> torch.Tensor:
    """Reward if only one foot is in contact with the ground (biped single-stance)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    latest_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, :, 2]
    contacts = latest_contact_forces > threshold
    single_contact = torch.sum(contacts.float(), dim=1) == 1
    return 1.0 * single_contact


def base_com_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base CoM height deviation from target using L2 squared kernel."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


class GaitReward(ManagerTermBase):
    """Phase-based gait reward that enforces alternating left/right steps.

    Uses the gait command (frequency, offset, duration) to compute desired
    contact states for each foot at each timestep. Penalizes:
    - Foot contact force when the foot should be in swing (in the air)
    - Foot velocity when the foot should be in stance (on the ground)

    This produces clean alternating walking gaits at the commanded frequency.
    From Tron bipedal locomotion example.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.sensor_cfg = cfg.params["sensor_cfg"]
        self.asset_cfg = cfg.params["asset_cfg"]

        # Resolve sensor and asset references
        self.contact_sensor: ContactSensor = env.scene.sensors[self.sensor_cfg.name]
        self.asset: Articulation = env.scene[self.asset_cfg.name]

        # Store configuration parameters
        self.force_scale = float(cfg.params["tracking_contacts_shaped_force"])
        self.vel_scale = float(cfg.params["tracking_contacts_shaped_vel"])
        self.force_sigma = cfg.params["gait_force_sigma"]
        self.vel_sigma = cfg.params["gait_vel_sigma"]
        self.kappa_gait_probs = cfg.params["kappa_gait_probs"]
        self.command_name = cfg.params["command_name"]
        self.dt = env.step_dt

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        tracking_contacts_shaped_force,
        tracking_contacts_shaped_vel,
        gait_force_sigma,
        gait_vel_sigma,
        kappa_gait_probs,
        command_name,
        sensor_cfg,
        asset_cfg,
    ) -> torch.Tensor:
        """Compute the gait reward.

        Combines force-based and velocity-based terms to encourage
        the desired contact pattern at the commanded stepping frequency.
        """
        gait_params = env.command_manager.get_command(self.command_name)

        # Compute desired contact states from gait phase
        desired_contact_states = self._compute_contact_targets(gait_params)

        # Force-based reward: penalize contact when foot should be in air
        foot_forces = torch.norm(
            self.contact_sensor.data.net_forces_w[:, self.sensor_cfg.body_ids], dim=-1
        )
        force_reward = self._compute_force_reward(foot_forces, desired_contact_states)

        # Velocity-based reward: penalize movement when foot should be on ground
        foot_velocities = torch.norm(
            self.asset.data.body_lin_vel_w[:, self.asset_cfg.body_ids], dim=-1
        )
        velocity_reward = self._compute_velocity_reward(foot_velocities, desired_contact_states)

        return force_reward + velocity_reward

    def _compute_contact_targets(self, gait_params: torch.Tensor) -> torch.Tensor:
        """Calculate desired contact states for the current timestep.

        Uses von Mises-like smoothing to produce continuous desired contact
        probabilities rather than hard binary contacts.
        """
        frequencies = gait_params[:, 0]
        offsets = gait_params[:, 1]
        durations = torch.cat(
            [
                gait_params[:, 2].view(self.num_envs, 1),
                gait_params[:, 2].view(self.num_envs, 1),
            ],
            dim=1,
        )

        # Compute gait phase from episode step counter
        gait_indices = torch.remainder(
            self._env.episode_length_buf * self.dt * frequencies, 1.0
        )

        # Compute per-foot phase (left foot = base phase, right foot = base + offset)
        foot_indices = torch.remainder(
            torch.cat(
                [
                    gait_indices.view(self.num_envs, 1),
                    (gait_indices + offsets + 1).view(self.num_envs, 1),
                ],
                dim=1,
            ),
            1.0,
        )

        # Determine stance and swing phases
        stance_idxs = foot_indices < durations
        swing_idxs = foot_indices > durations

        # Normalize foot indices within stance/swing phases
        foot_indices[stance_idxs] = torch.remainder(foot_indices[stance_idxs], 1) * (
            0.5 / durations[stance_idxs]
        )
        foot_indices[swing_idxs] = 0.5 + (
            torch.remainder(foot_indices[swing_idxs], 1) - durations[swing_idxs]
        ) * (0.5 / (1 - durations[swing_idxs]))

        # Smooth contact probabilities using normal CDF
        smoothing_cdf = distributions.normal.Normal(0, self.kappa_gait_probs).cdf
        desired_contact_states = smoothing_cdf(foot_indices) * (
            1 - smoothing_cdf(foot_indices - 0.5)
        ) + smoothing_cdf(foot_indices - 1) * (
            1 - smoothing_cdf(foot_indices - 1.5)
        )

        return desired_contact_states

    def _compute_force_reward(
        self, forces: torch.Tensor, desired_contacts: torch.Tensor
    ) -> torch.Tensor:
        """Force-based reward: penalize contact force when foot should be in air."""
        reward = torch.zeros_like(forces[:, 0])
        if self.force_scale < 0:
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * (
                    1 - torch.exp(-forces[:, i] ** 2 / self.force_sigma)
                )
        else:
            for i in range(forces.shape[1]):
                reward += (1 - desired_contacts[:, i]) * torch.exp(
                    -forces[:, i] ** 2 / self.force_sigma
                )
        return (reward / forces.shape[1]) * self.force_scale

    def _compute_velocity_reward(
        self, velocities: torch.Tensor, desired_contacts: torch.Tensor
    ) -> torch.Tensor:
        """Velocity-based reward: penalize foot movement when foot should be on ground."""
        reward = torch.zeros_like(velocities[:, 0])
        if self.vel_scale < 0:
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * (
                    1 - torch.exp(-velocities[:, i] ** 2 / self.vel_sigma)
                )
        else:
            for i in range(velocities.shape[1]):
                reward += desired_contacts[:, i] * torch.exp(
                    -velocities[:, i] ** 2 / self.vel_sigma
                )
        return (reward / velocities.shape[1]) * self.vel_scale


class ActionSmoothnessPenalty(ManagerTermBase):
    """Penalize large instantaneous changes in the network action output.

    Computes a second-order finite difference of actions to encourage smooth
    transitions. From Tron bipedal locomotion example.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_prev_action = None
        self.prev_action = None

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        current_action = env.action_manager.action.clone()

        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        # Ignore penalty during first few steps of episode
        startup_mask = env.episode_length_buf < 3
        penalty[startup_mask] = 0

        return penalty
