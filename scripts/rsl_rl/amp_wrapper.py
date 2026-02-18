# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper that records AMP observations each step for discriminator updates."""

import gymnasium as gym


def _get_amp_obs(env):
    from bascat_rl.tasks.manager_based.bascat_rl.mdp import amp_state_ostrich
    return amp_state_ostrich(env)


class AMPRolloutRecorder(gym.Wrapper):
    """Records AMP observations at each step. Call get_rollout_amp_obs() after a rollout, then clear_rollout_amp_obs()."""

    def __init__(self, env):
        super().__init__(env)
        self._amp_buffer = []

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        try:
            amp_obs = _get_amp_obs(self.unwrapped)
            self._amp_buffer.append(amp_obs.detach().cpu().clone())
        except Exception:
            pass
        return obs, reward, terminated, truncated, info

    def get_rollout_amp_obs(self):
        if not self._amp_buffer:
            return None
        import torch
        return torch.cat(self._amp_buffer, dim=0)

    def clear_rollout_amp_obs(self):
        self._amp_buffer.clear()
