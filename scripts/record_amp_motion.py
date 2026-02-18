# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record AMP reference motion from the env. Uses scripted gait by default; optional random actions."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Ostrich-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=2000, help="Number of steps to record (control steps)")
parser.add_argument("--out", type=str, default="ostrich_walk_amp.npz", help="Output .npz path")
parser.add_argument("--random", action="store_true", help="Use random actions instead of scripted gait")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
AppLauncher(args).app

import gymnasium as gym
import torch
import numpy as np

# Allow importing scripted_gait from same directory (scripts/) when run as python scripts/record_amp_motion.py
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

import bascat_rl.tasks  # noqa: F401
from bascat_rl.tasks.manager_based.bascat_rl.mdp import amp_state_ostrich
from bascat_rl.tasks.manager_based.bascat_rl.flat_env_cfg import OstrichFlatEnvCfg
from scripted_gait import ScriptedPointFootGait, rad_to_normalized_action

# Action scaling: JointPositionActionCfg(scale=0.5, use_default_offset=True) => action = (q_des - 0) / 0.5
ACTION_SCALE = 0.5
DEFAULT_JOINT_POS = np.zeros(6, dtype=np.float32)


def main():
    env_cfg = OstrichFlatEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make(args.task, cfg=env_cfg)
    buffer = []
    obs, _ = env.reset()
    device = env.unwrapped.device

    if args.random:
        n_actions = int(np.prod(env.action_space.shape))
    else:
        # Scripted gait: control step dt (decimation * sim.dt)
        step_dt = getattr(env.unwrapped, "step_dt", env_cfg.decimation * env_cfg.sim.dt)
        gait = ScriptedPointFootGait(dt=step_dt)

    for step in range(args.steps):
        if args.random:
            action = 0.1 * (torch.rand(args.num_envs, n_actions, device=device) * 2 - 1)
        else:
            q_des = gait.step()
            # Convert rad to normalized action [-1, 1] for env
            action_n = rad_to_normalized_action(q_des, scale=ACTION_SCALE, default_pos=DEFAULT_JOINT_POS)
            action = np.zeros((args.num_envs, 6), dtype=np.float32)
            action[0, :] = action_n
            for i in range(1, args.num_envs):
                action[i, :] = action_n
            action = torch.from_numpy(action).to(device=device, dtype=torch.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        amp_obs = amp_state_ostrich(env.unwrapped)
        buffer.append(amp_obs[0:1].detach().cpu().numpy())
        if step % 500 == 0:
            print(f"Recorded {step}/{args.steps}")
    env.close()
    amp_states = np.concatenate(buffer, axis=0)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(args.out, amp_states=amp_states)
    print(f"Saved {amp_states.shape} to {args.out}")


if __name__ == "__main__":
    main()
