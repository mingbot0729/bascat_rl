# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train with custom AMP: load reference .npz, train discriminator each iteration, add AMP reward."""

import argparse
import sys
import os

from isaaclab.app import AppLauncher
import cli_args  # noqa: E402

parser = argparse.ArgumentParser(description="Train with custom AMP (Adversarial Motion Priors).")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=2000)
parser.add_argument("--video_interval", type=int, default=20000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Ostrich-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--motion_file", type=str, required=True, help="Path to reference motion .npz (key: amp_states, shape (T, 22))")
parser.add_argument("--amp_weight", type=float, default=0.5, help="Weight for AMP reward term")
parser.add_argument("--distributed", action="store_true", default=False)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata
from packaging import version
RSL_RL_VERSION = "2.3.1"
if args_cli.distributed and version.parse(metadata.version("rsl-rl-lib")) < version.parse(RSL_RL_VERSION):
    exit(1)

import gymnasium as gym
import torch
import numpy as np
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner
from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, DirectMARLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
import bascat_rl.tasks  # noqa: F401

from amp_wrapper import AMPRolloutRecorder
from bascat_rl.tasks.manager_based.bascat_rl.amp_discriminator import AMPDiscriminator, update_discriminator

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.video:
        env_cfg.sim.render_interval = 1

    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name + "_amp")
    log_root_path = os.path.abspath(log_root_path)
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)
    print(f"[INFO] AMP training log dir: {log_dir}")

    # Load expert motion
    data = np.load(args_cli.motion_file)
    expert_states = data["amp_states"]  # (T, 22)
    expert_tensor = torch.from_numpy(expert_states).float()
    print(f"[INFO] Loaded expert motion: {expert_states.shape}")

    # Enable AMP reward and set weight
    if hasattr(env_cfg.rewards, "amp_reward"):
        env_cfg.rewards.amp_reward.weight = args_cli.amp_weight
        print(f"[INFO] AMP reward weight: {args_cli.amp_weight}")

    base_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(base_env.unwrapped, DirectMARLEnv):
        base_env = multi_agent_to_single_agent(base_env)

    # AMP: discriminator and recorder (keep reference to amp_wrapper for get/clear)
    device = torch.device(env_cfg.sim.device)
    discriminator = AMPDiscriminator(use_sigmoid=True).to(device)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    base_env.unwrapped.amp_discriminator = discriminator
    amp_wrapper = AMPRolloutRecorder(base_env)
    env = amp_wrapper
    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "train"),
            step_trigger=lambda s: s % args_cli.video_interval == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    max_it = agent_cfg.max_iterations
    for it in range(max_it):
        runner.learn(num_learning_iterations=1, init_at_random_ep_len=(it == 0))
        # Update discriminator from last rollout
        amp_obs = amp_wrapper.get_rollout_amp_obs()
        if amp_obs is not None and amp_obs.shape[0] > 0:
            d_loss = update_discriminator(
                discriminator, expert_tensor, amp_obs, d_optimizer, device,
                num_epochs=1, batch_size=256,
            )
            base_env.unwrapped.amp_discriminator = discriminator
            if it % 50 == 0:
                print(f"[AMP] iter {it} discriminator loss: {d_loss:.4f}")
        amp_wrapper.clear_rollout_amp_obs()

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
