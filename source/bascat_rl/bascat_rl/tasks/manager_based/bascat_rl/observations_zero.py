# observations_zero.py
import torch

def base_lin_vel_zero(env, asset_cfg=None):
    # Use a standard dtype; env.sim has no 'dtype' attr
    return torch.zeros(
        (env.num_envs, 3),
        device=env.device,
        dtype=torch.float32,              # <-- fix: don't use env.sim.dtype
    )
