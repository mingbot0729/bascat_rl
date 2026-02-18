# AMP (Adversarial Motion Priors) Implementation Guide for Ostrich

This guide explains **everything you need to do** to add AMP for better gait movement on your manager-based Ostrich velocity task.

---

## 1. What AMP Does

- **Goal**: Make the policy’s motion look like **reference motions** (e.g. walking clips).
- **Mechanism**: A **discriminator** network is trained to tell “reference” vs “policy” from a **motion state**. The policy gets reward for **fooling** the discriminator (looking like reference).
- **Result**: More natural, human/animal-like gaits and less “jittery” or unstable motion.

**You need**:
1. **Reference motion data** (.npz) for the Ostrich.
2. **AMP observation** in the env: same format as the data the discriminator sees.
3. **Discriminator** + **AMP reward** in the training loop.
4. **Runner** that supports AMP (e.g. amp-rsl-rl or custom integration).

---

## 2. High-Level Options

| Option | Pros | Cons |
|--------|------|------|
| **A. amp-rsl-rl** | AMP built into runner, less custom code | May need adapter for Isaac Lab ManagerBasedRLEnv; check compatibility |
| **B. Custom AMP in env** | Full control, stays on standard RSL-RL | You implement discriminator training + reward and motion loading |

Below we describe **Option B** in detail (custom AMP) so it works with your current **ManagerBasedRLEnv** and **OnPolicyRunner**. You can later swap to amp-rsl-rl if you adopt their runner.

---

## 3. Define the AMP State (Discriminator Input)

The discriminator must see the **same vector** for both:
- **Reference**: from your .npz motion.
- **Policy**: from the current simulation (same features, same order).

A minimal, robust choice for Ostrich (6 joints):

- Root height (1)
- Root orientation, e.g. projected gravity or quat (3 or 4)
- Joint positions (6)
- Root linear velocity (3)
- Root angular velocity (3)
- Joint velocities (6)

**Example dimension**: 1 + 3 + 6 + 3 + 3 + 6 = **22** (e.g. use projected gravity for orientation).

- If you use quat: 1 + 4 + 6 + 3 + 3 + 6 = **23**.

**You must**:
- Implement an **observation term** (or function) that returns this vector from `env` (from robot state).
- Save the **same vector** in your reference motion .npz (see below).
- Use this as **AMP observation** for both reference sampling and policy rollout.

---

## 4. Reference Motion Data (.npz)

### 4.1 What to store

- **AMP state** at each timestep (same 22- or 23-dim vector above).
- **Sampling rate**: Match your control rate (e.g. 50 Hz if `decimation=2`, `sim.dt=0.005`).
- **Length**: At least a few walking cycles (e.g. 5–10 s).

### 4.2 How to get the data

**Option 1 – Record from current sim (easiest)**  
- Add a small script that:
  - Runs the Ostrich env with a **scripted or hand-tuned controller** (e.g. sinusoids, PD targets, or a pre-trained policy).
  - At each step, computes the **same AMP state** you defined above and appends to a list.
  - Saves to `.npz`, e.g. `amp_states = (T, 22)` and optionally `dt` or `freq`.

**Option 2 – Motion capture**  
- If you have mocap for a similar robot, convert to your robot’s joint/root and export the same AMP state vector at your control rate.

**Option 3 – Use existing humanoid motions**  
- Not directly applicable: humanoid .npz in Isaac Lab is for humanoid body layout. For Ostrich you need Ostrich-specific AMP states.

### 4.3 .npz format (suggestion)

```python
# Save
np.savez(
    "ostrich_walk_amp.npz",
    amp_states=amp_states,   # shape (T, amp_obs_dim), e.g. (1000, 22)
    dt=0.02,                 # optional: time step
)
# Load
data = np.load("ostrich_walk_amp.npz")
amp_states = data["amp_states"]
```

- `amp_obs_dim` must match the AMP observation dimension in the env (e.g. 22 or 23).

---

## 5. Add AMP Observation to the Env

- **Where**: In your flat env config (e.g. `flat_env_cfg.py`), you already have observation groups (e.g. `policy`).  
- **Add**: Either:
  - A **separate observation group** used only for AMP (e.g. `amp`), or  
  - An **extra term** in an existing group that you later slice out for the discriminator.

Recommended: **separate group `amp`** so the policy does not get AMP obs as input; only the discriminator uses it.

### 5.1 Implement AMP state function

In your mdp (e.g. `observations.py` or a new `amp_observations.py`), implement a function that, given `env`, returns a tensor of shape `(num_envs, amp_obs_dim)`:

- Root height: `asset.data.root_pos_w[:, 2]`
- Projected gravity: `asset.data.projected_gravity_b` (if available) or compute from `asset.data.root_quat_w`
- Joint pos: `asset.data.joint_pos`
- Root lin vel: `asset.data.root_lin_vel_w`
- Root ang vel: `asset.data.root_ang_vel_w`
- Joint vel: `asset.data.joint_vel`

Stack in the **same order** as in the .npz. Register this as an observation term (e.g. `ObsTerm(func=amp_state_vector, ...)`).

### 5.2 Add observation group in config

- In the same env config where you have `observations.policy`, add e.g. `observations.amp` with a single term that uses that function.
- Set dimension to match (e.g. 22 or 23). No need to add noise for AMP (discriminator sees “clean” state).

---

## 6. Discriminator and AMP Reward

### 6.1 Discriminator network

- **Input**: `amp_obs_dim` (e.g. 22).
- **Output**: Single scalar (probability “real” or logit).
- **Architecture**: e.g. MLP 2–3 layers, hidden 256–512, output 1. Same as in AMP papers.

### 6.2 Training the discriminator

- **Real**: sample random timesteps from your reference .npz (`amp_states`).
- **Fake**: AMP observations from policy rollouts (from the env’s AMP observation group).
- **Loss**: Binary cross-entropy: real labels 1, fake labels 0 (or use gradient penalty / WGAN-style if you prefer).
- **When**: Typically every N policy steps, train the discriminator for K steps on a batch of (real, fake) pairs.

### 6.3 AMP reward

- **Formula**: `r_amp = -log(max(1 - D(s), epsilon))` so that looking “real” (D(s)→1) gives high reward.
- **Where**: Either:
  - **In the env**: Each step, run the current AMP observation through the discriminator and add `r_amp` as a reward term (you must pass the discriminator into the env or use a global/callback), or  
  - **In the runner**: If using an AMP-capable runner, it often computes this from stored AMP observations and adds it to the reward.

For **custom integration** with standard RSL-RL `OnPolicyRunner`, the easiest is to add the AMP reward **inside the env** as an extra reward term, and update the discriminator **outside** the env (e.g. in a wrapper or a custom training script that calls `env.step`, then updates D, then provides the updated D to the env for the next rollout). Alternatively you can use a **custom runner** that after each rollout: (1) updates D from buffer, (2) computes AMP rewards for the buffer, (3) runs PPO.

---

## 7. Integration Steps (Checklist)

1. **Decide AMP state**  
   - Fix the list of features and dimension (e.g. 22).

2. **Implement AMP state in code**  
   - One function from `env` → `(num_envs, amp_obs_dim)`.

3. **Add AMP observation group**  
   - In `flat_env_cfg.py` (or equivalent), add the group and the term that uses that function.

4. **Generate reference .npz**  
   - Recording script + scripted/PD/walk controller → `amp_states`, save .npz.

5. **Implement discriminator**  
   - PyTorch module, input `amp_obs_dim`, output 1.

6. **Implement discriminator training**  
   - Load .npz; each update sample (real from .npz, fake from last rollout’s AMP obs); backward on BCE loss.

7. **Add AMP reward**  
   - Either in env (with D passed in or updated globally) or in a custom training loop/runner using buffer AMP obs and current D.

8. **Wire into training**  
   - Use current `OnPolicyRunner` and add AMP reward as one term in the env, **or** switch to an AMP runner (e.g. amp-rsl-rl) if you use their env interface.

9. **Tune**  
   - Weight of AMP reward vs task reward (e.g. 0.1–1.0 for AMP); discriminator learning rate and update frequency.

---

## 8. Optional: Use amp-rsl-rl

- **Package**: [amp-rsl-rl](https://github.com/ami-iit/amp-rsl-rl) (or similar).
- **Steps**:
  1. Install: `pip install amp-rsl-rl` (or clone and install).
  2. Check if their runner accepts an Isaac Lab `ManagerBasedRLEnv` (or a wrapper that provides their expected interface).
  3. If they expect a specific env API (e.g. `get_amp_observations()`), add that to your env and pass AMP obs from your group.
  4. Configure motion file path and AMP observation dimension to match your .npz and `amp_obs_dim`.
  5. If their runner does not support ManagerBasedRLEnv, you may need to keep custom AMP (discriminator + reward in env + your runner).

---

## 9. Summary

| Step | Action |
|------|--------|
| 1 | Define AMP state (e.g. 22 dims: height, proj_gravity, joint_pos, root_lin_vel, root_ang_vel, joint_vel). |
| 2 | Implement that state as an observation term and add an `amp` observation group in the env config. |
| 3 | Record or export reference motion to .npz with the same 22-dim vector at your control rate. |
| 4 | Implement a discriminator (MLP, amp_obs_dim → 1) and train it on (reference, policy) AMP states. |
| 5 | Add AMP reward = -log(1 - D(s)) (and optionally clip for stability). |
| 6 | Integrate D training and AMP reward into your training loop or use an AMP-capable runner. |
| 7 | Tune AMP reward weight and D update frequency. |

Once these are in place, the policy will receive an extra reward for matching the reference motion distribution, which should yield better, more natural gait movement on the Ostrich.

---

## 10. Custom Implementation (Done for You)

The following are **already implemented** in this repo:

### 10.1 AMP state and observation group

- **`mdp/amp_observations.py`**: `amp_state_ostrich()` returns 22-dim vector (height, proj_gravity, joint_pos, root_lin_vel, root_ang_vel, joint_vel).
- **`flat_env_cfg.py`**: Observation group `observations.amp` is added in `__post_init__` via `AmpObsGroupCfg`.
- **`mdp/rewards.py`**: `amp_reward(env)` reads `env.unwrapped.amp_discriminator` and returns `-log(1 - D(amp_obs))`; if D is None, returns zeros.
- **Reward term**: `rewards.amp_reward` with default weight 0 (set > 0 when training with AMP).

### 10.2 Discriminator and training

- **`amp_discriminator.py`**: `AMPDiscriminator` (MLP 22→256→256→1) and `update_discriminator(expert, policy, optimizer, ...)`.
- **`scripts/rsl_rl/amp_wrapper.py`**: `AMPRolloutRecorder` records AMP obs each step; `get_rollout_amp_obs()` / `clear_rollout_amp_obs()`.
- **`scripts/rsl_rl/train_amp.py`**: Loads expert .npz, creates D, wraps env, sets `amp_discriminator`, runs PPO and updates D each iteration.

**Usage:**

1. **Record reference motion** (from repo root, with Isaac Sim):
   ```bash
   python scripts/record_amp_motion.py --task Isaac-Velocity-Flat-Ostrich-v0 --steps 2000 --out data/ostrich_walk_amp.npz
   ```
2. **Train with AMP:**
   ```bash
   cd scripts/rsl_rl && python train_amp.py --task Isaac-Velocity-Flat-Ostrich-v0 --motion_file ../../data/ostrich_walk_amp.npz --amp_weight 0.5
   ```

### 10.3 Legacy sketch (AMP state only)

Original sketch for `mdp/amp_observations.py`:

```python
"""AMP observation: state vector for the discriminator (must match reference .npz format)."""
from __future__ import annotations
import torch
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def amp_state_ostrich(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """AMP state for Ostrich: [root_height, proj_gravity(3), joint_pos(6), root_lin_vel(3), root_ang_vel(3), joint_vel(6)] = 22 dims."""
    asset = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2:3]                           # (N, 1)
    g = asset.data.projected_gravity_b                          # (N, 3)
    q = asset.data.joint_pos                                    # (N, 6)
    v_lin = asset.data.root_lin_vel_w                           # (N, 3)
    v_ang = asset.data.root_ang_vel_w                            # (N, 3)
    qd = asset.data.joint_vel                                    # (N, 6)
    return torch.cat([h, g, q, v_lin, v_ang, qd], dim=1)       # (N, 22)
```

Register this in your mdp `__init__.py` and use it in an observation term.

### 10.2 Add AMP observation group in env config

In `flat_env_cfg.py`, in the same place where you have `ObservationsCfg` / `PolicyCfg`, add an AMP group (if your base config allows extra groups). Example:

```python
# In the class that defines observations (e.g. inherit and override observations):
from . import mdp as mdp_local

# Add to observations:
# amp = ObsGroup(terms={"amp_state": ObsTerm(func=mdp_local.amp_state_ostrich)}, concatenate_terms=True)
```

And ensure the env’s observation space includes this group for the discriminator to read. (If using a custom training script, you can read the AMP observation from `obs["amp"]` or the same key you use.)

### 10.3 Record reference motion to .npz

Minimal script idea (run with Isaac Sim / your env):

```python
# scripts/record_amp_motion.py
import numpy as np
import gymnasium as gym
import bascat_rl.tasks  # noqa: F401

# Create env (e.g. Isaac-Velocity-Flat-Ostrich-v0), then:
# env = gym.make("Isaac-Velocity-Flat-Ostrich-v0", ...)
# For each step: action = your_controller(obs)  # e.g. scripted walk
# obs, reward, done, info = env.step(action)
# amp_obs = env.get_obs_from_group("amp")  # or however you expose AMP obs
# buffer.append(amp_obs[0].cpu().numpy())  # one env
# When done or T steps: np.savez("ostrich_walk.npz", amp_states=np.array(buffer))
```

Use the same `amp_state_ostrich` logic when recording so the .npz format matches exactly.

### 10.4 Discriminator and reward (pseudo-code)

```python
# Discriminator: MLP 22 -> 256 -> 256 -> 1
# AMP reward per step: r_amp = -log(clamp(1 - D(amp_obs), 1e-6, 1))
# Add to env as RewTerm with weight ~0.5–1.0, or add in runner from buffer.
# Train D on mixed batch: real = sample from .npz, fake = AMP obs from rollout; BCE loss.
```

This gives you everything you need to implement AMP for better gait movement on the Ostrich.
