# Sim2Sim: Isaac Lab → MuJoCo — Policy & Env Spec

Reference for porting the trained policy and matching environment timing in MuJoCo.

---

## Gait system: internal only (NOT in policy observations)

The gait command system (GaitReward) works **internally** during training to
enforce alternating left/right steps via the command manager and reward signal.
The policy does NOT see gait_phase or gait_command — no gait computation needed
at deployment.

---

## 1. Policy input (observations)

**Total observation dimension: 27**
Per-step layout (same as Isaac Lab `PolicyCfg`):

| Index | Name          | Dim | Description |
|-------|---------------|-----|-------------|
| 0–2   | base_ang_vel  | 3   | Base angular velocity in body frame (rad/s): `[wx, wy, wz]` |
| 3–5   | proj_gravity  | 3   | Gravity vector in base frame (unit): `[gx, gy, gz]` — upright ≈ (0, 0, -1) |
| 6–8   | vel_command   | 3   | Velocity command: `[lin_vel_x (m/s), lin_vel_y (m/s), ang_vel_z (rad/s)]`. No heading. |
| 9–14  | joint_pos     | 6   | Joint positions **relative to default** (rad). |
| 15–20 | joint_vel     | 6   | Joint velocities (rad/s). |
| 21–26 | last_action   | 6   | Previous step's action (same as policy output). |

**Joint order (same for joint_pos, joint_vel, action, last_action):**

1. `left_roll_joint`
2. `right_roll_joint`
3. `left_upper_leg_joint`
4. `right_upper_leg_joint`
5. `left_lower_leg_joint`
6. `right_lower_leg_joint`

---

## 2. Policy output (actions)

**Total action dimension: 6**

Actions are **target joint positions** (rad) for PD control:

- `scale = 0.25` → `q_target = default_pos + 0.25 * action`
- `use_default_offset = True`
- **Default joint positions (rad):** all zeros: `[0, 0, 0, 0, 0, 0]`

| Index | Joint name            | Default (rad) |
|-------|------------------------|---------------|
| 0     | left_roll_joint        | 0.0           |
| 1     | right_roll_joint        | 0.0           |
| 2     | left_upper_leg_joint   | 0.0           |
| 3     | right_upper_leg_joint  | 0.0           |
| 4     | left_lower_leg_joint   | 0.0           |
| 5     | right_lower_leg_joint  | 0.0           |

---

## 3. Environment timing (for MuJoCo)

| Setting              | Value   | Meaning |
|----------------------|--------|---------|
| **Physics (sim) dt** | 0.005 s | 200 Hz physics in Isaac Lab. |
| **Decimation**       | 4      | Policy runs every 4 physics steps. |
| **Control dt**       | 0.02 s | 4 × 0.005 = **20 ms** between policy steps → **50 Hz control**. |
| **Episode length**   | 30.0 s | Max episode duration. |
| **Max steps per episode** | 1500 | 30 / 0.02 = 1500 control steps. |

**For MuJoCo:**

- Set `model.opt.timestep = 0.005` (or 0.02 if you prefer one step per policy step).
- If you use 0.005 s: step physics 4 times per policy step (decimation = 4).
- If you use 0.02 s: step physics once per policy step (no decimation).

---

## 4. RSL-RL PPO network

From `OstrichFlatPPORunnerCfg`:

- **Actor/critic hidden dims:** `[512, 256, 128]`
- **Activation:** `"elu"`
- **Init noise std:** 1.0

Input size = 27, output size = 6.

---

## 5. Summary table

| Item              | Value |
|-------------------|--------|
| Obs dim           | 27     |
| Action dim        | 6      |
| Physics dt        | 5 ms   |
| Control dt        | 20 ms (50 Hz) |
| Decimation        | 4      |
| Action scale      | 0.25   |
| Default joint pos | All 0 rad |

How it works:
Isaac Lab maintains a ring buffer of the last 20 observation vectors (each 27-D)
Every policy step, the buffer is flattened into one vector: [obs_t-19, obs_t-18, ..., obs_t]
Total policy input: 27 × 20 = 540 dims
At episode start, the buffer is filled with copies of the first observation (so no garbage data)
For deployment (MuJoCo / real robot):
You need to maintain your own 20-step ring buffer of 27-D observations
Each policy step: push the new 27-D obs into the buffer, flatten all 20 into a 540-D vector, and feed that to the network
Note: The Tron example uses history_length=5. With 20, training will be slower (540-D input to MLP) but the policy gets more temporal context. If training is too slow or unstable, you can reduce to 10 or 5.