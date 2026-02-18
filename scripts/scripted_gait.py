# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Scripted point-foot gait generator for 2 legs, 3 DOF each. Drop-in for record_amp_motion."""

import numpy as np


class ScriptedPointFootGait:
    """
    Joint-space scripted gait for 2 legs, 3DOF each.
    Outputs q_des (6,) in order:
    [L_roll, L_hip, L_knee, R_roll, R_hip, R_knee]
    i.e. [left_roll, left_upper_leg, left_lower_leg, right_roll, right_upper_leg, right_lower_leg]
    """
    def __init__(self, dt: float):
        self.dt = float(dt)
        self.phase = 0.0

        # Step frequency (Hz). Start low.
        self.freq = 1.2
        self.omega = 2.0 * np.pi * self.freq

        # ---- Standing pose offsets (rad) ----
        # Point-foot needs some knee bend for stability.
        self.q0 = np.array([
            0.00,  # L roll
            0.10,  # L hip (upper_leg)
            -0.65, # L knee (lower_leg)
            0.00,  # R roll
            0.10,  # R hip
            -0.65, # R knee
        ], dtype=np.float32)

        # ---- Amplitudes (rad) ----
        self.A_roll = 0.06
        self.A_hip  = 0.25
        self.A_knee = 0.25

        # Extra knee bend during swing (rad)
        self.knee_swing_boost = 0.25

        # Optional: slight forward lean / bias (rad)
        self.hip_bias = 0.05

    def step(self) -> np.ndarray:
        self.phase += self.omega * self.dt
        phL = self.phase
        phR = self.phase + np.pi

        # Hip swing (sin)
        hipL = self.hip_bias + self.A_hip * np.sin(phL)
        hipR = self.hip_bias + self.A_hip * np.sin(phR)

        # Knee base follows hip (cos-shifted: bend when leg lifts)
        kneeL = -0.65 + self.A_knee * np.cos(phL)
        kneeR = -0.65 + self.A_knee * np.cos(phR)

        # Swing: add extra bend for foot clearance
        swingL = max(0.0, np.sin(phL))
        swingR = max(0.0, np.sin(phR))
        kneeL -= self.knee_swing_boost * swingL
        kneeR -= self.knee_swing_boost * swingR

        # Small roll for weight shifting (optional)
        rollL =  self.A_roll * np.sin(phL + np.pi/2)
        rollR = -self.A_roll * np.sin(phL + np.pi/2)

        q_des = np.array([rollL, hipL, kneeL, rollR, hipR, kneeR], dtype=np.float32)
        q_des = q_des + self.q0
        return q_des


def rad_to_normalized_action(q_des: np.ndarray, scale: float = 0.5, default_pos: np.ndarray | None = None) -> np.ndarray:
    """
    Convert desired joint positions (rad) to env action in [-1, 1].
    With JointPositionActionCfg(scale=0.5, use_default_offset=True):
        pos_target = default_pos + scale * action  =>  action = (q_des - default_pos) / scale
    """
    if default_pos is None:
        default_pos = np.zeros_like(q_des, dtype=np.float32)
    action = (q_des - default_pos) / scale
    return np.clip(action, -1.0, 1.0).astype(np.float32)
