# Copyright (c) 2022-2025, The Isaac Lab Project Developers.

# SPDX-License-Identifier: BSD-3-Clause



"""Custom actuator model with backlash (deadzone) support for sim2real transfer."""



from __future__ import annotations



import torch

from typing import Sequence



from isaaclab.actuators import DelayedPDActuatorCfg

from isaaclab.actuators.actuator_pd import DelayedPDActuator

from isaaclab.utils import configclass

from isaaclab.utils.types import ArticulationActions





class DelayedPDActuatorWithBacklash(DelayedPDActuator):

    """Delayed PD actuator with backlash (deadzone) on the position error.



    When the motor reverses direction, there is a small angular gap (backlash)

    where no torque is transmitted. This is modeled as a deadzone on the

    position error:



        if |error| < backlash/2:

            effective_error = 0

        else:

            effective_error = error - sign(error) * backlash/2



    This makes the PD controller "blind" to small position errors, simulating

    the mechanical play in real geared motors.

    """



    cfg: "DelayedPDActuatorWithBacklashCfg"



    def __init__(self, cfg: "DelayedPDActuatorWithBacklashCfg", *args, **kwargs):

        super().__init__(cfg, *args, **kwargs)

        # Parse backlash parameter (scalar or per-joint dict)

        backlash = cfg.backlash

        if isinstance(backlash, (int, float)):

            self._half_backlash = float(backlash) / 2.0

        else:

            # backlash is a dict like {".*_roll_joint": 0.02, ...}

            # Use largest value as a scalar fallback

            vals = list(backlash.values()) if isinstance(backlash, dict) else [0.0]

            self._half_backlash = max(vals) / 2.0



    def compute(

        self,

        control_action: ArticulationActions,

        joint_pos: torch.Tensor,

        joint_vel: torch.Tensor,

    ) -> ArticulationActions:

        # Apply delay (from DelayedPDActuator)

        control_action.joint_positions = self.positions_delay_buffer.compute(

            control_action.joint_positions  # type: ignore[arg-type]

        )

        control_action.joint_velocities = self.velocities_delay_buffer.compute(

            control_action.joint_velocities  # type: ignore[arg-type]

        )

        control_action.joint_efforts = self.efforts_delay_buffer.compute(

            control_action.joint_efforts  # type: ignore[arg-type]

        )



        # Compute position error with backlash deadzone

        error_pos = control_action.joint_positions - joint_pos

        if self._half_backlash > 0.0:

            # Deadzone: shrink error toward zero by half_backlash, clamp at zero

            sign = torch.sign(error_pos)

            abs_error = torch.abs(error_pos)

            effective_error = torch.clamp(abs_error - self._half_backlash, min=0.0) * sign

        else:

            effective_error = error_pos



        error_vel = control_action.joint_velocities - joint_vel



        # PD torque computation (same as IdealPDActuator)

        self.computed_effort = (

            self.stiffness * effective_error + self.damping * error_vel + control_action.joint_efforts

        )

        self.applied_effort = self._clip_effort(self.computed_effort)



        # Set computed actions

        control_action.joint_efforts = self.applied_effort

        control_action.joint_positions = None

        control_action.joint_velocities = None

        return control_action





@configclass

class DelayedPDActuatorWithBacklashCfg(DelayedPDActuatorCfg):

    """Configuration for a delayed PD actuator with backlash (deadzone).



    Backlash simulates gear play: when the position error is smaller than

    half the backlash angle, the effective error is reduced to zero.

    This creates a deadzone around the target position where no corrective

    torque is produced, mimicking real geared motor behavior.



    BACKLASH PARAMETER:

    -------------------

    - backlash: Total backlash angle [rad]. The deadzone is +/-backlash/2 around

      the target position. Typical values for small hobby servos: 0.01-0.05 rad

      (~0.6 deg - 3 deg). For high-quality gears: 0.005-0.01 rad.

    """



    class_type: type = DelayedPDActuatorWithBacklash

    backlash: float | dict[str, float] = 0.0

    """Total backlash angle in radians. Default 0 = no backlash."""





class StatefulBacklashPDActuator(DelayedPDActuator):

    """Delayed PD actuator with *stateful* mechanical backlash (hysteresis).



    This simulates true gear play:

    - When the motor reverses direction, it must traverse the gear gap before

      torque transmission re-engages.

    - While the gear is within the slack region, we output **zero torque**

      (both stiffness and damping effectively disconnected).



    Implementation details:

    - We track an internal slack state `s` per joint (in radians), clamped to

      [-backlash/2, +backlash/2].

    - The motor-side (rotor) position is modeled as: motor_pos = joint_pos + s.

    - When `s` is at a limit and commanded effort pushes further into that limit,

      the gear is engaged and torque is transmitted; otherwise torque is zero.

    """



    cfg: "StatefulBacklashPDActuatorCfg"



    def __init__(self, cfg: "StatefulBacklashPDActuatorCfg", *args, **kwargs):

        super().__init__(cfg, *args, **kwargs)



        backlash = cfg.backlash

        if isinstance(backlash, (int, float)):

            self._half_backlash = float(backlash) / 2.0

        else:

            vals = list(backlash.values()) if isinstance(backlash, dict) else [0.0]

            self._half_backlash = max(vals) / 2.0



        # Stateful buffers (allocated on first compute)

        self._slack_state: torch.Tensor | None = None

        # Use NaN to mark "needs init" rows after reset

        self._prev_target_pos: torch.Tensor | None = None

        self._prev_joint_pos: torch.Tensor | None = None



    def reset(self, env_ids: Sequence[int] | None = None):

        super().reset(env_ids)  # type: ignore[arg-type]

        # Mark requested env rows as needing re-init on next compute

        if self._slack_state is None:

            return

        if env_ids is None:

            self._slack_state.zero_()

            if self._prev_target_pos is not None:

                self._prev_target_pos.fill_(float("nan"))

            if self._prev_joint_pos is not None:

                self._prev_joint_pos.fill_(float("nan"))

        else:

            self._slack_state[env_ids] = 0.0

            if self._prev_target_pos is not None:

                self._prev_target_pos[env_ids] = float("nan")

            if self._prev_joint_pos is not None:

                self._prev_joint_pos[env_ids] = float("nan")



    def compute(

        self,

        control_action: ArticulationActions,

        joint_pos: torch.Tensor,

        joint_vel: torch.Tensor,

    ) -> ArticulationActions:

        # 1) Apply delay buffers (from DelayedPDActuator)

        control_action.joint_positions = self.positions_delay_buffer.compute(

            control_action.joint_positions  # type: ignore[arg-type]

        )

        control_action.joint_velocities = self.velocities_delay_buffer.compute(

            control_action.joint_velocities  # type: ignore[arg-type]

        )

        control_action.joint_efforts = self.efforts_delay_buffer.compute(

            control_action.joint_efforts  # type: ignore[arg-type]

        )



        target_pos = control_action.joint_positions



        # 2) Allocate state buffers on first call

        if self._slack_state is None:

            self._slack_state = torch.zeros_like(joint_pos)

            self._prev_target_pos = torch.full_like(joint_pos, float("nan"))

            self._prev_joint_pos = torch.full_like(joint_pos, float("nan"))



        assert self._prev_target_pos is not None

        assert self._prev_joint_pos is not None



        # 3) Initialize rows that were reset / never initialized

        needs_init = torch.isnan(self._prev_target_pos)

        if needs_init.any():

            self._prev_target_pos[needs_init] = target_pos[needs_init]

            self._prev_joint_pos[needs_init] = joint_pos[needs_init]

            self._slack_state[needs_init] = 0.0



        # 4) Update internal slack based on relative motor-vs-joint motion

        delta_target = target_pos - self._prev_target_pos

        delta_joint = joint_pos - self._prev_joint_pos

        raw_slack = self._slack_state + (delta_target - delta_joint)

        self._slack_state = torch.clamp(raw_slack, min=-self._half_backlash, max=self._half_backlash)



        # Update history for next frame (in-place to avoid extra allocations)

        self._prev_target_pos.copy_(target_pos)

        self._prev_joint_pos.copy_(joint_pos)



        # 5) Standard PD effort on the motor-side position (joint_pos + slack)

        motor_pos = joint_pos + self._slack_state

        error_pos = target_pos - motor_pos

        error_vel = control_action.joint_velocities - joint_vel

        computed_effort = self.stiffness * error_pos + self.damping * error_vel + control_action.joint_efforts



        # 6) Engagement logic: transmit torque only when slack is at a limit and pushing into that limit

        eps = 1e-6

        engaged_forward = (self._slack_state >= self._half_backlash - eps) & (computed_effort > 0)

        engaged_backward = (self._slack_state <= -self._half_backlash + eps) & (computed_effort < 0)

        is_engaged = engaged_forward | engaged_backward



        # 7) Disconnect when not engaged

        self.computed_effort = torch.where(is_engaged, computed_effort, torch.zeros_like(computed_effort))

        self.applied_effort = self._clip_effort(self.computed_effort)



        # 8) Set output action (efforts only)

        control_action.joint_efforts = self.applied_effort

        control_action.joint_positions = None

        control_action.joint_velocities = None

        return control_action





@configclass

class StatefulBacklashPDActuatorCfg(DelayedPDActuatorCfg):

    """Configuration for `StatefulBacklashPDActuator`."""



    class_type: type = StatefulBacklashPDActuator

    backlash: float | dict[str, float] = 0.0