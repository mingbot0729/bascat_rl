"""Configuration for Ostrich biped robot.

Follows Tron pointfoot_cfg.py pattern: simple articulation config with
actuator definitions. Uses DelayedPDActuator with backlash for sim2real.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from .actuators import StatefulBacklashPDActuatorCfg

_ASSETS_DIR = Path(__file__).resolve().parent
OSTRICH_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(_ASSETS_DIR / "dummy_robot" / "urdf" / "dummy_robot" / "dummy_robot.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        activate_contact_sensors=True,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.24),
        joint_pos={
            "left_roll_joint": 0.0,
            "left_upper_leg_joint": 0.0,
            "left_lower_leg_joint": 0.0,
            "right_roll_joint": 0.0,
            "right_upper_leg_joint": 0.0,
            "right_lower_leg_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": StatefulBacklashPDActuatorCfg(
            joint_names_expr=[".*_joint"],
            effort_limit=7.7,
            velocity_limit=22.0,
            stiffness={
                ".*_roll_joint": 15.0,
                ".*_upper_leg_joint": 15.0,
                ".*_lower_leg_joint": 15.0,
            },
            damping={
                ".*_roll_joint": 0.5,
                ".*_upper_leg_joint": 0.5,
                ".*_lower_leg_joint": 0.5,
            },
            armature={
                ".*_roll_joint": 0.0017,
                ".*_upper_leg_joint": 0.0017,
                ".*_lower_leg_joint": 0.0017,
            },
            friction={
                ".*_roll_joint": 0.09,
                ".*_upper_leg_joint": 0.09,
                ".*_lower_leg_joint": 0.09,
            },
            # Delay is in physics steps (see isaaclab.actuators.DelayedPDActuator).
            # With sim dt=0.005s, 14ms ≈ 2.8 steps -> use 3 steps (~15ms).
            min_delay=2,
            max_delay=4,
            backlash=0.05,
        ),
    },
)