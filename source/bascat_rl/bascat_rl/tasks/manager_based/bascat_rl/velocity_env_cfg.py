"""Base environment configuration for Ostrich biped locomotion.

Follows the Tron PFEnvCfg pattern: self-contained scene, MDP, observations,
events, rewards, terminations, curriculum - all in one file.
No inheritance from Isaac Lab's LocomotionVelocityRoughEnvCfg.
"""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from . import mdp as mdp_local


##############
# Scene
##############


@configclass
class OstrichSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the Ostrich biped robot."""

    # terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # sky light
    light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            color=(0.9, 0.9, 0.9),
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # robot (set in child config)
    robot: ArticulationCfg = MISSING

    # height scanner (set to None for blind, or RayCasterCfg for height scan)
    height_scanner: RayCasterCfg | None = None

    # contact sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=4,
        track_air_time=True,
        update_period=0.0,
    )


##############
# MDP
##############


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_roll_joint", "right_roll_joint",
            "left_upper_leg_joint", "right_upper_leg_joint",
            "left_lower_leg_joint", "right_lower_leg_joint",
        ],
        scale=0.25,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy. Per-step: 32 dims.

        Order: base_ang_vel(3), proj_gravity(3), vel_command(3),
               gait_phase(2), gait_command(3),
               joint_pos(6), joint_vel(6), last_action(6)

        With history_length=5 and flatten: 32 * 5 = 160 total input dims.
        The buffer stores the last 5 steps and flattens them into one vector:
        [obs_t-4, obs_t-3, obs_t-2, obs_t-1, obs_t]
        """

        # Robot base measurements
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        proj_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))

        # Velocity command (3D: lin_vel_x, lin_vel_y, ang_vel_z — no heading)
        vel_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        # Gait timing + parameters (metronome for the policy)
        gait_phase = ObsTerm(func=mdp_local.get_gait_phase)
        gait_command = ObsTerm(func=mdp_local.get_gait_command, params={"command_name": "gait_command"})

        # Robot joint measurements
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))

        # Last action
        last_action = ObsTerm(func=mdp.last_action)

        # NOTE: gait_phase and gait_command ARE in the policy observations (Option A).

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            # Stack 5 steps of history and flatten into one vector.
            self.history_length = 15
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.0, 5.0),
        rel_standing_envs=0.2,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=1.0,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 0.8),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.3, 0.3),
            heading=(0.0, 0.0),
        ),
    )

    gait_command = mdp_local.UniformGaitCommandCfg(
        resampling_time_range=(2.0, 4.0),
        debug_vis=False,
        ranges=mdp_local.UniformGaitCommandCfg.Ranges(
            frequencies=(2.5, 4.5),   # stepping frequency in Hz
            offsets=(0.5, 0.5),       # phase offset between L/R feet (0.5 = alternating)
            durations=(0.35, 0.5),     # fraction of cycle foot is on ground
        ),
    )


@configclass
class EventsCfg:
    """Event / domain randomization configuration."""

    # -- startup --
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "mass_distribution_params": (-0.05, 0.2),
            "operation": "add",
        },
    )
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.2, 0.8),
            "restitution_range": (0.0, 1.0),
            "num_buckets": 48,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- reset --
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "z": (-0.1, 0.1),
                "roll": (-0.1, 0.1),
                "pitch": (-0.1, 0.1),
                "yaw": (-0.1, 0.1),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (-0.2, 0.2),
            "velocity_range": (0.0, 0.0),
        },
    )

    # -- interval --
    push_robot = EventTerm(
        func=mdp_local.apply_external_force_torque_stochastic,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
            "force_range": {
                "x": (-10.0, 10.0),
                "y": (-10.0, 10.0),
                "z": (-10.0, 10.0),
            },
            "torque_range": {
                "x": (-2.0, 2.0),
                "y": (-2.0, 2.0),
                "z": (-2.0, 2.0),
            },
            "probability": 0.002,  # ~0.2% chance per step = random impulse every ~500 steps
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- tracking rewards --
    rew_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    rew_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    rew_no_fly = RewTerm(
        func=mdp_local.no_fly,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_lower_leg_link"),
            "threshold": 5.0,
        },
    )

    # -- penalties --
    pen_undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_roll_link", ".*_upper_leg_link", "base_link"],
            ),
            "threshold": 5.0,
        },
    )
    pen_lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.01)
    pen_ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    pen_action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    pen_flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.5)
    pen_joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-8.0e-05)
    pen_joint_accel = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-07)
    pen_joint_powers = RewTerm(func=mdp_local.joint_powers_l1, weight=-2.0e-05)
    pen_base_height = RewTerm(
        func=mdp_local.base_com_height,
        weight=-2.0,
        params={"target_height": 0.23},
    )
    pen_action_smoothness = RewTerm(func=mdp_local.ActionSmoothnessPenalty, weight=-0.2)  # type: ignore[arg-type]
    pen_joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-05)
    pen_joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)

    # -- gait reward --
    gait_reward = RewTerm(
        func=mdp_local.GaitReward,  # type: ignore[arg-type]
        weight=2.0,
        params={
            "tracking_contacts_shaped_force": -1.0,
            "tracking_contacts_shaped_vel": -1.0,
            "gait_force_sigma": 30.0,
            "gait_vel_sigma": 0.05,
            "kappa_gait_probs": 0.07,
            "command_name": "gait_command",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_lower_leg_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_lower_leg_link"),
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base_link"),
            "threshold": 20.0,
        },
    )


@configclass
class TerminationsRoughCfg(TerminationsCfg):
    """Terminations for rough terrain (adds terrain boundary check)."""

    terrain_out_of_bounds = DoneTerm(
        func=mdp_local.terrain_out_of_bounds,
        params={"distance_buffer": 3.0},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


########################
# Environment definition
########################


@configclass
class OstrichEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for the Ostrich locomotion environment."""

    # Scene settings
    scene: OstrichSceneCfg = OstrichSceneCfg(num_envs=2048, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 2
        self.episode_length_s = 60.0
        self.sim.render_interval = 1
        self.sim.dt = 0.005
        self.seed = 42
        # update sensor update periods
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
