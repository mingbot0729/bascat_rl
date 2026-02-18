1. Main Architecture Change
Old: Inherited from Isaac Lab's LocomotionVelocityRoughEnvCfg, which came with ~20 parent reward terms, parent events with wrong parameter names (friction_range), body names for quadrupeds (.*FOOT, .*THIGH), and lots of self.events.X = None overrides to disable things. Every parent update could break your code.
New: Self-contained OstrichEnvCfg inherits directly from ManagerBasedRLEnvCfg (the raw base class). Everything is defined explicitly - no hidden inherited terms. Like the Tron example.
2. Observations (Policy Input)
Observation	Dimensions	Noise
base_ang_vel	3	Uniform (-0.2, 0.2)
proj_gravity	3	Uniform (-0.05, 0.05)
joint_pos_rel	6	Uniform (-0.01, 0.01)
joint_vel_rel	6	Uniform (-1.5, 1.5)
last_action	6	None
vel_command	3	None
Total	27	
What changed:
Removed base_lin_vel (3 dims) -- the policy no longer gets privileged velocity info (more realistic for sim2real, since you don't have a perfect velocity sensor). The Tron example does the same.
Removed height_scan -- not needed for flat terrain (blind locomotion).
Old code had ~45+ observations (inherited base_lin_vel, height_scan, etc). Now it's a clean 27.
3. Actions (Policy Output)
Output	Dimensions
Joint position targets for 6 joints	6
scale=0.25 (old was 0.5) -- more conservative, matching Tron. The policy outputs a value, multiplied by 0.25, added to the default joint position.
No clip overrides (old had per-joint clip ranges) -- joints are naturally limited by URDF limits + soft_joint_pos_limit_factor.
4. Rewards (14 old -> 12 new)
Reward	Weight	Purpose	Old equivalent
rew_lin_vel_xy	+1.5	Track commanded XY velocity	track_lin_vel_xy_exp (was +4.0)
rew_ang_vel_z	+0.75	Track commanded yaw velocity	track_ang_vel_z_exp (was +4.0)
rew_no_fly	+1.0	Single-stance biped gait	NEW - encourages one foot on ground
pen_undesired_contacts	-0.5	Penalize body/thigh ground contact	similar but body names fixed
pen_lin_vel_z	-0.5	Penalize vertical bouncing	lin_vel_z_l2 (was absent)
pen_ang_vel_xy	-0.05	Penalize roll/pitch angular velocity	similar
pen_action_rate	-0.01	Smooth actions	similar
pen_flat_orientation	-1.0	Keep body level	was -0.01
pen_joint_vel	-5e-5	Penalize fast joints	NEW
pen_joint_accel	-2.5e-7	Penalize joint acceleration	similar
pen_joint_powers	-2e-5	Penalize torque * velocity	NEW (energy efficiency)
pen_base_height	-1.0	Keep CoM at 0.22m	was exp reward, now L2 penalty
pen_joint_torque	-2e-5	Penalize torque magnitude	similar
pen_joint_pos_limits	-1.0	Penalize hitting joint limits	similar
Removed from old:
contact_switch_penalty, foot_contact_force_penalty, foot_slip_penalty, feet_air_time, termination_penalty, joint_deviation_roll, joint_deviation_legs, base_height_target_exp, base_pos_xy_drift_l2 and all the commented-out/disabled terms.
Key difference: The old code had tracking weights at +4.0 (very dominant), with most penalties tiny. The new code has more balanced weights matching Tron's proven ratios.
5. Domain Randomization
Event	Old	New
Base mass	(-0.05, 0.2) add	(-0.05, 0.2) add -- same
Friction	Multiple separate events	Single: static (0.4, 1.2), dynamic (0.7, 0.9)
Actuator gains	scale (0.9, 1.1)	scale (0.9, 1.1) -- same
Reset base pose	Tiny ranges (0.0, 0.0)	Wide: (-0.5, 0.5) position, (-0.5, 0.5) velocity
Reset joints	(-0.1, 0.1)	(-0.5, 0.5) -- much wider
Push robot	vel (-0.2, 0.2) every 5-10s	vel (-0.3, 0.3) every 5-10s
Key difference: Much wider reset randomization. Old code reset the robot nearly upright with zero velocity. New code spawns with random position, orientation, and velocity -- forces the policy to learn recovery from any state.
6. Simulation Parameters
Parameter	Old	New
decimation	2	4
sim.dt	0.005	0.005
Control frequency	100 Hz	50 Hz
Episode length	50s	20s
Action scale	0.5	0.25
Standing envs	100% (zero cmd)	20%
Key difference: Old code was training standing only (zero velocity commands to all envs). New code commands actual walking: lin_vel_x=(-1.0, 1.0), with 20% standing. This is the Tron approach -- train walking from the start.
7. PPO Training
Parameter	Old	New (Tron)
Network	[256, 256, 128]	[512, 256, 128]
Learning rate	3e-4	1e-3
Entropy coef	0.005	0.01
Steps per env	32	24
Mini batches	8	4
Max iterations	15000	3001
Gradient clip	0.5	1.0
Init noise	0.8	1.0
Gamma	0.995	0.99
Key difference: Larger network, higher learning rate, more exploration (higher entropy + noise), fewer but faster iterations. The Tron settings are more aggressive but converge faster.
8. How to Tune
If robot falls immediately: Reduce reset_robot_base velocity range (e.g. to 0.2), and reduce command velocity ranges.
If robot doesn't walk: Increase rew_lin_vel_xy weight (e.g. 2.0 or 3.0).
If robot walks but wobbly: Increase pen_flat_orientation (e.g. -2.0), increase pen_ang_vel_xy (e.g. -0.1).
If robot crouches: Increase pen_base_height weight (e.g. -2.0) or adjust target_height.
If jerky motions: Increase pen_action_rate (e.g. -0.05) or pen_joint_vel.
If want standing first: Change command ranges to zero: lin_vel_x=(0,0), lin_vel_y=(0,0), ang_vel_z=(0,0) and rel_standing_envs=1.0.