# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
Simple Joint Test Script - Easy to use for comparing sim vs real robot

This script runs your robot through various joint tests and generates plots.
You can then compare these plots with data from your real robot.

Usage Examples:
    # Step response test on all joints
    python scripts/joint_test_simple.py
    
    # Sine wave test 
    python scripts/joint_test_simple.py --test sine
    
    # Test specific joint only
    python scripts/joint_test_simple.py --joint left_upper_leg_joint
    
    # Custom parameters
    python scripts/joint_test_simple.py --test step --target 0.5 --duration 3
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Simple Joint Test")
parser.add_argument("--test", type=str, default="step", choices=["step", "sine", "ramp", "chirp"],
                    help="Test type: step, sine, ramp, or chirp (frequency sweep)")
parser.add_argument("--joint", type=str, default=None, help="Test specific joint only (RECOMMENDED)")
parser.add_argument("--target", type=float, default=0.3, help="Target position (rad)")
parser.add_argument("--frequency", type=float, default=0.5, help="Sine frequency (Hz)")
parser.add_argument("--duration", type=float, default=5.0, help="Test duration (s)")
parser.add_argument("--no_gravity", action="store_true", help="Disable gravity for pure actuator test (RECOMMENDED)")
parser.add_argument("--save", action="store_true", help="Save plots and data")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

# Import robot config
import sys
sys.path.insert(0, r"C:\Users\dota2\OneDrive - National University of Singapore\Desktop\bascat_rl\bascat_rl\source\bascat_rl\bascat_rl\tasks\manager_based\bascat_rl")
from ostrich import OSTRICH_CFG


def run_test():
    """Run joint characterization test."""
    
    # Setup simulation
    sim_dt = 0.001  # 1ms physics timestep for high-resolution sampling
    decimation = 1  # No decimation - sample every physics step
    control_dt = sim_dt * decimation  # 1ms control/sampling rate
    
    sim = SimulationContext(sim_utils.SimulationCfg(dt=sim_dt, render_interval=decimation))
    sim.set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.3])
    
    # Ground
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    
    # Robot configuration
    robot_cfg = OSTRICH_CFG.copy()
    robot_cfg.prim_path = "/World/Robot"
    
    # Option: Disable gravity for pure actuator testing
    if args.no_gravity:
        print("\n[GRAVITY DISABLED] - Testing pure actuator response")
        robot_cfg.spawn.rigid_props.disable_gravity = True
    
    robot = Articulation(robot_cfg)
    
    sim.reset()
    
    # Initialize robot
    robot.reset()
    
    # Properly initialize the robot
    print("\nInitializing robot (settling for 1 second)...")
    zero_cmd = torch.zeros(1, len(robot.joint_names), device=robot.device)
    
    for i in range(int(1.0 / sim_dt)):
        # For DelayedPDActuatorCfg: set target, write, step, update (in this order)
        robot.set_joint_position_target(zero_cmd)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)
        
    sim.render()
    print("Robot initialized. Starting test...")
    
    # Get joint info
    joint_names = robot.joint_names
    num_joints = len(joint_names)
    joint_limits = robot.root_physx_view.get_dof_limits()[0].cpu().numpy()
    
    print("\n" + "="*70)
    print("ROBOT JOINT INFORMATION")
    print("="*70)
    for i, name in enumerate(joint_names):
        print(f"  [{i}] {name}: limits = ({joint_limits[i,0]:.3f}, {joint_limits[i,1]:.3f}) rad")
    print("="*70)
    
    # Find joint index if specified
    test_joints = list(range(num_joints))
    if args.joint:
        if args.joint in joint_names:
            test_joints = [joint_names.index(args.joint)]
            print(f"\nTesting only: {args.joint}")
        else:
            print(f"\nWarning: Joint '{args.joint}' not found. Testing all joints.")
    else:
        # Warn user about testing all joints
        if not args.no_gravity:
            print("\n" + "!"*70)
            print("WARNING: Testing ALL joints without --no_gravity")
            print("         Robot may fall/oscillate. For clean results, use:")
            print("         --joint <joint_name>  (test one joint)")
            print("         --no_gravity          (disable gravity)")
            print("!"*70)
    
    # Data storage
    time_data = []
    cmd_data = []
    pos_data = []
    vel_data = []
    
    # Generate test signal
    def get_command(t, joint_idx):
        """Generate command for given time and joint."""
        # Default: zero
        if joint_idx not in test_joints:
            return 0.0
            
        target = args.target
        freq = args.frequency
        
        if args.test == "step":
            # Step after 0.5 seconds
            return target if t > 0.5 else 0.0
            
        elif args.test == "sine":
            return target * np.sin(2 * np.pi * freq * t)
            
        elif args.test == "ramp":
            # Ramp up, hold, ramp down
            ramp_t = args.duration / 4
            if t < ramp_t:
                return target * t / ramp_t
            elif t < 3 * ramp_t:
                return target
            else:
                return target * (args.duration - t) / ramp_t
                
        elif args.test == "chirp":
            # Frequency sweep from 0.1 to 2 Hz
            f0, f1 = 0.1, 2.0
            freq_t = f0 + (f1 - f0) * t / args.duration
            return target * np.sin(2 * np.pi * freq_t * t)
            
        return 0.0
    
    # Run simulation
    num_steps = int(args.duration / control_dt)
    print(f"\nRunning {args.test} test for {args.duration}s ({num_steps} steps)...")
    print("Watch the simulation window - robot should be moving!")
    print("-" * 70)
    
    t = 0.0
    for step in range(num_steps):
        # Generate commands
        cmd = torch.zeros(1, num_joints, device=robot.device)
        for j in range(num_joints):
            cmd[0, j] = get_command(t, j)
        
        # Step physics (multiple times for decimation)
        # For DelayedPDActuatorCfg: must update state, set target, write, step EACH physics step
        for _ in range(decimation):
            # 1. Set position target
            robot.set_joint_position_target(cmd)
            # 2. Write actuator commands (computes torque from current state + target)
            robot.write_data_to_sim()
            # 3. Step physics
            sim.step()
            # 4. Update robot state from simulation
            robot.update(sim_dt)
            
        # Render to see the robot
        sim.render()
        
        # Record AFTER updating state
        time_data.append(t)
        cmd_data.append(cmd[0].cpu().numpy().copy())
        pos_data.append(robot.data.joint_pos[0].cpu().numpy().copy())
        vel_data.append(robot.data.joint_vel[0].cpu().numpy().copy())
        
        # Print live status every 25 steps (~0.5 seconds)
        if step % 25 == 0:
            pos = robot.data.joint_pos[0].cpu().numpy()
            print(f"  t={t:.2f}s | {joint_names[0]}: cmd={cmd[0,0]:.3f}, pos={pos[0]:.3f} | "
                  f"{joint_names[2]}: cmd={cmd[0,2]:.3f}, pos={pos[2]:.3f}")
        
        t += control_dt
    
    print("Test complete!\n")
    
    # Convert to arrays
    time_data = np.array(time_data)
    cmd_data = np.array(cmd_data)
    pos_data = np.array(pos_data)
    vel_data = np.array(vel_data)
    
    # Print statistics
    print("="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    for i in test_joints:
        name = joint_names[i]
        error = cmd_data[:, i] - pos_data[:, i]
        
        # Find rise time (10% to 90% of step)
        if args.test == "step":
            final_val = cmd_data[-1, i]
            if abs(final_val) > 0.01:
                try:
                    t10 = time_data[np.where(pos_data[:, i] >= 0.1 * final_val)[0][0]]
                    t90 = time_data[np.where(pos_data[:, i] >= 0.9 * final_val)[0][0]]
                    rise_time = t90 - t10
                except:
                    rise_time = float('nan')
            else:
                rise_time = float('nan')
        else:
            rise_time = float('nan')
            
        print(f"\n{name}:")
        print(f"  Final position: {pos_data[-1, i]:.4f} rad (target: {cmd_data[-1, i]:.4f})")
        print(f"  Max velocity:   {np.abs(vel_data[:, i]).max():.4f} rad/s")
        print(f"  Tracking error: mean={np.abs(error).mean():.4f}, max={np.abs(error).max():.4f} rad")
        if not np.isnan(rise_time):
            print(f"  Rise time (10-90%): {rise_time*1000:.1f} ms")
    print("="*70)
    
    # Plot results
    plot_joints = test_joints if len(test_joints) <= 3 else test_joints[:3]
    n_plots = len(plot_joints)
    
    fig, axes = plt.subplots(n_plots, 3, figsize=(15, 4 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'{args.test.upper()} Test Results - {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                 fontsize=14, fontweight='bold')
    
    for plot_i, joint_i in enumerate(plot_joints):
        name = joint_names[joint_i]
        
        # Position
        ax1 = axes[plot_i, 0]
        ax1.plot(time_data, cmd_data[:, joint_i], 'b--', label='Command', lw=2)
        ax1.plot(time_data, pos_data[:, joint_i], 'r-', label='Actual', lw=1.5)
        ax1.fill_between(time_data, cmd_data[:, joint_i], pos_data[:, joint_i], 
                        alpha=0.2, color='orange', label='Error')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (rad)')
        ax1.set_title(f'{name} - Position Tracking')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Velocity
        ax2 = axes[plot_i, 1]
        ax2.plot(time_data, vel_data[:, joint_i], 'g-', lw=1.5)
        ax2.axhline(y=0, color='k', linestyle='-', lw=0.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (rad/s)')
        ax2.set_title(f'{name} - Velocity')
        ax2.grid(True, alpha=0.3)
        
        # Phase plot (position vs velocity)
        ax3 = axes[plot_i, 2]
        scatter = ax3.scatter(pos_data[:, joint_i], vel_data[:, joint_i], 
                             c=time_data, cmap='viridis', s=2, alpha=0.7)
        ax3.axhline(y=0, color='k', linestyle='-', lw=0.5)
        ax3.axvline(x=0, color='k', linestyle='-', lw=0.5)
        ax3.set_xlabel('Position (rad)')
        ax3.set_ylabel('Velocity (rad/s)')
        ax3.set_title(f'{name} - Phase Plot')
        plt.colorbar(scatter, ax=ax3, label='Time (s)')
        ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if args.save:
        out_dir = os.path.join(os.path.dirname(__file__), "joint_test_results")
        os.makedirs(out_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        joint_str = args.joint if args.joint else "all"
        
        # Save plot
        plot_path = os.path.join(out_dir, f"{args.test}_{joint_str}_{timestamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved: {plot_path}")
        
        # Save data as CSV
        csv_path = os.path.join(out_dir, f"{args.test}_{joint_str}_{timestamp}.csv")
        header = "time," + ",".join([f"{n}_cmd,{n}_pos,{n}_vel" for n in joint_names])
        data = np.column_stack([time_data] + 
                               [np.column_stack([cmd_data[:,i], pos_data[:,i], vel_data[:,i]]) 
                                for i in range(num_joints)])
        np.savetxt(csv_path, data, delimiter=',', header=header, comments='')
        print(f"Data saved: {csv_path}")
    else:
        plt.show()
    
    simulation_app.close()


if __name__ == "__main__":
    run_test()
