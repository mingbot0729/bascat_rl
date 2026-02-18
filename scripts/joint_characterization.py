# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
Joint Characterization Script for Robot USD Testing

This script allows you to:
1. Load your robot USD without any RL policy
2. Send joint position commands (step, sine, ramp)
3. Record joint positions, velocities, torques over time
4. Plot results to compare commanded vs actual behavior

Usage:
    python scripts/joint_characterization.py --test_type step
    python scripts/joint_characterization.py --test_type sine --frequency 1.0
    python scripts/joint_characterization.py --test_type ramp
    python scripts/joint_characterization.py --test_type manual --joint left_upper_leg_joint --target 0.5
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os

# Isaac Sim imports
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Joint Characterization for Robot USD")
parser.add_argument("--test_type", type=str, default="step", 
                    choices=["step", "sine", "ramp", "manual", "interactive"],
                    help="Type of test to run")
parser.add_argument("--joint", type=str, default=None,
                    help="Specific joint to test (None = all joints)")
parser.add_argument("--target", type=float, default=0.3,
                    help="Target position for step/manual tests (radians)")
parser.add_argument("--frequency", type=float, default=0.5,
                    help="Frequency for sine wave test (Hz)")
parser.add_argument("--amplitude", type=float, default=0.3,
                    help="Amplitude for sine wave test (radians)")
parser.add_argument("--duration", type=float, default=5.0,
                    help="Test duration in seconds")
parser.add_argument("--save_plot", action="store_true",
                    help="Save plots to file instead of showing")
parser.add_argument("--headless", action="store_true",
                    help="Run in headless mode (no visualization)")

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import Isaac Lab modules
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

# Import robot config
import sys
sys.path.insert(0, r"C:\Users\dota2\OneDrive - National University of Singapore\Desktop\bascat_rl\bascat_rl\source\bascat_rl\bascat_rl\tasks\manager_based\bascat_rl")
from ostrich import OSTRICH_CFG


class JointCharacterizer:
    """Class to characterize robot joint behavior."""
    
    def __init__(self, sim_dt: float = 0.005, decimation: int = 4):
        self.sim_dt = sim_dt
        self.decimation = decimation
        self.control_dt = sim_dt * decimation
        
        # Data storage
        self.time_data = []
        self.joint_pos_cmd = []  # Commanded positions
        self.joint_pos_actual = []  # Actual positions
        self.joint_vel_actual = []  # Actual velocities
        self.joint_torque_actual = []  # Applied torques
        
        # Joint names (will be populated after robot loads)
        self.joint_names = []
        
    def setup_scene(self):
        """Setup the simulation scene with the robot."""
        # Create simulation context
        sim_cfg = sim_utils.SimulationCfg(
            dt=self.sim_dt,
            render_interval=self.decimation,
        )
        self.sim = SimulationContext(sim_cfg)
        
        # Set camera view
        self.sim.set_camera_view(eye=[2.0, 2.0, 1.5], target=[0.0, 0.0, 0.3])
        
        # Create ground plane
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)
        
        # Create robot
        robot_cfg = OSTRICH_CFG.copy()
        robot_cfg.prim_path = "/World/Robot"
        self.robot = Articulation(robot_cfg)
        
        # Play sim to initialize
        self.sim.reset()
        
        # Get joint names
        self.joint_names = self.robot.joint_names
        self.num_joints = len(self.joint_names)
        
        print(f"\n{'='*60}")
        print("Robot Joint Information:")
        print(f"{'='*60}")
        for i, name in enumerate(self.joint_names):
            limits = self.robot.root_physx_view.get_dof_limits()[0, i].cpu().numpy()
            print(f"  [{i}] {name}: limits = ({limits[0]:.3f}, {limits[1]:.3f}) rad")
        print(f"{'='*60}\n")
        
        # Initialize robot state
        self.robot.reset()
        
    def generate_command(self, t: float, test_type: str, joint_idx: int = None, **kwargs) -> torch.Tensor:
        """Generate joint position command based on test type."""
        cmd = torch.zeros(1, self.num_joints, device=self.robot.device)
        
        target = kwargs.get("target", 0.3)
        frequency = kwargs.get("frequency", 0.5)
        amplitude = kwargs.get("amplitude", 0.3)
        
        if test_type == "step":
            # Step input after 1 second
            if t > 1.0:
                if joint_idx is not None:
                    cmd[0, joint_idx] = target
                else:
                    # Apply to all joints (alternating signs for left/right)
                    for i, name in enumerate(self.joint_names):
                        if "left" in name:
                            cmd[0, i] = target
                        else:
                            cmd[0, i] = -target if "roll" in name else target
                            
        elif test_type == "sine":
            # Sinusoidal input
            value = amplitude * np.sin(2 * np.pi * frequency * t)
            if joint_idx is not None:
                cmd[0, joint_idx] = value
            else:
                for i, name in enumerate(self.joint_names):
                    phase = 0 if "left" in name else np.pi  # Opposite phase for left/right
                    cmd[0, i] = amplitude * np.sin(2 * np.pi * frequency * t + phase)
                    
        elif test_type == "ramp":
            # Linear ramp up, hold, then back down
            duration = kwargs.get("duration", 5.0)
            ramp_time = duration / 4
            if t < ramp_time:
                value = target * (t / ramp_time)
            elif t < 3 * ramp_time:
                value = target
            else:
                value = target * (1 - (t - 3 * ramp_time) / ramp_time)
            value = max(0, min(target, value))
            
            if joint_idx is not None:
                cmd[0, joint_idx] = value
            else:
                for i, name in enumerate(self.joint_names):
                    sign = 1 if "left" in name else -1
                    cmd[0, i] = sign * value if "roll" in name else value
                    
        elif test_type == "manual":
            # Constant target position
            if joint_idx is not None:
                cmd[0, joint_idx] = target
            else:
                cmd[0, :] = target
                
        return cmd
    
    def run_test(self, test_type: str, duration: float, joint_name: str = None, **kwargs):
        """Run the characterization test."""
        # Find joint index if specific joint requested
        joint_idx = None
        if joint_name is not None:
            if joint_name in self.joint_names:
                joint_idx = self.joint_names.index(joint_name)
            else:
                print(f"Warning: Joint '{joint_name}' not found. Testing all joints.")
                print(f"Available joints: {self.joint_names}")
        
        # Clear data
        self.time_data = []
        self.joint_pos_cmd = []
        self.joint_pos_actual = []
        self.joint_vel_actual = []
        self.joint_torque_actual = []
        
        # Calculate number of steps
        num_steps = int(duration / self.control_dt)
        
        print(f"\nRunning {test_type} test for {duration}s ({num_steps} control steps)...")
        print(f"Control frequency: {1/self.control_dt:.1f} Hz")
        print(f"Physics frequency: {1/self.sim_dt:.1f} Hz")
        
        # Reset robot to initial state
        self.robot.reset()
        
        t = 0.0
        for step in range(num_steps):
            # Generate command
            cmd = self.generate_command(t, test_type, joint_idx, duration=duration, **kwargs)
            
            # Apply command
            self.robot.set_joint_position_target(cmd)
            
            # Step physics (multiple times for decimation)
            for _ in range(self.decimation):
                self.sim.step()
            
            # Update robot state
            self.robot.update(self.control_dt)
            
            # Record data
            self.time_data.append(t)
            self.joint_pos_cmd.append(cmd.cpu().numpy().flatten())
            self.joint_pos_actual.append(self.robot.data.joint_pos.cpu().numpy().flatten())
            self.joint_vel_actual.append(self.robot.data.joint_vel.cpu().numpy().flatten())
            
            # Get applied torques if available
            if hasattr(self.robot.data, 'applied_torque'):
                self.joint_torque_actual.append(self.robot.data.applied_torque.cpu().numpy().flatten())
            else:
                self.joint_torque_actual.append(np.zeros(self.num_joints))
            
            t += self.control_dt
            
            # Print progress
            if step % 50 == 0:
                print(f"  Step {step}/{num_steps} (t={t:.2f}s)")
        
        print("Test complete!")
        
        # Convert to numpy arrays
        self.time_data = np.array(self.time_data)
        self.joint_pos_cmd = np.array(self.joint_pos_cmd)
        self.joint_pos_actual = np.array(self.joint_pos_actual)
        self.joint_vel_actual = np.array(self.joint_vel_actual)
        self.joint_torque_actual = np.array(self.joint_torque_actual)
        
    def plot_results(self, save_path: str = None, joint_name: str = None):
        """Plot the test results."""
        # Determine which joints to plot
        if joint_name is not None and joint_name in self.joint_names:
            joint_indices = [self.joint_names.index(joint_name)]
        else:
            joint_indices = list(range(self.num_joints))
        
        num_joints_to_plot = len(joint_indices)
        
        # Create figure with subplots
        fig, axes = plt.subplots(num_joints_to_plot, 3, figsize=(15, 4 * num_joints_to_plot))
        if num_joints_to_plot == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Joint Characterization Results - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                     fontsize=14, fontweight='bold')
        
        for plot_idx, joint_idx in enumerate(joint_indices):
            joint_name = self.joint_names[joint_idx]
            
            # Position plot (commanded vs actual)
            ax1 = axes[plot_idx, 0]
            ax1.plot(self.time_data, self.joint_pos_cmd[:, joint_idx], 'b--', 
                     label='Commanded', linewidth=2)
            ax1.plot(self.time_data, self.joint_pos_actual[:, joint_idx], 'r-', 
                     label='Actual', linewidth=1.5)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Position (rad)')
            ax1.set_title(f'{joint_name} - Position')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Position error
            error = self.joint_pos_cmd[:, joint_idx] - self.joint_pos_actual[:, joint_idx]
            ax1_twin = ax1.twinx()
            ax1_twin.fill_between(self.time_data, error, alpha=0.2, color='green', label='Error')
            ax1_twin.set_ylabel('Error (rad)', color='green')
            ax1_twin.tick_params(axis='y', labelcolor='green')
            
            # Velocity plot
            ax2 = axes[plot_idx, 1]
            ax2.plot(self.time_data, self.joint_vel_actual[:, joint_idx], 'g-', linewidth=1.5)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Velocity (rad/s)')
            ax2.set_title(f'{joint_name} - Velocity')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
            
            # Torque plot (if available)
            ax3 = axes[plot_idx, 2]
            ax3.plot(self.time_data, self.joint_torque_actual[:, joint_idx], 'm-', linewidth=1.5)
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Torque (Nm)')
            ax3.set_title(f'{joint_name} - Torque')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: {save_path}")
        else:
            plt.show()
            
    def plot_comparison(self, save_path: str = None):
        """Plot all joints overlaid for comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Joint Comparison - All Joints Overlaid', fontsize=14, fontweight='bold')
        
        colors = plt.cm.tab10(np.linspace(0, 1, self.num_joints))
        
        # Position comparison
        ax1 = axes[0, 0]
        for i, name in enumerate(self.joint_names):
            ax1.plot(self.time_data, self.joint_pos_actual[:, i], color=colors[i], 
                     label=name, linewidth=1.5)
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Position (rad)')
        ax1.set_title('Joint Positions')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Velocity comparison
        ax2 = axes[0, 1]
        for i, name in enumerate(self.joint_names):
            ax2.plot(self.time_data, self.joint_vel_actual[:, i], color=colors[i], 
                     label=name, linewidth=1.5)
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (rad/s)')
        ax2.set_title('Joint Velocities')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # Tracking error comparison
        ax3 = axes[1, 0]
        for i, name in enumerate(self.joint_names):
            error = self.joint_pos_cmd[:, i] - self.joint_pos_actual[:, i]
            ax3.plot(self.time_data, error, color=colors[i], label=name, linewidth=1.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Position Error (rad)')
        ax3.set_title('Tracking Error (Cmd - Actual)')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        # Torque comparison
        ax4 = axes[1, 1]
        for i, name in enumerate(self.joint_names):
            ax4.plot(self.time_data, self.joint_torque_actual[:, i], color=colors[i], 
                     label=name, linewidth=1.5)
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Torque (Nm)')
        ax4.set_title('Joint Torques')
        ax4.legend(loc='upper right', fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            base, ext = os.path.splitext(save_path)
            comparison_path = f"{base}_comparison{ext}"
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            print(f"Comparison plot saved to: {comparison_path}")
        else:
            plt.show()
            
    def save_data(self, filepath: str):
        """Save recorded data to CSV for external analysis."""
        import pandas as pd
        
        # Create dataframe
        data = {'time': self.time_data}
        for i, name in enumerate(self.joint_names):
            data[f'{name}_cmd'] = self.joint_pos_cmd[:, i]
            data[f'{name}_pos'] = self.joint_pos_actual[:, i]
            data[f'{name}_vel'] = self.joint_vel_actual[:, i]
            data[f'{name}_torque'] = self.joint_torque_actual[:, i]
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        print(f"\nData saved to: {filepath}")
        
    def print_statistics(self):
        """Print statistics about the test."""
        print(f"\n{'='*60}")
        print("Test Statistics:")
        print(f"{'='*60}")
        
        for i, name in enumerate(self.joint_names):
            error = self.joint_pos_cmd[:, i] - self.joint_pos_actual[:, i]
            print(f"\n{name}:")
            print(f"  Position: min={self.joint_pos_actual[:, i].min():.4f}, "
                  f"max={self.joint_pos_actual[:, i].max():.4f} rad")
            print(f"  Velocity: min={self.joint_vel_actual[:, i].min():.4f}, "
                  f"max={self.joint_vel_actual[:, i].max():.4f} rad/s")
            print(f"  Tracking Error: mean={error.mean():.4f}, std={error.std():.4f}, "
                  f"max_abs={np.abs(error).max():.4f} rad")
            print(f"  Torque: min={self.joint_torque_actual[:, i].min():.4f}, "
                  f"max={self.joint_torque_actual[:, i].max():.4f} Nm")


def main():
    """Main function."""
    # Create characterizer
    characterizer = JointCharacterizer(sim_dt=0.005, decimation=4)
    
    # Setup scene
    characterizer.setup_scene()
    
    # Run test
    characterizer.run_test(
        test_type=args_cli.test_type,
        duration=args_cli.duration,
        joint_name=args_cli.joint,
        target=args_cli.target,
        frequency=args_cli.frequency,
        amplitude=args_cli.amplitude,
    )
    
    # Print statistics
    characterizer.print_statistics()
    
    # Generate plots
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args_cli.save_plot:
        # Create output directory
        output_dir = os.path.join(os.path.dirname(__file__), "joint_characterization_results")
        os.makedirs(output_dir, exist_ok=True)
        
        plot_path = os.path.join(output_dir, f"joint_test_{args_cli.test_type}_{timestamp}.png")
        data_path = os.path.join(output_dir, f"joint_test_{args_cli.test_type}_{timestamp}.csv")
        
        characterizer.plot_results(save_path=plot_path, joint_name=args_cli.joint)
        characterizer.plot_comparison(save_path=plot_path)
        characterizer.save_data(data_path)
    else:
        characterizer.plot_results(joint_name=args_cli.joint)
        characterizer.plot_comparison()
    
    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    main()
