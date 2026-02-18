# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""
Interactive Joint Test Script

Control your robot joints in real-time and see live position/velocity data.
This is useful for comparing simulated vs real robot behavior.

Controls:
    - Number keys 1-6: Select joint
    - UP/DOWN arrows: Increase/Decrease target position
    - LEFT/RIGHT arrows: Fine adjust position
    - R: Reset all joints to zero
    - S: Start/Stop recording
    - P: Plot recorded data
    - Q: Quit

Usage:
    python scripts/joint_interactive_test.py
"""

import argparse
import numpy as np
import torch
from collections import deque
import time

# Isaac Sim imports
from isaaclab.app import AppLauncher

# Add argparse arguments
parser = argparse.ArgumentParser(description="Interactive Joint Testing")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
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


class InteractiveJointTest:
    """Interactive joint testing with keyboard control."""
    
    def __init__(self):
        self.sim_dt = 0.005
        self.decimation = 4
        self.control_dt = self.sim_dt * self.decimation
        
        # Recording
        self.is_recording = False
        self.record_data = {
            'time': [],
            'cmd': [],
            'pos': [],
            'vel': [],
        }
        self.start_time = 0
        
        # Real-time display buffer
        self.buffer_size = 200  # Last 200 samples (~4 seconds at 50Hz)
        self.pos_buffer = None
        self.vel_buffer = None
        self.cmd_buffer = None
        self.time_buffer = None
        
        # Control state
        self.selected_joint = 0
        self.target_positions = None
        self.position_increment = 0.05  # radians per key press
        self.fine_increment = 0.01
        
    def setup_scene(self):
        """Setup the simulation scene."""
        sim_cfg = sim_utils.SimulationCfg(dt=self.sim_dt, render_interval=self.decimation)
        self.sim = SimulationContext(sim_cfg)
        self.sim.set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.3])
        
        # Ground
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func("/World/ground", ground_cfg)
        
        # Robot
        robot_cfg = OSTRICH_CFG.copy()
        robot_cfg.prim_path = "/World/Robot"
        self.robot = Articulation(robot_cfg)
        
        self.sim.reset()
        
        self.joint_names = self.robot.joint_names
        self.num_joints = len(self.joint_names)
        
        # Initialize control state
        self.target_positions = torch.zeros(1, self.num_joints, device=self.robot.device)
        
        # Initialize buffers
        self.pos_buffer = deque(maxlen=self.buffer_size)
        self.vel_buffer = deque(maxlen=self.buffer_size)
        self.cmd_buffer = deque(maxlen=self.buffer_size)
        self.time_buffer = deque(maxlen=self.buffer_size)
        
        # Get joint limits
        self.joint_limits = self.robot.root_physx_view.get_dof_limits()[0].cpu().numpy()
        
        self.robot.reset()
        
        self.print_help()
        
    def print_help(self):
        """Print control help."""
        print("\n" + "="*70)
        print("INTERACTIVE JOINT TEST - Controls")
        print("="*70)
        print("\nJoint Selection:")
        for i, name in enumerate(self.joint_names):
            limits = self.joint_limits[i]
            print(f"  [{i+1}] {name}: limits = ({limits[0]:.2f}, {limits[1]:.2f}) rad")
        print("\nControls:")
        print("  UP/DOWN arrows    : Adjust target position (±0.05 rad)")
        print("  LEFT/RIGHT arrows : Fine adjust (±0.01 rad)")
        print("  R                 : Reset all joints to zero")
        print("  S                 : Start/Stop recording")
        print("  P                 : Plot recorded data (stops recording)")
        print("  H                 : Show this help")
        print("  Q or ESC          : Quit")
        print("="*70 + "\n")
        
    def print_status(self):
        """Print current joint status."""
        pos = self.robot.data.joint_pos[0].cpu().numpy()
        vel = self.robot.data.joint_vel[0].cpu().numpy()
        cmd = self.target_positions[0].cpu().numpy()
        
        print("\r", end="")  # Carriage return for in-place update
        
        status = f"Joint: [{self.selected_joint+1}] {self.joint_names[self.selected_joint]:25s} | "
        status += f"Cmd: {cmd[self.selected_joint]:+.3f} | "
        status += f"Pos: {pos[self.selected_joint]:+.3f} | "
        status += f"Vel: {vel[self.selected_joint]:+.3f} | "
        status += f"Err: {cmd[self.selected_joint] - pos[self.selected_joint]:+.3f}"
        
        if self.is_recording:
            status += " [REC]"
            
        print(status, end="", flush=True)
        
    def handle_keyboard(self):
        """Handle keyboard input using carb input."""
        import carb.input
        
        input_interface = carb.input.acquire_input_interface()
        keyboard = simulation_app.keyboard
        
        # Check for key events
        if keyboard is not None:
            # Number keys for joint selection
            for i in range(min(9, self.num_joints)):
                if keyboard.is_key_pressed(carb.input.KeyboardInput[f"KEY_{i+1}"]):
                    self.selected_joint = i
                    
            # Arrow keys for position adjustment
            if keyboard.is_key_pressed(carb.input.KeyboardInput.UP):
                self.adjust_target(self.position_increment)
            elif keyboard.is_key_pressed(carb.input.KeyboardInput.DOWN):
                self.adjust_target(-self.position_increment)
            elif keyboard.is_key_pressed(carb.input.KeyboardInput.RIGHT):
                self.adjust_target(self.fine_increment)
            elif keyboard.is_key_pressed(carb.input.KeyboardInput.LEFT):
                self.adjust_target(-self.fine_increment)
                
            # Other controls
            if keyboard.is_key_pressed(carb.input.KeyboardInput.R):
                self.reset_joints()
            elif keyboard.is_key_pressed(carb.input.KeyboardInput.S):
                self.toggle_recording()
            elif keyboard.is_key_pressed(carb.input.KeyboardInput.P):
                self.plot_data()
            elif keyboard.is_key_pressed(carb.input.KeyboardInput.H):
                self.print_help()
            elif keyboard.is_key_pressed(carb.input.KeyboardInput.Q) or \
                 keyboard.is_key_pressed(carb.input.KeyboardInput.ESCAPE):
                return False
                
        return True
        
    def adjust_target(self, delta: float):
        """Adjust target position for selected joint."""
        new_value = self.target_positions[0, self.selected_joint].item() + delta
        # Clamp to joint limits
        low, high = self.joint_limits[self.selected_joint]
        new_value = max(low * 0.9, min(high * 0.9, new_value))
        self.target_positions[0, self.selected_joint] = new_value
        
    def reset_joints(self):
        """Reset all joints to zero."""
        self.target_positions.zero_()
        print("\n[RESET] All joints reset to zero")
        
    def toggle_recording(self):
        """Toggle recording state."""
        if self.is_recording:
            self.is_recording = False
            print("\n[RECORDING STOPPED]")
        else:
            self.is_recording = True
            self.record_data = {'time': [], 'cmd': [], 'pos': [], 'vel': []}
            self.start_time = time.time()
            print("\n[RECORDING STARTED]")
            
    def record_step(self, t: float):
        """Record current data."""
        pos = self.robot.data.joint_pos[0].cpu().numpy().copy()
        vel = self.robot.data.joint_vel[0].cpu().numpy().copy()
        cmd = self.target_positions[0].cpu().numpy().copy()
        
        # Update buffers (always)
        self.pos_buffer.append(pos)
        self.vel_buffer.append(vel)
        self.cmd_buffer.append(cmd)
        self.time_buffer.append(t)
        
        # Record if recording
        if self.is_recording:
            self.record_data['time'].append(t - self.start_time)
            self.record_data['cmd'].append(cmd)
            self.record_data['pos'].append(pos)
            self.record_data['vel'].append(vel)
            
    def plot_data(self):
        """Plot recorded data."""
        import matplotlib.pyplot as plt
        
        if not self.record_data['time']:
            print("\n[No data to plot - start recording first with 'S']")
            return
            
        self.is_recording = False
        
        t = np.array(self.record_data['time'])
        cmd = np.array(self.record_data['cmd'])
        pos = np.array(self.record_data['pos'])
        vel = np.array(self.record_data['vel'])
        
        # Create plot
        fig, axes = plt.subplots(self.num_joints, 2, figsize=(14, 3 * self.num_joints))
        if self.num_joints == 1:
            axes = axes.reshape(1, -1)
            
        fig.suptitle('Recorded Joint Data', fontsize=14, fontweight='bold')
        
        for i, name in enumerate(self.joint_names):
            # Position
            ax1 = axes[i, 0]
            ax1.plot(t, cmd[:, i], 'b--', label='Command', linewidth=2)
            ax1.plot(t, pos[:, i], 'r-', label='Actual', linewidth=1.5)
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Position (rad)')
            ax1.set_title(f'{name} - Position')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Velocity
            ax2 = axes[i, 1]
            ax2.plot(t, vel[:, i], 'g-', linewidth=1.5)
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('Velocity (rad/s)')
            ax2.set_title(f'{name} - Velocity')
            ax2.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.show()
        
        print("\n[Plot displayed - close window to continue]")
        
    def run(self):
        """Main run loop."""
        print("\nStarting interactive test...")
        print("Use number keys 1-6 to select joints, arrows to adjust position")
        print("Press 'H' for help, 'Q' to quit\n")
        
        t = 0.0
        step_count = 0
        
        try:
            while simulation_app.is_running():
                # Apply commands
                self.robot.set_joint_position_target(self.target_positions)
                
                # Step simulation
                for _ in range(self.decimation):
                    self.sim.step()
                    
                self.robot.update(self.control_dt)
                
                # Record data
                self.record_step(t)
                
                # Print status (every 10 steps to avoid flickering)
                if step_count % 10 == 0:
                    self.print_status()
                
                t += self.control_dt
                step_count += 1
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            
        print("\nTest complete!")


def main():
    """Main function with simple keyboard input loop."""
    tester = InteractiveJointTest()
    tester.setup_scene()
    
    print("\n" + "="*70)
    print("SIMPLE MODE: Enter commands in terminal")
    print("="*70)
    print("\nCommands:")
    print("  set <joint_idx> <position>  : Set joint target (e.g., 'set 0 0.3')")
    print("  all <position>              : Set all joints to position")
    print("  sine <joint_idx> <amp> <freq> <duration> : Run sine test")
    print("  step <joint_idx> <position> <duration>   : Run step test")
    print("  reset                       : Reset all to zero")
    print("  plot                        : Plot recorded data")
    print("  record                      : Toggle recording")
    print("  status                      : Show current joint states")
    print("  help                        : Show this help")
    print("  quit                        : Exit")
    print("="*70 + "\n")
    
    t = 0.0
    running = True
    command_queue = []
    
    # For automated tests
    test_running = False
    test_generator = None
    
    while simulation_app.is_running() and running:
        # Check for terminal input (non-blocking would be better but this is simpler)
        # For now, we'll run a simple command loop
        
        # Apply current target
        tester.robot.set_joint_position_target(tester.target_positions)
        
        # Step simulation
        for _ in range(tester.decimation):
            tester.sim.step()
            
        tester.robot.update(tester.control_dt)
        tester.record_step(t)
        
        t += tester.control_dt
        
        # Process commands from queue
        if command_queue:
            cmd = command_queue.pop(0)
            parts = cmd.strip().lower().split()
            
            if not parts:
                continue
                
            action = parts[0]
            
            if action == 'set' and len(parts) >= 3:
                try:
                    joint_idx = int(parts[1])
                    position = float(parts[2])
                    if 0 <= joint_idx < tester.num_joints:
                        tester.target_positions[0, joint_idx] = position
                        print(f"Set {tester.joint_names[joint_idx]} to {position:.3f} rad")
                except ValueError:
                    print("Invalid format. Use: set <joint_idx> <position>")
                    
            elif action == 'all' and len(parts) >= 2:
                try:
                    position = float(parts[1])
                    tester.target_positions[:, :] = position
                    print(f"Set all joints to {position:.3f} rad")
                except ValueError:
                    print("Invalid format. Use: all <position>")
                    
            elif action == 'reset':
                tester.reset_joints()
                
            elif action == 'status':
                pos = tester.robot.data.joint_pos[0].cpu().numpy()
                vel = tester.robot.data.joint_vel[0].cpu().numpy()
                cmd = tester.target_positions[0].cpu().numpy()
                print("\nCurrent Joint States:")
                print("-" * 60)
                for i, name in enumerate(tester.joint_names):
                    print(f"  {name:25s}: cmd={cmd[i]:+.3f}, pos={pos[i]:+.3f}, vel={vel[i]:+.3f}")
                print("-" * 60)
                    
            elif action == 'record':
                tester.toggle_recording()
                
            elif action == 'plot':
                tester.plot_data()
                
            elif action == 'help':
                tester.print_help()
                
            elif action == 'quit':
                running = False
                
        # Every 2 seconds, prompt for input (blocking)
        if int(t * 10) % 100 == 0 and int(t) > 0:
            try:
                user_input = input("\nCommand (or press Enter to continue): ")
                if user_input.strip():
                    command_queue.append(user_input)
            except EOFError:
                pass
                
    simulation_app.close()


if __name__ == "__main__":
    main()
