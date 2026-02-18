# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper that logs robot leg contact forces, motor torques, height, and projected gravity.
Adds live Contact Forces and Motor Torques panels to the right-side debug window when the UI is available."""

import gymnasium as gym
import torch

try:
    from isaaclab.utils.math import quat_apply_inverse
except ImportError:
    quat_apply_inverse = None

# Optional: for live debug window panel (only in Isaac Sim with UI)
try:
    import omni.ui  # type: ignore[import-untyped]
    _OMNI_UI_AVAILABLE = True
except ImportError:
    _OMNI_UI_AVAILABLE = False

# Optional: visual markers for leg-ground contact (green sphere when in contact)
try:
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
    _MARKERS_AVAILABLE = True
except ImportError:
    _MARKERS_AVAILABLE = False


def _find_env_with_scene(env):
    """Unwrap until we find an env with a .scene attribute (ManagerBasedRLEnv)."""
    e = env
    while e is not None:
        if hasattr(e, "scene") and getattr(e.scene, "sensors", None) is not None:
            return e
        e = getattr(e, "env", getattr(e, "unwrapped", None))
        if e is env:
            break
    return None


def _get_lower_leg_body_ids(scene, robot_name="robot"):
    """Return body indices for links whose name contains 'lower_leg_link'."""
    try:
        robot = scene[robot_name]
    except KeyError:
        return None
    body_names = getattr(robot, "body_names", None)
    if body_names is None:
        return None
    import re
    pattern = re.compile(r".*lower_leg_link.*")
    return [i for i, name in enumerate(body_names) if pattern.match(name)]


def _get_contact_sensor_body_ids_and_names(scene, robot_name="robot", sensor_name="contact_forces"):
    """Return (body_indices, body_names) for bodies used by the contact sensor, or (None, None)."""
    try:
        robot = scene[robot_name]
        sensor = scene.sensors.get(sensor_name)
    except (KeyError, AttributeError):
        return None, None
    if sensor is None or not hasattr(sensor, "cfg") or sensor.cfg is None:
        return None, None
    body_ids = getattr(sensor.cfg, "body_ids", None)
    if body_ids is None:
        # Sensor may use all bodies; use leg body names if available
        body_ids = _get_lower_leg_body_ids(scene, robot_name)
    body_names = getattr(robot, "body_names", None)
    if body_names is not None and body_ids is not None:
        names = [body_names[i] if i < len(body_names) else f"body_{i}" for i in body_ids]
        return body_ids, names
    return body_ids, []


class ContactForceLoggingWrapper(gym.Wrapper):
    """Logs average leg contact force every log_interval steps.
    When the env has a window (non-headless), adds a 'Contact Forces (Live)' panel to the right-side debug window."""

    def __init__(self, env, log_interval=500, robot_name="robot", sensor_name="contact_forces"):
        super().__init__(env)
        self._log_interval = log_interval
        self._robot_name = robot_name
        self._sensor_name = sensor_name
        self._step_count = 0
        self._base_env = None
        self._leg_body_ids = None
        # Live debug panel (right-side window) — contact forces
        self._live_label = None
        self._live_body_ids = None
        self._live_body_names = None
        self._live_panel_built = False
        # Live debug panel — motor torques
        self._torque_label = None
        self._torque_panel_built = False
        self._joint_names = None
        # Visual markers: green sphere at foot when leg-ground contact, gray when not
        self._foot_markers = None
        self._foot_markers_created = False
        # Resolved foot indices (robot-side for positions, sensor-side for forces)
        self._robot_left_idx = None
        self._robot_right_idx = None
        self._sensor_left_idx = None
        self._sensor_right_idx = None
        self._foot_idx_resolved = False
        self._build_live_panel()

    def _build_live_panel(self):
        """Add a 'Contact Forces (Live)' panel with numeric values to the right-side IsaacLab window."""
        if not _OMNI_UI_AVAILABLE:
            if self._step_count == 0:
                print("[ContactForceLoggingWrapper] omni.ui not available (run inside Isaac Sim with GUI for live values).")
            return
        if self._live_panel_built:
            return
        base = _find_env_with_scene(self.env)
        if base is None:
            if self._step_count == 0:
                print("[ContactForceLoggingWrapper] Could not find env with scene.")
            return
        window = getattr(base, "window", None)
        if window is None:
            if self._step_count <= 2:
                print("[ContactForceLoggingWrapper] No window yet (run without --headless).")
            return
        elements = getattr(window, "ui_window_elements", None)
        if not elements:
            if self._step_count == 0:
                print("[ContactForceLoggingWrapper] Window has no ui_window_elements.")
            return
        scene = getattr(base, "scene", None)
        if scene is None or not hasattr(scene, "sensors"):
            return
        # Use main_vstack so the panel is a top-level section (below Simulation / Viewer / Scene Debug)
        target_stack = elements.get("main_vstack")
        if target_stack is None:
            if self._step_count == 0:
                print("[ContactForceLoggingWrapper] main_vstack not found. Keys:", list(elements.keys()))
            return
        try:
            with target_stack:
                frame = omni.ui.CollapsableFrame(
                    title="Contact Forces (Live values)",
                    width=omni.ui.Fraction(1),
                    height=80,
                    collapsed=False,
                )
                with frame:
                    self._live_label = omni.ui.Label(
                        "— N (no data yet)",
                        word_wrap=True,
                        width=omni.ui.Fraction(1),
                        height=60,
                    )
            self._live_panel_built = True
            print("[ContactForceLoggingWrapper] Live panel added to IsaacLab window. Scroll to the BOTTOM of the right panel to see 'Contact Forces (Live values)'.")
            # Resolve body ids/names for display
            if self._sensor_name in scene.sensors:
                self._live_body_ids, self._live_body_names = _get_contact_sensor_body_ids_and_names(
                    scene, self._robot_name, self._sensor_name
                )
                if self._live_body_ids is None:
                    self._live_body_ids = _get_lower_leg_body_ids(scene, self._robot_name)
                    if self._live_body_ids is not None:
                        try:
                            robot = scene[self._robot_name]
                            names = getattr(robot, "body_names", None)
                        except (KeyError, TypeError):
                            names = None
                        if names is not None:
                            self._live_body_names = [names[i] if i < len(names) else f"body_{i}" for i in self._live_body_ids]
                        else:
                            self._live_body_names = [f"body_{i}" for i in self._live_body_ids]
        except Exception as e:
            import traceback
            print(f"[ContactForceLoggingWrapper] Failed to add live panel: {e}")
            traceback.print_exc()

    def _ensure_cached(self):
        if self._base_env is not None:
            return
        self._base_env = _find_env_with_scene(self.env)
        scene = getattr(self._base_env, "scene", None) if self._base_env is not None else None
        if scene is not None and hasattr(scene, "sensors") and self._sensor_name in scene.sensors:
            self._leg_body_ids = _get_lower_leg_body_ids(scene, self._robot_name)

    def _update_live_panel(self):
        """Update the right-side debug panel with current contact force readings (env 0)."""
        scene = getattr(self._base_env, "scene", None) if self._base_env is not None else None
        if self._live_label is None or scene is None or self._sensor_name not in getattr(scene, "sensors", {}):
            return
        try:
            sensor = scene.sensors[self._sensor_name]
            # net_forces_w_history: (num_envs, history, num_bodies, 3)
            forces_w = sensor.data.net_forces_w_history
            if forces_w is None or forces_w.numel() == 0:
                self._live_label.text = "— N (no data)"
                return
            # Use env 0, latest history step
            env0 = forces_w[0]
            latest = env0[-1] if env0.dim() > 1 else env0
            mag = latest.norm(dim=-1).float()
            body_ids = self._live_body_ids
            names = self._live_body_names or []
            if body_ids is not None and len(body_ids) <= mag.shape[0]:
                lines = []
                leg_mag = mag[body_ids]
                for i, bid in enumerate(body_ids):
                    name = names[i] if i < len(names) else f"body_{bid}"
                    lines.append(f"{name}: {mag[bid].item():.1f} N")
                summary = leg_mag[leg_mag > 1.0]
                avg_contact = summary.mean().item() if summary.numel() > 0 else 0.0
                contact_frac = (leg_mag > 1.0).float().mean().item()
                lines.append(f"avg(in contact)={avg_contact:.1f} N  contact_frac={contact_frac:.2f}")
                self._live_label.text = "\n".join(lines)
            else:
                # Fallback: show per-body for first few bodies
                n_show = min(8, mag.shape[0])
                lines = [f"body_{i}: {mag[i].item():.1f} N" for i in range(n_show)]
                lines.append(f"max={mag.max().item():.1f} N")
                self._live_label.text = "\n".join(lines)
        except Exception as e:
            self._live_label.text = f"Error: {e}"

    def _build_torque_panel(self):
        """Add a 'Motor Torques (Live)' collapsible panel to the right-side IsaacLab window."""
        if not _OMNI_UI_AVAILABLE or self._torque_panel_built:
            return
        base = _find_env_with_scene(self.env)
        if base is None:
            return
        window = getattr(base, "window", None)
        if window is None:
            return
        elements = getattr(window, "ui_window_elements", None)
        if not elements:
            return
        target_stack = elements.get("main_vstack")
        if target_stack is None:
            return
        scene = getattr(base, "scene", None)
        if scene is None:
            return
        try:
            robot = scene[self._robot_name]
            self._joint_names = list(getattr(robot, "joint_names", []))
            num_joints = len(self._joint_names)
            if num_joints == 0:
                return
            # Calculate frame height: ~18px per joint line + header + avg line
            frame_h = max(80, 20 * (num_joints + 2))
            label_h = max(60, 18 * (num_joints + 2))
            with target_stack:
                frame = omni.ui.CollapsableFrame(
                    title="Motor Torques (Live values)",
                    width=omni.ui.Fraction(1),
                    height=frame_h,
                    collapsed=False,
                )
                with frame:
                    self._torque_label = omni.ui.Label(
                        "— Nm (no data yet)",
                        word_wrap=True,
                        width=omni.ui.Fraction(1),
                        height=label_h,
                    )
            self._torque_panel_built = True
            print(f"[ContactForceLoggingWrapper] Motor Torques panel added. Joints: {self._joint_names}")
        except Exception as e:
            if self._step_count <= 2:
                import traceback
                print(f"[ContactForceLoggingWrapper] Failed to add torque panel: {e}")
                traceback.print_exc()

    def _update_torque_panel(self):
        """Update the Motor Torques panel with per-joint applied torque for env 0."""
        if self._torque_label is None or self._base_env is None:
            return
        scene = getattr(self._base_env, "scene", None)
        if scene is None:
            return
        try:
            robot = scene[self._robot_name]
            # applied_torque: (num_envs, num_joints)
            torques = getattr(robot.data, "applied_torque", None)
            if torques is None:
                self._torque_label.text = "— Nm (applied_torque not available)"
                return
            env0_torques = torques[0].float()  # (num_joints,) — signed: + one direction, - opposite
            names = self._joint_names or []
            lines = []
            for j in range(env0_torques.shape[0]):
                name = names[j] if j < len(names) else f"joint_{j}"
                short = name.replace("_joint", "").replace("_", " ")
                val = env0_torques[j].item()  # keep sign
                bar_len = min(int(abs(val) / 7.7 * 10), 10)
                bar = ("+" if val >= 0 else "-") + "|" * bar_len  # show direction
                lines.append(f"{short:>16s}: {val:+7.2f} Nm  {bar}")
            # Summary: signed avg and max magnitude (signed so you see if one side dominates)
            signed_avg = torques.float().mean().item()
            max_abs = torques.float().abs().max().item()
            lines.append(f"{'(env 0 signed)':>16s}: avg={signed_avg:+.2f}  max|tau|={max_abs:.2f} Nm")
            self._torque_label.text = "\n".join(lines)
        except Exception as e:
            self._torque_label.text = f"Error: {e}"

    def _ensure_foot_markers(self):
        """Create visualization markers for leg-ground contact (once)."""
        if not _MARKERS_AVAILABLE or self._foot_markers_created or self._base_env is None:
            return
        scene = getattr(self._base_env, "scene", None)
        if scene is None or self._leg_body_ids is None or len(self._leg_body_ids) < 2:
            return
        try:
            cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/foot_contact_markers",
                markers={
                    "contact": sim_utils.SphereCfg(
                        radius=0.015,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                    ),
                    "no_contact": sim_utils.SphereCfg(
                        radius=0.008,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4)),
                    ),
                },
            )
            self._foot_markers = VisualizationMarkers(cfg)
            self._foot_markers_created = True
        except Exception as e:
            if self._step_count <= 2:
                print(f"[ContactForceLoggingWrapper] Foot markers not created: {e}")

    def _resolve_foot_indices(self):
        """Resolve left/right lower-leg indices separately in robot body list and sensor body list.
        This is the critical fix: the sensor may order bodies differently from the robot articulation."""
        if self._foot_idx_resolved:
            return
        scene = getattr(self._base_env, "scene", None)
        if scene is None:
            return
        try:
            robot = scene[self._robot_name]
            sensor = scene.sensors[self._sensor_name]
        except (KeyError, AttributeError):
            return
        # --- Robot-side indices (for body positions) ---
        robot_body_names = list(getattr(robot, "body_names", []))
        for i, name in enumerate(robot_body_names):
            nl = name.lower()
            if "lower_leg" in nl and "left" in nl:
                self._robot_left_idx = i
            elif "lower_leg" in nl and "right" in nl:
                self._robot_right_idx = i
        # --- Sensor-side indices (for force data) ---
        # Try sensor.body_names first (Isaac Lab ContactSensor exposes this)
        sensor_body_names = getattr(sensor, "body_names", None)
        if sensor_body_names is not None and len(sensor_body_names) > 0:
            sensor_body_names = list(sensor_body_names)
            for i, name in enumerate(sensor_body_names):
                nl = name.lower()
                if "lower_leg" in nl and "left" in nl:
                    self._sensor_left_idx = i
                elif "lower_leg" in nl and "right" in nl:
                    self._sensor_right_idx = i
        # Try sensor.find_bodies() as a second approach
        if self._sensor_left_idx is None or self._sensor_right_idx is None:
            find_fn = getattr(sensor, "find_bodies", None)
            if find_fn is not None:
                try:
                    ids, names = find_fn(".*lower_leg.*")
                    if ids is not None and len(ids) >= 2:
                        for idx, name in zip(ids, names):
                            nl = name.lower()
                            if "left" in nl:
                                self._sensor_left_idx = idx
                            elif "right" in nl:
                                self._sensor_right_idx = idx
                except Exception:
                    pass
        # Fallback: assume sensor uses same ordering as robot
        if self._sensor_left_idx is None:
            self._sensor_left_idx = self._robot_left_idx
        if self._sensor_right_idx is None:
            self._sensor_right_idx = self._robot_right_idx
        self._foot_idx_resolved = True
        # One-time diagnostic print
        sensor_body_list = list(sensor_body_names) if sensor_body_names else "N/A"
        force_shape = "N/A"
        try:
            force_shape = list(sensor.data.net_forces_w_history.shape)
        except Exception:
            pass
        print(f"[ContactForceLoggingWrapper] === Foot marker index mapping ===")
        print(f"  Robot bodies:  {list(enumerate(robot_body_names))}")
        print(f"  Sensor bodies: {sensor_body_list}")
        print(f"  Force tensor shape: {force_shape}")
        print(f"  ROBOT  left_lower_leg={self._robot_left_idx}  right_lower_leg={self._robot_right_idx}")
        print(f"  SENSOR left_lower_leg={self._sensor_left_idx}  right_lower_leg={self._sensor_right_idx}")

    def _update_foot_markers(self):
        """Update foot contact markers at the foot tip, scaled by force magnitude.
        Green when in contact with ground, gray when in air."""
        if self._foot_markers is None or self._base_env is None:
            return
        scene = getattr(self._base_env, "scene", None)
        if scene is None or self._sensor_name not in getattr(scene, "sensors", {}):
            return
        # Resolve indices once
        if not self._foot_idx_resolved:
            self._resolve_foot_indices()
        if (self._robot_left_idx is None or self._robot_right_idx is None
                or self._sensor_left_idx is None or self._sensor_right_idx is None):
            return
        try:
            robot = scene[self._robot_name]
            sensor = scene.sensors[self._sensor_name]
            num_envs = robot.data.body_pos_w.shape[0]
            # === POSITIONS from robot (indexed by robot body list) ===
            left_pos = robot.data.body_pos_w[:, self._robot_left_idx, :].clone()   # (N, 3)
            right_pos = robot.data.body_pos_w[:, self._robot_right_idx, :].clone()  # (N, 3)
            # Offset marker to foot tip: lower_leg origin is at knee joint,
            # foot tip is ~0.10 m below (from URDF geometry).  Clamp above ground.
            LOWER_LEG_TIP_OFFSET = 0.10
            left_pos[:, 2] = torch.clamp(left_pos[:, 2] - LOWER_LEG_TIP_OFFSET, min=0.003)
            right_pos[:, 2] = torch.clamp(right_pos[:, 2] - LOWER_LEG_TIP_OFFSET, min=0.003)
            # === FORCES from sensor (indexed by sensor body list) ===
            forces_all = sensor.data.net_forces_w_history[:, -1, :, :]  # (N, num_sensor_bodies, 3)
            left_force = forces_all[:, self._sensor_left_idx, :].norm(dim=-1)   # (N,)
            right_force = forces_all[:, self._sensor_right_idx, :].norm(dim=-1)  # (N,)
            # Combine: all left markers first, then all right markers
            positions = torch.cat([left_pos, right_pos], dim=0)  # (2*N, 3)
            # Contact state
            in_contact = torch.cat([left_force > 1.0, right_force > 1.0], dim=0)
            marker_indices = torch.where(in_contact, 0, 1).to(torch.int32)
            # Scale markers with contact force magnitude
            # base_scale 0.5 (small when no contact), grows with force, capped at 5.0
            force_mags = torch.cat([left_force, right_force], dim=0)  # (2*N,)
            scale_factor = torch.clamp(0.5 + force_mags / 40.0, 0.5, 5.0)
            scales = scale_factor.unsqueeze(-1).expand(-1, 3)  # (2*N, 3)
            # Orientations (identity quaternion)
            device = positions.device
            quats = torch.zeros(2 * num_envs, 4, device=device, dtype=positions.dtype)
            quats[:, 0] = 1.0
            self._foot_markers.visualize(positions, quats, marker_indices=marker_indices, scales=scales)
        except Exception as e:
            if self._step_count <= 5:
                print(f"[ContactForceLoggingWrapper] Foot marker update error: {e}")
                import traceback
                traceback.print_exc()

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._step_count += 1
        self._ensure_cached()
        # Retry building live panels for the first 30 steps (window may appear after a few frames)
        if not self._live_panel_built and self._step_count <= 30:
            self._build_live_panel()
        if not self._torque_panel_built and self._step_count <= 30:
            self._build_torque_panel()
        # Update live panels
        if self._live_label is not None and self._base_env is not None:
            self._update_live_panel()
        if self._torque_label is not None and self._base_env is not None:
            self._update_torque_panel()
        # Visual markers for leg-ground contact (green = in contact, gray = no contact)
        if not self._foot_markers_created and self._base_env is not None and self._step_count <= 5:
            self._ensure_foot_markers()
        if self._foot_markers is not None and self._base_env is not None:
            self._update_foot_markers()
        # Periodic console logging
        if self._base_env is not None and self._step_count % self._log_interval == 0:
            scene = getattr(self._base_env, "scene", None)
            if scene is not None:
                try:
                    robot = scene[self._robot_name]
                    # Average base height (m)
                    height = robot.data.root_pos_w[:, 2].float().mean().item()
                    # Average projected gravity xyz in body frame (unit vector, z should be ~-1 when upright)
                    if quat_apply_inverse is not None:
                        n = robot.data.root_quat_w.shape[0]
                        g_world = torch.zeros(n, 3, device=robot.data.root_quat_w.device, dtype=robot.data.root_quat_w.dtype)
                        g_world[:, 2] = -1.0
                        proj_grav = quat_apply_inverse(robot.data.root_quat_w, g_world)  # (N, 3)
                        pg_mean = proj_grav.float().mean(dim=0).cpu()
                        pg_x, pg_y, pg_z = pg_mean[0].item(), pg_mean[1].item(), pg_mean[2].item()
                    else:
                        pg_x = pg_y = pg_z = 0.0
                    print(
                        f"[TrainLog] step={self._step_count} "
                        f"avg_height={height:.3f} m "
                        f"proj_gravity_xyz=({pg_x:.3f}, {pg_y:.3f}, {pg_z:.3f})"
                    )
                    # --- Motor torques: signed average per joint (mean over envs), and max magnitude ---
                    torques = getattr(robot.data, "applied_torque", None)
                    if torques is not None:
                        torques_f = torques.float()
                        joint_names = self._joint_names or list(getattr(robot, "joint_names", []))
                        per_joint_avg = torques_f.mean(dim=0)  # signed average per joint
                        parts = []
                        for j in range(per_joint_avg.shape[0]):
                            name = joint_names[j] if j < len(joint_names) else f"j{j}"
                            short = name.replace("_joint", "")
                            parts.append(f"{short}={per_joint_avg[j].item():+.2f}")
                        max_abs = torques_f.abs().max().item()
                        print(
                            f"         torque (signed avg Nm): max|tau|={max_abs:.2f}  "
                            + "  ".join(parts)
                        )
                    # --- Leg contact forces (if available) — use sensor-side indices ---
                    sensor_leg_ids = None
                    if self._foot_idx_resolved and self._sensor_left_idx is not None and self._sensor_right_idx is not None:
                        sensor_leg_ids = [self._sensor_left_idx, self._sensor_right_idx]
                    elif self._leg_body_ids is not None:
                        sensor_leg_ids = self._leg_body_ids  # fallback
                    if sensor_leg_ids is not None and self._sensor_name in getattr(scene, "sensors", {}):
                        sensor = scene.sensors[self._sensor_name]
                        forces = sensor.data.net_forces_w_history[:, :, sensor_leg_ids, :].norm(dim=-1).float()
                        in_contact = forces > 1.0  # threshold 1 N
                        contact_frac = in_contact.float().mean().item()
                        avg_all = forces.mean().item()
                        max_force = forces.max().item()
                        when_contact = forces[in_contact]
                        avg_when_contact = when_contact.mean().item() if when_contact.numel() > 0 else 0.0
                        print(
                            f"         leg_force: avg_all={avg_all:.1f} N  avg_when_contact={avg_when_contact:.1f} N  "
                            f"max={max_force:.1f} N  contact_frac={contact_frac:.2f}"
                        )
                except Exception as e:
                    print(f"[TrainLog] log failed: {e}")
        return obs, reward, terminated, truncated, info
