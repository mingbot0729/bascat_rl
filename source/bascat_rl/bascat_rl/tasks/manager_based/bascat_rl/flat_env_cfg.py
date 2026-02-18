"""Ostrich flat and rough terrain environment configurations.

Follows the Tron pointfoot_env_cfg.py pattern:
- PFBaseEnvCfg  -> OstrichBaseEnvCfg  (sets robot, contact body names)
- PFBlindFlatEnvCfg -> OstrichBlindFlatEnvCfg  (disables height scan, flat terrain)
- PFBlindRoughEnvCfg -> OstrichBlindRoughEnvCfg  (uneven terrain 0-5cm, denser)
"""

from isaaclab.utils import configclass

from .ostrich import OSTRICH_CFG
from .terrains_cfg import ROUGH_TERRAINS_CFG, ROUGH_TERRAINS_PLAY_CFG
from .velocity_env_cfg import OstrichEnvCfg, TerminationsRoughCfg


######################
# Ostrich Base Environment
######################


@configclass
class OstrichBaseEnvCfg(OstrichEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Set robot
        self.scene.robot = OSTRICH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Robot-specific body names for events/terminations
        self.events.add_base_mass.params["asset_cfg"].body_names = "base_link"
        self.terminations.base_contact.params["sensor_cfg"].body_names = "base_link"

        # Update viewport camera
        self.viewer.origin_type = "env"


@configclass
class OstrichBaseEnvCfg_PLAY(OstrichBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for play
        self.scene.num_envs = 32

        # Disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.add_base_mass = None


############################
# Ostrich Blind Flat Environment
############################


@configclass
class OstrichBlindFlatEnvCfg(OstrichBaseEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # No height scanner for blind locomotion
        self.scene.height_scanner = None

        # No terrain curriculum for flat terrain
        self.curriculum.terrain_levels = None


@configclass
class OstrichBlindFlatEnvCfg_PLAY(OstrichBaseEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()

        # No height scanner for blind locomotion
        self.scene.height_scanner = None

        # No terrain curriculum for flat terrain
        self.curriculum.terrain_levels = None


############################
# Ostrich Blind Rough Environment
############################


@configclass
class OstrichBlindRoughEnvCfg(OstrichBaseEnvCfg):
    """Ostrich on uneven terrain (0-5cm height, denser bumps). Blind locomotion."""

    def __post_init__(self):
        # Set rough terminations before super so parent's base_contact params apply
        self.terminations = TerminationsRoughCfg()
        super().__post_init__()

        # No height scanner for blind locomotion
        self.scene.height_scanner = None

        # Rough terrain from generator
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class OstrichBlindRoughEnvCfg_PLAY(OstrichBaseEnvCfg_PLAY):
    """Play variant for rough terrain - smaller scene, fixed difficulty."""

    def __post_init__(self):
        # Set rough terminations before super so parent's base_contact params apply
        self.terminations = TerminationsRoughCfg()
        super().__post_init__()

        # No height scanner for blind locomotion
        self.scene.height_scanner = None

        # Rough terrain, spawn anywhere (full difficulty)
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = ROUGH_TERRAINS_PLAY_CFG
        self.scene.terrain.max_init_terrain_level = None
