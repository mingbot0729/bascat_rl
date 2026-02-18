"""Terrain configurations for Ostrich rough terrain training.

Copied from the Tron bipedal_locomotion reference (PF/terrains_cfg.py),
adapted to use isaaclab.terrains imports.
"""

from isaaclab.terrains import (
    HfRandomUniformTerrainCfg,
    HfWaveTerrainCfg,
    MeshPlaneTerrainCfg,
    MeshRandomGridTerrainCfg,
    TerrainGeneratorCfg,
)

#############################
# Rough Terrain Configuration
#############################

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        # "flat": MeshPlaneTerrainCfg(proportion=0.25),
        # "waves": HfWaveTerrainCfg(
        #     proportion=0.01, amplitude_range=(0.01, 0.06), num_waves=10, border_width=0.25
        # ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.25, grid_width=0.15, grid_height_range=(0.01, 0.03), platform_width=0.1
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.25, noise_range=(0.01, 0.03), noise_step=0.01, border_width=0.25
        ),
    },
    curriculum=True,
    difficulty_range=(0.0, 1.0),
)

ROUGH_TERRAINS_PLAY_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=True,
    sub_terrains={
        "waves": HfWaveTerrainCfg(
            proportion=0.33, amplitude_range=(0.01, 0.06), num_waves=10, border_width=0.25
        ),
        "boxes": MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.33, grid_height_range=(0.01, 0.04), platform_width=2.0
        ),
        "random_rough": HfRandomUniformTerrainCfg(
            proportion=0.34, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
    },
    curriculum=False,
    difficulty_range=(1.0, 1.0),
)
