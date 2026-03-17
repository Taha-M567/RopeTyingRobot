"""Articulated rope-chain configuration for Isaac Lab.

Converts the generated URDF to USD and provides an ``ArticulationCfg``
factory function, mirroring the pattern in ``so100_config.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

# Ensure sibling modules (generate_rope_urdf) are importable.
_SELF_DIR = Path(__file__).resolve().parent
if str(_SELF_DIR) not in sys.path:
    sys.path.insert(0, str(_SELF_DIR))

from generate_rope_urdf import generate_rope_urdf  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROPE_NUM_SEGMENTS = 16
ROPE_URDF_FILENAME = "rope_chain.urdf"
ROPE_USD_FILENAME = "rope_chain_from_urdf.usd"

ROPE_PITCH_JOINT_NAMES = [f"rope_pitch_{i}" for i in range(ROPE_NUM_SEGMENTS - 1)]
ROPE_YAW_JOINT_NAMES = [f"rope_yaw_{i}" for i in range(ROPE_NUM_SEGMENTS - 1)]

ROPE_ALL_JOINT_NAMES: list[str] = []
for _i in range(ROPE_NUM_SEGMENTS - 1):
    ROPE_ALL_JOINT_NAMES.append(f"rope_pitch_{_i}")
    ROPE_ALL_JOINT_NAMES.append(f"rope_yaw_{_i}")

ROPE_SEGMENT_BODY_NAMES = [f"rope_seg_{i}" for i in range(ROPE_NUM_SEGMENTS)]
ROPE_HINGE_BODY_NAMES = [f"rope_hinge_{i}" for i in range(ROPE_NUM_SEGMENTS - 1)]

ROPE_DEFAULT_JOINT_POS: dict[str, float] = {name: 0.0 for name in ROPE_ALL_JOINT_NAMES}


# ---------------------------------------------------------------------------
# USD conversion
# ---------------------------------------------------------------------------
def ensure_rope_usd(
    asset_dir: Path | None = None,
    force_conversion: bool = False,
) -> Path:
    """Generate the rope URDF (if missing) and convert to USD.

    Args:
        asset_dir: Directory for URDF / USD files. Defaults to this
            script's parent directory.
        force_conversion: Re-generate URDF and re-convert even if files
            already exist.

    Returns:
        Path to the USD file.
    """
    resolved_dir = asset_dir if asset_dir is not None else _SELF_DIR
    urdf_path = resolved_dir / ROPE_URDF_FILENAME
    usd_path = resolved_dir / ROPE_USD_FILENAME

    # Generate URDF if needed
    if not urdf_path.exists() or force_conversion:
        generate_rope_urdf(output_path=urdf_path)

    # Convert URDF -> USD if needed
    if usd_path.exists() and not force_conversion:
        return usd_path

    converter_cfg = UrdfConverterCfg(
        asset_path=str(urdf_path),
        usd_dir=str(resolved_dir),
        usd_file_name=ROPE_USD_FILENAME,
        fix_base=False,
        merge_fixed_joints=False,
        force_usd_conversion=force_conversion,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=20.0,
                damping=5.0,
            ),
            target_type="position",
        ),
    )
    converter = UrdfConverter(converter_cfg)
    return Path(converter.usd_path)


# ---------------------------------------------------------------------------
# ArticulationCfg factory
# ---------------------------------------------------------------------------
def create_rope_articulation_cfg(
    asset_dir: Path | None = None,
    force_conversion: bool = False,
) -> ArticulationCfg:
    """Create an ``ArticulationCfg`` for the articulated rope chain.

    The rope is a passive articulation: zero stiffness, low damping,
    self-collisions enabled for crossing detection, and higher solver
    iterations for chain stability.
    """
    usd_path = ensure_rope_usd(
        asset_dir=asset_dir, force_conversion=force_conversion,
    )

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=50.0,
                max_angular_velocity=50.0,
                max_depenetration_velocity=10.0,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.25, 0.0, 0.05),
            joint_pos=ROPE_DEFAULT_JOINT_POS,
        ),
        actuators={
            "rope_joints": ImplicitActuatorCfg(
                joint_names_expr=["rope_pitch_.*", "rope_yaw_.*"],
                effort_limit_sim=10.0,
                stiffness=20.0,
                damping=5.0,
            ),
        },
    )
