"""Articulated rope-chain configuration for Isaac Lab.

Converts the generated URDF to USD and provides an ``ArticulationCfg``
factory function, mirroring the pattern in ``so100_config.py``.

Also provides utilities for loading and saving knot configurations
(joint angle presets) from YAML files.
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path

import yaml

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

# Joint limits (from generate_rope_urdf: ±120 deg).
ROPE_JOINT_LIMIT_RAD = math.radians(120.0)

KNOT_CONFIGS_DIR = _SELF_DIR / "knot_configs"

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# USD post-processing: adjacent-body collision filter
# ---------------------------------------------------------------------------
def _apply_adjacent_body_filter(
    usd_path: Path,
    num_segments: int = ROPE_NUM_SEGMENTS,
) -> None:
    """Disable self-collisions between adjacent rope segments in the USD.

    PhysX auto-filters direct parent↔child pairs, but seg_i and
    seg_{i+1} are separated by a hinge link in the kinematic tree, so
    they are NOT auto-filtered.  Without this filter they collide
    constantly at the hinge point, causing jitter.

    Uses ``PhysxFilteredPairsAPI`` which stores filtered pairs as a
    flat relationship list where consecutive path pairs form the
    disabled collision pairs (element 0↔1, element 2↔3, ...).
    """
    from pxr import PhysxSchema, Sdf, Usd

    stage = Usd.Stage.Open(str(usd_path))
    root_prim = stage.GetDefaultPrim()
    root_path = root_prim.GetPath()

    filtered_api = PhysxSchema.PhysxFilteredPairsAPI.Apply(root_prim)

    targets = []
    for i in range(num_segments - 1):
        targets.append(Sdf.Path(f"{root_path}/rope_seg_{i}"))
        targets.append(Sdf.Path(f"{root_path}/rope_seg_{i + 1}"))

    filtered_api.GetFilteredPairsRel().SetTargets(targets)
    stage.Save()
    logger.info(
        "Applied adjacent-body collision filter (%d pairs) to %s",
        num_segments - 1,
        usd_path.name,
    )


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
        replace_cylinders_with_capsules=True,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.5,
            ),
            target_type="none",
        ),
    )
    converter = UrdfConverter(converter_cfg)
    result_path = Path(converter.usd_path)

    # Post-process: disable collisions between adjacent segments.
    _apply_adjacent_body_filter(result_path, num_segments=ROPE_NUM_SEGMENTS)

    return result_path


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
                max_linear_velocity=10.0,
                max_angular_velocity=20.0,
                max_depenetration_velocity=2.0,
                enable_gyroscopic_forces=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.002,
                rest_offset=0.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=8,
                sleep_threshold=0.001,
                stabilization_threshold=0.0005,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.25, 0.0, 0.05),
            joint_pos=ROPE_DEFAULT_JOINT_POS,
        ),
        actuators={
            "rope_joints": ImplicitActuatorCfg(
                joint_names_expr=["rope_pitch_.*", "rope_yaw_.*"],
                effort_limit_sim=0.5,
                stiffness=0.0,
                damping=0.5,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Knot configuration I/O
# ---------------------------------------------------------------------------
def load_knot_config(path: str | Path) -> dict[str, float]:
    """Load a knot configuration from a YAML file.

    The YAML must contain a ``joint_angles`` mapping with all 30 rope
    joint names as keys and angle values (radians) within the joint
    limits (±2.094 rad).

    Args:
        path: Path to the YAML file.  If a plain filename (no
            directory separator), it is resolved relative to the
            ``assets/knot_configs/`` directory.

    Returns:
        Dict mapping joint names to angle values (radians).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: On missing joints or out-of-range angles.
    """
    p = Path(path)
    if not p.is_absolute() and p.parent == Path("."):
        p = KNOT_CONFIGS_DIR / p

    if not p.exists():
        raise FileNotFoundError(f"Knot config not found: {p}")

    with open(p) as f:
        data = yaml.safe_load(f)

    angles: dict[str, float] = data.get("joint_angles", {})

    # Validate: all joints present.
    expected = set(ROPE_ALL_JOINT_NAMES)
    got = set(angles.keys())
    missing = expected - got
    if missing:
        raise ValueError(
            f"Knot config {p.name} is missing joints: "
            f"{sorted(missing)}"
        )

    # Validate: values within joint limits.
    limit = ROPE_JOINT_LIMIT_RAD
    for name, val in angles.items():
        if abs(val) > limit:
            raise ValueError(
                f"Joint {name}={val:.4f} rad exceeds limit "
                f"±{limit:.4f} rad in {p.name}"
            )

    logger.info("Loaded knot config '%s' from %s", data.get("name", "?"), p)
    return {k: float(v) for k, v in angles.items()}


def save_knot_config(
    joint_angles: dict[str, float],
    path: str | Path,
    name: str = "custom",
    description: str = "",
) -> Path:
    """Save a knot configuration to a YAML file.

    Args:
        joint_angles: Dict mapping joint names to angles (radians).
        path: Output file path.  If a plain filename, resolved
            relative to ``assets/knot_configs/``.
        name: Human-readable name stored in the YAML.
        description: Optional description stored in the YAML.

    Returns:
        The resolved output path.
    """
    p = Path(path)
    if not p.is_absolute() and p.parent == Path("."):
        p = KNOT_CONFIGS_DIR / p

    p.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "name": name,
        "description": description,
        "joint_angles": {
            k: round(float(v), 6) for k, v in joint_angles.items()
        },
    }

    with open(p, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info("Saved knot config '%s' to %s", name, p)
    return p
