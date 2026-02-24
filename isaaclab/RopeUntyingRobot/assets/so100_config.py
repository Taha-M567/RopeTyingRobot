"""SO-100 asset helpers for Isaac Lab.

This module converts the local SO-100 URDF into USD and builds an
``ArticulationCfg`` that can be used by Isaac Lab scenes.
"""

from __future__ import annotations

import re
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

SO100_URDF_FILENAME = "so100.urdf"
SO100_USD_FILENAME = "so100_from_urdf.usd"
SO100_PATCHED_URDF_FILENAME = "so100_mesh_fixed.urdf"
SO100_FAST_SANDBOX_URDF_FILENAME = "so100_mesh_fixed_fast.urdf"

SO100_DEFAULT_JOINT_POS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 1.0,
    "elbow_flex": -1.2,
    "wrist_flex": 0.3,
    "wrist_roll": 0.0,
    "gripper": 0.15,
}


def _extract_mesh_paths(urdf_text: str) -> list[str]:
    """Extract mesh file references from URDF text."""
    return re.findall(r'<mesh\s+filename="([^"]+)"', urdf_text)


def _find_missing_meshes(urdf_dir: Path, mesh_paths: list[str]) -> list[str]:
    """Return relative mesh paths that do not exist next to the URDF."""
    missing: list[str] = []
    for mesh_rel_path in mesh_paths:
        # Skip package:// or other URI styles.
        if "://" in mesh_rel_path:
            continue
        if not (urdf_dir / mesh_rel_path).exists():
            missing.append(mesh_rel_path)
    return missing


def _repair_mesh_paths_if_needed(urdf_path: Path) -> Path:
    """Repair common mesh path issues and return a URDF path to convert.

    Some URDF files reference meshes as ``assets/foo.stl``. If the URDF is
    itself already inside an ``assets/`` folder, this points to
    ``assets/assets/foo.stl`` and fails. In that case we write a patched URDF
    beside the original that uses ``foo.stl`` paths.
    """
    urdf_text = urdf_path.read_text(encoding="utf-8")
    mesh_paths = _extract_mesh_paths(urdf_text)
    if not mesh_paths:
        return urdf_path

    missing = _find_missing_meshes(urdf_path.parent, mesh_paths)
    if not missing:
        return urdf_path

    patched_text = urdf_text.replace('filename="assets/', 'filename="')
    if patched_text == urdf_text:
        missing_preview = ", ".join(missing[:5])
        raise FileNotFoundError(
            "SO-100 mesh files are missing relative to the URDF. "
            f"First missing entries: {missing_preview}"
        )

    patched_mesh_paths = _extract_mesh_paths(patched_text)
    patched_missing = _find_missing_meshes(urdf_path.parent, patched_mesh_paths)
    if patched_missing:
        missing_preview = ", ".join(patched_missing[:5])
        raise FileNotFoundError(
            "SO-100 mesh files are still missing after path repair. "
            "Place the mesh files beside the URDF or fix mesh paths in the "
            f"URDF. First missing entries: {missing_preview}"
        )

    patched_path = urdf_path.parent / SO100_PATCHED_URDF_FILENAME
    patched_path.write_text(patched_text, encoding="utf-8")
    return patched_path


def _strip_collision_blocks(urdf_text: str) -> str:
    """Remove collision tags to avoid expensive convex cooking at startup."""
    # Non-greedy block removal keeps visual and inertial sections intact.
    return re.sub(
        r"<collision>.*?</collision>",
        "",
        urdf_text,
        flags=re.DOTALL,
    )


def _create_fast_sandbox_urdf(urdf_path: Path) -> Path:
    """Create a URDF variant optimized for quick sandbox startup."""
    urdf_text = urdf_path.read_text(encoding="utf-8")
    fast_text = _strip_collision_blocks(urdf_text)
    fast_path = urdf_path.parent / SO100_FAST_SANDBOX_URDF_FILENAME
    fast_path.write_text(fast_text, encoding="utf-8")
    return fast_path


def ensure_so100_usd(
    asset_dir: Path,
    force_urdf_conversion: bool = False,
    use_fast_sandbox_urdf: bool = True,
) -> Path:
    """Convert SO-100 URDF to USD if needed and return USD path."""
    urdf_path = asset_dir / SO100_URDF_FILENAME
    if not urdf_path.exists():
        raise FileNotFoundError(f"SO-100 URDF not found: {urdf_path}")

    usd_path = asset_dir / SO100_USD_FILENAME
    if usd_path.exists() and not force_urdf_conversion:
        return usd_path

    urdf_for_conversion = _repair_mesh_paths_if_needed(urdf_path)
    if use_fast_sandbox_urdf:
        urdf_for_conversion = _create_fast_sandbox_urdf(urdf_for_conversion)

    converter_cfg = UrdfConverterCfg(
        asset_path=str(urdf_for_conversion),
        usd_dir=str(asset_dir),
        usd_file_name=SO100_USD_FILENAME,
        fix_base=True,
        merge_fixed_joints=True,
        force_usd_conversion=force_urdf_conversion,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
            target_type="none",
        ),
    )
    converter = UrdfConverter(converter_cfg)
    return Path(converter.usd_path)


SO100_ARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

SO100_GRIPPER_JOINT_NAMES = [
    "gripper",
]


def create_so100_rl_articulation_cfg(
    asset_dir: Path | None = None,
    force_urdf_conversion: bool = False,
) -> ArticulationCfg:
    """Create an ``ArticulationCfg`` for RL training with split actuators.

    Unlike :func:`create_so100_articulation_cfg`, this variant:
    - Uses the full-collision URDF (never strips collision meshes).
    - Separates arm and gripper into distinct actuator groups with
      tuned PD gains for manipulation tasks.
    """
    resolved_asset_dir = (
        asset_dir if asset_dir is not None
        else Path(__file__).resolve().parent
    )
    usd_path = ensure_so100_usd(
        asset_dir=resolved_asset_dir,
        force_urdf_conversion=force_urdf_conversion,
        use_fast_sandbox_urdf=False,
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
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=SO100_DEFAULT_JOINT_POS,
        ),
        actuators={
            "so100_arm": ImplicitActuatorCfg(
                joint_names_expr=SO100_ARM_JOINT_NAMES,
                effort_limit_sim=40.0,
                stiffness=20.0,
                damping=2.0,
            ),
            "so100_gripper": ImplicitActuatorCfg(
                joint_names_expr=SO100_GRIPPER_JOINT_NAMES,
                effort_limit_sim=20.0,
                stiffness=40.0,
                damping=10.0,
            ),
        },
    )


def create_so100_articulation_cfg(
    asset_dir: Path | None = None,
    force_urdf_conversion: bool = False,
    use_fast_sandbox_urdf: bool = True,
) -> ArticulationCfg:
    """Create an ``ArticulationCfg`` for the SO-100 arm."""
    resolved_asset_dir = (
        asset_dir if asset_dir is not None else Path(__file__).resolve().parent
    )
    usd_path = ensure_so100_usd(
        asset_dir=resolved_asset_dir,
        force_urdf_conversion=force_urdf_conversion,
        use_fast_sandbox_urdf=use_fast_sandbox_urdf,
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
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                sleep_threshold=0.005,
                stabilization_threshold=0.001,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=SO100_DEFAULT_JOINT_POS,
        ),
        actuators={
            "so100_arm": ImplicitActuatorCfg(
                joint_names_expr=list(SO100_DEFAULT_JOINT_POS.keys()),
                effort_limit_sim=40.0,
                stiffness=20.0,
                damping=2.0,
            ),
        },
    )
