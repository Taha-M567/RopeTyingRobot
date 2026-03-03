"""SO-100 sandbox scene with camera-driven rope preprocessing.

This script:
1) Spawns the SO-100 arm from a local URDF asset.
2) Adds a table and deformable rope to the scene.
3) Adds a camera sensor in the Isaac Lab scene.
4) Runs the existing OpenCV rope preprocessing pipeline on rendered RGB frames.

RTX 50-series (Blackwell) GPU workarounds
==========================================
Isaac Sim 5.1 has known rendering issues on RTX 50-series GPUs (5060 Ti,
5070, 5080, 5090). Symptoms include hangs, black screens, and unresponsive
viewports. NVIDIA has acknowledged this across Isaac Sim 4.5–5.1.

Diagnostic / workaround flags (try in order):

  1. --headless        Bypass the viewport entirely. If this works, the
                       simulation core is fine and the problem is renderer +
                       GPU.

  2. Vulkan backend    Use the Vulkan rendering backend instead of the
                       default. Pass via --kit_args:
                         python so100_sandbox.py --kit_args "--/renderer/active=vulkan"

  3. --livestream 1    Run headless but stream the viewport over WebRTC so
                       you can view it in a browser at http://localhost:8211.

  4. Upgrade driver    Ensure you have the latest Game-Ready / Studio driver
                       from NVIDIA (>=580.88 required, latest recommended).

See: IsaacLab GitHub issues #1888, #3466, #3564 for discussion.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import MISSING
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher

# Resolve project paths before loading the rest of the stack.
SCRIPT_PATH = Path(__file__).resolve()
EXT_ROOT = SCRIPT_PATH.parents[1]
ASSET_DIR = EXT_ROOT / "assets"
REPO_ROOT = SCRIPT_PATH.parents[3]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ASSET_DIR) not in sys.path:
    sys.path.insert(0, str(ASSET_DIR))

# Use a writable local runtime root for Kit settings/cache to avoid permission
# issues when Isaac Sim is installed in read-only site-packages locations.
RUNTIME_ROOT = (EXT_ROOT / ".kit_runtime").resolve()
RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)
(RUNTIME_ROOT / "optix_cache").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("OPTIX_CACHE_PATH", str((RUNTIME_ROOT / "optix_cache").resolve()))

# Always ensure --portable-root is in kit args so config/cache go to
# .kit_runtime.  If the user also supplies --kit_args, prepend ours.
_portable_root_arg = f"--portable-root {RUNTIME_ROOT}"
if "--kit_args" in sys.argv:
    _kit_idx = sys.argv.index("--kit_args")
    if _kit_idx + 1 < len(sys.argv):
        sys.argv[_kit_idx + 1] = (
            f"{_portable_root_arg} {sys.argv[_kit_idx + 1]}"
        )
    else:
        sys.argv.append(_portable_root_arg)
else:
    sys.argv += ["--kit_args", _portable_root_arg]


def _get_dist_version(dist_name: str) -> str | None:
    """Return an installed distribution version, if available."""
    try:
        return importlib_metadata.version(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _assert_python_env_compatibility() -> None:
    """Fail fast for known Isaac Sim package conflicts."""
    isaacsim_version = _get_dist_version("isaacsim")
    numpy_version = _get_dist_version("numpy")
    if (
        isaacsim_version is not None
        and isaacsim_version.startswith("5.1")
        and numpy_version is not None
    ):
        numpy_major = int(numpy_version.split(".", maxsplit=1)[0])
        if numpy_major >= 2:
            raise RuntimeError(
                "Incompatible environment detected for Isaac Sim 5.1: "
                f"numpy=={numpy_version}. "
                "This can crash camera startup in omni.syntheticdata. "
                "Fix in env_isaaclab with: "
                "pip install --force-reinstall \"numpy==1.26.4\""
            )


parser = argparse.ArgumentParser(
    description="Spawn SO-100 with camera and run rope preprocessing per frame."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable Fabric API and use USD I/O operations.",
)
parser.add_argument(
    "--camera_width",
    type=int,
    default=640,
    help="Camera image width.",
)
parser.add_argument(
    "--camera_height",
    type=int,
    default=480,
    help="Camera image height.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Max simulation steps (0 means run until app closes).",
)
parser.add_argument(
    "--log_interval",
    type=int,
    default=30,
    help="Print pipeline summary every N steps.",
)
parser.add_argument(
    "--save_interval",
    type=int,
    default=120,
    help="Save visualization image every N steps (0 disables saving).",
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="output/so100_preprocess",
    help="Output directory for saved pipeline images.",
)
parser.add_argument(
    "--show",
    action="store_true",
    default=False,
    help="Show a live OpenCV preview window.",
)
parser.add_argument(
    "--force_urdf_conversion",
    action="store_true",
    default=False,
    help="Force URDF->USD conversion even if USD already exists.",
)
parser.add_argument(
    "--full_collision",
    action="store_true",
    default=False,
    help=(
        "Keep collision meshes from URDF during conversion. "
        "Default uses a faster sandbox URDF without collision blocks."
    ),
)
parser.add_argument(
    "--perception_config",
    type=str,
    default="src/configs/perception_config.yaml",
    help="Path to perception YAML config.",
)
parser.add_argument(
    "--skip_env_check",
    action="store_true",
    default=False,
    help="Skip startup checks for known Isaac Sim dependency conflicts.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Validate dependency compatibility before launching Kit.
if not args_cli.skip_env_check:
    _assert_python_env_compatibility()
# The camera pipeline requires render products.
args_cli.enable_cameras = True

# Launch Omniverse app.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import cv2
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from so100_config import create_so100_articulation_cfg
from src.perception.keypoint_detection import detect_keypoints
from src.perception.keypoint_mask import create_keypoint_class_mask
from src.perception.rope_segmentation import segment_rope
from src.perception.skeletonization import extract_path, skeletonize_rope
from src.perception.state_estimation import estimate_rope_state
from src.utils.config_loader import load_config


logger = logging.getLogger(__name__)
# Isaac Lab reconfigures the root logger after AppLauncher init, so
# basicConfig is a no-op by this point.  Attach a handler directly.
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)


@configclass
class So100PerceptionSceneCfg(InteractiveSceneCfg):
    """Scene containing SO-100 arm and an RGB camera."""

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.CuboidCfg(
            size=(4.0, 4.0, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.18, 0.18, 0.18),
                roughness=0.9,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025)),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=3000.0,
            color=(0.85, 0.85, 0.85),
        ),
    )

    robot: ArticulationCfg = MISSING

    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/sim_camera",
        update_period=0.0,
        height=args_cli.camera_height,
        width=args_cli.camera_width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.22, 0.0, 0.10),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.4, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.45, 0.35, 0.25),
                roughness=0.8,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, 0.0, -0.02)),
    )

    rope = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Rope",
        spawn=sim_utils.MeshCylinderCfg(
            radius=0.00175,
            height=0.45,
            axis="X",
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(
                self_collision=True,
                solver_position_iteration_count=32,
                vertex_velocity_damping=0.05,
                simulation_hexahedral_resolution=10,
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.DeformableBodyMaterialCfg(
                density=700.0,
                youngs_modulus=2000000.0,
                poissons_ratio=0.40,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 1.0, 1.0),
                roughness=0.7,
            ),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(0.25, 0.0, 0.05),
        ),
    )


def _resolve_path(path_str: str, base_dir: Path) -> Path:
    """Resolve relative paths against a base directory."""
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _tensor_rgb_to_bgr(rgb_tensor: torch.Tensor) -> np.ndarray:
    """Convert Isaac Lab camera RGB tensor to OpenCV BGR image."""
    rgb_np = rgb_tensor.detach().cpu().numpy()
    if rgb_np.shape[-1] == 4:
        rgb_np = rgb_np[..., :3]

    if rgb_np.dtype != np.uint8:
        max_value = float(np.max(rgb_np)) if rgb_np.size > 0 else 0.0
        if max_value <= 1.0:
            rgb_np = np.clip(rgb_np * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            rgb_np = np.clip(rgb_np, 0.0, 255.0).astype(np.uint8)

    return cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)


def _draw_pipeline_overlay(
    frame_bgr: np.ndarray,
    rope_mask: np.ndarray,
    keypoints: list[Any],
    path_graph: dict[str, Any] | None,
    fallback_path: np.ndarray,
) -> np.ndarray:
    """Draw segmentation, keypoints, and path information."""
    vis = frame_bgr.copy()

    mask_region = rope_mask > 0
    if np.any(mask_region):
        overlay = vis.copy()
        overlay[mask_region] = (0, 120, 255)
        vis = cv2.addWeighted(vis, 0.75, overlay, 0.25, 0.0)

    for kp in keypoints:
        x = int(round(kp.position[0]))
        y = int(round(kp.position[1]))
        color = (0, 255, 0) if kp.keypoint_type == "endpoint" else (0, 0, 255)
        radius = 4 if kp.keypoint_type == "endpoint" else 6
        cv2.circle(vis, (x, y), radius, color, -1)

    edges = []
    if path_graph is not None:
        edges = path_graph.get("edges", [])
    if edges:
        for edge in edges:
            if edge is None or len(edge) < 2:
                continue
            points = edge.astype(np.int32)
            for idx in range(len(points) - 1):
                cv2.line(
                    vis,
                    tuple(points[idx]),
                    tuple(points[idx + 1]),
                    (255, 50, 50),
                    2,
                )
    elif fallback_path is not None and len(fallback_path) >= 2:
        points = fallback_path.astype(np.int32)
        for idx in range(len(points) - 1):
            cv2.line(
                vis,
                tuple(points[idx]),
                tuple(points[idx + 1]),
                (255, 50, 50),
                2,
            )

    return vis


def _run_perception_pipeline(
    frame_bgr: np.ndarray,
    perception_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run segmentation, keypoints, skeletonization, and state estimation."""
    pipeline_cfg = perception_cfg.get("pipeline", {})
    segmentation_cfg = perception_cfg.get("segmentation", {})
    keypoint_cfg = perception_cfg.get("keypoint_detection", {})
    keypoint_mask_cfg = perception_cfg.get("keypoint_mask", {})
    skeleton_cfg = perception_cfg.get("skeletonization", {})

    disable_keypoints = bool(pipeline_cfg.get("disable_keypoint_extraction", False))
    disable_skeleton = bool(pipeline_cfg.get("disable_skeletonization", False))

    rope_mask_obj = segment_rope(frame_bgr, config=segmentation_cfg)
    rope_mask = rope_mask_obj.mask

    if disable_keypoints:
        keypoints = []
    else:
        keypoints = detect_keypoints(rope_mask, config=keypoint_cfg)

    _ = create_keypoint_class_mask(rope_mask, keypoints, config=keypoint_mask_cfg)

    if disable_skeleton:
        path = np.array([], dtype=np.float32).reshape(0, 2)
    else:
        skeleton = skeletonize_rope(rope_mask, config=skeleton_cfg)
        path = extract_path(skeleton)

    rope_state = estimate_rope_state(keypoints, path)

    vis = _draw_pipeline_overlay(
        frame_bgr=frame_bgr,
        rope_mask=rope_mask,
        keypoints=keypoints,
        path_graph=rope_state.path_graph,
        fallback_path=rope_state.path,
    )

    metrics = {
        "confidence": float(rope_mask_obj.confidence),
        "endpoints": len(rope_state.endpoints),
        "crossings": len(rope_state.crossings),
        "path_points": int(len(rope_state.path)),
    }
    return vis, metrics


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    perception_cfg: dict[str, Any],
) -> None:
    """Run simulation and process camera frames through perception pipeline."""
    output_dir = _resolve_path(args_cli.output_dir, EXT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_dt = sim.get_physics_dt()
    step_count = 0

    while simulation_app.is_running():
        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            break

        # Keep the arm in default pose for stable image testing.
        targets = scene["robot"].data.default_joint_pos
        scene["robot"].set_joint_position_target(targets)
        scene.write_data_to_sim()

        sim.step()
        scene.update(sim_dt)

        camera_output = scene["camera"].data.output
        if "rgb" not in camera_output:
            step_count += 1
            continue

        rgb = camera_output["rgb"][0]
        if rgb.numel() == 0:
            step_count += 1
            continue

        frame_bgr = _tensor_rgb_to_bgr(rgb)
        vis, metrics = _run_perception_pipeline(frame_bgr, perception_cfg)

        if step_count % max(1, args_cli.log_interval) == 0:
            rope_pos = scene["rope"].data.nodal_pos_w
            rope_center = rope_pos.mean(dim=1).squeeze(0)
            logger.info(
                "step=%d conf=%.3f endpoints=%d crossings=%d"
                " path_points=%d rope_center=(%.3f, %.3f, %.3f)",
                step_count,
                metrics["confidence"],
                metrics["endpoints"],
                metrics["crossings"],
                metrics["path_points"],
                rope_center[0].item(),
                rope_center[1].item(),
                rope_center[2].item(),
            )

        if args_cli.save_interval > 0 and step_count % args_cli.save_interval == 0:
            frame_path = output_dir / f"frame_{step_count:06d}.png"
            cv2.imwrite(str(frame_path), vis)

        if args_cli.show:
            # Write latest frame for the external viewer process.
            latest_path = output_dir / "latest.png"
            cv2.imwrite(str(latest_path), vis)

        step_count += 1


def main() -> None:
    """Main entrypoint."""
    perception_cfg_path = _resolve_path(args_cli.perception_config, REPO_ROOT)
    if not perception_cfg_path.exists():
        raise FileNotFoundError(
            f"Perception config not found: {perception_cfg_path}"
        )

    logger.info("Loading perception config from %s", perception_cfg_path)
    config = load_config(perception_cfg_path)
    perception_cfg = config.get("perception", {})

    logger.info(
        "Preparing SO-100 USD (fast_sandbox=%s, force_conversion=%s)",
        not args_cli.full_collision,
        args_cli.force_urdf_conversion,
    )
    so100_cfg = create_so100_articulation_cfg(
        asset_dir=ASSET_DIR,
        force_urdf_conversion=args_cli.force_urdf_conversion,
        use_fast_sandbox_urdf=not args_cli.full_collision,
    )

    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        device=args_cli.device,
        use_fabric=not args_cli.disable_fabric,
    )
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.0, 1.0, 0.8], target=[0.0, 0.0, 0.2])

    scene_cfg = So100PerceptionSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.5,
        replicate_physics=False,
    )
    scene_cfg.robot = so100_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    logger.info("SO-100 + camera scene initialized. Running simulation...")

    run_simulator(sim, scene, perception_cfg)


if __name__ == "__main__":
    main()
    # Isaac Sim 5.1 + RTX 50-series can crash inside omni.syntheticdata
    # during shutdown.  Suppress the segfault so the process exits cleanly
    # after the simulation has already finished.
    try:
        simulation_app.close(wait_for_replicator=False)
    except Exception:
        pass
