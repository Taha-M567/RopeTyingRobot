"""SO-100 sandbox scene with camera-driven rope preprocessing.

This script:
1) Spawns the SO-100 arm from a local URDF asset.
2) Adds a table and articulated rope chain to the scene.
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
parser.add_argument(
    "--rope_pos",
    type=float,
    nargs=3,
    default=None,
    metavar=("X", "Y", "Z"),
    help="Rope spawn position as X Y Z (default: 0.25 0.0 0.05).",
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

import datetime

import carb.input
import cv2
import numpy as np
import omni.appwindow
import omni.ui as ui
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from rope_config import (
    ROPE_ALL_JOINT_NAMES,
    ROPE_NUM_SEGMENTS,
    create_rope_articulation_cfg,
    save_knot_config,
)
from generate_rope_urdf import _interpolate_color
from so100_config import create_so100_articulation_cfg
from src.perception.keypoint_detection import detect_keypoints
from src.perception.keypoint_mask import create_keypoint_class_mask
from src.perception.rope_segmentation import segment_rope
from src.perception.skeletonization import extract_path, skeletonize_rope
from src.perception.state_estimation import estimate_rope_state
from src.utils.config_loader import load_config


class RopeKeyboardController:
    """Move and bend the rope with keyboard keys during simulation.

    Translation (whole rope):
        I / K  — forward / backward  (X)
        J / L  — left / right        (Y)
        U / O  — up / down           (Z)

    Joint manipulation:
        , / .  — select previous / next hinge joint
        W / S  — bend pitch of selected joint  (+/-)
        A / D  — bend yaw of selected joint    (+/-)

    Other:
        R      — reset rope to spawn position and straighten all joints
        P      — save current rope joint angles as a knot config YAML
    """

    _NUM_HINGES = ROPE_NUM_SEGMENTS - 1
    _ANGLE_STEP = 0.174  # ~10 degrees per keypress

    def __init__(
        self,
        scene: "InteractiveScene",
        default_pos: tuple[float, ...],
        step_size: float = 0.02,
    ):
        self._scene = scene
        self._step = step_size
        self._default_pos = default_pos
        self._offset = [0.0, 0.0, 0.0]
        self._reset = False
        self._save_config = False

        # Joint bending state: track target angles for every joint.
        num_joints = self._NUM_HINGES * 2  # pitch + yaw per hinge
        self._joint_targets = [0.0] * num_joints
        self._selected_hinge = 0
        self._joints_dirty = False

        # Hinge selection marker — bright magenta sphere, invisible to cameras.
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/SelectedHinge",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.012,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 1.0),
                        emissive_color=(0.5, 0.0, 0.5),
                    ),
                ),
            },
        )
        self._hinge_marker = VisualizationMarkers(marker_cfg)

        # Controls overlay HUD.
        self._hinge_label: ui.Label | None = None
        self._build_overlay()

        app_window = omni.appwindow.get_default_app_window()
        keyboard = app_window.get_keyboard()
        self._input = carb.input.acquire_input_interface()
        self._sub = self._input.subscribe_to_keyboard_events(keyboard, self._on_key)

    def _build_overlay(self) -> None:
        """Create a floating HUD window with the control reference."""
        self._overlay = ui.Window(
            "Rope Controls",
            width=260,
            height=310,
        )
        _title = {"font_size": 14, "color": 0xFF00DDFF}  # cyan
        _key = {"font_size": 13}
        _status = {"font_size": 14, "color": 0xFFFF55FF}  # magenta
        with self._overlay.frame:
            with ui.VStack(spacing=3):
                ui.Spacer(height=2)
                ui.Label("  MOVE ROPE", style=_title)
                ui.Label("    I / K      Forward / Back  (X)", style=_key)
                ui.Label("    J / L      Left / Right    (Y)", style=_key)
                ui.Label("    U / O      Up / Down       (Z)", style=_key)
                ui.Spacer(height=4)
                ui.Label("  BEND JOINTS", style=_title)
                ui.Label("    , / .       Prev / Next hinge", style=_key)
                ui.Label("    W / S      Pitch  +/-", style=_key)
                ui.Label("    A / D      Yaw    +/-", style=_key)
                ui.Spacer(height=4)
                ui.Label("  OTHER", style=_title)
                ui.Label("    R            Reset all", style=_key)
                ui.Label("    P            Save knot config", style=_key)
                ui.Spacer(height=6)
                self._hinge_label = ui.Label(
                    f"  Hinge: 0 / {self._NUM_HINGES - 1}",
                    style=_status,
                )

    def _update_hinge_label(self) -> None:
        if self._hinge_label is not None:
            self._hinge_label.text = f"  Hinge: {self._selected_hinge} / {self._NUM_HINGES - 1}"

    def _on_key(self, event: carb.input.KeyboardEvent, *args, **kwargs) -> bool:
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        k = event.input

        # --- whole-rope translation ---
        if k == carb.input.KeyboardInput.I:
            self._offset[0] += self._step
        elif k == carb.input.KeyboardInput.K:
            self._offset[0] -= self._step
        elif k == carb.input.KeyboardInput.J:
            self._offset[1] += self._step
        elif k == carb.input.KeyboardInput.L:
            self._offset[1] -= self._step
        elif k == carb.input.KeyboardInput.U:
            self._offset[2] += self._step
        elif k == carb.input.KeyboardInput.O:
            self._offset[2] -= self._step

        # --- joint selection ---
        elif k == carb.input.KeyboardInput.COMMA:
            self._selected_hinge = max(0, self._selected_hinge - 1)
            self._update_hinge_label()
        elif k == carb.input.KeyboardInput.PERIOD:
            self._selected_hinge = min(self._NUM_HINGES - 1, self._selected_hinge + 1)
            self._update_hinge_label()

        # --- joint bending ---
        elif k == carb.input.KeyboardInput.W:
            idx = self._selected_hinge * 2  # pitch
            self._joint_targets[idx] += self._ANGLE_STEP
            self._joints_dirty = True
        elif k == carb.input.KeyboardInput.S:
            idx = self._selected_hinge * 2
            self._joint_targets[idx] -= self._ANGLE_STEP
            self._joints_dirty = True
        elif k == carb.input.KeyboardInput.A:
            idx = self._selected_hinge * 2 + 1  # yaw
            self._joint_targets[idx] += self._ANGLE_STEP
            self._joints_dirty = True
        elif k == carb.input.KeyboardInput.D:
            idx = self._selected_hinge * 2 + 1
            self._joint_targets[idx] -= self._ANGLE_STEP
            self._joints_dirty = True

        # --- reset ---
        elif k == carb.input.KeyboardInput.R:
            self._reset = True

        # --- save knot config ---
        elif k == carb.input.KeyboardInput.P:
            self._save_config = True

        return True

    def apply(self) -> None:
        """Call once per sim step to apply any pending rope changes."""
        rope = self._scene["rope"]

        # Full reset: root position + orientation + all joint angles.
        if self._reset:
            state = rope.data.root_state_w.clone()
            state[:, 0] = self._default_pos[0]
            state[:, 1] = self._default_pos[1]
            state[:, 2] = self._default_pos[2]
            state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0])  # identity quat
            state[:, 7:] = 0.0  # zero velocities
            rope.write_root_state_to_sim(state)

            # Snap all joint positions and velocities to zero immediately.
            n = rope.num_joints
            rope.write_joint_state_to_sim(
                torch.zeros(1, n, device=rope.device),
                torch.zeros(1, n, device=rope.device),
            )

            self._joint_targets = [0.0] * (self._NUM_HINGES * 2)
            self._joints_dirty = True
            self._reset = False

        # Translate whole rope.
        dx, dy, dz = self._offset
        if dx or dy or dz:
            state = rope.data.root_state_w.clone()
            state[:, 0] += dx
            state[:, 1] += dy
            state[:, 2] += dz
            state[:, 7:] = 0.0
            rope.write_root_state_to_sim(state)
            self._offset = [0.0, 0.0, 0.0]

        # Set joint position targets so the actuator drives bends.
        if self._joints_dirty:
            targets = torch.tensor(
                [self._joint_targets],
                dtype=torch.float32,
                device=rope.device,
            )
            rope.set_joint_position_target(targets)
            self._joints_dirty = False

        # Save current joint angles as a knot config YAML.
        if self._save_config:
            joint_pos = rope.data.joint_pos[0]  # (num_joints,)
            angles: dict[str, float] = {}
            for i, name in enumerate(ROPE_ALL_JOINT_NAMES):
                angles[name] = joint_pos[i].item()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"knot_{timestamp}.yaml"
            path = save_knot_config(
                joint_angles=angles,
                path=filename,
                name=f"knot_{timestamp}",
                description="Saved from sandbox keyboard controller.",
            )
            logger.info("Knot config saved → %s", path)
            self._save_config = False

        # Update hinge selection marker.
        # Body order: seg_0, hinge_0, seg_1, hinge_1, ...
        # So hinge i is at body index 2*i + 1.
        hinge_body_idx = self._selected_hinge * 2 + 1
        hinge_pos = rope.data.body_pos_w[:1, hinge_body_idx, :]  # (1, 3)
        self._hinge_marker.visualize(translations=hinge_pos)



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
                diffuse_color=(0.02, 0.02, 0.02),
                roughness=1.0,
                metallic=0.0,
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
            pos=(0.25, 0.0, 0.50),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )

    # Side-view camera: observes arm pose relative to table and rope.
    # Mounted on robot base (fixed at origin, so offset = world position).
    # Looks along +Y toward the scene; right = +X, up = +Z.
    side_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/side_camera",
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
            pos=(0.25, -0.60, 0.15),
            rot=(0.7071, 0.7071, 0.0, 0.0),
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
                diffuse_color=(0.05, 0.05, 0.05),
                roughness=1.0,
                metallic=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.25, 0.0, -0.02)),
    )

    rope: ArticulationCfg = MISSING


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


def _compose_dual_view(
    top_down: np.ndarray,
    side_view: np.ndarray | None,
) -> np.ndarray:
    """Horizontally concatenate top-down and side-view camera frames.

    The side view is resized to match the top-down height.  Text labels
    are drawn on each half.  Returns *top_down* unchanged when
    *side_view* is ``None`` (camera not yet ready).
    """
    label_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(top_down, "Top-Down", (10, 25), font, 0.6, label_color, 2)

    if side_view is None:
        return top_down

    h_td = top_down.shape[0]
    h_sv, w_sv = side_view.shape[:2]
    if h_sv != h_td:
        scale = h_td / h_sv
        side_view = cv2.resize(side_view, (int(w_sv * scale), h_td))

    cv2.putText(side_view, "Side View", (10, 25), font, 0.6, label_color, 2)

    return np.hstack([top_down, side_view])


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


def _apply_rope_colors(sim: sim_utils.SimulationContext) -> None:
    """Apply Green->Yellow->Red->Blue gradient PreviewSurface materials to rope segments."""
    stage = sim.stage
    num_segments = ROPE_NUM_SEGMENTS
    num_hinges = num_segments - 1

    # Apply materials to segment links (cylinders).
    for i in range(num_segments):
        t = i / max(num_segments - 1, 1)
        r, g, b, _ = _interpolate_color(t)
        mat_path = f"/World/Looks/rope_mat_{i}"
        prim_path = f"/World/envs/env_0/Rope/rope_seg_{i}"
        mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(r, g, b))
        mat_cfg.func(mat_path, mat_cfg)
        sim_utils.bind_visual_material(prim_path, mat_path)

    # Apply matching materials to hinge links (spheres).
    for i in range(num_hinges):
        t = i / max(num_segments - 1, 1)
        r, g, b, _ = _interpolate_color(t)
        mat_path = f"/World/Looks/rope_hinge_mat_{i}"
        prim_path = f"/World/envs/env_0/Rope/rope_hinge_{i}"
        mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(r, g, b))
        mat_cfg.func(mat_path, mat_cfg)
        sim_utils.bind_visual_material(prim_path, mat_path)

    logger.info(
        "Applied %d gradient materials (Green->Yellow->Red->Blue) to rope.",
        num_segments + num_hinges,
    )


def _spawn_camera_fov_prims(
    sim: sim_utils.SimulationContext,
    cam_width: int,
    cam_height: int,
) -> None:
    """Spawn permanent USD sphere prims outlining camera frustums.

    Must be called BEFORE ``sim.reset()`` so the Fabric Scene Delegate
    picks up the new geometry.  Uses the known camera config poses rather
    than sensor data (which is not populated until after reset).

    Spawns frustum markers for both the top-down and side-view cameras.
    """
    from pxr import Gf, UsdGeom

    # Shared camera intrinsics (must match PinholeCameraCfg in scene config).
    focal_length = 24.0
    h_aperture = 20.955
    w, h = cam_width, cam_height
    fx = focal_length * w / h_aperture
    fy = fx
    cx, cy = w / 2.0, h / 2.0

    stage = sim.stage
    n_per = 8
    table_z = 0.01

    def _spawn_single_fov(
        cam_pos: np.ndarray,
        R: np.ndarray,
        xform_path: str,
        mat_name: str,
        diffuse: tuple[float, float, float],
        emissive: tuple[float, float, float],
        depth: float = 0.40,
    ) -> int:
        """Spawn frustum sphere prims for one camera. Returns prim count."""
        # Normalised ray directions for 4 image corners (OpenGL: -Z forward).
        corners_px = [(0, 0), (w, 0), (w, h), (0, h)]
        ray_dirs: list[np.ndarray] = []
        for u, v in corners_px:
            rc = np.array([(u - cx) / fx, -(v - cy) / fy, -1.0])
            rc /= np.linalg.norm(rc)
            ray_dirs.append(R @ rc)

        far_pts = [cam_pos + depth * d for d in ray_dirs]

        pts: list[np.ndarray] = []

        # Camera → each far corner (4 edges).
        for fp in far_pts:
            for j in range(n_per + 1):
                t = j / n_per
                pts.append(cam_pos * (1 - t) + fp * t)

        # Far-plane rectangle (4 edges).
        for i in range(4):
            p0, p1 = far_pts[i], far_pts[(i + 1) % 4]
            for j in range(1, n_per):
                t = j / n_per
                pts.append(p0 * (1 - t) + p1 * t)

        # Table-surface footprint.
        table_pts: list[np.ndarray] = []
        for d in ray_dirs:
            if abs(d[2]) > 1e-6:
                tv = (table_z - cam_pos[2]) / d[2]
                if tv > 0:
                    table_pts.append(cam_pos + tv * d)
        if len(table_pts) >= 2:
            for i in range(len(table_pts)):
                p0 = table_pts[i]
                p1 = table_pts[(i + 1) % len(table_pts)]
                for j in range(n_per + 1):
                    t = j / n_per
                    pts.append(p0 * (1 - t) + p1 * t)

        # Material.
        mat_path = f"/World/Looks/{mat_name}"
        mat_cfg = sim_utils.PreviewSurfaceCfg(
            diffuse_color=diffuse,
            emissive_color=emissive,
        )
        mat_cfg.func(mat_path, mat_cfg)

        # Spawn spheres under a guide-purpose Xform.
        xform = UsdGeom.Xform.Define(stage, xform_path)
        xform.GetPurposeAttr().Set("guide")
        for i, pt in enumerate(pts):
            pp = f"{xform_path}/s{i}"
            sphere = UsdGeom.Sphere.Define(stage, pp)
            sphere.GetRadiusAttr().Set(0.01)
            sphere.AddTranslateOp().Set(
                Gf.Vec3d(float(pt[0]), float(pt[1]), float(pt[2]))
            )
            sim_utils.bind_visual_material(pp, mat_path)

        return len(pts)

    # --- Top-down camera (cyan) ---
    # pos=(0.25, 0.0, 0.50), identity rotation → looks along -Z.
    top_pos = np.array([0.25, 0.0, 0.50])
    top_R = np.eye(3)
    n = _spawn_single_fov(
        top_pos, top_R,
        "/World/CameraFOV_TopDown", "fov_mat_top",
        diffuse=(0.0, 1.0, 1.0), emissive=(0.0, 0.5, 0.5),
    )
    logger.info(
        "Top-down FOV: %d prims at (%.3f, %.3f, %.3f).",
        n, *top_pos,
    )

    # --- Side camera (orange) ---
    # pos=(0.25, -0.60, 0.15), 90° about X → looks along +Y.
    # OpenGL rotation: right=+X, up=+Z, -forward=-Y.
    side_pos = np.array([0.25, -0.60, 0.15])
    side_R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ])
    n = _spawn_single_fov(
        side_pos, side_R,
        "/World/CameraFOV_Side", "fov_mat_side",
        diffuse=(1.0, 0.6, 0.0), emissive=(0.5, 0.3, 0.0),
        depth=0.50,
    )
    logger.info(
        "Side FOV: %d prims at (%.3f, %.3f, %.3f).",
        n, *side_pos,
    )


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene,
    perception_cfg: dict[str, Any],
    rope_controller: RopeKeyboardController | None = None,
) -> None:
    """Run simulation and process camera frames through perception pipeline."""
    output_dir = _resolve_path(args_cli.output_dir, EXT_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    sim_dt = sim.get_physics_dt()
    step_count = 0

    while simulation_app.is_running():
        if args_cli.max_steps > 0 and step_count >= args_cli.max_steps:
            break

        # Apply any pending keyboard-driven rope movement.
        if rope_controller is not None:
            rope_controller.apply()

        # Keep the arm in default pose for stable image testing.
        targets = scene["robot"].data.default_joint_pos
        scene["robot"].set_joint_position_target(targets)
        scene.write_data_to_sim()

        sim.step()
        scene.update(sim_dt)

        try:
            camera_output = scene["camera"].data.output
        except RuntimeError:
            # Camera buffers are not yet allocated on early frames.
            step_count += 1
            continue
        if "rgb" not in camera_output:
            step_count += 1
            continue

        rgb = camera_output["rgb"][0]
        if rgb.numel() == 0:
            step_count += 1
            continue

        frame_bgr = _tensor_rgb_to_bgr(rgb)
        vis, metrics = _run_perception_pipeline(frame_bgr, perception_cfg)

        # Side camera — raw RGB for arm pose observation (no perception).
        side_frame_bgr = None
        try:
            side_output = scene["side_camera"].data.output
            if "rgb" in side_output:
                side_rgb = side_output["rgb"][0]
                if side_rgb.numel() > 0:
                    side_frame_bgr = _tensor_rgb_to_bgr(side_rgb)
        except RuntimeError:
            pass  # Side camera buffers not yet allocated.

        composite = _compose_dual_view(vis, side_frame_bgr)

        if step_count % max(1, args_cli.log_interval) == 0:
            rope_pos = scene["rope"].data.body_pos_w
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
            cv2.imwrite(str(frame_path), composite)

        if args_cli.show:
            # Write latest frame for the external viewer process.
            latest_path = output_dir / "latest.png"
            cv2.imwrite(str(latest_path), composite)

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
    sim.set_camera_view(eye=[0.5, 0.5, 0.6], target=[0.25, 0.0, 0.02])

    rope_cfg = create_rope_articulation_cfg(
        asset_dir=ASSET_DIR,
        force_conversion=args_cli.force_urdf_conversion,
    )

    scene_cfg = So100PerceptionSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=2.5,
        replicate_physics=True,
    )
    scene_cfg.robot = so100_cfg.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Override rope spawn position if provided via CLI.
    if args_cli.rope_pos is not None:
        from isaaclab.assets import ArticulationCfg as _ArtCfg

        rope_cfg = rope_cfg.replace(
            init_state=_ArtCfg.InitialStateCfg(
                pos=tuple(args_cli.rope_pos),
                joint_pos=rope_cfg.init_state.joint_pos,
            ),
        )
    scene_cfg.rope = rope_cfg.replace(prim_path="{ENV_REGEX_NS}/Rope")
    scene = InteractiveScene(scene_cfg)

    _apply_rope_colors(sim)
    _spawn_camera_fov_prims(sim, args_cli.camera_width, args_cli.camera_height)
    sim.reset()

    # Warm-up: camera sensor needs scene.update() to populate data.
    for _ in range(3):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())

    # Debug: verify rope exists and dump initial body positions.
    rope_art = scene["rope"]
    rope_pos = rope_art.data.body_pos_w
    logger.info(
        "Rope debug: num_bodies=%d, shape=%s",
        rope_art.num_bodies, rope_pos.shape,
    )
    logger.info(
        "Rope body positions (first env):\n%s", rope_pos[0],
    )
    logger.info("SO-100 + camera scene initialized. Running simulation...")

    # Resolve the default rope position for the keyboard controller.
    rope_default_pos = rope_cfg.init_state.pos
    rope_ctrl = RopeKeyboardController(scene, default_pos=rope_default_pos)
    logger.info(
        "Rope keyboard controls active: I/K=X, J/L=Y, U/O=Z, R=reset"
    )

    run_simulator(sim, scene, perception_cfg, rope_controller=rope_ctrl)


if __name__ == "__main__":
    main()
    # Isaac Sim 5.1 + RTX 50-series can crash inside omni.syntheticdata
    # during shutdown.  Suppress the segfault so the process exits cleanly
    # after the simulation has already finished.
    try:
        simulation_app.close(wait_for_replicator=False)
    except Exception:
        pass
