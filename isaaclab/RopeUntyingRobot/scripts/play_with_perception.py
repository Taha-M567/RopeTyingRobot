"""Roll out a trained policy while replacing GT endpoint obs with perception output.

Validates the sim-to-real perception bridge before any real hardware is
involved.  Uses the same scene as ``so100_sandbox.py`` (top-down camera
+ rope), loads the JIT-exported policy from ``logs/.../exported/policy.pt``,
and at each control tick:

  1. Reads the top-down camera RGB.
  2. Runs the OpenCV perception pipeline.
  3. Picks the two highest-confidence endpoints, back-projects them to
     world XY on the table plane.
  4. Splices those 6 values into the 24-D observation vector in place
     of the ground-truth ``rope_endpoint_pos`` slice.
  5. Steps the policy → applies the action.

Reports the median final-step distance from the EE to the nearer rope
endpoint across all completed episodes.

Usage::

    python scripts/play_with_perception.py \\
        --task RopeReachEnd-SO100-v0 \\
        --policy logs/rsl_rl/rope_reach_so100/<run>/exported/policy.pt \\
        --num_envs 4 --num_episodes 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

SCRIPT_PATH = Path(__file__).resolve()
EXT_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

parser = argparse.ArgumentParser(
    description="Play a trained policy with perception-derived endpoints.",
)
parser.add_argument(
    "--task",
    type=str,
    default="RopeReachEnd-SO100-v0",
    help="Gym task ID.",
)
parser.add_argument(
    "--policy",
    type=str,
    required=True,
    help="Path to the JIT-exported policy (logs/.../exported/policy.pt).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=4,
    help="Number of parallel environments.",
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=20,
    help="Stop after this many completed episodes across all envs.",
)
parser.add_argument(
    "--perception_config",
    type=str,
    default="src/configs/perception_config.yaml",
    help="Path to perception YAML config.",
)
parser.add_argument(
    "--camera_width", type=int, default=640, help="Sim camera width.",
)
parser.add_argument(
    "--camera_height", type=int, default=480, help="Sim camera height.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
from statistics import median

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils.assets import retrieve_file_path

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import RopeUntyingRobot.tasks  # noqa: F401
from perception_runner import (  # noqa: E402  — local helper
    run_perception,
    select_two_endpoints,
    tensor_rgb_to_bgr,
)
from src.utils.config_loader import load_config  # noqa: E402
from src.utils.geometry import CameraPose, pixel_to_table_world  # noqa: E402

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"),
    )
    logger.addHandler(_handler)
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Top-down sim camera — must mirror so100_sandbox.py exactly so the policy
# sees the same view it would see at deployment.
# ---------------------------------------------------------------------------
_TOP_DOWN_CAM_POS = (0.25, 0.0, 0.50)
_TOP_DOWN_FOCAL_LENGTH = 24.0  # mm
_TOP_DOWN_HORIZONTAL_APERTURE = 20.955  # mm
_TABLE_Z = 0.0  # rope rests at z ≈ 0 on the table top.


def _top_down_camera_cfg(width: int, height: int) -> CameraCfg:
    return CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/perception_cam",
        update_period=0.0,
        height=height,
        width=width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_TOP_DOWN_FOCAL_LENGTH,
            focus_distance=400.0,
            horizontal_aperture=_TOP_DOWN_HORIZONTAL_APERTURE,
            clipping_range=(0.05, 10.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=_TOP_DOWN_CAM_POS,
            rot=(1.0, 0.0, 0.0, 0.0),  # identity in ROS convention.
            convention="ros",
        ),
    )


def _build_top_down_intrinsics(width: int, height: int) -> np.ndarray:
    fx = _TOP_DOWN_FOCAL_LENGTH * width / _TOP_DOWN_HORIZONTAL_APERTURE
    fy = fx
    return np.array(
        [
            [fx, 0.0, width / 2.0],
            [0.0, fy, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _build_top_down_pose() -> CameraPose:
    # Camera looks straight down: +Z_cam = −Z_world, +X_cam = +X_world,
    # +Y_cam = −Y_world (OpenCV convention).
    R = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )
    return CameraPose(position=np.array(_TOP_DOWN_CAM_POS), rotation=R)


def _load_perception_cfg(path_str: str) -> dict:
    path = Path(path_str)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    cfg = load_config(path)
    return cfg.get("perception", {})


def _nearest_endpoint_distance_world(
    ee_pos_w: torch.Tensor,
    endpoints_w: torch.Tensor,
) -> torch.Tensor:
    """``ee_pos_w``: (N, 3); ``endpoints_w``: (N, 2, 3) → (N,) nearest distance."""
    d = torch.norm(endpoints_w - ee_pos_w[:, None, :], dim=-1)
    return d.min(dim=-1).values


def main() -> None:
    policy_path = retrieve_file_path(args_cli.policy)
    perception_cfg = _load_perception_cfg(args_cli.perception_config)

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    # Attach a top-down perception camera that mirrors so100_sandbox.py.
    env_cfg.scene.perception_cam = _top_down_camera_cfg(
        args_cli.camera_width, args_cli.camera_height,
    )

    env = gym.make(args_cli.task, cfg=env_cfg)
    device = env.unwrapped.device

    logger.info("Loading JIT policy from %s", policy_path)
    policy = torch.jit.load(policy_path, map_location=device)
    policy.eval()

    intrinsics = _build_top_down_intrinsics(
        args_cli.camera_width, args_cli.camera_height,
    )
    cam_pose = _build_top_down_pose()

    # Pre-compute the index range of the rope_endpoint_pos slice inside the
    # 24-D obs vector.  Layout (set in EndpointObservationsCfg.PolicyCfg):
    #   joint_pos_rel (5) | joint_vel_rel (5) | rope_endpoints (6) | ee (3) | last_action (5)
    ENDPOINT_OBS_START = 10
    ENDPOINT_OBS_END = 16
    EE_OBS_START = 16
    EE_OBS_END = 19

    obs_dict, _ = env.reset()
    obs = obs_dict["policy"]

    last_endpoints_px: list[np.ndarray | None] = [None] * args_cli.num_envs
    final_distances: list[float] = []
    completed_episodes = 0

    with torch.inference_mode():
        while completed_episodes < args_cli.num_episodes:
            # --- Replace the rope-endpoint slice with perception output. ---
            camera_data = env.unwrapped.scene["perception_cam"].data.output
            if "rgb" in camera_data:
                rgb_batch = camera_data["rgb"]  # (N, H, W, 3 or 4)
                for env_idx in range(args_cli.num_envs):
                    frame_bgr = tensor_rgb_to_bgr(rgb_batch[env_idx])
                    perception_out = run_perception(frame_bgr, perception_cfg)
                    try:
                        endpoints_px = select_two_endpoints(
                            keypoints=_collect_endpoint_keypoints(perception_out),
                            endpoints_xy=perception_out.rope_state.endpoints,
                            fallback=last_endpoints_px[env_idx],
                        )
                    except ValueError as err:
                        logger.warning("env %d: %s", env_idx, err)
                        continue
                    last_endpoints_px[env_idx] = endpoints_px
                    world_xyz = pixel_to_table_world(
                        endpoints_px, intrinsics, cam_pose, _TABLE_Z,
                    )
                    if np.any(np.isnan(world_xyz)):
                        logger.warning(
                            "env %d: pixel→world produced NaN; keeping GT.",
                            env_idx,
                        )
                        continue
                    # Subtract env origin to match the env-relative training frame.
                    env_origin = (
                        env.unwrapped.scene.env_origins[env_idx]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    relative = world_xyz - env_origin[None, :]
                    obs_vec = relative.reshape(-1).astype(np.float32)
                    obs[env_idx, ENDPOINT_OBS_START:ENDPOINT_OBS_END] = (
                        torch.from_numpy(obs_vec).to(device)
                    )

            actions = policy(obs)
            obs_dict, _, terminated, truncated, _ = env.step(actions)
            dones = (terminated | truncated)

            # On any terminated/truncated env, record final EE-to-nearest-end distance.
            if dones.any():
                rope = env.unwrapped.scene["rope"]
                ee_sensor = env.unwrapped.scene["ee_frame"]
                from RopeUntyingRobot.tasks.manager_based.ropeuntyingrobot.mdp.observations import (  # noqa: E501
                    _last_segment_body_index,
                    _segment_tip_w,
                )

                end0 = rope.data.body_pos_w[:, 0, :]
                end1 = _segment_tip_w(rope, _last_segment_body_index(rope))
                ee_pos = ee_sensor.data.target_pos_w[:, 0, :]
                endpoints_w = torch.stack([end0, end1], dim=1)
                dist = _nearest_endpoint_distance_world(ee_pos, endpoints_w)

                for env_idx in torch.nonzero(dones, as_tuple=False).flatten():
                    final_distances.append(float(dist[env_idx].item()))
                    completed_episodes += 1
                    last_endpoints_px[int(env_idx)] = None
                    if completed_episodes >= args_cli.num_episodes:
                        break

            obs = obs_dict["policy"]

    env.close()

    if final_distances:
        med = median(final_distances)
        mean = sum(final_distances) / len(final_distances)
        logger.info(
            "Completed %d episodes | mean nearest-endpoint distance=%.4f m | median=%.4f m",
            len(final_distances), mean, med,
        )
        # Acceptance threshold from the plan: median < 5 cm.
        if med >= 0.05:
            logger.warning(
                "Median final distance %.4f m exceeds 5 cm threshold; "
                "tune perception_config.yaml before real-robot deployment.",
                med,
            )
    else:
        logger.warning("No episodes completed; nothing to report.")


def _collect_endpoint_keypoints(perception_out) -> list:
    """Recover endpoint Keypoint objects from the overlay step.

    ``RopeState`` already strips confidences; rebuild a minimal Keypoint
    list from ``rope_state.endpoints`` so ``select_two_endpoints`` works.
    Confidences default to 1.0 since the state estimator already filtered
    out low-confidence ones.
    """
    from src.perception.keypoint_detection import Keypoint

    return [
        Keypoint(position=pos, keypoint_type="endpoint", confidence=1.0)
        for pos in perception_out.rope_state.endpoints
    ]


if __name__ == "__main__":
    main()
    try:
        simulation_app.close(wait_for_replicator=False)
    except Exception:
        pass
