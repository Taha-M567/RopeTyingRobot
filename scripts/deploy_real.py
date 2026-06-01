"""Deploy a trained RopeReachEnd policy on the physical SO-100.

Pipeline (30 Hz control tick):

    real camera ── perception ── pixel_to_table_world ──► endpoints[6]
    SO-100 read joints ─────────────────────────────────► joints_pos[5], joints_vel[5]
                                                          last_action[5], ee_pos[3]
                                                                │
                                                       build 24-D obs tensor
                                                                │
                                                         JIT policy.forward
                                                                │
                                                   SafeSO100.step (clamped writes)

Run with ``--dry-run`` to validate every stage with motor writes
suppressed.

Forward kinematics for the EE position uses ``urdfpy`` against the
sim's SO-100 URDF so the FK output matches what the policy saw at
training time.  Cross-check against the sim FrameTransformer before
trusting it on real motors.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.hardware.camera import Camera, CameraCalibration  # noqa: E402
from src.hardware.so100_arm import (  # noqa: E402
    SO100_ARM_JOINT_NAMES,
    SafeSO100,
    SO100Arm,
    hold_for,
)
from src.utils.config_loader import load_config  # noqa: E402
from src.utils.geometry import CameraPose, pixel_to_table_world  # noqa: E402

# Reuse the same perception runner the sim-eval script uses, so behavior
# is identical between the sim-camera validation and real deployment.
sys.path.insert(
    0, str(REPO_ROOT / "isaaclab" / "RopeUntyingRobot" / "scripts"),
)
from perception_runner import run_perception, select_two_endpoints  # noqa: E402

logger = logging.getLogger(__name__)

# Observation slice layout — must match EndpointObservationsCfg.PolicyCfg.
OBS_DIM = 24
JOINT_POS_REL_START, JOINT_POS_REL_END = 0, 5
JOINT_VEL_REL_START, JOINT_VEL_REL_END = 5, 10
ENDPOINT_START, ENDPOINT_END = 10, 16
EE_START, EE_END = 16, 19
LAST_ACTION_START, LAST_ACTION_END = 19, 24

# Default joint pose used by the sim env for ``joint_pos_rel`` normalization.
SO100_DEFAULT_JOINT_POS = np.array(
    [0.0, 1.0, -1.2, 0.3, 0.0], dtype=np.float64,
)


def _load_camera_config(path: Path) -> tuple[Camera, np.ndarray, CameraPose, float]:
    cfg = load_config(path)["camera"]
    calib_cfg = cfg.get("calibration", {})
    K = np.array(calib_cfg["camera_matrix"], dtype=np.float64)
    dist = np.array(calib_cfg["dist_coeffs"], dtype=np.float64).reshape(-1)
    image_size = (
        int(cfg["image_size"]["width"]),
        int(cfg["image_size"]["height"]),
    )

    extrinsics = cfg["extrinsics"]
    if extrinsics.get("position") is None or extrinsics.get("rotation") is None:
        raise ValueError(
            "camera_config.yaml is missing extrinsics.position/rotation; "
            "fill these in before deploying.",
        )
    pose = CameraPose(
        position=np.array(extrinsics["position"], dtype=np.float64),
        rotation=np.array(extrinsics["rotation"], dtype=np.float64),
    )
    z_table = float(cfg.get("table_z", 0.0))

    calibration = CameraCalibration(
        camera_matrix=K, dist_coeffs=dist, image_size=image_size,
    )
    camera = Camera(cfg["device_id"], calibration=calibration)
    return camera, K, pose, z_table


def _build_fk(urdf_path: Path):
    """Return a callable that maps 5-D joint positions to EE world XYZ.

    Wraps ``urdfpy`` so we depend on it only when actually deploying.
    """
    try:
        from urdfpy import URDF
    except ImportError as err:
        raise ImportError(
            "urdfpy is required for real-robot FK. Install with: pip install urdfpy",
        ) from err

    robot = URDF.load(str(urdf_path))
    ee_link_name = "gripper"

    def fk(joint_positions: np.ndarray) -> np.ndarray:
        cfg = {
            name: float(joint_positions[idx])
            for idx, name in enumerate(SO100_ARM_JOINT_NAMES)
        }
        fk_dict = robot.link_fk(cfg=cfg)
        ee_link = next(link for link in robot.links if link.name == ee_link_name)
        T = fk_dict[ee_link]
        return T[:3, 3].astype(np.float64)

    return fk


def _build_obs(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    endpoints_world_rel: np.ndarray,
    ee_world: np.ndarray,
    last_action: np.ndarray,
) -> torch.Tensor:
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[JOINT_POS_REL_START:JOINT_POS_REL_END] = (
        joint_pos - SO100_DEFAULT_JOINT_POS
    )
    obs[JOINT_VEL_REL_START:JOINT_VEL_REL_END] = joint_vel
    obs[ENDPOINT_START:ENDPOINT_END] = endpoints_world_rel.reshape(-1)
    obs[EE_START:EE_END] = ee_world
    obs[LAST_ACTION_START:LAST_ACTION_END] = last_action
    return torch.from_numpy(obs).unsqueeze(0)


def _undistort_pixels(
    uv: np.ndarray, K: np.ndarray, dist: np.ndarray,
) -> np.ndarray:
    """Apply ``cv2.undistortPoints`` if distortion is nontrivial."""
    if dist is None or not np.any(np.abs(dist) > 1e-9):
        return uv
    pts = uv.reshape(-1, 1, 2).astype(np.float64)
    undist = cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)
    return undist


def run(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    camera, K, cam_pose, z_table = _load_camera_config(
        REPO_ROOT / args.camera_config,
    )
    perception_cfg = load_config(REPO_ROOT / args.perception_config)["perception"]

    fk = _build_fk(REPO_ROOT / args.urdf)
    arm = SO100Arm(port=args.port, urdf_path=REPO_ROOT / args.urdf)
    safe_arm = SafeSO100(arm, max_step_rad=args.max_step_rad)

    policy = torch.jit.load(str(REPO_ROOT / args.policy))
    policy.eval()

    camera.connect()
    arm.connect()
    logger.info("Hardware connected. Dry-run=%s", args.dry_run)

    dt = 1.0 / args.hz
    last_action = np.zeros(5, dtype=np.float64)
    last_joint_pos: np.ndarray | None = None
    last_endpoints_px: np.ndarray | None = None
    steps = int(args.duration_s * args.hz)

    try:
        for step in range(steps):
            t0 = time.time()

            # 1) Camera → perception → world endpoints.
            frame = camera.capture()
            perception_out = run_perception(frame, perception_cfg)
            try:
                endpoints_px = select_two_endpoints(
                    keypoints=_endpoint_keypoints(perception_out),
                    endpoints_xy=perception_out.rope_state.endpoints,
                    fallback=last_endpoints_px,
                )
            except ValueError as err:
                logger.warning("Endpoint select failed: %s; holding pose.", err)
                if last_joint_pos is not None and not args.dry_run:
                    safe_arm.step(last_joint_pos)
                _sleep_remaining(t0, dt)
                continue
            last_endpoints_px = endpoints_px

            endpoints_px_u = _undistort_pixels(
                endpoints_px, K, camera.calibration.dist_coeffs,
            )
            endpoints_world = pixel_to_table_world(
                endpoints_px_u, K, cam_pose, z_table,
            )
            if np.any(np.isnan(endpoints_world)):
                logger.warning(
                    "pixel→world produced NaN; holding pose this tick.",
                )
                _sleep_remaining(t0, dt)
                continue

            # 2) Read joints; compute EE via FK; compute velocity by finite diff.
            joint_pos = arm.read_joint_positions()
            if last_joint_pos is None:
                joint_vel = np.zeros_like(joint_pos)
            else:
                joint_vel = (joint_pos - last_joint_pos) / dt
            last_joint_pos = joint_pos.copy()
            ee_world = fk(joint_pos)

            # 3) Assemble obs (robot-base frame, identical to training env-relative).
            obs = _build_obs(
                joint_pos=joint_pos,
                joint_vel=joint_vel,
                endpoints_world_rel=endpoints_world,
                ee_world=ee_world,
                last_action=last_action,
            )

            # 4) Policy → action → joint targets (action scale matches sim).
            with torch.inference_mode():
                action = policy(obs).squeeze(0).cpu().numpy().astype(np.float64)
            action_scale = 0.5
            target = SO100_DEFAULT_JOINT_POS + action_scale * action
            last_action = action

            # 5) Dispatch (or skip if dry-run).
            if args.dry_run:
                logger.info(
                    "[dry] step=%d action=%s target=%s endpoints=%s",
                    step, np.round(action, 3),
                    np.round(target, 3),
                    np.round(endpoints_world, 3),
                )
            else:
                safe_arm.step(target)

            _sleep_remaining(t0, dt)
    except KeyboardInterrupt:
        logger.info("Interrupted; holding pose.")
    finally:
        if not args.dry_run:
            hold_for(safe_arm, duration_s=0.5, hz=args.hz)
        arm.disconnect()
        camera.disconnect()

    return 0


def _endpoint_keypoints(perception_out):
    """Reconstruct a list of endpoint Keypoint objects."""
    from src.perception.keypoint_detection import Keypoint

    return [
        Keypoint(position=pos, keypoint_type="endpoint", confidence=1.0)
        for pos in perception_out.rope_state.endpoints
    ]


def _sleep_remaining(t0: float, dt: float) -> None:
    rem = dt - (time.time() - t0)
    if rem > 0:
        time.sleep(rem)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        required=True,
        help="Path to JIT-exported policy (logs/.../exported/policy.pt).",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=Path("isaaclab/RopeUntyingRobot/assets/so100.urdf"),
    )
    parser.add_argument(
        "--camera-config",
        type=Path,
        default=Path("src/configs/camera_config.yaml"),
    )
    parser.add_argument(
        "--perception-config",
        type=Path,
        default=Path("src/configs/perception_config.yaml"),
    )
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0")
    parser.add_argument("--hz", type=float, default=30.0)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument(
        "--max-step-rad", type=float, default=0.1,
        help="Per-step joint delta cap.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip motor writes; print actions only.",
    )
    args = parser.parse_args()
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
