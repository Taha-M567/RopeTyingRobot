"""Custom observation terms for the rope-reaching task."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# ---------------------------------------------------------------------------
# Resolve rope geometry constants from the asset module.  The asset directory
# lives two levels above this file's package root (see env_cfg).
# ---------------------------------------------------------------------------
_ASSET_DIR = Path(__file__).resolve().parents[6] / "assets"
if str(_ASSET_DIR) not in sys.path:
    sys.path.insert(0, str(_ASSET_DIR))

from rope_config import ROPE_NUM_SEGMENTS  # noqa: E402

# Rope geometry — total length must match ``generate_rope_urdf.total_length``.
_ROPE_TOTAL_LENGTH = 0.45
_ROPE_SEG_LEN = _ROPE_TOTAL_LENGTH / ROPE_NUM_SEGMENTS
_LAST_SEGMENT_BODY_NAME = f"rope_seg_{ROPE_NUM_SEGMENTS - 1}"


def _last_segment_body_index(rope: Articulation) -> int:
    """Index of the rope's terminal segment within ``body_pos_w``."""
    return rope.body_names.index(_LAST_SEGMENT_BODY_NAME)


def _segment_tip_w(rope: Articulation, body_index: int) -> torch.Tensor:
    """World-frame position of the *far* end of a rope segment.

    Each segment's body origin sits at the start of the cylinder; the
    physical tip is ``seg_len`` along the segment's local +X axis.
    Returns a tensor of shape ``(num_envs, 3)``.
    """
    seg_pos_w = rope.data.body_pos_w[:, body_index, :]
    seg_quat_w = rope.data.body_quat_w[:, body_index, :]
    local_offset = torch.tensor(
        [_ROPE_SEG_LEN, 0.0, 0.0],
        device=seg_pos_w.device,
        dtype=seg_pos_w.dtype,
    ).expand(seg_pos_w.shape[0], 3)
    return seg_pos_w + quat_apply(seg_quat_w, local_offset)


def rope_com_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """Rope centre-of-mass position relative to each environment's origin.

    Computes the mean of all body positions for the articulated chain and
    subtracts the env origin so values are consistent across parallel envs.

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    rope: Articulation = env.scene[asset_cfg.name]
    com_w = rope.data.body_pos_w.mean(dim=1)
    return com_w - env.scene.env_origins


def ee_pos_w(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector position relative to each environment's origin.

    The FrameTransformer must have exactly one target frame (the EE link).

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    sensor: FrameTransformer = env.scene[sensor_cfg.name]
    ee_w = sensor.data.target_pos_w[:, 0, :]
    return ee_w - env.scene.env_origins


def rope_endpoint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """Concatenated positions of the rope's two endpoints, env-relative.

    The first endpoint is the root of ``rope_seg_0``; the second is the
    far tip of ``rope_seg_{N-1}`` (offset by one segment length along the
    segment's local +X axis).  Both are returned in the env-local frame
    so values are independent of which parallel env they come from.

    Returns:
        Tensor of shape ``(num_envs, 6)`` laid out as ``[end0_xyz, end1_xyz]``.
    """
    rope: Articulation = env.scene[asset_cfg.name]
    origins = env.scene.env_origins

    end0_w = rope.data.body_pos_w[:, 0, :]
    end1_w = _segment_tip_w(rope, _last_segment_body_index(rope))

    end0 = end0_w - origins
    end1 = end1_w - origins
    return torch.cat([end0, end1], dim=-1)
