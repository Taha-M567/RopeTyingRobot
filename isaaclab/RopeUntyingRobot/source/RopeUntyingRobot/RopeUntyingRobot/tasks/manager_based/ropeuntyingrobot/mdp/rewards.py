"""Reward terms for the rope-reaching task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .observations import _last_segment_body_index, _segment_tip_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reaching_rope(
    env: ManagerBasedRLEnv,
    sigma: float = 0.1,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense reward: 1 - tanh(||ee - rope_com|| / sigma).

    Provides a smooth gradient from anywhere in the workspace toward
    the rope centre-of-mass.
    """
    rope: Articulation = env.scene[rope_cfg.name]
    ee_sensor: FrameTransformer = env.scene[ee_frame_cfg.name]

    rope_com = rope.data.body_pos_w.mean(dim=1)  # (N, 3)
    ee_pos = ee_sensor.data.target_pos_w[:, 0, :]  # (N, 3)

    distance = torch.norm(ee_pos - rope_com, dim=-1)
    return 1.0 - torch.tanh(distance / sigma)


def close_to_rope(
    env: ManagerBasedRLEnv,
    threshold: float = 0.02,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Binary bonus when the end-effector is within *threshold* of rope COM."""
    rope: Articulation = env.scene[rope_cfg.name]
    ee_sensor: FrameTransformer = env.scene[ee_frame_cfg.name]

    rope_com = rope.data.body_pos_w.mean(dim=1)
    ee_pos = ee_sensor.data.target_pos_w[:, 0, :]

    distance = torch.norm(ee_pos - rope_com, dim=-1)
    return (distance < threshold).float()


def _nearest_endpoint_distance(
    env: ManagerBasedRLEnv,
    rope_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Per-env distance from the EE to the nearer of the rope's two ends.

    Computed in world frame; env origins cancel in the subtraction so the
    result is identical across parallel envs.
    """
    rope: Articulation = env.scene[rope_cfg.name]
    ee_sensor: FrameTransformer = env.scene[ee_frame_cfg.name]

    end0 = rope.data.body_pos_w[:, 0, :]
    end1 = _segment_tip_w(rope, _last_segment_body_index(rope))
    ee_pos = ee_sensor.data.target_pos_w[:, 0, :]

    d0 = torch.norm(ee_pos - end0, dim=-1)
    d1 = torch.norm(ee_pos - end1, dim=-1)
    return torch.minimum(d0, d1)


def reaching_nearest_endpoint(
    env: ManagerBasedRLEnv,
    sigma: float = 0.1,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Dense reward toward whichever rope end is closer to the EE."""
    d_min = _nearest_endpoint_distance(env, rope_cfg, ee_frame_cfg)
    return 1.0 - torch.tanh(d_min / sigma)


def close_to_nearest_endpoint(
    env: ManagerBasedRLEnv,
    threshold: float = 0.02,
    rope_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Binary bonus when the EE is within *threshold* of either rope end."""
    d_min = _nearest_endpoint_distance(env, rope_cfg, ee_frame_cfg)
    return (d_min < threshold).float()
