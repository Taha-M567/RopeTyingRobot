"""Reward terms for the rope-reaching task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import DeformableObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

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
    rope: DeformableObject = env.scene[rope_cfg.name]
    ee_sensor: FrameTransformer = env.scene[ee_frame_cfg.name]

    rope_com = rope.data.nodal_pos_w.mean(dim=1)  # (N, 3)
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
    rope: DeformableObject = env.scene[rope_cfg.name]
    ee_sensor: FrameTransformer = env.scene[ee_frame_cfg.name]

    rope_com = rope.data.nodal_pos_w.mean(dim=1)
    ee_pos = ee_sensor.data.target_pos_w[:, 0, :]

    distance = torch.norm(ee_pos - rope_com, dim=-1)
    return (distance < threshold).float()
