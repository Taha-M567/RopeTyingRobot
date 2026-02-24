"""Custom observation terms for the rope-reaching task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import DeformableObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rope_com_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """Centre-of-mass position of the rope in world frame.

    Computes the mean of all nodal positions for the deformable body.

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    rope: DeformableObject = env.scene[asset_cfg.name]
    # nodal_pos_w: (num_envs, num_nodes, 3)
    return rope.data.nodal_pos_w.mean(dim=1)


def ee_pos_w(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector position in world frame from a FrameTransformer.

    The FrameTransformer must have exactly one target frame (the EE link).

    Returns:
        Tensor of shape ``(num_envs, 3)``.
    """
    sensor: FrameTransformer = env.scene[sensor_cfg.name]
    # target_pos_w: (num_envs, num_targets, 3) — squeeze target dim
    return sensor.data.target_pos_w[:, 0, :]
