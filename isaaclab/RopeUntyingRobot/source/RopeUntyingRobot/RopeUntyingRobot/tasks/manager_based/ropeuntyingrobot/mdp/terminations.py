"""Termination terms for the rope-reaching task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import DeformableObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def rope_below_height(
    env: ManagerBasedRLEnv,
    min_height: float = -0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> torch.Tensor:
    """Terminate if the rope centre-of-mass falls below *min_height*."""
    rope: DeformableObject = env.scene[asset_cfg.name]
    rope_com_z = rope.data.nodal_pos_w.mean(dim=1)[:, 2]
    return rope_com_z < min_height
