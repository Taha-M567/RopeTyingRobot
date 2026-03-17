"""Event terms for the rope-reaching task.

Provides custom reset functions that randomize the articulated rope
chain's initial configuration (root pose and joint angles) across
parallel environments and episodes.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_articulated_rope(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_offset_range: dict[str, tuple[float, float]],
    rotation_range: tuple[float, float] = (0.0, 2.0 * math.pi),
    joint_noise_std: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> None:
    """Reset the articulated rope chain with randomized root pose and joint angles.

    Applies three layers of randomization:

    1. **Global position offset** — translates the rope root by a
       random XY offset within the specified range.
    2. **Z-axis rotation** — rotates the rope around the Z axis by a
       random angle, so it can face any direction on the table.
    3. **Joint angle noise** — adds small Gaussian noise to each
       joint's position, creating random bends along the chain.

    Args:
        env: The environment instance.
        env_ids: Tensor of environment indices being reset.
        position_offset_range: Dict with keys ``"x"``, ``"y"``, ``"z"``
            each mapping to a ``(min, max)`` tuple for the global
            translation offset.
        rotation_range: ``(min_angle, max_angle)`` in radians for the
            random Z-axis rotation.
        joint_noise_std: Standard deviation (in radians) of Gaussian
            noise added to each joint angle. Set to ``0.0`` to disable.
        asset_cfg: Scene entity configuration identifying the rope
            asset.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    device = env.device

    # -- Root state: (N, 13) = pos(3) + quat_wxyz(4) + lin_vel(3) + ang_vel(3)
    root_state = asset.data.default_root_state[env_ids].clone()

    # Random position offset
    for i, axis in enumerate(("x", "y", "z")):
        lo, hi = position_offset_range.get(axis, (0.0, 0.0))
        if lo != hi:
            root_state[:, i] += torch.empty(
                num_envs, device=device,
            ).uniform_(lo, hi)

    # Random Z-axis rotation (quaternion wxyz, replaces default identity)
    theta = torch.empty(num_envs, device=device).uniform_(
        rotation_range[0], rotation_range[1],
    )
    half = theta * 0.5
    root_state[:, 3] = torch.cos(half)  # w
    root_state[:, 4] = 0.0              # x
    root_state[:, 5] = 0.0              # y
    root_state[:, 6] = torch.sin(half)  # z

    # Zero velocities
    root_state[:, 7:] = 0.0

    asset.write_root_state_to_sim(root_state, env_ids=env_ids)

    # -- Joint state: default positions + Gaussian noise, zero velocity --
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_vel = torch.zeros_like(joint_pos)

    if joint_noise_std > 0.0:
        joint_pos += torch.randn_like(joint_pos) * joint_noise_std

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
