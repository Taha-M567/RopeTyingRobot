"""Event terms for the rope-reaching task.

Provides custom reset functions that randomize the rope's initial
configuration (orientation, shape, and position) across parallel
environments and episodes.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import DeformableObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_rope_randomized(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_offset_range: dict[str, tuple[float, float]],
    rotation_range: tuple[float, float] = (0.0, 2.0 * math.pi),
    per_node_noise_std: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("rope"),
) -> None:
    """Reset rope with randomized orientation, shape, and position.

    Applies three layers of randomization to the rope's default nodal
    state:

    1. **Z-axis rotation** — rotates the entire rope around its COM by
       a random angle, so it can face any direction on the table.
    2. **Per-node XY perturbation** — adds small independent Gaussian
       noise to each node's XY position, creating bends and curves.
    3. **Global position offset** — translates the whole rope by a
       random XY offset within the specified range.

    All randomization is XY-plane only (no Z perturbation) since the
    rope lies flat on a table.

    Args:
        env: The environment instance.
        env_ids: Tensor of environment indices being reset.
        position_offset_range: Dict with keys ``"x"``, ``"y"``, ``"z"``
            each mapping to a ``(min, max)`` tuple for the global
            translation offset.
        rotation_range: ``(min_angle, max_angle)`` in radians for the
            random Z-axis rotation applied to the whole rope.
        per_node_noise_std: Standard deviation (in metres) of
            independent Gaussian noise added to each node's XY
            position. Set to ``0.0`` to disable.
        asset_cfg: Scene entity configuration identifying the rope
            asset.
    """
    asset: DeformableObject = env.scene[asset_cfg.name]
    num_envs = len(env_ids)
    device = env.device

    # (N, num_nodes, 6): cols 0-2 = pos, cols 3-5 = vel
    nodal_state = asset.data.default_nodal_state_w[env_ids].clone()
    num_nodes = nodal_state.shape[1]

    positions = nodal_state[:, :, :3]  # (N, num_nodes, 3)

    # -- 1. Z-axis rotation around COM --
    com = positions.mean(dim=1, keepdim=True)  # (N, 1, 3)
    centered = positions - com  # (N, num_nodes, 3)

    theta = torch.empty(num_envs, device=device).uniform_(
        rotation_range[0], rotation_range[1]
    )
    cos_t = torch.cos(theta)  # (N,)
    sin_t = torch.sin(theta)  # (N,)

    # Rotate XY components: [x', y'] = R(theta) @ [x, y]
    x = centered[:, :, 0]  # (N, num_nodes)
    y = centered[:, :, 1]  # (N, num_nodes)
    x_rot = cos_t[:, None] * x - sin_t[:, None] * y
    y_rot = sin_t[:, None] * x + cos_t[:, None] * y

    centered[:, :, 0] = x_rot
    centered[:, :, 1] = y_rot

    positions = centered + com

    # -- 2. Per-node XY perturbation --
    if per_node_noise_std > 0.0:
        noise = torch.randn(
            num_envs, num_nodes, 2, device=device
        ) * per_node_noise_std
        positions[:, :, :2] += noise

    # -- 3. Global position offset --
    offset = torch.zeros(num_envs, 1, 3, device=device)
    for i, axis in enumerate(("x", "y", "z")):
        lo, hi = position_offset_range.get(axis, (0.0, 0.0))
        if lo != hi:
            offset[:, 0, i] = torch.empty(
                num_envs, device=device
            ).uniform_(lo, hi)
    positions += offset

    # Write back and zero velocities
    nodal_state[:, :, :3] = positions
    nodal_state[:, :, 3:] = 0.0

    asset.write_nodal_state_to_sim(nodal_state, env_ids=env_ids)
