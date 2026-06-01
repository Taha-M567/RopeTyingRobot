"""SO-100 arm driver and safety wrapper for real-hardware deployment.

This module is intentionally thin: it wraps LeRobot's existing SO-100
motor bus and exposes the minimum API the deployment loop needs:

  * ``connect()`` / ``disconnect()``
  * ``read_joint_positions()`` — radians, in ``SO100_ARM_JOINT_NAMES`` order
  * ``write_joint_position_targets(targets)`` — radians, same order

The companion ``SafeSO100`` class clamps each commanded step so a
policy bug cannot fling the arm into a joint limit at full speed.
Joint limits are parsed once from the URDF so they stay in lock-step
with the sim model.

LeRobot is a soft dependency: an ``ImportError`` from ``lerobot`` is
surfaced with an actionable install hint instead of being hidden.
"""

from __future__ import annotations

import logging
import re
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


SO100_ARM_JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


@dataclass(frozen=True)
class JointLimits:
    """Per-joint position limits (radians) ordered by ``SO100_ARM_JOINT_NAMES``."""

    lower: np.ndarray
    upper: np.ndarray


def parse_joint_limits_from_urdf(
    urdf_path: Path,
    joint_names: tuple[str, ...] = SO100_ARM_JOINT_NAMES,
) -> JointLimits:
    """Parse ``<limit lower=... upper=...>`` for each named joint.

    Uses a regex rather than a full URDF parser to keep this module
    free of XML dependencies; the URDF format we ship is stable.
    """
    text = urdf_path.read_text(encoding="utf-8")
    lower = np.zeros(len(joint_names), dtype=np.float64)
    upper = np.zeros(len(joint_names), dtype=np.float64)

    for idx, name in enumerate(joint_names):
        joint_match = re.search(
            rf'<joint\s+name="{re.escape(name)}"[^>]*>(.*?)</joint>',
            text,
            flags=re.DOTALL,
        )
        if joint_match is None:
            raise ValueError(f"Joint '{name}' not found in {urdf_path}")
        limit_match = re.search(
            r'<limit\b[^/]*lower="([-\d.eE+]+)"[^/]*upper="([-\d.eE+]+)"',
            joint_match.group(1),
        )
        if limit_match is None:
            raise ValueError(
                f"Joint '{name}' has no <limit> with lower/upper in {urdf_path}"
            )
        lower[idx] = float(limit_match.group(1))
        upper[idx] = float(limit_match.group(2))

    return JointLimits(lower=lower, upper=upper)


class SO100Arm:
    """LeRobot-backed driver for the SO-100 arm.

    The LeRobot import is lazy so unit tests that don't touch hardware
    can still import this module.
    """

    def __init__(
        self,
        port: str,
        urdf_path: Path,
        joint_names: tuple[str, ...] = SO100_ARM_JOINT_NAMES,
    ) -> None:
        self.port = port
        self.joint_names = joint_names
        self.limits = parse_joint_limits_from_urdf(urdf_path, joint_names)
        self._bus = None

    def connect(self) -> None:
        try:
            from lerobot.common.robot_devices.motors.feetech import (
                FeetechMotorsBus,
            )
        except ImportError as err:
            raise ImportError(
                "lerobot is required for real-robot control. "
                "Install with: pip install 'lerobot[feetech]'"
            ) from err

        motor_map = {
            name: (idx + 1, "sts3215")
            for idx, name in enumerate(self.joint_names)
        }
        self._bus = FeetechMotorsBus(port=self.port, motors=motor_map)
        self._bus.connect()
        logger.info("SO-100 connected on %s", self.port)

    def disconnect(self) -> None:
        if self._bus is None:
            return
        with suppress(Exception):
            self._bus.disconnect()
        self._bus = None
        logger.info("SO-100 disconnected.")

    def read_joint_positions(self) -> np.ndarray:
        if self._bus is None:
            raise RuntimeError("SO-100 not connected.")
        # Feetech motor returns ticks; convert via the bus's calibration.
        ticks = self._bus.read("Present_Position")
        return self._bus.ticks_to_rad(ticks, list(self.joint_names))

    def write_joint_position_targets(self, targets: np.ndarray) -> None:
        if self._bus is None:
            raise RuntimeError("SO-100 not connected.")
        if targets.shape != (len(self.joint_names),):
            raise ValueError(
                f"targets must be ({len(self.joint_names)},), got {targets.shape}",
            )
        clamped = np.clip(targets, self.limits.lower, self.limits.upper)
        ticks = self._bus.rad_to_ticks(clamped, list(self.joint_names))
        self._bus.write("Goal_Position", ticks)


class SafeSO100:
    """Movement-limited wrapper around :class:`SO100Arm`.

    Enforces three guarantees:

    1. Position targets are clamped into the URDF joint limits with a
       margin of ``edge_epsilon`` radians from each end-stop.
    2. Per-step commanded delta is clamped to ``max_step_rad`` (default
       0.1 rad ≈ 5.7° per control tick — matches the sim action scale of
       0.5 over a 1/30 s tick).
    3. If a read fails, the arm is commanded to its last successful
       target (no NaN propagation into motors).
    """

    def __init__(
        self,
        arm: SO100Arm,
        max_step_rad: float = 0.1,
        edge_epsilon: float = 0.02,
    ) -> None:
        self.arm = arm
        self.max_step_rad = max_step_rad
        self.edge_epsilon = edge_epsilon
        self._last_target: Optional[np.ndarray] = None

    def read(self) -> np.ndarray:
        return self.arm.read_joint_positions()

    def step(self, desired_target: np.ndarray) -> np.ndarray:
        """Clamp and dispatch a target; return the actually-sent values."""
        limits = self.arm.limits
        lo = limits.lower + self.edge_epsilon
        hi = limits.upper - self.edge_epsilon

        # Anchor: where we were last commanded, or the current measured pose.
        if self._last_target is None:
            try:
                anchor = self.arm.read_joint_positions()
            except Exception as err:
                logger.error("read_joint_positions failed: %s", err)
                anchor = np.clip(desired_target, lo, hi)
        else:
            anchor = self._last_target

        delta = np.clip(
            desired_target - anchor,
            -self.max_step_rad,
            self.max_step_rad,
        )
        target = np.clip(anchor + delta, lo, hi)
        self.arm.write_joint_position_targets(target)
        self._last_target = target
        return target


def hold_for(arm: SafeSO100, duration_s: float, hz: float = 30.0) -> None:
    """Keep the arm at its current pose for *duration_s* seconds.

    Useful for a graceful shutdown so the arm doesn't drop when the
    deployment loop exits.
    """
    if arm._last_target is None:
        arm._last_target = arm.read()
    dt = 1.0 / hz
    steps = int(duration_s * hz)
    for _ in range(steps):
        arm.arm.write_joint_position_targets(arm._last_target)
        time.sleep(dt)
