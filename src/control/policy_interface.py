"""Interface for learned policies from LeRobot.

This module provides a clean interface for policy inference.
"""

from typing import Dict, Optional

import numpy as np


class PolicyInterface:
    """Interface for loading and executing learned policies.

    This class wraps LeRobot policies to provide a consistent interface
    for both simulation and hardware execution.
    """

    def __init__(self, policy_path: str, config: dict):
        """Initialize policy interface.

        Args:
            policy_path: Path to saved policy model
            config: Configuration dictionary
        """
        self.policy_path = policy_path
        self.config = config
        self.policy = None  # Will be loaded from LeRobot

    def load(self) -> None:
        """Load the policy model from disk."""
        # TODO: Implement policy loading using LeRobot
        pass

    def predict(
        self,
        observation: Dict[str, np.ndarray],
    ) -> np.ndarray:
        """Predict action from observation.

        Args:
            observation: Dictionary containing sensor observations
                (e.g., 'image', 'state', 'proprioception')

        Returns:
            Action array (e.g., joint angles, end-effector pose)

        Raises:
            RuntimeError: If policy is not loaded
        """
        if self.policy is None:
            raise RuntimeError("Policy not loaded. Call load() first.")

        # TODO: Implement policy inference
        return np.array([])

    def reset(self) -> None:
        """Reset policy internal state (for recurrent policies)."""
        # TODO: Implement state reset if needed
        pass
