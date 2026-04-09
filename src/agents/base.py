from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch


class Agent(ABC, torch.nn.Module):
    """Base class for all online off-policy agents.

    Subclasses must implement `get_action` and `update`.
    Both methods use the same convention as `run_online.py`:
      - `get_action` operates on numpy arrays (single observation)
      - `update` operates on torch tensors (batches from the replay buffer)
    """

    @abstractmethod
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Sample an action for a single observation.

        Args:
            obs: numpy array of shape (ob_dim,)

        Returns:
            action: numpy array of shape (ac_dim,)
        """

    @abstractmethod
    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        step: int,
    ) -> Dict[str, float]:
        """Update the agent from a batch of transitions.

        Args:
            obs:        (B, ob_dim)
            actions:    (B, ac_dim)
            rewards:    (B,)
            next_obs:   (B, ob_dim)
            dones:      (B,)
            step:       current training step (useful for scheduling)

        Returns:
            dict of scalar metrics for logging
        """
