"""Vectorized environment interface definitions used by rl_base runners."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class VecEnv(ABC):
    """Minimal vectorized environment contract consumed by OnPolicyRunner."""

    num_envs: int
    num_actions: int
    max_episode_length: int | torch.Tensor
    episode_length_buf: torch.Tensor
    device: torch.device
    cfg: object

    @abstractmethod
    def get_observations(self) -> tuple[torch.Tensor, dict]:
        """Return the latest policy observations from the vectorized environment."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> tuple[torch.Tensor, dict]:
        """Reset environment, module, or buffer state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Advance the environment wrapper by one action step."""
        raise NotImplementedError
