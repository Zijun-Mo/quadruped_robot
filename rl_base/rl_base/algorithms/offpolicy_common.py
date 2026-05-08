"""Training algorithm implementation for offpolicy common policies."""

from __future__ import annotations

import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


def resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve the device value from configuration or runtime state."""
    if device is None or str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


def set_global_seed(seed: int | None) -> None:
    """Seed Python, NumPy, and Torch random number generators."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_mlp(input_dim: int, output_dim: int, hidden_dims: Iterable[int], activation: type[nn.Module] = nn.ReLU):
    """Build the MLP component."""
    dims = [int(input_dim), *[int(d) for d in hidden_dims], int(output_dim)]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())
    return nn.Sequential(*layers)


@torch.no_grad()
def polyak_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    """Apply Polyak averaging from source parameters into target parameters."""
    for src, dst in zip(source.parameters(), target.parameters()):
        dst.data.mul_(1.0 - tau).add_(src.data, alpha=tau)


def ensure_pt_path(path: str | os.PathLike) -> Path:
    """Ensure the pt path invariant holds."""
    path = Path(path)
    if path.suffix != ".pt":
        path = path.with_suffix(".pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def dump_pickle(path: str | os.PathLike, obj) -> None:
    """Serialize an object to disk with pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


@dataclass
class ReplayBatch:
    """Mini-batch of replayed transitions sampled for off-policy updates."""
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    """Fixed-size replay buffer for off-policy transitions."""
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...], capacity: int, device: torch.device):
        """Initialize ReplayBuffer with configuration, tensor shapes, and runtime state."""
        self.obs_shape = tuple(obs_shape)
        self.action_shape = tuple(action_shape)
        self.capacity = int(capacity)
        self.device = device
        self.observations = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.timeouts = np.zeros((self.capacity, 1), dtype=np.float32)
        self.pos = 0
        self.full = False

    def add(self, obs, action, reward: float, next_obs, done: bool, timeout: bool = False) -> None:
        """Insert one transition into the replay buffer."""
        self.observations[self.pos] = np.asarray(obs, dtype=np.float32)
        self.actions[self.pos] = np.asarray(action, dtype=np.float32)
        self.rewards[self.pos] = float(reward)
        self.next_observations[self.pos] = np.asarray(next_obs, dtype=np.float32)
        self.dones[self.pos] = float(done)
        self.timeouts[self.pos] = float(timeout)
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def add_batch(self, observations, actions, rewards, next_observations, dones, timeouts=None) -> None:
        """Insert a batch of transitions into the replay buffer."""
        observations = np.asarray(observations)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards).reshape(-1)
        next_observations = np.asarray(next_observations)
        dones = np.asarray(dones).reshape(-1)
        if timeouts is None:
            timeouts = np.zeros_like(dones, dtype=bool)
        else:
            timeouts = np.asarray(timeouts).reshape(-1)
        for i in range(observations.shape[0]):
            self.add(observations[i], actions[i], float(rewards[i]), next_observations[i], bool(dones[i]), bool(timeouts[i]))

    def sample(self, batch_size: int) -> ReplayBatch:
        """Sample a random mini-batch of transitions from replay storage."""
        size = len(self)
        idx = np.random.randint(0, size, size=int(batch_size))
        dones = self.dones[idx] * (1.0 - self.timeouts[idx])
        return ReplayBatch(
            torch.as_tensor(self.observations[idx], device=self.device, dtype=torch.float32),
            torch.as_tensor(self.actions[idx], device=self.device, dtype=torch.float32),
            torch.as_tensor(self.rewards[idx], device=self.device, dtype=torch.float32),
            torch.as_tensor(self.next_observations[idx], device=self.device, dtype=torch.float32),
            torch.as_tensor(dones, device=self.device, dtype=torch.float32),
        )

    def __len__(self) -> int:
        """Implement Python len protocol behavior."""
        return self.capacity if self.full else self.pos

    def state_dict(self) -> dict:
        """Return serializable state for checkpointing."""
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "next_observations": self.next_observations,
            "dones": self.dones,
            "timeouts": self.timeouts,
            "pos": self.pos,
            "full": self.full,
            "capacity": self.capacity,
            "obs_shape": self.obs_shape,
            "action_shape": self.action_shape,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Load replay-buffer state from a serialized off-policy checkpoint."""
        for key in ("observations", "actions", "rewards", "next_observations", "dones"):
            getattr(self, key)[:] = state_dict[key]
        self.timeouts[:] = state_dict.get("timeouts", np.zeros_like(self.timeouts))
        self.pos = int(state_dict["pos"])
        self.full = bool(state_dict["full"])


class TensorboardLogger:
    """Experiment logging adapter for tensorboard logger."""
    def __init__(self, log_dir: str | None):
        """Initialize TensorboardLogger with configuration, tensor shapes, and runtime state."""
        self.name_to_value = {}
        self.writer = None
        if log_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir)
            except Exception:
                self.writer = None

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a scalar metric value to the backing writer."""
        value = float(value)
        self.name_to_value[tag] = value
        if self.writer is not None:
            self.writer.add_scalar(tag, value, step)

    def close(self) -> None:
        """Release logger or writer resources."""
        if self.writer is not None:
            self.writer.close()
