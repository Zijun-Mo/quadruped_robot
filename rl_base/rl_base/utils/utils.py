"""Utility helpers for utils support in rl_base."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import Callable

import torch
from torch import nn


def resolve_nn_activation(act_name: str) -> nn.Module:
    """Resolve the neural network activation value from configuration or runtime state."""
    name = act_name.lower()
    if name == "elu":
        return nn.ELU()
    if name == "selu":
        return nn.SELU()
    if name == "relu":
        return nn.ReLU()
    if name == "crelu":
        return nn.ReLU()
    if name == "lrelu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Invalid activation function: {act_name}")


def split_and_pad_trajectories(tensor: torch.Tensor, dones: torch.Tensor):
    """Split the and pad trajectories tensor into its expected parts."""
    dones = dones.squeeze(-1).to(dtype=torch.bool)
    trajectories = []
    for env_idx in range(tensor.shape[1]):
        start = 0
        done_ids = torch.nonzero(dones[:, env_idx], as_tuple=False).flatten().tolist()
        for end in done_ids:
            trajectories.append(tensor[start : end + 1, env_idx])
            start = end + 1
        if start < tensor.shape[0]:
            trajectories.append(tensor[start:, env_idx])
    if not trajectories:
        return tensor[:, :0], torch.zeros(tensor.shape[0], 0, dtype=torch.bool, device=tensor.device)
    max_len = max(traj.shape[0] for traj in trajectories)
    padded = torch.zeros(max_len, len(trajectories), *tensor.shape[2:], dtype=tensor.dtype, device=tensor.device)
    masks = torch.zeros(max_len, len(trajectories), dtype=torch.bool, device=tensor.device)
    for idx, traj in enumerate(trajectories):
        padded[: traj.shape[0], idx].copy_(traj)
        masks[: traj.shape[0], idx] = True
    return padded, masks


def unpad_trajectories(trajectories: torch.Tensor, masks: torch.Tensor):
    """Restore padded trajectory tensors to their original flattened layout."""
    if masks is None:
        return trajectories
    if masks.dim() > 2:
        masks = masks.squeeze(-1)
    return trajectories.transpose(0, 1)[masks.transpose(0, 1)].view(-1, *trajectories.shape[2:])


def store_code_state(logdir, repositories) -> list[str]:
    """Snapshot git repository state into the experiment log directory."""
    try:
        import git
    except Exception:
        return []
    git_dir = Path(logdir) / "git"
    git_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for repo_file in repositories:
        try:
            repo = git.Repo(repo_file, search_parent_directories=True)
            name = Path(repo.working_tree_dir).name
            path = git_dir / f"{name}.diff"
            status = repo.git.status("--short")
            diff = repo.git.diff("HEAD")
            path.write_text(f"# git status --short\n{status}\n\n# git diff HEAD\n{diff}\n", encoding="utf-8")
            written.append(str(path))
        except Exception:
            continue
    return written


def string_to_callable(name: str) -> Callable:
    """Resolve a dotted Python path string into a callable object."""
    if ":" not in name:
        raise ValueError(f"Callable string must use 'module:attribute' form: {name}")
    module_name, attr_name = name.split(":", 1)
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    if not callable(value):
        raise ValueError(f"Resolved object is not callable: {name}")
    return value


def make_mlp(input_dim: int, hidden_dims, output_dim: int, activation="elu") -> nn.Sequential:
    """Create the MLP helper."""
    dims = [int(input_dim), *[int(d) for d in hidden_dims], int(output_dim)]
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(resolve_nn_activation(activation))
    return nn.Sequential(*layers)


def ensure_dir(path: str | os.PathLike):
    """Ensure the dir invariant holds."""
    Path(path).mkdir(parents=True, exist_ok=True)
