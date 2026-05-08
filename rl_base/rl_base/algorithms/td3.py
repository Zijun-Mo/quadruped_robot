"""Training algorithm implementation for TD3 policies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .offpolicy_common import (
    ReplayBuffer,
    TensorboardLogger,
    build_mlp,
    dump_pickle,
    ensure_pt_path,
    polyak_update,
    resolve_device,
    set_global_seed,
)


class TD3Actor(nn.Module):
    """Policy actor network used by the t d3 algorithm."""
    def __init__(self, obs_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray, hidden_dims=(400, 300)):
        """Initialize TD3Actor with configuration, tensor shapes, and runtime state."""
        super().__init__()
        self.backbone = build_mlp(obs_dim, action_dim, hidden_dims, nn.ReLU)
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Run the forward pass for this module."""
        raw = torch.tanh(self.backbone(obs))
        return self.action_low + (raw + 1.0) * 0.5 * (self.action_high - self.action_low)


class TD3Critic(nn.Module):
    """Critic network used by the t d3 algorithm."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=(400, 300)):
        """Initialize TD3Critic with configuration, tensor shapes, and runtime state."""
        super().__init__()
        self.q1 = build_mlp(obs_dim + action_dim, 1, hidden_dims, nn.ReLU)
        self.q2 = build_mlp(obs_dim + action_dim, 1, hidden_dims, nn.ReLU)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """Run the forward pass for this module."""
        x = torch.cat((obs, action), dim=-1)
        return self.q1(x), self.q2(x)

    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor):
        """Evaluate only the first critic branch for actor updates."""
        return self.q1(torch.cat((obs, action), dim=-1))


class TD3:
    """Training algorithm implementation for TD3."""
    def __init__(
        self,
        policy: str,
        env,
        learning_rate: float = 1.0e-3,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        tensorboard_log: str | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: str | torch.device | None = "auto",
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        exploration_noise_std: float = 0.1,
    ):
        """Initialize TD3 with configuration, tensor shapes, and runtime state."""
        if policy != "MlpPolicy":
            raise ValueError("TD3 only supports policy='MlpPolicy'.")
        set_global_seed(seed)
        self.env = env
        self.device = resolve_device(device)
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.learning_starts = int(learning_starts)
        self.batch_size = int(batch_size)
        self.tau = tau
        self.gamma = gamma
        self.train_freq = int(train_freq)
        self.gradient_steps = int(gradient_steps)
        self.policy_delay = int(policy_delay)
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.exploration_noise_std = exploration_noise_std
        self.total_timesteps = 0
        self.total_env_steps = 0
        self.total_updates = 0
        self.logger = TensorboardLogger(tensorboard_log)

        self.obs_shape = tuple(env.observation_space.shape)
        self.action_shape = tuple(env.action_space.shape)
        self.obs_dim = int(np.prod(self.obs_shape))
        self.action_dim = int(np.prod(self.action_shape))
        self.action_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
        self.action_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
        self.actor = TD3Actor(self.obs_dim, self.action_dim, self.action_low, self.action_high).to(self.device)
        self.actor_target = TD3Actor(self.obs_dim, self.action_dim, self.action_low, self.action_high).to(self.device)
        self.critic = TD3Critic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_target = TD3Critic(self.obs_dim, self.action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(self.obs_shape, self.action_shape, buffer_size, self.device)

    def _ensure_batched_obs(self, obs) -> np.ndarray:
        """Normalize inputs so the internal batched observations invariant holds."""
        arr = np.asarray(obs, dtype=np.float32)
        if arr.ndim == len(self.obs_shape):
            arr = arr[None, ...]
        return arr

    def _reshape_action_batch(self, action_batch_flat: np.ndarray) -> np.ndarray:
        """Reshape the internal action batch tensor to the expected batch layout."""
        return action_batch_flat.reshape(action_batch_flat.shape[0], *self.action_shape)

    def _sample_random_actions(self) -> np.ndarray:
        """Sample random actions from the environment action range."""
        n_envs = getattr(self.env, "num_envs", 1)
        return np.stack([self.env.action_space.sample() for _ in range(n_envs)], axis=0).astype(np.float32)

    def _apply_terminal_obs_and_timeouts(self, infos, next_obs_batch, done_batch, timeout_batch):
        """Patch terminal next observations and timeout masks into a replay transition."""
        if isinstance(infos, dict):
            terminal = infos.get("terminal_observation")
            if terminal is not None:
                terminal = self._ensure_batched_obs(terminal)
                next_obs_batch = np.where(done_batch.reshape(-1, 1), terminal.reshape(next_obs_batch.shape), next_obs_batch)
            timeout = infos.get("TimeLimit.truncated", infos.get("time_outs", None))
            if timeout is not None:
                timeout_batch = np.asarray(timeout, dtype=bool).reshape(-1)
        return next_obs_batch, timeout_batch

    def _predict_actions(self, obs_batch: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Predict policy actions for the current observation batch."""
        obs_t = torch.as_tensor(obs_batch.reshape(obs_batch.shape[0], -1), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            actions = self.actor(obs_t).cpu().numpy()
        if not deterministic:
            actions += np.random.normal(0.0, self.exploration_noise_std, size=actions.shape).astype(np.float32)
        actions = np.clip(actions, self.action_low, self.action_high)
        return self._reshape_action_batch(actions)

    def _train_step(self) -> dict[str, float]:
        """Run one TD3 gradient update from a replay batch."""
        batch = self.replay_buffer.sample(self.batch_size)
        with torch.no_grad():
            noise = torch.randn_like(batch.actions) * self.target_policy_noise
            noise = torch.clamp(noise, -self.target_noise_clip, self.target_noise_clip)
            next_actions = torch.clamp(self.actor_target(batch.next_observations.view(batch.next_observations.shape[0], -1)) + noise, torch.as_tensor(self.action_low, device=self.device), torch.as_tensor(self.action_high, device=self.device))
            target_q1, target_q2 = self.critic_target(batch.next_observations.view(batch.next_observations.shape[0], -1), next_actions)
            target_q = batch.rewards + (1.0 - batch.dones) * self.gamma * torch.min(target_q1, target_q2)
        obs = batch.observations.view(batch.observations.shape[0], -1)
        actions = batch.actions.view(batch.actions.shape[0], -1)
        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss_value = 0.0
        if self.total_updates % self.policy_delay == 0:
            actor_actions = self.actor(obs)
            actor_loss = -self.critic.q1_forward(obs, actor_actions).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            polyak_update(self.actor, self.actor_target, self.tau)
            polyak_update(self.critic, self.critic_target, self.tau)
            actor_loss_value = float(actor_loss.item())
        self.total_updates += 1
        return {"critic_loss": float(critic_loss.item()), "actor_loss": actor_loss_value}

    def learn(self, total_timesteps: int, log_interval: int = 10, tb_log_name: str = "TD3"):
        """Run the off-policy training loop for the requested number of timesteps."""
        obs, _ = self.env.reset()
        obs = self._ensure_batched_obs(obs)
        n_envs = obs.shape[0]
        while self.total_timesteps < total_timesteps:
            if self.total_timesteps < self.learning_starts:
                actions = self._sample_random_actions()
            else:
                actions = self._predict_actions(obs)
            next_obs, rewards, terminated, truncated, infos = self.env.step(actions)
            next_obs = self._ensure_batched_obs(next_obs)
            done = np.asarray(terminated, dtype=bool).reshape(n_envs) | np.asarray(truncated, dtype=bool).reshape(n_envs)
            timeout = np.asarray(truncated, dtype=bool).reshape(n_envs)
            next_obs, timeout = self._apply_terminal_obs_and_timeouts(infos, next_obs, done, timeout)
            self.replay_buffer.add_batch(obs, actions, rewards, next_obs, done, timeout)
            obs = next_obs
            self.total_env_steps += 1
            self.total_timesteps += n_envs
            if len(self.replay_buffer) >= self.batch_size and self.total_timesteps >= self.learning_starts and self.total_env_steps % self.train_freq == 0:
                for _ in range(self.gradient_steps):
                    metrics = self._train_step()
                    for key, value in metrics.items():
                        self.logger.add_scalar(f"{tb_log_name}/{key}", value, self.total_timesteps)
        return self

    def save(self, save_path: str | Path) -> None:
        """Save model and optimizer state to disk."""
        path = ensure_pt_path(save_path)
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "config": {
                    "algo": "td3",
                    "learning_rate": self.learning_rate,
                    "buffer_size": self.buffer_size,
                    "learning_starts": self.learning_starts,
                    "batch_size": self.batch_size,
                    "tau": self.tau,
                    "gamma": self.gamma,
                    "train_freq": self.train_freq,
                    "gradient_steps": self.gradient_steps,
                    "policy_delay": self.policy_delay,
                    "target_policy_noise": self.target_policy_noise,
                    "target_noise_clip": self.target_noise_clip,
                    "exploration_noise_std": self.exploration_noise_std,
                    "total_timesteps": self.total_timesteps,
                    "total_env_steps": self.total_env_steps,
                    "total_updates": self.total_updates,
                },
            },
            path,
        )

    def save_replay_buffer(self, path: str | Path) -> None:
        """Save replay buffer state to disk."""
        dump_pickle(path, self.replay_buffer.state_dict())
