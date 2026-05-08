"""Training algorithm implementation for SAC policies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

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


class SACActor(nn.Module):
    """Policy actor network used by the s a c algorithm."""
    def __init__(self, obs_dim: int, action_dim: int, action_low: np.ndarray, action_high: np.ndarray, hidden_dims=(256, 256)):
        """Initialize SACActor with configuration, tensor shapes, and runtime state."""
        super().__init__()
        self.backbone = build_mlp(obs_dim, hidden_dims[-1] if hidden_dims else 256, hidden_dims[:-1], nn.ReLU)
        last_dim = hidden_dims[-1] if hidden_dims else 256
        self.mean = nn.Linear(last_dim, action_dim)
        self.log_std = nn.Linear(last_dim, action_dim)
        self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))

    def _distribution_params(self, obs: torch.Tensor):
        """Return mean and standard-deviation tensors for the Gaussian policy."""
        features = self.backbone(obs)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), -20, 2)
        return mean, log_std

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Run the forward pass for this module."""
        action, _ = self.action_log_prob(obs, deterministic=deterministic)
        return action

    def action_log_prob(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample squashed actions and return corrected log probabilities."""
        mean, log_std = self._distribution_params(obs)
        if deterministic:
            z = mean
        else:
            z = Normal(mean, log_std.exp()).rsample()
        squashed = torch.tanh(z)
        action = self.action_low + (squashed + 1.0) * 0.5 * (self.action_high - self.action_low)
        log_prob = Normal(mean, log_std.exp()).log_prob(z)
        log_prob -= torch.log(1.0 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob


class SACCritic(nn.Module):
    """Critic network used by the s a c algorithm."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_dims=(256, 256)):
        """Initialize SACCritic with configuration, tensor shapes, and runtime state."""
        super().__init__()
        self.q1 = build_mlp(obs_dim + action_dim, 1, hidden_dims, nn.ReLU)
        self.q2 = build_mlp(obs_dim + action_dim, 1, hidden_dims, nn.ReLU)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        """Run the forward pass for this module."""
        x = torch.cat((obs, action), dim=-1)
        return self.q1(x), self.q2(x)


class SAC:
    """Training algorithm implementation for SAC."""
    def __init__(
        self,
        policy: str,
        env,
        learning_rate: float = 3.0e-4,
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
        ent_coef: str | float = "auto",
        target_update_interval: int = 1,
        target_entropy: str | float = "auto",
    ):
        """Initialize SAC with configuration, tensor shapes, and runtime state."""
        if policy != "MlpPolicy":
            raise ValueError("SAC only supports policy='MlpPolicy'.")
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
        self.target_update_interval = int(target_update_interval)
        self.ent_coef_setting = ent_coef
        self.target_entropy_setting = target_entropy
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
        self.actor = SACActor(self.obs_dim, self.action_dim, self.action_low, self.action_high).to(self.device)
        self.critic = SACCritic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_target = SACCritic(self.obs_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(self.obs_shape, self.action_shape, buffer_size, self.device)
        self._setup_entropy_coef()

    def _setup_entropy_coef(self) -> None:
        """Configure fixed or learned entropy coefficient state."""
        if self.target_entropy_setting == "auto":
            self.target_entropy = -float(self.action_dim)
        else:
            self.target_entropy = float(self.target_entropy_setting)
        if self.ent_coef_setting == "auto":
            self.log_ent_coef = torch.zeros(1, device=self.device, requires_grad=True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=self.learning_rate)
            self.ent_coef_tensor = None
        else:
            self.log_ent_coef = None
            self.ent_coef_optimizer = None
            self.ent_coef_tensor = torch.tensor(float(self.ent_coef_setting), device=self.device)

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

    def _apply_terminal_obs(self, infos, next_obs_batch: np.ndarray) -> np.ndarray:
        """Replace next observations with terminal observations where available."""
        if isinstance(infos, dict) and infos.get("terminal_observation") is not None:
            terminal = self._ensure_batched_obs(infos["terminal_observation"])
            return terminal.reshape(next_obs_batch.shape)
        return next_obs_batch

    def _predict_actions(self, obs_batch: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Predict policy actions for the current observation batch."""
        obs_t = torch.as_tensor(obs_batch.reshape(obs_batch.shape[0], -1), device=self.device, dtype=torch.float32)
        with torch.no_grad():
            actions = self.actor(obs_t, deterministic=deterministic).cpu().numpy()
        actions = np.clip(actions, self.action_low, self.action_high)
        return self._reshape_action_batch(actions)

    def _current_ent_coef(self) -> torch.Tensor:
        """Return the active entropy coefficient tensor."""
        if self.log_ent_coef is not None:
            return self.log_ent_coef.exp()
        return self.ent_coef_tensor

    def _train_step(self) -> dict[str, float]:
        """Run one SAC gradient update from a replay batch."""
        batch = self.replay_buffer.sample(self.batch_size)
        obs = batch.observations.view(batch.observations.shape[0], -1)
        actions = batch.actions.view(batch.actions.shape[0], -1)
        next_obs = batch.next_observations.view(batch.next_observations.shape[0], -1)

        with torch.no_grad():
            next_actions, next_log_prob = self.actor.action_log_prob(next_obs)
            next_q1, next_q2 = self.critic_target(next_obs, next_actions)
            next_q = torch.min(next_q1, next_q2) - self._current_ent_coef() * next_log_prob
            target_q = batch.rewards + (1.0 - batch.dones) * self.gamma * next_q

        q1, q2 = self.critic(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_prob = self.actor.action_log_prob(obs)
        q1_pi, q2_pi = self.critic(obs, new_actions)
        ent_coef = self._current_ent_coef().detach()
        actor_loss = (ent_coef * log_prob - torch.min(q1_pi, q2_pi)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        ent_coef_loss_value = 0.0
        if self.log_ent_coef is not None:
            ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            ent_coef_loss_value = float(ent_coef_loss.item())

        if self.total_updates % self.target_update_interval == 0:
            polyak_update(self.critic, self.critic_target, self.tau)
        self.total_updates += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "ent_coef": float(self._current_ent_coef().item()),
            "ent_coef_loss": ent_coef_loss_value,
        }

    def learn(self, total_timesteps: int, log_interval: int = 10, tb_log_name: str = "SAC"):
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
            terminal_next_obs = self._apply_terminal_obs(infos, next_obs)
            self.replay_buffer.add_batch(obs, actions, rewards, terminal_next_obs, done, timeout)
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
                "critic": self.critic.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "log_ent_coef": None if self.log_ent_coef is None else self.log_ent_coef.detach().cpu(),
                "ent_coef_optimizer": None if self.ent_coef_optimizer is None else self.ent_coef_optimizer.state_dict(),
                "ent_coef_tensor": None if self.ent_coef_tensor is None else self.ent_coef_tensor.detach().cpu(),
                "config": {
                    "algo": "sac",
                    "learning_rate": self.learning_rate,
                    "buffer_size": self.buffer_size,
                    "learning_starts": self.learning_starts,
                    "batch_size": self.batch_size,
                    "tau": self.tau,
                    "gamma": self.gamma,
                    "train_freq": self.train_freq,
                    "gradient_steps": self.gradient_steps,
                    "target_update_interval": self.target_update_interval,
                    "ent_coef_setting": self.ent_coef_setting,
                    "target_entropy_setting": self.target_entropy_setting,
                    "target_entropy": self.target_entropy,
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
