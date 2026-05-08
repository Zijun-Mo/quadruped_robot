"""Training algorithm implementation for PPO policies."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from rl_base.storage import RolloutStorage


class PPO:
    """Training algorithm implementation for PPO."""
    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=1,
        clip_param=0.2,
        gamma=0.998,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        device="cpu",
        normalize_advantage_per_mini_batch=False,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        """Initialize PPO with configuration, tensor shapes, and runtime state."""
        self.policy = policy
        self.actor_critic = policy
        self.device = torch.device(device)
        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.storage = None
        self.transition = RolloutStorage.Transition()
        self.rnd = None
        self.rnd_optimizer = None

    def init_storage(self, training_type, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, actions_shape):
        """Allocate rollout storage with shapes derived from the environment."""
        self.storage = RolloutStorage(
            training_type,
            num_envs,
            num_transitions_per_env,
            actor_obs_shape,
            critic_obs_shape,
            actions_shape,
            device=self.device,
        )

    def act(self, obs, critic_obs):
        """Sample actions from the current policy distribution."""
        if hasattr(self.policy, "get_hidden_states"):
            self.transition.hidden_states = self.policy.get_hidden_states()
        actions = self.policy.act(obs)
        values = self.policy.evaluate(critic_obs)
        self.transition.observations = obs.detach()
        self.transition.privileged_observations = critic_obs.detach()
        self.transition.actions = actions.detach()
        self.transition.values = values.detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(actions).detach().unsqueeze(-1)
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        return actions.detach()

    def process_env_step(self, rewards, dones, infos):
        """Record environment feedback and update rollout bookkeeping after a step."""
        rewards = rewards.to(self.device).view(-1, 1)
        dones = dones.to(self.device).view(-1, 1).bool()
        if "time_outs" in infos:
            timeouts = infos["time_outs"].to(self.device).view(-1, 1).float()
            rewards = rewards + self.gamma * self.transition.values * timeouts
        self.transition.rewards = rewards.detach()
        self.transition.dones = dones
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, last_critic_obs):
        """Compute bootstrapped returns and advantages for the rollout buffer."""
        with torch.no_grad():
            last_values = self.policy.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):
        """Run one optimization update and return training statistics."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        generator = self.storage.recurrent_mini_batch_generator if getattr(self.policy, "is_recurrent", False) else self.storage.mini_batch_generator
        num_updates = 0
        for batch in generator(self.num_mini_batches, self.num_learning_epochs):
            (
                obs_batch,
                critic_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                rnd_state_batch,
                privileged_actions_batch,
            ) = batch
            if self.normalize_advantage_per_mini_batch:
                advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)
            if getattr(self.policy, "is_recurrent", False):
                self.policy.reset()
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch).unsqueeze(-1)
            value_batch = self.policy.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy.unsqueeze(-1)

            if self.schedule == "adaptive" and self.desired_kl is not None:
                with torch.no_grad():
                    kl = torch.sum(
                        torch.log(sigma_batch / (old_sigma_batch + 1e-8) + 1e-8)
                        + (old_sigma_batch.pow(2) + (old_mu_batch - mu_batch).pow(2)) / (2.0 * sigma_batch.pow(2))
                        - 0.5,
                        dim=-1,
                    ).mean()
                    if kl > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif kl < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
            surrogate = -advantages_batch * ratio
            surrogate_clipped = -advantages_batch * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            entropy_loss = entropy_batch.mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += float(value_loss.item())
            mean_surrogate_loss += float(surrogate_loss.item())
            mean_entropy += float(entropy_loss.item())
            num_updates += 1
        self.storage.clear()
        denom = max(1, num_updates)
        return {
            "value_function": mean_value_loss / denom,
            "surrogate": mean_surrogate_loss / denom,
            "entropy": mean_entropy / denom,
        }

    def broadcast_parameters(self):
        """Broadcast model parameters across distributed workers."""
        return None

    def reduce_parameters(self):
        """Average gradients or parameters across distributed workers."""
        return None
