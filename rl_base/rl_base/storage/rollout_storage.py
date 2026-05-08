"""Rollout storage utilities for on-policy reinforcement learning."""

from __future__ import annotations

import torch


class RolloutStorage:
    """Fixed-horizon rollout buffer for policy, value, and distillation batches."""
    class Transition:
        """Mutable one-step transition assembled by runners before insertion."""
        def __init__(self):
            """Initialize Transition with configuration, tensor shapes, and runtime state."""
            self.observations = None
            self.privileged_observations = None
            self.actions = None
            self.privileged_actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None
            self.rnd_state = None

        def clear(self):
            """Reset stored buffers and transient rollout state."""
            self.__init__()

    def __init__(
        self,
        training_type,
        num_envs,
        num_transitions_per_env,
        obs_shape,
        privileged_obs_shape,
        actions_shape,
        rnd_state_shape=None,
        device="cpu",
    ):
        """Initialize RolloutStorage with configuration, tensor shapes, and runtime state."""
        self.training_type = training_type
        self.device = torch.device(device)
        self.num_envs = int(num_envs)
        self.num_transitions_per_env = int(num_transitions_per_env)
        self.step = 0
        obs_shape = tuple(obs_shape)
        privileged_obs_shape = tuple(privileged_obs_shape)
        actions_shape = tuple(actions_shape)

        self.observations = torch.zeros(self.num_transitions_per_env, self.num_envs, *obs_shape, device=self.device)
        self.privileged_observations = torch.zeros(
            self.num_transitions_per_env, self.num_envs, *privileged_obs_shape, device=self.device
        )
        self.actions = torch.zeros(self.num_transitions_per_env, self.num_envs, *actions_shape, device=self.device)
        self.privileged_actions = torch.zeros_like(self.actions)
        self.rewards = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.dones = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device, dtype=torch.bool)

        # Rollout tensors are stored as [time, env, ...] and flattened only when
        # generating mini-batches for optimizers.
        self.values = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.actions_log_prob = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.mu = torch.zeros_like(self.actions)
        self.sigma = torch.zeros_like(self.actions)
        self.returns = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)
        self.advantages = torch.zeros(self.num_transitions_per_env, self.num_envs, 1, device=self.device)

        self.rnd_state = None
        if rnd_state_shape is not None:
            self.rnd_state = torch.zeros(self.num_transitions_per_env, self.num_envs, *rnd_state_shape, device=self.device)
        self.saved_hidden_states_a = None
        self.saved_hidden_states_c = None

    def add_transitions(self, transition: Transition):
        """Append a transition to storage and advance the rollout cursor."""
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow.")
        self.observations[self.step].copy_(transition.observations)
        priv_obs = transition.privileged_observations
        if priv_obs is None:
            priv_obs = transition.observations
        self.privileged_observations[self.step].copy_(priv_obs)
        self.actions[self.step].copy_(transition.actions)
        if transition.privileged_actions is not None:
            self.privileged_actions[self.step].copy_(transition.privileged_actions)
        if transition.values is not None:
            self.values[self.step].copy_(transition.values)
        if transition.actions_log_prob is not None:
            self.actions_log_prob[self.step].copy_(transition.actions_log_prob)
        if transition.action_mean is not None:
            self.mu[self.step].copy_(transition.action_mean)
        if transition.action_sigma is not None:
            self.sigma[self.step].copy_(transition.action_sigma)
        self.rewards[self.step].copy_(transition.rewards)
        self.dones[self.step].copy_(transition.dones)
        if self.rnd_state is not None and transition.rnd_state is not None:
            self.rnd_state[self.step].copy_(transition.rnd_state)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def _save_hidden_states(self, hidden_states):
        """Persist recurrent hidden states for the current rollout step."""
        if hidden_states is None:
            return
        actor_states, critic_states = hidden_states
        if self.saved_hidden_states_a is None:
            self.saved_hidden_states_a = self._alloc_hidden(actor_states)
            self.saved_hidden_states_c = self._alloc_hidden(critic_states)
        self._copy_hidden(self.saved_hidden_states_a, actor_states, self.step)
        self._copy_hidden(self.saved_hidden_states_c, critic_states, self.step)

    def clear(self):
        """Reset stored buffers and transient rollout state."""
        self.step = 0

    def compute_returns(self, last_values, gamma, lam, normalize_advantage: bool = True):
        """Compute bootstrapped returns and advantages for the rollout buffer."""
        advantage = 0
        for step in reversed(range(self.num_transitions_per_env)):
            next_values = last_values if step == self.num_transitions_per_env - 1 else self.values[step + 1]
            next_is_not_terminal = 1.0 - self.dones[step].float()
            # Generalized advantage estimation runs backward through the rollout
            # while masking terminal transitions.
            delta = self.rewards[step] + gamma * next_values * next_is_not_terminal - self.values[step]
            advantage = delta + gamma * lam * next_is_not_terminal * advantage
            self.returns[step] = advantage + self.values[step]
        self.advantages = self.returns - self.values
        if normalize_advantage:
            self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def generator(self):
        """Yield rollout samples for optimization."""
        batch_size = self.num_envs * self.num_transitions_per_env
        yield (
            self.observations.reshape(batch_size, *self.observations.shape[2:]),
            self.privileged_observations.reshape(batch_size, *self.privileged_observations.shape[2:]),
            self.actions.reshape(batch_size, *self.actions.shape[2:]),
            self.privileged_actions.reshape(batch_size, *self.privileged_actions.shape[2:]),
            self.dones.reshape(batch_size, 1),
        )

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """Yield shuffled feed-forward mini-batches from stored rollouts."""
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = max(1, batch_size // int(num_mini_batches))
        flat = self._flat_tensors()
        for _ in range(int(num_epochs)):
            indices = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, mini_batch_size):
                batch_idx = indices[start : start + mini_batch_size]
                yield tuple(x[batch_idx] if x is not None else None for x in flat)

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        """Yield padded recurrent mini-batches with masks and initial hidden states."""
        yield from self.mini_batch_generator(num_mini_batches, num_epochs)

    def _flat_tensors(self):
        """Flatten rollout tensors from [time, env, ...] to [batch, ...]."""
        batch_size = self.num_envs * self.num_transitions_per_env
        rnd_state = None if self.rnd_state is None else self.rnd_state.reshape(batch_size, *self.rnd_state.shape[2:])
        return (
            self.observations.reshape(batch_size, *self.observations.shape[2:]),
            self.privileged_observations.reshape(batch_size, *self.privileged_observations.shape[2:]),
            self.actions.reshape(batch_size, *self.actions.shape[2:]),
            self.values.reshape(batch_size, 1),
            self.advantages.reshape(batch_size, 1),
            self.returns.reshape(batch_size, 1),
            self.actions_log_prob.reshape(batch_size, 1),
            self.mu.reshape(batch_size, *self.mu.shape[2:]),
            self.sigma.reshape(batch_size, *self.sigma.shape[2:]),
            None,
            None,
            rnd_state,
            self.privileged_actions.reshape(batch_size, *self.privileged_actions.shape[2:]),
        )

    def _alloc_hidden(self, hidden_states):
        """Allocate tensors for the internal hidden state."""
        if hidden_states is None:
            return None
        if isinstance(hidden_states, tuple):
            return tuple(torch.zeros(self.num_transitions_per_env, *h.shape, device=h.device, dtype=h.dtype) for h in hidden_states)
        return torch.zeros(self.num_transitions_per_env, *hidden_states.shape, device=hidden_states.device, dtype=hidden_states.dtype)

    @staticmethod
    def _copy_hidden(dst, src, step):
        """Copy the internal hidden tensors into storage."""
        if dst is None or src is None:
            return
        if isinstance(src, tuple):
            for d, s in zip(dst, src):
                d[step].copy_(s)
        else:
            dst[step].copy_(src)
