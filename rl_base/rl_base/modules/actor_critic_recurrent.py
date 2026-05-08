"""PyTorch policy and value-network module definitions for actor critic recurrent."""

from __future__ import annotations

from .actor_critic import ActorCritic
from rl_base.networks import Memory


class ActorCriticRecurrent(ActorCritic):
    """Actor-critic network module for actor critic recurrent policies."""
    is_recurrent = True

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=1.0,
        **kwargs,
    ):
        """Initialize ActorCriticRecurrent with configuration, tensor shapes, and runtime state."""
        super().__init__(
            rnn_hidden_dim,
            rnn_hidden_dim,
            num_actions,
            actor_hidden_dims,
            critic_hidden_dims,
            activation,
            init_noise_std,
            **kwargs,
        )
        self.memory_a = Memory(num_actor_obs, rnn_type, rnn_num_layers, rnn_hidden_dim)
        self.memory_c = Memory(num_critic_obs, rnn_type, rnn_num_layers, rnn_hidden_dim)

    def reset(self, dones=None):
        """Reset environment, module, or buffer state."""
        self.memory_a.reset(dones)
        self.memory_c.reset(dones)

    def act(self, observations, masks=None, hidden_states=None):
        """Sample actions from the current policy distribution."""
        h = None if hidden_states is None else hidden_states[0]
        features = self.memory_a(observations, masks=masks, hidden_states=h)
        return super().act(features)

    def act_inference(self, observations):
        """Compute deterministic actions for inference without sampling noise."""
        features = self.memory_a(observations)
        return self.actor(features)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """Evaluate value predictions for the provided observations."""
        h = None if hidden_states is None else hidden_states[1]
        features = self.memory_c(critic_observations, masks=masks, hidden_states=h)
        return self.critic(features)

    def get_hidden_states(self):
        """Return recurrent hidden states in the format expected by storage or runners."""
        return self.memory_a.hidden_states, self.memory_c.hidden_states
