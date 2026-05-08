"""Feed-forward actor-critic policy and value networks."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from rl_base.utils.utils import make_mlp


class ActorCritic(nn.Module):
    """Gaussian actor-critic module with separate actor and critic MLPs."""
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=(256, 256, 256),
        critic_hidden_dims=(256, 256, 256),
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        """Initialize ActorCritic with configuration, tensor shapes, and runtime state."""
        super().__init__()
        Normal.set_default_validate_args(False)
        self.num_actions = int(num_actions)
        self.actor = make_mlp(int(num_actor_obs), actor_hidden_dims, int(num_actions), activation)
        self.critic = make_mlp(int(num_critic_obs), critic_hidden_dims, 1, activation)
        self.noise_std_type = noise_std_type
        if noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(torch.ones(num_actions) * float(init_noise_std)))
        else:
            self.std = nn.Parameter(torch.ones(num_actions) * float(init_noise_std))
        self.distribution: Normal | None = None

    @staticmethod
    def init_weights(sequential, scales):
        """Initialize linear layer weights with the configured gains."""
        linear_idx = 0
        for module in sequential:
            if isinstance(module, nn.Linear):
                scale = scales[min(linear_idx, len(scales) - 1)]
                nn.init.orthogonal_(module.weight, gain=scale)
                nn.init.constant_(module.bias, 0.0)
                linear_idx += 1

    def reset(self, dones=None):
        """No-op reset hook for the non-recurrent policy API."""
        return None

    def forward(self):
        """Run the forward pass for this module."""
        raise NotImplementedError

    @property
    def action_mean(self):
        """Return the action mean value."""
        return self.distribution.mean

    @property
    def action_std(self):
        """Return the action standard deviation value."""
        return self.distribution.stddev

    @property
    def entropy(self):
        """Return the entropy value."""
        return self.distribution.entropy().sum(dim=-1)

    def _std(self):
        """Return the current action standard-deviation tensor."""
        if hasattr(self, "log_std"):
            return torch.exp(self.log_std)
        return torch.clamp(self.std, min=1e-6)

    def update_distribution(self, observations):
        """Build the action distribution from policy features."""
        mean = self.actor(observations)
        std = self._std().expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations, **kwargs):
        """Sample actions from the current policy distribution."""
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        """Return log probabilities for actions under the current distribution."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        """Compute deterministic actions for inference without sampling noise."""
        return self.actor(observations)

    def evaluate(self, critic_observations, **kwargs):
        """Evaluate value predictions for the provided observations."""
        return self.critic(critic_observations)

    def load_state_dict(self, state_dict, strict=True):
        """Load actor-critic module parameters from a checkpoint."""
        super().load_state_dict(state_dict, strict=strict)
        return True
