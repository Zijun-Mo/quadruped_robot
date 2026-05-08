"""Terrain-aware actor-critic networks that fuse proprioception with height maps."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Normal

from rl_base.utils import resolve_nn_activation
from rl_base.utils.utils import make_mlp


class TerrainAwareActorCritic(nn.Module):
    """Actor-critic policy that appends encoded terrain-height features to core observations."""
    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        *,
        height_obs_dim: int,
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        critic_hidden_dims: Sequence[int] = (256, 256, 256),
        fusion_encoder_dims: Sequence[int] | None = (256, 128, 96),
        height_cnn_channels: Sequence[int] = (16, 32),
        height_map_shape: Tuple[int, int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        height_encoder_dims: Sequence[int] | None = None,
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        build_critic: bool = True,
        **kwargs,
    ):
        """Initialize TerrainAwareActorCritic with configuration, tensor shapes, and runtime state."""
        super().__init__()
        Normal.set_default_validate_args(False)
        self.num_actions = int(num_actions)
        self.height_dim = int(height_obs_dim or 0)
        self.build_critic = build_critic
        self.noise_std_type = noise_std_type
        self.height_map_shape = self._resolve_height_map_shape(self.height_dim, height_map_shape)

        # Height samples are stored as a flattened tail in the observation vector and are
        # reshaped to [batch, 1, height, width] only when the CNN encoder is used.
        if self.height_dim > 0:
            self.height_encoder, height_embedding_dim = self._build_height_cnn(
                self.height_map_shape, height_cnn_channels, activation
            )
        else:
            self.height_encoder = nn.Identity()
            height_embedding_dim = 0

        actor_core_dim = int(num_actor_obs) - self.height_dim
        critic_core_dim = int(num_critic_obs) - self.height_dim
        if actor_core_dim < 0 or critic_core_dim < 0:
            raise ValueError("height_obs_dim cannot exceed observation dimension.")

        fusion_dims = tuple(fusion_encoder_dims or ())
        self.actor_fusion_encoder, self.actor_fusion_dim = self._build_fusion_encoder(
            actor_core_dim + height_embedding_dim, fusion_dims, activation
        )
        if build_critic:
            self.critic_fusion_encoder, self.critic_fusion_dim = self._build_fusion_encoder(
                critic_core_dim + height_embedding_dim, fusion_dims, activation
            )
        else:
            self.critic_fusion_encoder = None
            self.critic_fusion_dim = self.actor_fusion_dim

        self.actor = self._build_head(self.actor_fusion_dim, actor_hidden_dims, num_actions, activation)
        if build_critic:
            self.critic = self._build_head(self.critic_fusion_dim, critic_hidden_dims, 1, activation)

        if noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(torch.ones(num_actions) * float(init_noise_std)))
        else:
            self.std = nn.Parameter(torch.ones(num_actions) * float(init_noise_std))
        self.distribution = None

    @staticmethod
    def _resolve_height_map_shape(height_dim, explicit_shape):
        """Resolve the 2-D height-map shape from explicit config or factorization."""
        if height_dim <= 0:
            return (0, 0)
        if explicit_shape is not None:
            if math.prod(explicit_shape) != height_dim:
                raise ValueError(f"height_map_shape={explicit_shape} does not match height_obs_dim={height_dim}")
            return tuple(explicit_shape)
        best = (1, height_dim)
        best_delta = height_dim
        for rows in range(1, int(math.sqrt(height_dim)) + 1):
            if height_dim % rows == 0:
                cols = height_dim // rows
                delta = abs(cols - rows)
                if delta < best_delta:
                    best = (rows, cols)
                    best_delta = delta
        return best

    @staticmethod
    def _build_head(input_dim, hidden_dims, output_dim, activation_name):
        """Build an MLP head for policy means or value predictions."""
        return make_mlp(input_dim, hidden_dims, output_dim, activation_name)

    @staticmethod
    def _build_height_cnn(map_shape, channels, activation_name):
        """Build the CNN encoder for flattened terrain-height maps."""
        layers = []
        in_channels = 1
        h, w = map_shape
        for out_channels in channels:
            # Track the spatial size explicitly so the later linear head sees the
            # correct flattened CNN embedding dimension.
            layers.append(nn.Conv2d(in_channels, int(out_channels), kernel_size=3, stride=2))
            layers.append(resolve_nn_activation(activation_name))
            h = (h - 3) // 2 + 1
            w = (w - 3) // 2 + 1
            if h <= 0 or w <= 0:
                raise ValueError(f"Height map shape {map_shape} is too small for two stride-2 conv layers.")
            in_channels = int(out_channels)
        layers.append(nn.Flatten())
        return nn.Sequential(*layers), int(in_channels * h * w)

    @staticmethod
    def _build_fusion_encoder(input_dim, hidden_dims, activation_name):
        """Build the encoder that fuses core observations and terrain features."""
        if not hidden_dims:
            return nn.Identity(), int(input_dim)
        layers = []
        dims = [int(input_dim), *[int(d) for d in hidden_dims]]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(resolve_nn_activation(activation_name))
        return nn.Sequential(*layers), dims[-1]

    def reset(self, dones=None):
        """Reset environment, module, or buffer state."""
        return None

    def get_hidden_states(self):
        """Return recurrent hidden states in the format expected by storage or runners."""
        return None, None

    def detach_hidden_states(self, dones=None):
        """Detach recurrent hidden states from the current autograd graph."""
        return None

    def _split_obs(self, obs, height_dim):
        """Split the internal observations tensor into its expected parts."""
        if height_dim <= 0:
            return obs, None
        # The environment appends terrain samples after proprioceptive features.
        return obs[..., :-height_dim], obs[..., -height_dim:]

    def _encode_height(self, height, height_dim):
        """Encode flattened terrain-height observations into latent features."""
        if height_dim <= 0:
            return None
        # CNNs expect channel-first image-like input, while the environment provides a flat vector.
        height = height.reshape(height.shape[0], 1, *self.height_map_shape)
        return self.height_encoder(height)

    def _prepare_features(self, observations, height_dim, fusion_encoder):
        """Split observations and concatenate proprioceptive and terrain features."""
        core, height = self._split_obs(observations, height_dim)
        if height_dim > 0:
            height_features = self._encode_height(height, height_dim)
            core = torch.cat((core, height_features), dim=-1)
        return fusion_encoder(core)

    def update_distribution(self, features):
        """Build the action distribution from policy features."""
        mean = self.actor(features)
        std = self._std().expand_as(mean)
        self.distribution = Normal(mean, std)

    def _std(self):
        """Return the current terrain-aware action standard-deviation tensor."""
        if hasattr(self, "log_std"):
            return torch.exp(self.log_std)
        return torch.clamp(self.std, min=1e-6)

    def act(self, observations, masks=None, hidden_states=None):
        """Sample actions from the current policy distribution."""
        features = self._prepare_features(observations, self.height_dim, self.actor_fusion_encoder)
        self.update_distribution(features)
        return self.distribution.sample()

    def act_inference(self, observations):
        """Compute deterministic actions for inference without sampling noise."""
        features = self._prepare_features(observations, self.height_dim, self.actor_fusion_encoder)
        return self.actor(features)

    def evaluate(self, critic_observations, masks=None, hidden_states=None):
        """Evaluate value predictions for the provided observations."""
        if not self.build_critic:
            raise RuntimeError("TerrainAwareActorCritic was created with build_critic=False.")
        features = self._prepare_features(critic_observations, self.height_dim, self.critic_fusion_encoder)
        return self.critic(features)

    def evaluate_actor_features(self, observations):
        """Return actor-side latent features for distillation or diagnostics."""
        return self._prepare_features(observations, self.height_dim, self.actor_fusion_encoder)

    def get_actions_log_prob(self, actions):
        """Return log probabilities for actions under the current distribution."""
        return self.distribution.log_prob(actions).sum(dim=-1)

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

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load a terrain-aware actor-critic checkpoint, optionally ignoring critic keys."""
        if not self.build_critic:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("critic")}
            strict = False
        super().load_state_dict(state_dict, strict=strict)
        return True
