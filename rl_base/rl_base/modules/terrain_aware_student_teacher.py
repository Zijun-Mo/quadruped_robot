"""Terrain-aware student-teacher module for recurrent policy distillation."""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
from torch import nn
from torch.distributions import Normal

from rl_base.networks import Memory
from rl_base.utils.utils import make_mlp

from .terrain_aware_actor_critic import TerrainAwareActorCritic


class TerrainAwareStudentTeacher(nn.Module):
    """Recurrent student policy that imitates a frozen terrain-aware teacher."""
    is_recurrent = True

    def __init__(
        self,
        num_student_obs: int,
        num_teacher_obs: int,
        num_actions: int,
        *,
        teacher_height_obs_dim: int,
        student_height_obs_dim: int = 0,
        fusion_encoder_dims: Sequence[int] | None = (256, 128, 96),
        height_cnn_channels: Sequence[int] = (16, 32),
        height_map_shape: Tuple[int, int] | None = None,
        height_encoder_dims: Sequence[int] | None = None,
        teacher_actor_hidden_dims: Sequence[int] | None = None,
        teacher_critic_hidden_dims: Sequence[int] | None = None,
        actor_hidden_dims: Sequence[int] = (512, 256, 128),
        critic_hidden_dims: Sequence[int] = (512, 256, 128),
        student_encoder_hidden_dims: Sequence[int] | None = None,
        student_policy_hidden_dims: Sequence[int] = (256, 256, 256),
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        rnn_type: str = "lstm",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        ensemble_size: int = 1,
        encoder_seeds=None,
        **kwargs,
    ):
        """Initialize TerrainAwareStudentTeacher with configuration, tensor shapes, and runtime state."""
        super().__init__()
        Normal.set_default_validate_args(False)
        teacher_actor_hidden_dims = teacher_actor_hidden_dims or (512, 256, 128)
        teacher_critic_hidden_dims = teacher_critic_hidden_dims or (512, 256, 128)
        self.teacher_height_dim = int(teacher_height_obs_dim)
        self.student_height_dim = int(student_height_obs_dim or 0)
        # Student height samples, when present, are stripped before the recurrent memory.
        self.student_core_dim = int(num_student_obs) - self.student_height_dim
        self.teacher = TerrainAwareActorCritic(
            num_teacher_obs,
            num_teacher_obs,
            num_actions,
            height_obs_dim=self.teacher_height_dim,
            actor_hidden_dims=teacher_actor_hidden_dims,
            critic_hidden_dims=teacher_critic_hidden_dims,
            fusion_encoder_dims=fusion_encoder_dims,
            height_cnn_channels=height_cnn_channels,
            height_map_shape=height_map_shape,
            activation=activation,
            init_noise_std=init_noise_std,
            noise_std_type=noise_std_type,
        )
        self.teacher_actor_fusion_dim = self.teacher.actor_fusion_dim
        self.memory_s = Memory(self.student_core_dim, rnn_type, rnn_num_layers, rnn_hidden_dim)
        encoder_hidden = tuple(student_encoder_hidden_dims or (256, 256))
        self.student_encoder = self._build_mlp(rnn_hidden_dim, encoder_hidden, self.teacher_actor_fusion_dim, activation)
        self.student_policy_head = self._build_mlp(
            self.teacher_actor_fusion_dim, student_policy_hidden_dims, num_actions, activation
        )
        self.student = nn.ModuleDict({"encoder": self.student_encoder, "policy": self.student_policy_head})
        if noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(torch.ones(num_actions) * float(init_noise_std)))
        else:
            self.std = nn.Parameter(torch.ones(num_actions) * float(init_noise_std))
        self.noise_std_type = noise_std_type
        self.distribution = None
        self.loaded_teacher = False

    @staticmethod
    def _build_mlp(input_dim, hidden_dims, output_dim, activation_name):
        """Build an MLP with the repository activation helper."""
        return make_mlp(input_dim, hidden_dims, output_dim, activation_name)

    @staticmethod
    def _split_obs(obs, height_dim):
        """Split the internal observations tensor into its expected parts."""
        if height_dim <= 0:
            return obs, None
        # Height observations follow the core observation features at the tail.
        return obs[..., :-height_dim], obs[..., -height_dim:]

    def _student_core_features(self, observations, masks=None, hidden_states=None):
        """Encode non-height student observations through the recurrent memory."""
        core, _ = self._split_obs(observations, self.student_height_dim)
        return self.memory_s(core, masks=masks, hidden_states=hidden_states)

    def update_distribution(self, features):
        """Build the action distribution from policy features."""
        mean = self.student_policy_head(features)
        std = self._std().expand_as(mean)
        self.distribution = Normal(mean, std)

    def _std(self):
        """Return the current student action standard-deviation tensor."""
        if hasattr(self, "log_std"):
            return torch.exp(self.log_std)
        return torch.clamp(self.std, min=1e-6)

    def act(self, observations, masks=None, hidden_states=None):
        """Sample actions from the current policy distribution."""
        rnn_features = self._student_core_features(observations, masks=masks, hidden_states=hidden_states)
        latent = self.student_encoder(rnn_features)
        self.update_distribution(latent)
        return self.distribution.sample()

    def act_inference(self, observations, *, return_latent: bool = False):
        """Compute deterministic actions for inference without sampling noise."""
        rnn_features = self._student_core_features(observations)
        latent = self.student_encoder(rnn_features)
        actions = self.student_policy_head(latent)
        if return_latent:
            return actions, latent
        return actions

    def evaluate(self, teacher_observations):
        """Return deterministic teacher actions for privileged observations."""
        with torch.no_grad():
            return self.teacher.act_inference(teacher_observations)

    def evaluate_feature(self, teacher_observations):
        """Return frozen teacher features for student distillation."""
        with torch.no_grad():
            return self.teacher.evaluate_actor_features(teacher_observations)

    def get_student_latent(self, observations, masks=None, hidden_states=None):
        """Encode student observations into the latent feature space."""
        return self.student_encoder(self._student_core_features(observations, masks=masks, hidden_states=hidden_states))

    def get_actions_log_prob(self, actions):
        """Return log probabilities for actions under the current distribution."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def reset(self, dones=None, hidden_states=None):
        """Reset environment, module, or buffer state."""
        self.memory_s.reset(dones, hidden_states=hidden_states)

    def get_hidden_states(self):
        """Return recurrent hidden states in the format expected by storage or runners."""
        return self.memory_s.hidden_states, None

    def detach_hidden_states(self, dones=None):
        """Detach recurrent hidden states from the current autograd graph."""
        self.memory_s.detach_hidden_states(dones)

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

    def load_state_dict(self, state_dict, strict: bool = False):
        """Load full student-teacher weights or recover a terrain-aware teacher checkpoint."""
        keys = list(state_dict.keys())
        # New checkpoints store both teacher and student namespaces; older teacher-only
        # checkpoints store actor/critic/height modules at the root.
        if any(k.startswith("teacher.") for k in keys) or any(k.startswith("student_encoder.") for k in keys):
            super().load_state_dict(state_dict, strict=False)
            self.loaded_teacher = True
            return True

        teacher_state = {}
        for key, value in state_dict.items():
            if key.startswith(("height_encoder.", "actor_fusion_encoder.", "critic_fusion_encoder.", "actor.", "critic.", "std", "log_std")):
                teacher_state[key] = value
        if teacher_state:
            self.teacher.load_state_dict(teacher_state, strict=False)
            self.loaded_teacher = True
            return False

        super().load_state_dict(state_dict, strict=False)
        self.loaded_teacher = True
        return True
