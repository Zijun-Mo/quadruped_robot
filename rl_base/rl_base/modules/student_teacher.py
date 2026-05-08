"""Feed-forward student-teacher policy module for action distillation."""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from rl_base.utils.utils import make_mlp


class StudentTeacher(nn.Module):
    """Student policy paired with a teacher action network for distillation."""
    is_recurrent = False

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_dims=(256, 256, 256),
        teacher_hidden_dims=(256, 256, 256),
        activation="elu",
        init_noise_std=0.1,
        **kwargs,
    ):
        """Initialize StudentTeacher with configuration, tensor shapes, and runtime state."""
        super().__init__()
        Normal.set_default_validate_args(False)
        self.student = make_mlp(num_student_obs, student_hidden_dims, num_actions, activation)
        self.teacher = make_mlp(num_teacher_obs, teacher_hidden_dims, num_actions, activation)
        self.std = nn.Parameter(torch.ones(num_actions) * float(init_noise_std))
        self.distribution = None
        self.loaded_teacher = False

    def reset(self, dones=None, hidden_states=None):
        """No-op reset hook for the non-recurrent distillation policy API."""
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

    def update_distribution(self, observations):
        """Build the action distribution from policy features."""
        mean = self.student(observations)
        self.distribution = Normal(mean, torch.clamp(self.std, min=1e-6).expand_as(mean))

    def act(self, observations):
        """Sample actions from the current policy distribution."""
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations):
        """Compute deterministic actions for inference without sampling noise."""
        return self.student(observations)

    def evaluate(self, teacher_observations):
        """Return deterministic teacher actions for privileged observations."""
        with torch.no_grad():
            return self.teacher(teacher_observations)

    def get_actions_log_prob(self, actions):
        """Return log probabilities for actions under the current distribution."""
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        """Return recurrent hidden states in the format expected by storage or runners."""
        return None

    def detach_hidden_states(self, dones=None):
        """Detach recurrent hidden states from the current autograd graph."""
        return None

    def load_state_dict(self, state_dict, strict=True):
        """Load student-teacher weights or recover a teacher-only actor checkpoint."""
        keys = list(state_dict.keys())
        # Teacher pretraining checkpoints save the teacher as an actor network; full
        # distillation checkpoints include both student and teacher namespaces.
        if any(k.startswith("student.") for k in keys):
            super().load_state_dict(state_dict, strict=False)
            self.loaded_teacher = True
            return True
        teacher_state = {k.removeprefix("actor."): v for k, v in state_dict.items() if k.startswith("actor.")}
        if teacher_state:
            self.teacher.load_state_dict(teacher_state, strict=False)
            self.loaded_teacher = True
            return False
        super().load_state_dict(state_dict, strict=False)
        self.loaded_teacher = True
        return True
