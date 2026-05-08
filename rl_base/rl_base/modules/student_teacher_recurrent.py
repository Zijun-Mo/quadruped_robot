"""PyTorch policy and value-network module definitions for student teacher recurrent."""

from __future__ import annotations

from .student_teacher import StudentTeacher
from rl_base.networks import Memory
from rl_base.utils.utils import make_mlp


class StudentTeacherRecurrent(StudentTeacher):
    """Student-teacher policy module for student teacher recurrent distillation."""
    is_recurrent = True

    def __init__(
        self,
        num_student_obs,
        num_teacher_obs,
        num_actions,
        student_hidden_dims=(256, 256, 256),
        teacher_hidden_dims=(256, 256, 256),
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        init_noise_std=0.1,
        teacher_recurrent=False,
        **kwargs,
    ):
        """Initialize StudentTeacherRecurrent with configuration, tensor shapes, and runtime state."""
        self.teacher_recurrent = teacher_recurrent
        if teacher_recurrent:
            teacher_in = rnn_hidden_dim
        else:
            teacher_in = num_teacher_obs
        super().__init__(
            rnn_hidden_dim,
            teacher_in,
            num_actions,
            student_hidden_dims,
            teacher_hidden_dims,
            activation,
            init_noise_std,
            **kwargs,
        )
        self.memory_s = Memory(num_student_obs, rnn_type, rnn_num_layers, rnn_hidden_dim)
        if teacher_recurrent:
            self.memory_t = Memory(num_teacher_obs, rnn_type, rnn_num_layers, rnn_hidden_dim)

    def reset(self, dones=None, hidden_states=None):
        """Reset environment, module, or buffer state."""
        self.memory_s.reset(dones)
        if self.teacher_recurrent:
            self.memory_t.reset(dones)

    def act(self, observations):
        """Sample actions from the current policy distribution."""
        features = self.memory_s(observations)
        return super().act(features)

    def act_inference(self, observations):
        """Compute deterministic actions for inference without sampling noise."""
        features = self.memory_s(observations)
        return self.student(features)

    def evaluate(self, teacher_observations):
        """Return deterministic recurrent teacher actions for privileged observations."""
        if self.teacher_recurrent:
            teacher_observations = self.memory_t(teacher_observations)
        return super().evaluate(teacher_observations)

    def get_hidden_states(self):
        """Return recurrent hidden states in the format expected by storage or runners."""
        return self.memory_s.hidden_states, self.memory_t.hidden_states if self.teacher_recurrent else None

    def detach_hidden_states(self, dones=None):
        """Detach recurrent hidden states from the current autograd graph."""
        self.memory_s.detach_hidden_states(dones)
        if self.teacher_recurrent:
            self.memory_t.detach_hidden_states(dones)
