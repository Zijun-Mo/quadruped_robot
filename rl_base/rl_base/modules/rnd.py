"""PyTorch policy and value-network module definitions for random network distillation."""

from __future__ import annotations

import torch
from torch import nn

from rl_base.modules.normalizer import EmpiricalDiscountedVariationNormalization, EmpiricalNormalization
from rl_base.utils.utils import make_mlp


class RandomNetworkDistillation(nn.Module):
    """Random Network Distillation module with fixed target and trainable predictor."""
    def __init__(
        self,
        num_states: int,
        num_outputs: int,
        predictor_hidden_dims: list[int],
        target_hidden_dims: list[int],
        activation: str = "elu",
        weight: float = 0.0,
        state_normalization: bool = False,
        reward_normalization: bool = False,
        device: str = "cpu",
        weight_schedule: dict | None = None,
    ):
        """Initialize RandomNetworkDistillation with configuration, tensor shapes, and runtime state."""
        super().__init__()
        self.predictor = self._build_mlp(num_states, predictor_hidden_dims, num_outputs, activation).to(device)
        self.target = self._build_mlp(num_states, target_hidden_dims, num_outputs, activation).to(device)
        for param in self.target.parameters():
            param.requires_grad = False
        self.weight = float(weight)
        self.weight_schedule = weight_schedule or {"mode": "constant"}
        self.state_normalizer = EmpiricalNormalization((num_states,)).to(device) if state_normalization else None
        self.reward_normalizer = (
            EmpiricalDiscountedVariationNormalization((1,)).to(device) if reward_normalization else None
        )
        self.step = 0

    def get_intrinsic_reward(self, rnd_state):
        """Compute intrinsic reward from prediction error in the random network distillation head."""
        if self.state_normalizer is not None:
            rnd_state = self.state_normalizer(rnd_state)
        with torch.no_grad():
            target = self.target(rnd_state)
        pred = self.predictor(rnd_state)
        reward = (pred.detach() - target).pow(2).mean(dim=-1, keepdim=True)
        if self.reward_normalizer is not None:
            reward = self.reward_normalizer(reward)
        return reward * self._current_weight(), rnd_state

    def forward(self, *args, **kwargs):
        """Run the forward pass for this module."""
        raise RuntimeError("Use get_intrinsic_reward() for RandomNetworkDistillation.")

    def train(self, mode: bool = True):
        """Switch the module to training mode and return itself."""
        self.predictor.train(mode)
        self.target.eval()
        if self.state_normalizer is not None:
            self.state_normalizer.train(mode)
        if self.reward_normalizer is not None:
            self.reward_normalizer.train(mode)
        return self

    def eval(self):
        """Switch the module to evaluation mode and return itself."""
        return self.train(False)

    @staticmethod
    def _build_mlp(input_dims, hidden_dims, output_dims, activation_name="elu"):
        """Build an MLP with the repository activation helper."""
        return make_mlp(input_dims, hidden_dims, output_dims, activation_name)

    def _current_weight(self):
        """Return the active intrinsic-reward weight for RND."""
        mode = self.weight_schedule.get("mode", "constant")
        if mode == "step":
            final_step = int(self.weight_schedule.get("final_step", 0))
            final_value = float(self.weight_schedule.get("final_value", self.weight))
            return final_value if self.step >= final_step else self.weight
        if mode == "linear":
            initial_step = int(self.weight_schedule.get("initial_step", 0))
            final_step = int(self.weight_schedule.get("final_step", initial_step + 1))
            final_value = float(self.weight_schedule.get("final_value", self.weight))
            if self.step <= initial_step:
                return self.weight
            if self.step >= final_step:
                return final_value
            p = (self.step - initial_step) / max(1, final_step - initial_step)
            return self.weight + p * (final_value - self.weight)
        return self.weight
