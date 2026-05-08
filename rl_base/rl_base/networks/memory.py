"""Recurrent memory wrapper used by rl_base policy modules."""

from __future__ import annotations

import torch
from torch import nn

from rl_base.utils import unpad_trajectories


class Memory(nn.Module):
    """Small LSTM/GRU wrapper for online inference and recurrent batches."""

    def __init__(self, input_size: int, type: str = "lstm", num_layers: int = 1, hidden_size: int = 256):
        """Initialize Memory with configuration, tensor shapes, and runtime state."""
        super().__init__()
        rnn_type = type.lower()
        if rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        elif rnn_type == "gru":
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        else:
            raise ValueError(f"Unsupported RNN type: {type}")
        self.type = rnn_type
        self.hidden_states = None

    def forward(self, input: torch.Tensor, masks: torch.Tensor | None = None, hidden_states=None) -> torch.Tensor:
        """Encode observations with online hidden state or padded rollout masks."""
        if masks is not None:
            if input.dim() == 2:
                input = input.unsqueeze(0)
            if hidden_states is None:
                hidden_states = self._zeros(input.shape[1], input.device, input.dtype)
            # Training batches arrive as padded [time, env, feature] sequences and are
            # unpadded after the RNN so losses can use the flattened mini-batch layout.
            out, _ = self.rnn(input, hidden_states)
            return unpad_trajectories(out, masks)

        if input.dim() == 1:
            input = input.unsqueeze(0)
        batch_size = input.shape[0]
        if hidden_states is not None:
            out, _ = self.rnn(input.unsqueeze(0), hidden_states)
            return out.squeeze(0)
        if self.hidden_states is None or self._hidden_batch_size(self.hidden_states) != batch_size:
            self.hidden_states = self._zeros(batch_size, input.device, input.dtype)
        out, self.hidden_states = self.rnn(input.unsqueeze(0), self.hidden_states)
        return out.squeeze(0)

    def reset(self, dones: torch.Tensor | None = None, hidden_states=None):
        """Clear all hidden state or only the entries for completed environments."""
        if hidden_states is not None:
            self.hidden_states = hidden_states
            return
        if self.hidden_states is None:
            return
        if dones is None:
            self.hidden_states = None
            return
        dones = dones.reshape(-1).to(dtype=torch.bool, device=self._hidden_device(self.hidden_states))
        if self.type == "lstm":
            h, c = self.hidden_states
            h[:, dones, :] = 0.0
            c[:, dones, :] = 0.0
        else:
            self.hidden_states[:, dones, :] = 0.0

    def detach_hidden_states(self, dones: torch.Tensor | None = None):
        """Detach recurrent hidden states from the current autograd graph."""
        if self.hidden_states is None:
            return
        if self.type == "lstm":
            h, c = self.hidden_states
            self.hidden_states = (h.detach(), c.detach())
        else:
            self.hidden_states = self.hidden_states.detach()
        if dones is not None:
            self.reset(dones)

    def _zeros(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Allocate zero hidden states with the requested batch size and device."""
        shape = (self.rnn.num_layers, batch_size, self.rnn.hidden_size)
        if self.type == "lstm":
            return (torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype))
        return torch.zeros(shape, device=device, dtype=dtype)

    @staticmethod
    def _hidden_batch_size(hidden_states) -> int:
        """Return the batch dimension encoded by a hidden-state tuple or tensor."""
        if isinstance(hidden_states, tuple):
            return hidden_states[0].shape[1]
        return hidden_states.shape[1]

    @staticmethod
    def _hidden_device(hidden_states) -> torch.device:
        """Return the device used by a hidden-state tuple or tensor."""
        if isinstance(hidden_states, tuple):
            return hidden_states[0].device
        return hidden_states.device
