"""PyTorch policy and value-network module definitions for normalizer."""

from __future__ import annotations

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Tracks running mean and variance for observation normalization."""
    def __init__(self, shape, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization with configuration, tensor shapes, and runtime state."""
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape))
        self.register_buffer("_var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(0.0))

    @property
    def mean(self):
        """Return the mean value."""
        return self._mean.clone()

    @property
    def std(self):
        """Return the standard deviation value."""
        return torch.sqrt(self._var + self.eps).clone()

    def forward(self, x):
        """Run the forward pass for this module."""
        if self.training:
            self.update(x)
        return (x - self._mean.to(x.device)) / torch.sqrt(self._var.to(x.device) + self.eps)

    def update(self, x):
        """Run one optimization update and return training statistics."""
        if self.until is not None and self.count >= self.until:
            return
        x = x.detach()
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, unbiased=False)
        batch_count = x.shape[0]
        if self.count.item() == 0:
            self._mean.copy_(batch_mean)
            self._var.copy_(batch_var + self.eps)
            self.count += batch_count
            return
        delta = batch_mean - self._mean
        total = self.count + batch_count
        new_mean = self._mean + delta * batch_count / total
        m_a = self._var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total
        self._mean.copy_(new_mean)
        self._var.copy_(m2 / total)
        self.count.copy_(total)

    def inverse(self, y):
        """Map normalized values back into the original observation scale."""
        return y * torch.sqrt(self._var.to(y.device) + self.eps) + self._mean.to(y.device)


class DiscountedAverage:
    """Maintains an exponential moving average of discounted values."""
    def __init__(self, gamma):
        """Initialize DiscountedAverage with configuration, tensor shapes, and runtime state."""
        self.gamma = gamma
        self.avg = None

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        """Run one optimization update and return training statistics."""
        if self.avg is None or self.avg.shape != rew.shape:
            self.avg = torch.zeros_like(rew)
        self.avg = self.gamma * self.avg + rew
        return self.avg


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Normalizes values using discounted empirical variation estimates."""
    def __init__(self, shape, eps=1e-2, gamma=0.99, until=None):
        """Initialize EmpiricalDiscountedVariationNormalization with configuration, tensor shapes, and runtime state."""
        super().__init__()
        self.discounted_average = DiscountedAverage(gamma)
        self.emp_norm = EmpiricalNormalization(shape, eps=eps, until=until)

    def forward(self, rew):
        """Run the forward pass for this module."""
        discounted = self.discounted_average.update(rew)
        if self.training:
            self.emp_norm.update(discounted)
        return rew / self.emp_norm.std.to(rew.device)
