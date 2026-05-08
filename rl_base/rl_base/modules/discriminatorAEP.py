"""PyTorch policy and value-network module definitions for discriminator a e p."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from rl_base.utils import resolve_nn_activation


class Discriminator(nn.Module):
    """Adversarial discriminator used to score policy and expert features."""
    def __init__(
        self,
        input_dim: int,
        hidden_layer_sizes: list[int],
        *,
        device: str = "cpu",
        loss_type: str = "BCEWithLogits",
        eta_wgan: float = 0.3,
        use_minibatch_std: bool = True,
    ):
        """Initialize Discriminator with configuration, tensor shapes, and runtime state."""
        super().__init__()
        self.loss_type = loss_type
        self.eta_wgan = eta_wgan
        self.use_minibatch_std = use_minibatch_std
        dims = [input_dim, *hidden_layer_sizes, 1]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(resolve_nn_activation("elu"))
        self.net = nn.Sequential(*layers).to(device)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Run the forward pass for this module."""
        return self.net(latent)

    def classify(self, latent: torch.Tensor) -> torch.Tensor:
        """Return discriminator logits for policy or expert features."""
        return torch.sigmoid(self.forward(latent))

    def generator_loss(self, student_latent: torch.Tensor) -> torch.Tensor:
        """Compute the generator-side adversarial loss."""
        logits = self.forward(student_latent)
        if self.loss_type.lower().startswith("wasser"):
            return -logits.mean()
        return F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))

    def discriminator_loss(self, student_latent: torch.Tensor, teacher_latent: torch.Tensor) -> torch.Tensor:
        """Compute the discriminator loss for expert and policy samples."""
        student_logits = self.forward(student_latent.detach())
        teacher_logits = self.forward(teacher_latent.detach())
        if self.loss_type.lower().startswith("wasser"):
            return student_logits.mean() - teacher_logits.mean()
        loss_s = F.binary_cross_entropy_with_logits(student_logits, torch.zeros_like(student_logits))
        loss_t = F.binary_cross_entropy_with_logits(teacher_logits, torch.ones_like(teacher_logits))
        return 0.5 * (loss_s + loss_t)

    def compute_grad_pen(self, teacher_latent, student_latent, lambda_=10.0) -> torch.Tensor:
        """Compute the discriminator gradient penalty term."""
        alpha = torch.rand(teacher_latent.shape[0], 1, device=teacher_latent.device)
        mixed = alpha * teacher_latent + (1 - alpha) * student_latent
        mixed.requires_grad_(True)
        logits = self.forward(mixed)
        grad = torch.autograd.grad(logits.sum(), mixed, create_graph=True, retain_graph=True, only_inputs=True)[0]
        return lambda_ * (grad.norm(2, dim=1) - 1).pow(2).mean()

    @staticmethod
    def _minibatch_std_scalar(h: torch.Tensor) -> torch.Tensor:
        """Compute the scalar minibatch-standard-deviation feature used by the discriminator."""
        return h.std(dim=0, unbiased=False).mean().expand(h.shape[0], 1)
