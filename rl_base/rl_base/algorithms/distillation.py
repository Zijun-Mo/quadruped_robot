"""Training algorithm implementation for distillation policies."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from rl_base.storage import RolloutStorage


class Distillation:
    """Training algorithm implementation for Distillation."""
    def __init__(
        self,
        policy,
        num_learning_epochs=1,
        num_mini_batches=4,
        clip_param=0.2,
        gamma=0.99,
        lam=0.95,
        value_loss_coef=1.0,
        entropy_coef=0.0,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule="fixed",
        desired_kl=0.01,
        bc_loss_coef=1.0,
        RL_loss_coef=1.0,
        use_action_imitation_reward=False,
        action_imitation_reward_coef=1.0,
        use_mse_loss=True,
        device="cpu",
        uncertainty_abs_coef=1.0,
        uncertainty_delta_coef=2.0,
        uncertainty_delta_threshold=0.03,
        uncertainty_max=1.0,
        uncertainty_min=0.0,
        uncertainty_warmup_iters=500,
        uncertainty_ema_beta=0.01,
        uncertainty_eps=1e-6,
        curriculum_enable=False,
        curriculum_start_iter=1000,
        curriculum_ramp_iters=1500,
        curriculum_final_rl_coef=1.0,
        curriculum_final_bc_coef=0.0,
        curriculum_noise_start: float | None = None,
        curriculum_noise_target=0.8,
        curriculum_noise_handover_to_rl=True,
        curriculum_type="linear",
        multi_gpu_cfg: dict | None = None,
        **kwargs,
    ):
        """Initialize Distillation with configuration, tensor shapes, and runtime state."""
        self.policy = policy
        self.device = torch.device(device)
        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.bc_loss_coef = bc_loss_coef
        self.RL_loss_coef = RL_loss_coef
        self.use_action_imitation_reward = use_action_imitation_reward
        self.action_imitation_reward_coef = action_imitation_reward_coef
        self.use_mse_loss = use_mse_loss
        self.curriculum_enable = curriculum_enable
        self.curriculum_start_iter = int(curriculum_start_iter)
        self.curriculum_ramp_iters = int(curriculum_ramp_iters)
        self.curriculum_final_rl_coef = curriculum_final_rl_coef
        self.curriculum_final_bc_coef = curriculum_final_bc_coef
        self.curriculum_noise_start = curriculum_noise_start
        self.curriculum_noise_target = curriculum_noise_target
        self.curriculum_noise_handover_to_rl = curriculum_noise_handover_to_rl
        self.curriculum_type = curriculum_type
        self.update_counter = 0
        self.active_bc_coef = bc_loss_coef
        self.active_rl_coef = RL_loss_coef
        self.storage = None
        self.transition = RolloutStorage.Transition()
        params = list({id(p): p for p in self.policy.parameters() if p.requires_grad}.values())
        self.optimizer = torch.optim.Adam(params, lr=learning_rate)
        self.discriminator = None

        teacher = getattr(self.policy, "teacher", None)
        if teacher is not None:
            for name, param in teacher.named_parameters():
                if "critic" not in name:
                    # Keep the teacher policy fixed while allowing critic values to train when exposed.
                    param.requires_grad_(False)

    def init_storage(self, training_type, num_envs, num_transitions_per_env, student_obs_shape, teacher_obs_shape, actions_shape):
        """Allocate rollout storage with shapes derived from the environment."""
        self.storage = RolloutStorage(
            "rl",
            num_envs,
            num_transitions_per_env,
            student_obs_shape,
            teacher_obs_shape,
            actions_shape,
            device=self.device,
        )

    def compute_returns(self, last_teacher_obs):
        """Compute bootstrapped returns and advantages for the rollout buffer."""
        with torch.no_grad():
            last_values = self._teacher_value(last_teacher_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def act(self, obs, teacher_obs):
        """Sample actions from the current policy distribution."""
        if hasattr(self.policy, "get_hidden_states"):
            self.transition.hidden_states = self.policy.get_hidden_states()
        actions = self.policy.act(obs)
        with torch.no_grad():
            teacher_actions = self.policy.evaluate(teacher_obs)
            values = self._teacher_value(teacher_obs)
        self.transition.observations = obs.detach()
        self.transition.privileged_observations = teacher_obs.detach()
        self.transition.actions = actions.detach()
        self.transition.privileged_actions = teacher_actions.detach()
        self.transition.values = values.detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(actions).detach().unsqueeze(-1)
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        return actions.detach()

    def process_env_step(self, rewards, dones, infos):
        """Record environment feedback and update rollout bookkeeping after a step."""
        rewards = rewards.to(self.device).view(-1, 1)
        dones = dones.to(self.device).view(-1, 1).bool()
        if self.use_action_imitation_reward and self.transition.privileged_actions is not None:
            imitation = -torch.square(self.transition.actions - self.transition.privileged_actions).mean(dim=-1, keepdim=True)
            rewards = rewards + self.action_imitation_reward_coef * imitation
        if "time_outs" in infos:
            timeouts = infos["time_outs"].to(self.device).view(-1, 1).float()
            # Time-limit truncations bootstrap from the teacher value instead of being
            # treated as terminal failures.
            rewards = rewards + self.gamma * self.transition.values * timeouts
        self.transition.rewards = rewards.detach()
        self.transition.dones = dones
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def _compute_curriculum_progress(self, update_idx: int) -> float:
        """Compute normalized curriculum progress for the current update index."""
        if not self.curriculum_enable:
            return 0.0
        if update_idx < self.curriculum_start_iter:
            return 0.0
        return min(1.0, (update_idx - self.curriculum_start_iter) / max(1, self.curriculum_ramp_iters))

    @staticmethod
    def _mix_with_progress(start_value: float, end_value: float, progress: float) -> float:
        """Linearly interpolate between two scalar curriculum endpoints."""
        return float(start_value + (end_value - start_value) * progress)

    def _get_current_noise_std(self) -> float:
        """Return the mean action-noise standard deviation currently used by the policy."""
        if hasattr(self.policy, "log_std"):
            return float(torch.exp(self.policy.log_std).mean().item())
        if hasattr(self.policy, "std"):
            return float(torch.clamp(self.policy.std, min=1e-6).mean().item())
        return 0.0

    def _set_policy_noise_std(self, target_std: float) -> None:
        """Set scalar or log-parameterized policy exploration noise."""
        with torch.no_grad():
            if hasattr(self.policy, "log_std"):
                self.policy.log_std.fill_(torch.log(torch.tensor(float(target_std), device=self.policy.log_std.device)))
            elif hasattr(self.policy, "std"):
                self.policy.std.fill_(float(target_std))

    def _update_curriculum_state(self) -> None:
        """Update active BC/RL loss weights and optional exploration noise."""
        progress = self._compute_curriculum_progress(self.update_counter)
        if self.curriculum_enable:
            self.active_bc_coef = self._mix_with_progress(self.bc_loss_coef, self.curriculum_final_bc_coef, progress)
            self.active_rl_coef = self._mix_with_progress(self.RL_loss_coef, self.curriculum_final_rl_coef, progress)
            if self.curriculum_noise_start is not None:
                self._set_policy_noise_std(
                    self._mix_with_progress(self.curriculum_noise_start, self.curriculum_noise_target, progress)
                )
        else:
            self.active_bc_coef = self.bc_loss_coef
            self.active_rl_coef = self.RL_loss_coef

    def update(self):
        """Run one optimization update and return training statistics."""
        self._update_curriculum_state()
        means = {
            "mse_loss": 0.0,
            "bc_loss": 0.0,
            "surrogate_loss": 0.0,
            "value_function": 0.0,
            "latent/student_mean_norm": 0.0,
            "latent/teacher_mean_norm": 0.0,
        }
        generator = self.storage.recurrent_mini_batch_generator if getattr(self.policy, "is_recurrent", False) else self.storage.mini_batch_generator
        num_updates = 0
        for batch in generator(self.num_mini_batches, self.num_learning_epochs):
            (
                obs_batch,
                teacher_obs_batch,
                actions_batch,
                target_values_batch,
                advantages_batch,
                returns_batch,
                old_actions_log_prob_batch,
                old_mu_batch,
                old_sigma_batch,
                hid_states_batch,
                masks_batch,
                rnd_state_batch,
                privileged_actions_batch,
            ) = batch
            if getattr(self.policy, "is_recurrent", False):
                self.policy.reset()
            if hasattr(self.policy, "get_student_latent"):
                student_latent = self.policy.get_student_latent(obs_batch, masks=masks_batch, hidden_states=hid_states_batch)
                teacher_latent = self.policy.evaluate_feature(teacher_obs_batch)
                self.policy.update_distribution(student_latent)
            else:
                self.policy.act(obs_batch)
                student_latent = self.policy.action_mean
                teacher_latent = privileged_actions_batch

            # The update combines latent distillation, action imitation, clipped PPO,
            # and value losses; active coefficients are curriculum-controlled.
            actions_log_prob = self.policy.get_actions_log_prob(actions_batch).unsqueeze(-1)
            value_batch = self._teacher_value(teacher_obs_batch)
            entropy_batch = self.policy.entropy.unsqueeze(-1)
            bc_loss = F.mse_loss(self.policy.action_mean, privileged_actions_batch)
            mse_loss = F.mse_loss(student_latent, teacher_latent)
            ratio = torch.exp(actions_log_prob - old_actions_log_prob_batch)
            surrogate = -advantages_batch * ratio
            surrogate_clipped = -advantages_batch * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_loss = torch.max((value_batch - returns_batch).pow(2), (value_clipped - returns_batch).pow(2)).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                mse_loss
                + self.active_bc_coef * bc_loss
                + self.active_rl_coef * surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
            )
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            means["mse_loss"] += float(mse_loss.item())
            means["bc_loss"] += float(bc_loss.item())
            means["surrogate_loss"] += float(surrogate_loss.item())
            means["value_function"] += float(value_loss.item())
            means["latent/student_mean_norm"] += float(student_latent.norm(dim=-1).mean().item())
            means["latent/teacher_mean_norm"] += float(teacher_latent.norm(dim=-1).mean().item())
            num_updates += 1

        self.storage.clear()
        self.update_counter += 1
        denom = max(1, num_updates)
        for key in means:
            means[key] /= denom
        means.update(
            {
                "distillation/mean_action_imitation_reward": 0.0,
                "distillation/curriculum_progress": self._compute_curriculum_progress(self.update_counter),
                "distillation/active_bc_coef": self.active_bc_coef,
                "distillation/active_rl_coef": self.active_rl_coef,
                "distillation/target_noise_std": self._get_current_noise_std(),
            }
        )
        return means

    def _normalize_uncertainty(self, u: torch.Tensor) -> torch.Tensor:
        """Normalize uncertainty values to the [0, 1] range for curriculum weighting."""
        u_min, u_max = u.min(), u.max()
        return (u - u_min) / (u_max - u_min + 1e-6)

    def _teacher_value(self, teacher_obs):
        """Return teacher critic values, falling back to zeros when no critic is exposed."""
        teacher = getattr(self.policy, "teacher", None)
        if teacher is not None and hasattr(teacher, "evaluate"):
            return teacher.evaluate(teacher_obs)
        return torch.zeros(teacher_obs.shape[0], 1, device=teacher_obs.device)

    def broadcast_parameters(self):
        """Broadcast model parameters across distributed workers."""
        return None

    def reduce_parameters(self, params=None):
        """Average gradients or parameters across distributed workers."""
        return None

    def _reduce_module_gradients(self, module):
        """Placeholder for distributed gradient reduction in compatible runners."""
        return None
