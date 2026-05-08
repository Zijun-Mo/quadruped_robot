"""Training runner utilities that connect environments, algorithms, and logging."""

from __future__ import annotations

import os
import time
from collections import deque
from copy import deepcopy
from pathlib import Path

import torch

from rl_base.algorithms import Distillation, PPO
from rl_base.modules import (
    ActorCritic,
    ActorCriticRecurrent,
    EmpiricalNormalization,
    StudentTeacher,
    StudentTeacherRecurrent,
    TerrainAwareActorCritic,
    TerrainAwareStudentTeacher,
)
from rl_base.utils import store_code_state


POLICY_CLASSES = {
    "ActorCritic": ActorCritic,
    "ActorCriticRecurrent": ActorCriticRecurrent,
    "StudentTeacher": StudentTeacher,
    "StudentTeacherRecurrent": StudentTeacherRecurrent,
    "TerrainAwareActorCritic": TerrainAwareActorCritic,
    "TerrainAwareStudentTeacher": TerrainAwareStudentTeacher,
}

ALGORITHM_CLASSES = {"PPO": PPO, "Distillation": Distillation}


class DummyWriter:
    """Experiment logging adapter for dummy writer."""
    def add_scalar(self, *args, **kwargs):
        """Log a scalar metric value to the backing writer."""
        return None

    def log_config(self, *args, **kwargs):
        """Log configuration files or dictionaries for experiment tracking."""
        return None

    def save_model(self, *args, **kwargs):
        """Persist a model checkpoint through the experiment writer."""
        return None

    def save_file(self, *args, **kwargs):
        """Persist an auxiliary file through the experiment writer."""
        return None

    def stop(self):
        """Flush and close the experiment writer."""
        return None


class TensorboardSummaryWriter(DummyWriter):
    """Experiment logging adapter for tensorboard summary writer."""
    def __init__(self, log_dir):
        """Initialize TensorboardSummaryWriter with configuration, tensor shapes, and runtime state."""
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir)
        except Exception:
            self.writer = None

    def add_scalar(self, *args, **kwargs):
        """Log a scalar metric value to the backing writer."""
        if self.writer is not None:
            self.writer.add_scalar(*args, **kwargs)

    def stop(self):
        """Flush and close the experiment writer."""
        if self.writer is not None:
            self.writer.close()


class OnPolicyRunner:
    """Coordinates rollouts, policy updates, normalization, logging, and checkpoints."""
    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device="cpu"):
        """Initialize OnPolicyRunner with configuration, tensor shapes, and runtime state."""
        self.cfg = deepcopy(train_cfg)
        self.env = env
        self.log_dir = log_dir
        self.device = torch.device(device if not (str(device).startswith("cuda") and not torch.cuda.is_available()) else "cpu")
        self.git_status_repos = []
        self.current_learning_iteration = 0
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self.rewbuffer = deque(maxlen=100)
        self.lenbuffer = deque(maxlen=100)
        self.cur_reward_sum = torch.zeros(env.num_envs, device=self.device)
        self.cur_episode_length = torch.zeros(env.num_envs, device=self.device)
        self._configure_multi_gpu()

        alg_cfg = deepcopy(self.cfg["algorithm"])
        policy_cfg = deepcopy(self.cfg["policy"])
        alg_class_name = alg_cfg.pop("class_name")
        policy_class_name = policy_cfg.pop("class_name")
        self.training_type = "distillation" if alg_class_name == "Distillation" else "rl"

        obs, extras = self.env.get_observations()
        obs = obs.to(self.device)
        obs_groups = extras.get("observations", {})
        critic_obs = obs_groups.get("critic", obs).to(self.device)
        teacher_obs = obs_groups.get("teacher", critic_obs).to(self.device)
        self._debug_print_observation_breakdown(obs, extras)

        policy_cls = POLICY_CLASSES[policy_class_name]
        if "StudentTeacher" in policy_class_name:
            self.policy = policy_cls(obs.shape[-1], teacher_obs.shape[-1], env.num_actions, **policy_cfg).to(self.device)
            storage_priv_shape = teacher_obs.shape[1:]
        else:
            self.policy = policy_cls(obs.shape[-1], critic_obs.shape[-1], env.num_actions, **policy_cfg).to(self.device)
            storage_priv_shape = critic_obs.shape[1:]

        self.empirical_normalization = bool(self.cfg.get("empirical_normalization", False))
        self.obs_normalizer = None
        self.privileged_obs_normalizer = None
        if self.empirical_normalization:
            self.obs_normalizer = EmpiricalNormalization(obs.shape[1:]).to(self.device)
            self.privileged_obs_normalizer = EmpiricalNormalization(storage_priv_shape).to(self.device)

        alg_cls = ALGORITHM_CLASSES[alg_class_name]
        self.alg = alg_cls(self.policy, device=self.device, **alg_cfg)
        self.num_steps_per_env = int(self.cfg["num_steps_per_env"])
        self.save_interval = int(self.cfg.get("save_interval", 100))
        self.alg.init_storage(
            self.training_type,
            env.num_envs,
            self.num_steps_per_env,
            obs.shape[1:],
            storage_priv_shape,
            (env.num_actions,),
        )
        self.writer = self._make_writer()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        """Run the off-policy training loop for the requested number of timesteps."""
        if init_at_random_ep_len and hasattr(self.env, "episode_length_buf"):
            try:
                max_len = int(self.env.max_episode_length)
                self.env.episode_length_buf[:] = torch.randint_like(self.env.episode_length_buf, high=max(1, max_len))
            except Exception:
                pass

        obs, extras = self.env.get_observations()
        obs = obs.to(self.device)
        start_iter = self.current_learning_iteration
        for it in range(start_iter, start_iter + int(num_learning_iterations)):
            start = time.time()
            self.train_mode()
            with torch.no_grad():
                for _ in range(self.num_steps_per_env):
                    obs = self._normalize_obs(obs)
                    priv_obs = self._get_privileged_obs(obs, extras)
                    actions = self.alg.act(obs, priv_obs)
                    next_obs, rewards, dones, infos = self.env.step(actions)
                    next_obs = next_obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    self._update_episode_buffers(rewards, dones, infos)
                    obs, extras = next_obs, infos
                last_priv_obs = self._get_privileged_obs(self._normalize_obs(obs), extras)
                self.alg.compute_returns(last_priv_obs)
            losses = self.alg.update()
            self.current_learning_iteration = it + 1
            self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
            self.tot_time += time.time() - start
            self.log({"it": it, "losses": losses, "collection_time": time.time() - start})
            if self.log_dir is not None and it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
        return None

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        """Write training statistics for the current iteration."""
        it = locs.get("it", self.current_learning_iteration)
        losses = locs.get("losses", {})
        for name, value in losses.items():
            self.writer.add_scalar(f"Loss/{name}", value, it)
        if len(self.rewbuffer) > 0:
            self.writer.add_scalar("Train/mean_reward", torch.tensor(list(self.rewbuffer)).mean().item(), it)
        if len(self.lenbuffer) > 0:
            self.writer.add_scalar("Train/mean_episode_length", torch.tensor(list(self.lenbuffer)).float().mean().item(), it)
        if hasattr(self.policy, "action_std") and getattr(self.policy, "distribution", None) is not None:
            self.writer.add_scalar("Policy/mean_noise_std", self.policy.action_std.mean().item(), it)
        fps = int(self.tot_timesteps / max(self.tot_time, 1e-6))
        self.writer.add_scalar("Perf/total_fps", fps, it)
        print(f"[INFO] Iteration {it}: " + ", ".join(f"{k}={float(v):.4f}" for k, v in losses.items()))

    def save(self, path: str, infos=None):
        """Save model and optimizer state to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration - 1,
            "infos": infos,
        }
        if self.obs_normalizer is not None:
            checkpoint["obs_norm_state_dict"] = self.obs_normalizer.state_dict()
        if self.privileged_obs_normalizer is not None:
            checkpoint["privileged_obs_norm_state_dict"] = self.privileged_obs_normalizer.state_dict()
        if getattr(self.alg, "rnd", None) is not None:
            checkpoint["rnd_state_dict"] = self.alg.rnd.state_dict()
            checkpoint["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        if getattr(self.alg, "discriminator", None) is not None:
            checkpoint["discriminator_state_dict"] = self.alg.discriminator.state_dict()
        torch.save(checkpoint, path)
        self.writer.save_model(path, checkpoint["iter"])
        if self.log_dir is not None:
            for diff_file in store_code_state(self.log_dir, self.git_status_repos):
                self.writer.save_file(diff_file)

    def load(self, path: str, load_optimizer: bool = True):
        """Load model and optimizer state from a checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        loaded_for_resume = self.policy.load_state_dict(state_dict, strict=False)
        if loaded_for_resume and load_optimizer and "optimizer_state_dict" in checkpoint:
            try:
                self.alg.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as exc:
                print(f"[WARN] Optimizer state was not loaded: {exc}")
        if loaded_for_resume:
            self.current_learning_iteration = int(checkpoint.get("iter", -1)) + 1
        if self.obs_normalizer is not None and "obs_norm_state_dict" in checkpoint:
            self.obs_normalizer.load_state_dict(checkpoint["obs_norm_state_dict"])
        if self.privileged_obs_normalizer is not None and "privileged_obs_norm_state_dict" in checkpoint:
            self.privileged_obs_normalizer.load_state_dict(checkpoint["privileged_obs_norm_state_dict"])
        return checkpoint.get("infos", None)

    def get_inference_policy(self, device=None):
        """Return a callable policy that normalizes observations before inference."""
        device = torch.device(device or self.device)
        self.eval_mode()
        self.policy.to(device)
        normalizer = self.obs_normalizer
        if normalizer is not None:
            normalizer.to(device)
            normalizer.eval()

            def policy(obs):
                """Return a callable inference policy for actor-critic or student-teacher modules."""
                return self.policy.act_inference(normalizer(obs.to(device)))

            return policy

        def policy(obs):
            """Return a callable inference policy for actor-critic or student-teacher modules."""
            return self.policy.act_inference(obs.to(device))

        return policy

    def train_mode(self):
        """Switch runner-managed modules into training mode."""
        self.policy.train()
        if self.obs_normalizer is not None:
            self.obs_normalizer.train()
        if self.privileged_obs_normalizer is not None:
            self.privileged_obs_normalizer.train()

    def eval_mode(self):
        """Switch runner-managed modules into evaluation mode."""
        self.policy.eval()
        if self.obs_normalizer is not None:
            self.obs_normalizer.eval()
        if self.privileged_obs_normalizer is not None:
            self.privileged_obs_normalizer.eval()

    def add_git_repo_to_log(self, repo_file_path):
        """Record git metadata for the repository associated with a source file."""
        self.git_status_repos.append(repo_file_path)

    def _configure_multi_gpu(self):
        """Configure distributed rank, device, and seed settings for multi-GPU runs."""
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.is_distributed = self.world_size > 1
        if self.is_distributed and torch.distributed.is_available() and not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")

    def _debug_print_observation_breakdown(self, obs: torch.Tensor, extras: dict) -> None:
        """Print observation-group shapes to help debug environment wrappers."""
        obs_groups = extras.get("observations", {})
        pieces = [f"policy={tuple(obs.shape)}"]
        for key, value in obs_groups.items():
            if torch.is_tensor(value):
                pieces.append(f"{key}={tuple(value.shape)}")
        print("[INFO] Observation breakdown: " + ", ".join(pieces))

    def _get_privileged_obs(self, obs, extras):
        """Return teacher or critic observations from wrapper extras."""
        obs_groups = extras.get("observations", {}) if isinstance(extras, dict) else {}
        # Distillation prefers a teacher observation group, while PPO uses critic observations.
        if self.training_type == "distillation":
            priv = obs_groups.get("teacher", obs_groups.get("critic", obs))
        else:
            priv = obs_groups.get("critic", obs)
        priv = priv.to(self.device)
        if self.privileged_obs_normalizer is not None:
            priv = self.privileged_obs_normalizer(priv)
        return priv

    def _normalize_obs(self, obs):
        """Move observations to the runner device and apply the optional normalizer."""
        obs = obs.to(self.device)
        if self.obs_normalizer is not None:
            return self.obs_normalizer(obs)
        return obs

    def _update_episode_buffers(self, rewards, dones, infos):
        """Update reward and length statistics for completed episodes."""
        rewards = rewards.view(-1)
        dones = dones.view(-1).bool()
        self.cur_reward_sum += rewards
        self.cur_episode_length += 1
        done_ids = torch.nonzero(dones, as_tuple=False).flatten()
        if done_ids.numel() > 0:
            for idx in done_ids:
                self.rewbuffer.append(float(self.cur_reward_sum[idx].item()))
                self.lenbuffer.append(float(self.cur_episode_length[idx].item()))
            self.cur_reward_sum[done_ids] = 0.0
            self.cur_episode_length[done_ids] = 0.0

    def _make_writer(self):
        """Create the internal writer helper."""
        if self.log_dir is None:
            return DummyWriter()
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        logger = self.cfg.get("logger", "tensorboard")
        if logger == "wandb":
            from rl_base.utils.wandb_utils import WandbSummaryWriter

            return WandbSummaryWriter(self.log_dir, flush_secs=10, cfg=self.cfg)
        if logger == "neptune":
            from rl_base.utils.neptune_utils import NeptuneSummaryWriter

            return NeptuneSummaryWriter(self.log_dir, flush_secs=10, cfg=self.cfg)
        return TensorboardSummaryWriter(self.log_dir)
