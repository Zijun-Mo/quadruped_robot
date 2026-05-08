"""Isaac Lab adapter classes, configuration objects, and policy exporters for rl_base."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Literal

import torch

from rl_base.env import VecEnv

try:
    from isaaclab.utils import configclass
except Exception:
    def configclass(cls):
        """Fallback decorator that mimics Isaac Lab configclass with dataclass support."""
        cls = dataclass(cls)

        def to_dict(self):
            """Recursively convert dataclass fields to a plain dictionary."""
            def convert(value):
                """Convert nested dataclasses, dictionaries, and sequences to serializable values."""
                if hasattr(value, "to_dict"):
                    return value.to_dict()
                if isinstance(value, dict):
                    return {k: convert(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return type(value)(convert(v) for v in value)
                return copy.deepcopy(value)

            return {k: convert(getattr(self, k)) for k in self.__dataclass_fields__}

        cls.to_dict = to_dict
        return cls


@configclass
class RlBasePpoActorCriticCfg:
    """Configuration container for the rl_base PPO actor-critic module."""
    class_name: str = "ActorCritic"
    init_noise_std: float = 1.0
    noise_std_type: Literal["scalar", "log"] = "scalar"
    actor_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    critic_hidden_dims: list[int] = field(default_factory=lambda: [256, 256, 256])
    activation: str = "elu"


@configclass
class RlBasePpoAlgorithmCfg:
    """Configuration container for the rl_base PPO algorithm."""
    class_name: str = "PPO"
    value_loss_coef: float = 1.0
    use_clipped_value_loss: bool = True
    clip_param: float = 0.2
    entropy_coef: float = 0.0
    num_learning_epochs: int = 1
    num_mini_batches: int = 1
    learning_rate: float = 1.0e-3
    schedule: str = "fixed"
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    max_grad_norm: float = 1.0
    normalize_advantage_per_mini_batch: bool = False
    symmetry_cfg: dict | None = None
    rnd_cfg: dict | None = None


@configclass
class RlBaseDistillationAlgorithmCfg:
    """Configuration container for the rl_base distillation algorithm."""
    class_name: str = "Distillation"
    num_learning_epochs: int = 1
    num_mini_batches: int = 4
    learning_rate: float = 1.0e-3
    gradient_length: int = 15
    max_grad_norm: float = 1.0
    optimizer: str = "adam"
    loss_type: str = "BCEWithLogits"


@configclass
class RlBaseOnPolicyRunnerCfg:
    """Configuration container for the rl_base on-policy runner."""
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = 24
    max_iterations: int = 1500
    empirical_normalization: bool = False
    clip_actions: float | None = None
    save_interval: int = 50
    experiment_name: str = ""
    run_name: str = ""
    logger: str = "tensorboard"
    neptune_project: str = ""
    wandb_project: str = ""
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
    class_name: str = "OnPolicyRunner"
    policy: RlBasePpoActorCriticCfg = field(default_factory=RlBasePpoActorCriticCfg)
    algorithm: RlBasePpoAlgorithmCfg = field(default_factory=RlBasePpoAlgorithmCfg)


class RlBaseVecEnvWrapper(VecEnv):
    """
    Environment wrapper that adapts Isaac Lab vectorized environments to rl_base expectations.
    """
    def __init__(self, env, clip_actions: float | None = None):
        """Initialize RlBaseVecEnvWrapper with configuration, tensor shapes, and runtime state."""
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = int(getattr(self.unwrapped, "num_envs", 1))
        self.device = torch.device(getattr(self.unwrapped, "device", "cpu"))
        self.max_episode_length = getattr(self.unwrapped, "max_episode_length", 0)
        self._modify_action_space()
        self.num_actions = self._infer_num_actions()
        self._last_obs, self._last_extras = self.reset()

    def __str__(self):
        """Implement Python str protocol behavior."""
        return f"RlBaseVecEnvWrapper({self.env})"

    __repr__ = __str__

    @property
    def cfg(self):
        """Return the wrapped environment configuration object when available."""
        return getattr(self.unwrapped, "cfg", None)

    @property
    def render_mode(self):
        """Return the render mode advertised by the wrapped environment."""
        return getattr(self.env, "render_mode", None)

    @property
    def observation_space(self):
        """Return the wrapped environment observation space."""
        return self.env.observation_space

    @property
    def action_space(self):
        """Return the wrapped environment action space."""
        return self.env.action_space

    @classmethod
    def class_name(cls) -> str:
        """Return the wrapped environment class name."""
        return cls.__name__

    @property
    def unwrapped(self):
        """Return the innermost wrapped environment object."""
        return self.env.unwrapped

    def get_observations(self):
        """Return the latest policy observations from the vectorized environment."""
        return self._last_obs, self._last_extras

    @property
    def episode_length_buf(self):
        """Return the wrapped environment episode-length buffer."""
        return getattr(self.unwrapped, "episode_length_buf")

    @episode_length_buf.setter
    def episode_length_buf(self, value):
        """Set the wrapped environment episode-length buffer."""
        setattr(self.unwrapped, "episode_length_buf", value)

    def seed(self, seed: int = -1) -> int:
        """Forward seeding to the wrapped environment when supported."""
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return seed

    def reset(self):
        """Reset environment, module, or buffer state."""
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        self._last_obs, self._last_extras = self._process_observations(obs, info)
        return self._last_obs, self._last_extras

    def step(self, actions: torch.Tensor):
        """Advance the environment wrapper by one action step."""
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -float(self.clip_actions), float(self.clip_actions))
        result = self.env.step(actions)
        # Support both Gymnasium's five-value API and older VecEnv-style four-value API.
        if len(result) == 5:
            obs, rewards, terminated, truncated, info = result
            dones = torch.as_tensor(terminated, device=self.device).bool() | torch.as_tensor(truncated, device=self.device).bool()
            time_outs = torch.as_tensor(truncated, device=self.device).bool()
        elif len(result) == 4:
            obs, rewards, dones, info = result
            dones = torch.as_tensor(dones, device=self.device).bool()
            time_outs = torch.zeros_like(dones, dtype=torch.bool, device=self.device)
        else:
            raise ValueError(f"Unexpected env.step return length: {len(result)}")
        obs, extras = self._process_observations(obs, info)
        extras["time_outs"] = time_outs
        if isinstance(info, dict) and "log" in info:
            extras["log"] = info["log"]
        rewards = torch.as_tensor(rewards, device=self.device, dtype=torch.float32)
        self._last_obs, self._last_extras = obs, extras
        return obs, rewards, dones, extras

    def close(self):
        """Release logger or writer resources."""
        return self.env.close()

    def _modify_action_space(self):
        """Hook for subclasses that need to adjust the action space."""
        return None

    def _infer_num_actions(self) -> int:
        """Infer the flattened action dimension from the wrapped environment."""
        if hasattr(self.unwrapped, "num_actions"):
            return int(self.unwrapped.num_actions)
        space = getattr(self.env, "single_action_space", getattr(self.env, "action_space", None))
        if space is not None and getattr(space, "shape", None):
            shape = space.shape
            if len(shape) > 1 and shape[0] == self.num_envs:
                return int(torch.tensor(shape[1:]).prod().item())
            return int(torch.tensor(shape).prod().item())
        action_manager = getattr(self.unwrapped, "action_manager", None)
        if action_manager is not None and hasattr(action_manager, "total_action_dim"):
            return int(action_manager.total_action_dim)
        raise RuntimeError("Could not infer number of actions from environment.")

    def _process_observations(self, obs, info=None):
        """Normalize Isaac Lab observation dictionaries into rl_base policy/extras tensors."""
        info = info or {}
        if isinstance(obs, dict):
            obs_dict = {k: v for k, v in obs.items() if torch.is_tensor(v)}
            policy_obs = obs.get("policy")
            if policy_obs is None:
                # Some Isaac Lab tasks expose only one tensor group; use it as policy observations.
                policy_obs = next(iter(obs_dict.values()))
            extras = {"observations": obs_dict}
        else:
            policy_obs = obs
            extras = {"observations": {}}
        if isinstance(info, dict):
            info_obs = info.get("observations")
            if isinstance(info_obs, dict):
                extras["observations"].update({k: v for k, v in info_obs.items() if torch.is_tensor(v)})
        return torch.as_tensor(policy_obs, device=self.device, dtype=torch.float32), extras


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter that serializes policies through the torch policy path."""
    def __init__(self, policy, normalizer=None):
        """Initialize _TorchPolicyExporter with configuration, tensor shapes, and runtime state."""
        super().__init__()
        self.policy = copy.deepcopy(policy).cpu().eval()
        self.normalizer = copy.deepcopy(normalizer).cpu().eval() if normalizer is not None else torch.nn.Identity()

    def forward(self, x):
        """Run the forward pass for this module."""
        x = self.normalizer(x)
        return self.policy.act_inference(x)

    @torch.jit.export
    def reset(self):
        """Reset environment, module, or buffer state."""
        if hasattr(self.policy, "reset"):
            self.policy.reset()

    def export(self, path, filename):
        """Export the copied actor-critic policy as a TorchScript module."""
        os.makedirs(path, exist_ok=True)
        scripted = torch.jit.script(self)
        full_path = os.path.join(path, filename)
        scripted.save(full_path)
        return full_path


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter that serializes policies through the ONNX policy path."""
    def __init__(self, policy, normalizer=None, verbose=False):
        """Initialize _OnnxPolicyExporter with configuration, tensor shapes, and runtime state."""
        super().__init__()
        self.policy = copy.deepcopy(policy).cpu().eval()
        self.normalizer = copy.deepcopy(normalizer).cpu().eval() if normalizer is not None else torch.nn.Identity()
        self.verbose = verbose

    def forward(self, x):
        """Run the forward pass for this module."""
        x = self.normalizer(x)
        return self.policy.act_inference(x)

    def export(self, path, filename):
        """Export the copied actor-critic policy as an ONNX graph."""
        os.makedirs(path, exist_ok=True)
        obs_dim = getattr(self.policy, "actor", None)
        if hasattr(self.policy, "actor") and len(self.policy.actor) > 0:
            input_dim = self.policy.actor[0].in_features
        elif hasattr(self.policy, "student_policy_head"):
            # Student-teacher policies export from raw student observations, not latent features.
            input_dim = getattr(self.policy.memory_s.rnn, "input_size", 1)
        else:
            raise ValueError("Could not infer ONNX export input dimension.")
        sample = torch.zeros(1, input_dim)
        full_path = os.path.join(path, filename)
        torch.onnx.export(self, sample, full_path, opset_version=11, input_names=["obs"], output_names=["actions"])
        return full_path


def export_policy_as_jit(policy: object, normalizer: object | None, path: str, filename="policy.pt"):
    """Export the policy as jit artifact."""
    return _TorchPolicyExporter(policy, normalizer).export(path, filename)


def export_policy_as_onnx(policy: object, path: str, normalizer: object | None = None, filename="policy.onnx", verbose=False):
    """Export the policy as ONNX artifact."""
    return _OnnxPolicyExporter(policy, normalizer, verbose=verbose).export(path, filename)
