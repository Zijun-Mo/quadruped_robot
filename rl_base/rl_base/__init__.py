"""Local reinforcement learning library used by this repository."""

from __future__ import annotations

__version__ = "2.3.3"

_EXPORTS = {
    "PPO": ("rl_base.algorithms", "PPO"),
    "Distillation": ("rl_base.algorithms", "Distillation"),
    "TD3": ("rl_base.algorithms", "TD3"),
    "SAC": ("rl_base.algorithms", "SAC"),
    "OnPolicyRunner": ("rl_base.runners", "OnPolicyRunner"),
    "ActorCritic": ("rl_base.modules", "ActorCritic"),
    "ActorCriticRecurrent": ("rl_base.modules", "ActorCriticRecurrent"),
    "StudentTeacher": ("rl_base.modules", "StudentTeacher"),
    "StudentTeacherRecurrent": ("rl_base.modules", "StudentTeacherRecurrent"),
    "TerrainAwareActorCritic": ("rl_base.modules", "TerrainAwareActorCritic"),
    "TerrainAwareStudentTeacher": ("rl_base.modules", "TerrainAwareStudentTeacher"),
    "EmpiricalNormalization": ("rl_base.modules", "EmpiricalNormalization"),
    "Normalizer": ("rl_base.modules", "Normalizer"),
    "RlBaseVecEnvWrapper": ("rl_base.isaaclab_support", "RlBaseVecEnvWrapper"),
    "RlBaseOnPolicyRunnerCfg": ("rl_base.isaaclab_support", "RlBaseOnPolicyRunnerCfg"),
    "RlBasePpoActorCriticCfg": ("rl_base.isaaclab_support", "RlBasePpoActorCriticCfg"),
    "RlBasePpoAlgorithmCfg": ("rl_base.isaaclab_support", "RlBasePpoAlgorithmCfg"),
    "RlBaseDistillationAlgorithmCfg": ("rl_base.isaaclab_support", "RlBaseDistillationAlgorithmCfg"),
    "export_policy_as_jit": ("rl_base.isaaclab_support", "export_policy_as_jit"),
    "export_policy_as_onnx": ("rl_base.isaaclab_support", "export_policy_as_onnx"),
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str):
    """Implement Python getattr protocol behavior."""
    if name not in _EXPORTS:
        raise AttributeError(f"module 'rl_base' has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
