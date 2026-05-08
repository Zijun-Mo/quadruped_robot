# rl_base

`rl_base` is the local reinforcement-learning library used by this workspace. It provides PyTorch policy modules, algorithms, rollout storage, runners, logging adapters, and Isaac Lab compatibility wrappers.

## Contents

- `rl_base/algorithms/`: PPO, distillation, TD3, SAC, and off-policy utilities.
- `rl_base/modules/`: actor-critic, recurrent actor-critic, student-teacher, terrain-aware policies, normalization, RND, and discriminator modules.
- `rl_base/storage/`: rollout storage for on-policy and distillation training.
- `rl_base/networks/`: recurrent memory wrappers.
- `rl_base/runners/`: on-policy runner orchestration for environments, algorithms, logging, checkpoints, and inference export.
- `rl_base/env/`: vectorized environment interface expected by runners.
- `rl_base/isaaclab_support.py`: Isaac Lab config classes, vectorized environment adapter, and policy exporters.

## Installation

Install in editable mode from the repository root:

```bash
conda activate env_isaacsim
python -m pip install -e rl_base
```

The package depends on PyTorch, NumPy, GitPython, ONNX, and an Isaac Lab environment when using `isaaclab_support.py`.

## Typical Use

The top-level launcher and Unitree scripts instantiate this package through config classes rather than direct command-line entrypoints:

```bash
bash run.sh teacher_train
bash run.sh baseline_train
bash run.sh baseline_td3
bash run.sh baseline_sac
```

Policy and algorithm class names are selected from the task agent configs under:

```text
unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/
```

## Validation

Run a syntax check from the repository root:

```bash
conda run -n env_isaacsim python -m compileall -q rl_base
```

This package is documented with English module, class, and function docstrings so future changes should keep public APIs and non-obvious tensor or recurrent-state behavior documented.
