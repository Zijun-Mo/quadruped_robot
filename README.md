# Quadruped Robot RL Workspace

This repository combines a local `rl_base` reinforcement-learning library with Unitree Isaac Lab tasks, training scripts, evaluation tools, and deployment assets.

The current workflow focuses on Unitree Go2 velocity tracking with teacher policies, terrain-aware student distillation, PPO-style on-policy training, and TD3/SAC off-policy baselines.

## Repository Layout

- `run.sh`: top-level launcher for training, playback, evaluation, and smoke tests.
- `rl_base/`: local PyTorch RL library with algorithms, policy modules, rollout storage, runners, and Isaac Lab adapters.
- `unitree_rl_lab/source/unitree_rl_lab/`: Isaac Lab extension with Unitree robot assets, locomotion tasks, mimic tasks, MDP terms, and agent configs.
- `unitree_rl_lab/scripts/rl_base/`: teacher, baseline, playback, evaluation, and checkpoint-inspection scripts.
- `unitree_rl_lab/scripts/mimic/`: motion data conversion and replay utilities.
- `unitree_rl_lab/deploy/`: sim2sim and sim2real deployment code and third-party runtime assets.

Generated outputs are written under `unitree_rl_lab/logs/`, `unitree_rl_lab/outputs/`, and `unitree_rl_lab/wandb/`.

## Environment

Use a conda environment with Isaac Sim and Isaac Lab installed. The launcher defaults to:

```bash
conda activate env_isaacsim
```

You can override it per command:

```bash
CONDA_ENV_NAME=isaaclab-pip-50 bash run.sh teacher_train
```

`run.sh` also defaults `WANDB_MODE=offline` to avoid login blocking. Set `WANDB_MODE=online` and `WANDB_API_KEY` if online logging is required.

## Installation

Install both Python packages in editable mode from the repository root:

```bash
conda activate env_isaacsim
python -m pip install -e rl_base
python -m pip install -e unitree_rl_lab/source/unitree_rl_lab
```

Robot description paths are configured in:

```text
unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py
```

Set `UNITREE_MODEL_DIR` for USD assets or `UNITREE_ROS_DIR` for URDF assets according to your local Isaac Lab setup.

## Quick Start

Show available launcher modes:

```bash
bash run.sh --help
```

Run a short end-to-end smoke workflow:

```bash
bash run.sh smoke_all
```

Train and evaluate the default Go2 teacher:

```bash
bash run.sh teacher_train
bash run.sh teacher_play
```

Train and evaluate the default Go2 baseline student:

```bash
bash run.sh baseline_train
bash run.sh baseline_play
bash run.sh baseline_eval
```

Run off-policy baselines:

```bash
bash run.sh baseline_td3
bash run.sh baseline_sac
```

## Launcher Modes

`run.sh` supports these modes:

- `smoke_all`: small end-to-end check covering teacher, baseline, playback, evaluation, TD3, and SAC.
- `teacher_train`: train the terrain-aware teacher policy.
- `teacher_play`: replay the latest or specified teacher checkpoint.
- `baseline_train`: train the default student baseline with the configured distillation curriculum.
- `baseline_pure_bc`: train the student with behavior cloning only.
- `baseline_pure_rl`: train the student with RL loss only.
- `baseline_play`: replay the latest or specified student checkpoint.
- `baseline_eval`: evaluate a baseline checkpoint.
- `baseline_td3`: train an off-policy TD3 baseline.
- `baseline_sac`: train an off-policy SAC baseline.

Common overrides:

```bash
SMOKE=1 bash run.sh teacher_train
NUM_ENVS=256 MAX_ITERATIONS=10 bash run.sh baseline_train
TOTAL_TIMESTEPS=200000 RL_ALGO=td3 bash run.sh baseline_td3
TEACHER_CKPT=/abs/path/model_100.pt bash run.sh teacher_play
BASELINE_CKPT=/abs/path/model_100.pt bash run.sh baseline_eval
```

Defaults are defined near the top of `run.sh`, including task names, device, log roots, checkpoint roots, batch sizes, and off-policy hyperparameters.

## Training Configuration

Teacher and baseline behavior is configured mainly through:

```text
unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/
unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/
```

For the default baseline, `TerrainAwareDistillationAlgorithmCfg` controls the BC/RL curriculum, loss weights, entropy coefficient, and exploration noise schedule.

Useful settings:

```python
# PPO/RL-heavy training
RL_loss_coef = 1.0
entropy_coef = 0.01

# Behavior-cloning-heavy training
bc_loss_coef = 1.0
use_mse_loss = True
```

Prefer using `baseline_pure_bc` or `baseline_pure_rl` when you want those modes without editing config files.

## Validation

For documentation or syntax-only changes:

```bash
conda run -n env_isaacsim python -m compileall -q rl_base unitree_rl_lab/source unitree_rl_lab/scripts
```

For runtime checks, use smoke mode first:

```bash
SMOKE=1 bash run.sh teacher_train
SMOKE=1 bash run.sh baseline_train
```

## Deployment

Deployment assets and C++ controllers live under `unitree_rl_lab/deploy/`. Build and runtime dependencies depend on the selected robot and simulator target. See `unitree_rl_lab/README.md` for the upstream Unitree RL Lab deployment notes and robot-specific setup details.

## Acknowledgements

This workspace builds on:

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab)
- [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab)
- [MuJoCo](https://github.com/google-deepmind/mujoco)
- [robot_lab](https://github.com/fan-ziqi/robot_lab)
- [whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking)
