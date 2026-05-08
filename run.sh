#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNITREE_RL_LAB_DIR="${PROJECT_ROOT}/unitree_rl_lab"

# -------------------------------
# Environment setup
# -------------------------------
CONDA_BASE="$(conda info --base 2>/dev/null || true)"
if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    . "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME:-env_isaacsim}"
else
    echo "[WARN] Conda not found or not initialized. Falling back to current Python."
fi

# Keep IsaacSim's own Python libs first; drop host anaconda paths from LD_LIBRARY_PATH.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
    export LD_LIBRARY_PATH
    LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v anaconda3 | paste -sd: || true)"
    export LD_LIBRARY_PATH
fi

# Default to offline mode to avoid login blocking; override with WANDB_MODE=online if needed.
export WANDB_MODE="${WANDB_MODE:-offline}"
# if you wan to use wandb online
# export WANDB_API_KEY="your_wandb_api_key_here"

cd "${UNITREE_RL_LAB_DIR}"

# -------------------------------
# Runtime knobs (all overridable)
# -------------------------------
MODE="${1:-baseline_td3}"
SMOKE="${SMOKE:-0}"
if [[ "${MODE}" == "smoke_all" ]]; then
    SMOKE=1
fi

TASK_TEACHER="${TASK_TEACHER:-Unitree-Go2-Velocity-Teacher-v0}"
TASK_BASELINE="${TASK_BASELINE:-Unitree-Go2-Velocity-lab-Rough-Env-v0}"

DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-64}"
HEADLESS_FLAG="${HEADLESS_FLAG:---headless}"

VIDEO_LENGTH="${VIDEO_LENGTH:-600}"
EVAL_DURATION="${EVAL_DURATION:-30}"

MAX_ITERATIONS="${MAX_ITERATIONS:-1}"
RL_ALGO="${RL_ALGO:-td3}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-50000}"
LR="${LR:-3e-4}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
BUFFER_SIZE="${BUFFER_SIZE:-1000000}"
LEARNING_STARTS="${LEARNING_STARTS:-10000}"
TRAIN_FREQ="${TRAIN_FREQ:-1}"
GRAD_STEPS="${GRAD_STEPS:-1}"
TAU="${TAU:-0.005}"
GAMMA="${GAMMA:-0.99}"
OFFPOLICY_SAVE_INTERVAL="${OFFPOLICY_SAVE_INTERVAL:-5000}"

TEACHER_LOG_ROOT="${TEACHER_LOG_ROOT:-${PROJECT_ROOT}/unitree_rl_lab/logs}"
BASELINE_LOG_ROOT="${BASELINE_LOG_ROOT:-${PROJECT_ROOT}/unitree_rl_lab/logs/student_baseline}"
TEACHER_CKPT_ROOT="${TEACHER_CKPT_ROOT:-${PROJECT_ROOT}/unitree_rl_lab/logs/rl_base}"
BASELINE_CKPT_ROOT="${BASELINE_CKPT_ROOT:-${PROJECT_ROOT}/unitree_rl_lab/logs/student_baseline/rl_base_mle}"

TEACHER_RUN_NAME="${TEACHER_RUN_NAME:-teacher_run}"
BASELINE_RUN_NAME="${BASELINE_RUN_NAME:-baseline_run}"

if [[ "${SMOKE}" == "1" ]]; then
    NUM_ENVS="${SMOKE_NUM_ENVS:-1}"
    MAX_ITERATIONS="${SMOKE_MAX_ITERATIONS:-1}"
    VIDEO_LENGTH="${SMOKE_VIDEO_LENGTH:-4}"
    EVAL_DURATION="${SMOKE_EVAL_DURATION:-0.2}"
    TOTAL_TIMESTEPS="${SMOKE_TOTAL_TIMESTEPS:-64}"
    BATCH_SIZE="${SMOKE_BATCH_SIZE:-8}"
    BUFFER_SIZE="${SMOKE_BUFFER_SIZE:-1024}"
    LEARNING_STARTS="${SMOKE_LEARNING_STARTS:-4}"
    TRAIN_FREQ="${SMOKE_TRAIN_FREQ:-1}"
    GRAD_STEPS="${SMOKE_GRAD_STEPS:-1}"
    OFFPOLICY_SAVE_INTERVAL="${SMOKE_OFFPOLICY_SAVE_INTERVAL:-1}"
    TEACHER_RUN_NAME="${SMOKE_TEACHER_RUN_NAME:-smoke_runsh_teacher}"
    BASELINE_RUN_NAME="${SMOKE_BASELINE_RUN_NAME:-smoke_runsh_baseline}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    echo "[INFO] Smoke mode enabled: NUM_ENVS=${NUM_ENVS}, MAX_ITERATIONS=${MAX_ITERATIONS}, TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS}"
fi

# -------------------------------
# Helpers
# -------------------------------
latest_checkpoint() {
    local root="$1"
    if [[ ! -d "$root" ]]; then
        return 0
    fi
    # Only use real checkpoint weights like model_0.pt; ignore debug artifacts like model_0_debug_data.pt.
    find "$root" -type f -name "model_[0-9]*.pt" -print | sort -V | tail -n 1
}

require_file() {
    local path="$1"
    local hint="$2"
    if [[ -z "$path" || ! -f "$path" ]]; then
        echo "[ERROR] Missing checkpoint: ${path:-<empty>}"
        echo "[HINT] ${hint}"
        exit 1
    fi
}

TEACHER_CKPT="${TEACHER_CKPT:-$(latest_checkpoint "${TEACHER_CKPT_ROOT}")}" 
BASELINE_CKPT="${BASELINE_CKPT:-$(latest_checkpoint "${BASELINE_CKPT_ROOT}")}" 

refresh_checkpoints() {
    if [[ -z "${TEACHER_CKPT:-}" || ! -f "${TEACHER_CKPT}" ]]; then
        TEACHER_CKPT="$(latest_checkpoint "${TEACHER_CKPT_ROOT}")"
    fi
    if [[ -z "${BASELINE_CKPT:-}" || ! -f "${BASELINE_CKPT}" ]]; then
        BASELINE_CKPT="$(latest_checkpoint "${BASELINE_CKPT_ROOT}")"
    fi
}

ensure_teacher_checkpoint() {
    refresh_checkpoints
    if [[ "${SMOKE}" == "1" && ( -z "${TEACHER_CKPT:-}" || ! -f "${TEACHER_CKPT}" ) ]]; then
        echo "[INFO] Smoke teacher checkpoint missing; generating one with teacher_train."
        run_teacher_train
        TEACHER_CKPT="$(latest_checkpoint "${TEACHER_CKPT_ROOT}")"
    fi
    require_file "${TEACHER_CKPT}" "Run 'bash run.sh teacher_train' first or set TEACHER_CKPT=/abs/path/model_x.pt"
}

ensure_baseline_checkpoint() {
    refresh_checkpoints
    if [[ "${SMOKE}" == "1" && ( -z "${BASELINE_CKPT:-}" || ! -f "${BASELINE_CKPT}" ) ]]; then
        echo "[INFO] Smoke baseline checkpoint missing; generating one with baseline_train."
        ensure_teacher_checkpoint
        run_baseline_train
        BASELINE_CKPT="$(latest_checkpoint "${BASELINE_CKPT_ROOT}")"
    fi
    require_file "${BASELINE_CKPT}" "Run 'bash run.sh baseline_train' first or set BASELINE_CKPT=/abs/path/model_x.pt"
}

usage() {
    cat <<USAGE
Usage: bash run.sh <mode>

Modes:
  smoke_all
  teacher_train
  teacher_play
  baseline_train
  baseline_pure_bc
  baseline_pure_rl
  baseline_play
  baseline_eval
  baseline_td3
  baseline_sac

Examples:
  bash run.sh teacher_train
  SMOKE=1 bash run.sh teacher_train
  bash run.sh smoke_all
  MAX_ITERATIONS=10 NUM_ENVS=256 bash run.sh baseline_train
  TOTAL_TIMESTEPS=200000 RL_ALGO=td3 bash run.sh baseline_td3
USAGE
}

run_teacher_train() {
    python scripts/rl_base/train_teacher.py \
        "${HEADLESS_FLAG}" \
        --task "${TASK_TEACHER}" \
        --num_envs "${NUM_ENVS}" \
        --max_iterations "${MAX_ITERATIONS}" \
        --log_root "${TEACHER_LOG_ROOT}" \
        --run_name "${TEACHER_RUN_NAME}" \
        --device "${DEVICE}"
}

run_teacher_play() {
    ensure_teacher_checkpoint
    python scripts/rl_base/play_teacher.py \
        "${HEADLESS_FLAG}" \
        --task "${TASK_TEACHER}" \
        --checkpoint "${TEACHER_CKPT}" \
        --num_envs "${NUM_ENVS}" \
        --device "${DEVICE}" \
        --video \
        --video_length "${VIDEO_LENGTH}"
}

run_baseline_train_common() {
    local run_name="$1"
    shift
    ensure_teacher_checkpoint
    python scripts/rl_base/train_baseline.py \
        "${HEADLESS_FLAG}" \
        --task "${TASK_BASELINE}" \
        --num_envs "${NUM_ENVS}" \
        --max_iterations "${MAX_ITERATIONS}" \
        --log_root "${BASELINE_LOG_ROOT}" \
        --resume_path "${TEACHER_CKPT}" \
        --run_name "${run_name}" \
        --device "${DEVICE}" \
        "$@"
}

run_baseline_train() {
    run_baseline_train_common "${BASELINE_RUN_NAME}"
}

run_baseline_pure_bc() {
    run_baseline_train_common "${BASELINE_RUN_NAME}_pure_bc" \
        agent.algorithm.curriculum_enable=False \
        agent.algorithm.RL_loss_coef=0.0 \
        agent.algorithm.bc_loss_coef=1.0 \
        agent.algorithm.use_mse_loss=True \
        agent.algorithm.entropy_coef=0.0
}

run_baseline_pure_rl() {
    run_baseline_train_common "${BASELINE_RUN_NAME}_pure_rl" \
        agent.algorithm.curriculum_enable=False \
        agent.algorithm.RL_loss_coef=1.0 \
        agent.algorithm.bc_loss_coef=0.0 \
        agent.algorithm.use_mse_loss=False \
        agent.algorithm.entropy_coef=0.01
}

run_baseline_play() {
    ensure_baseline_checkpoint
    python scripts/rl_base/play_baseline.py \
        "${HEADLESS_FLAG}" \
        --task "${TASK_BASELINE}" \
        --checkpoint "${BASELINE_CKPT}" \
        --num_envs "${NUM_ENVS}" \
        --device "${DEVICE}" \
        --video \
        --video_length "${VIDEO_LENGTH}"
}

run_baseline_eval() {
    ensure_baseline_checkpoint
    python scripts/rl_base/eval_baseline_policy.py \
        "${HEADLESS_FLAG}" \
        --task "${TASK_BASELINE}" \
        --checkpoint "${BASELINE_CKPT}" \
        --eval_duration "${EVAL_DURATION}" \
        --num_envs "${NUM_ENVS}" \
        --device "${DEVICE}"
}

run_baseline_offpolicy() {
    local algo="$1"
    python scripts/rl_base/train_baseline.py \
        "${HEADLESS_FLAG}" \
        --task "${TASK_BASELINE}" \
        --num_envs "${NUM_ENVS}" \
        --log_root "${BASELINE_LOG_ROOT}" \
        --run_name "${BASELINE_RUN_NAME}_${algo}" \
        --device "${DEVICE}" \
        --rl_algorithm "${algo}" \
        --offpolicy_total_timesteps "${TOTAL_TIMESTEPS}" \
        --offpolicy_learning_rate "${LR}" \
        --offpolicy_batch_size "${BATCH_SIZE}" \
        --offpolicy_buffer_size "${BUFFER_SIZE}" \
        --offpolicy_learning_starts "${LEARNING_STARTS}" \
        --offpolicy_train_freq "${TRAIN_FREQ}" \
        --offpolicy_gradient_steps "${GRAD_STEPS}" \
        --offpolicy_tau "${TAU}" \
        --offpolicy_gamma "${GAMMA}" \
        --offpolicy_save_interval "${OFFPOLICY_SAVE_INTERVAL}"
}

run_smoke_all() {
    run_teacher_train
    TEACHER_CKPT="$(latest_checkpoint "${TEACHER_CKPT_ROOT}")"
    run_teacher_play
    run_baseline_train
    BASELINE_CKPT="$(latest_checkpoint "${BASELINE_CKPT_ROOT}")"
    run_baseline_pure_bc
    run_baseline_pure_rl
    BASELINE_CKPT="$(latest_checkpoint "${BASELINE_CKPT_ROOT}")"
    run_baseline_play
    run_baseline_eval
    run_baseline_offpolicy td3
    run_baseline_offpolicy sac
}

case "${MODE}" in
    smoke_all)
        run_smoke_all
        ;;
    teacher_train)
        run_teacher_train
        ;;
    teacher_play)
        run_teacher_play
        ;;
    baseline_train)
        run_baseline_train
        ;;
    baseline_pure_bc)
        run_baseline_pure_bc
        ;;
    baseline_pure_rl)
        run_baseline_pure_rl
        ;;
    baseline_play)
        run_baseline_play
        ;;
    baseline_eval)
        run_baseline_eval
        ;;
    baseline_td3)
        run_baseline_offpolicy td3
        ;;
    baseline_sac)
        run_baseline_offpolicy sac
        ;;
    -h|--help|help)
        usage
        ;;
    *)
        echo "[ERROR] Unknown mode: ${MODE}"
        usage
        exit 1
    ;;
esac
