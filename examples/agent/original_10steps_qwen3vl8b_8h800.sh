#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEEPEYES_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

load_env_file() {
  local env_file="$1"
  if [[ -f "${env_file}" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "${env_file}"
    set +a
  fi
}

load_env_file "${DEEPEYES_DIR}/.deepeyes.env"
load_env_file "${DEEPEYES_DIR}/.hf.env"
load_env_file "${DEEPEYES_DIR}/.mimo.env"
load_env_file "${DEEPEYES_DIR}/.wandb.env"

if [[ -n "${DEEPEYES_CONDA_INIT:-}" ]]; then
  eval "${DEEPEYES_CONDA_INIT}"
fi

DEEPEYES_CONDA_ENV="${DEEPEYES_CONDA_ENV:-verl}"
if [[ -n "${DEEPEYES_CONDA_ENV}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available. Set DEEPEYES_CONDA_INIT in DeepEyes/.deepeyes.env." >&2
    exit 1
  fi
  conda activate "${DEEPEYES_CONDA_ENV}"
fi

export CUDA_VISIBLE_DEVICES="${DEEPEYES_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}"
export HF_HOME="${HF_HOME:-/home/deeplearn/dataDisk/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${DEEPEYES_DIR}:${PYTHONPATH:-}"
export WORLD_SIZE="${WORLD_SIZE:-1}"

if [[ -n "${DEEPEYES_HF_ENDPOINT:-}" && -z "${HF_ENDPOINT:-}" ]]; then
  export HF_ENDPOINT="${DEEPEYES_HF_ENDPOINT}"
fi

export LLM_AS_A_JUDGE_BASE="${LLM_AS_A_JUDGE_BASE:-${MIMO_BASE_URL:-https://api.xiaomimimo.com/v1}}"
export LLM_AS_A_JUDGE_MODEL="${LLM_AS_A_JUDGE_MODEL:-${MIMO_MODEL:-mimo-v2.5}}"
if [[ -z "${LLM_AS_A_JUDGE_API_KEY:-}" && -n "${MIMO_API_KEY:-}" ]]; then
  export LLM_AS_A_JUDGE_API_KEY="${MIMO_API_KEY}"
fi
if [[ -z "${LLM_AS_A_JUDGE_API_KEY:-}" && -z "${MIMO_API_KEY:-}" && -z "${OPENAI_API_KEY:-}" ]]; then
  echo "ERROR: MiMo judge requires MIMO_API_KEY, LLM_AS_A_JUDGE_API_KEY, or OPENAI_API_KEY." >&2
  exit 1
fi

PROJECT_NAME="agent_vlagent"
EXPERIMENT_NAME="${DEEPEYES_EXPERIMENT_NAME:-deepeyes_original_10steps_8h800}"
SAVE_CHECKPOINT_DIR="${DEEPEYES_SAVE_CHECKPOINT_DIR:-/home/deeplearn/dataDisk/deepeyes_original_step_timing}"
LOG_DIR="${DEEPEYES_LOG_DIR:-${SAVE_CHECKPOINT_DIR}/logs}"
mkdir -p "${SAVE_CHECKPOINT_DIR}" "${LOG_DIR}" "${DEEPEYES_DIR}/logs"
TOTAL_TRAINING_STEPS="${DEEPEYES_TOTAL_TRAINING_STEPS:-10}"

if ! [[ "${TOTAL_TRAINING_STEPS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: DEEPEYES_TOTAL_TRAINING_STEPS must be a positive integer." >&2
  exit 1
fi

REF_MODEL_PATH="${DEEPEYES_MODEL:-/home/deeplearn/dataDisk/qwen3-vl-8b}"

if [[ -n "${DEEPEYES_TRAIN_FILES:-}" ]]; then
  TRAIN_FILES="${DEEPEYES_TRAIN_FILES}"
elif [[ -n "${DEEPEYES_DATA_BASEDIR:-}" ]]; then
  VISUAL_DATASET_TRAIN_0_1_2="${DEEPEYES_DATA_BASEDIR}/data_0.1.2_visual_toolbox_v2.parquet"
  VISUAL_DATASET_TRAIN_0_8="${DEEPEYES_DATA_BASEDIR}/minghao_data_vnew/data_v0.8_visual_toolbox_v2.parquet"
  EUREKA_DATASET_TRAIN="${DEEPEYES_DATA_BASEDIR}/data_thinklite_reasoning_acc.parquet"
  TRAIN_FILES="[${VISUAL_DATASET_TRAIN_0_1_2},${VISUAL_DATASET_TRAIN_0_8},${EUREKA_DATASET_TRAIN}]"
else
  SOURCE_TRAIN_FILE="${DEEPEYES_SOURCE_TRAIN_FILE:-/home/deeplearn/dataDisk/zwz_dataset/reannotate_verl_format_teacher_crop_train23k/train.parquet}"
  ADAPTED_DATA_DIR="${DEEPEYES_ADAPTED_DATA_DIR:-${SAVE_CHECKPOINT_DIR}/data}"
  ADAPTED_TRAIN_FILE="${DEEPEYES_ADAPTED_TRAIN_FILE:-${ADAPTED_DATA_DIR}/reannotate_train23k_deepeyes_interface.parquet}"
  mkdir -p "${ADAPTED_DATA_DIR}"
  if [[ ! -f "${ADAPTED_TRAIN_FILE}" || "${DEEPEYES_REBUILD_DATA:-0}" == "1" ]]; then
    python3 "${DEEPEYES_DIR}/examples/data_preprocess/prepare_deepeyes_interface.py" \
      --input "${SOURCE_TRAIN_FILE}" \
      --output "${ADAPTED_TRAIN_FILE}" \
      --overwrite
  fi
  TRAIN_FILES="[${ADAPTED_TRAIN_FILE}]"
fi

if [[ -n "${DEEPEYES_VAL_FILES:-}" ]]; then
  VAL_FILES="${DEEPEYES_VAL_FILES}"
elif [[ -n "${EUREKA_DATASET_TRAIN:-}" ]]; then
  VAL_FILES="[${EUREKA_DATASET_TRAIN}]"
else
  VAL_FILES="${TRAIN_FILES}"
fi

cd "${DEEPEYES_DIR}"

LOG_FILE="${LOG_DIR}/${EXPERIMENT_NAME}_$(date +%Y%m%d_%H%M%S).log"
START_SECONDS="$(date +%s)"

{
  python3 -m verl.trainer.main_ppo \
    +debug=False \
    +vs_debug=False \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=256 \
    data.max_prompt_length=8192 \
    data.max_response_length=20480 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.model.path="${REF_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    "actor_rollout_ref.actor.checkpoint.contents=['model','hf_model','optimizer','extra']" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.agent.activate_agent=True \
    actor_rollout_ref.rollout.agent.tool_name_key=env_name \
    actor_rollout_ref.rollout.agent.single_response_max_tokens=10240 \
    actor_rollout_ref.rollout.agent.max_turns=5 \
    actor_rollout_ref.rollout.agent.concurrent_workers=1 \
    actor_rollout_ref.rollout.agent.show_tqdm=True \
    trainer.critic_warmup=0 \
    "trainer.logger=['console','wandb','rl_logging_board']" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes="${WORLD_SIZE}" \
    trainer.save_freq=8 \
    trainer.test_freq=10000 \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${SAVE_CHECKPOINT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
    "+trainer.tensorboard_dir=${SAVE_CHECKPOINT_DIR}/logs/tensorboard" \
    "+trainer.rl_logging_board_dir=${SAVE_CHECKPOINT_DIR}/logs/rl_logging_board" \
    trainer.total_epochs=32 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    "$@"
} 2>&1 | tee "${LOG_FILE}"

END_SECONDS="$(date +%s)"
ELAPSED_SECONDS="$((END_SECONDS - START_SECONDS))"
AVG_SECONDS="$(python3 - <<PY
elapsed = float("${ELAPSED_SECONDS}")
steps = float("${TOTAL_TRAINING_STEPS}")
print(f"{elapsed / steps:.3f}")
PY
)"

{
  echo "deepeyes_total_wall_sec=${ELAPSED_SECONDS}"
  echo "deepeyes_total_training_steps=${TOTAL_TRAINING_STEPS}"
  echo "deepeyes_avg_wall_sec_per_step=${AVG_SECONDS}"
} | tee -a "${LOG_FILE}"
