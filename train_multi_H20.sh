#!/bin/bash
set -euo pipefail
export DEBUG_MODE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
# ---- NCCL 多卡通信调优 (H20/H800 NVLS 兼容性修复) ----
export NCCL_NVLS_ENABLE=0
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
# ---- XLA / GPU 吞吐调优 ----
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_enable_while_loop_double_buffering=true"
# ---- 强制离线 ----
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# ################## ----(1) LeRobot 数据集根目录 ----
export XDG_CACHE_HOME=/mnt/vepfs/vbot/lzx/RoboDojo/data
export HF_LEROBOT_HOME=/mnt/vepfs/vbot/lzx/RoboDojo/data/huggingface/lerobot
# ################## ----(1.1) LeRobot 数据集软链接示例 ----
# mkdir -p /mnt/vepfs/vbot/lzx/RoboDojo/data/huggingface/lerobot && \
# ln -s /mnt/vepfs/vbot/lzx/RoboDojo/data/new/stack_bowls /mnt/vepfs/vbot/lzx/RoboDojo/data/huggingface/lerobot/stack_bowls
# ---- 限制视频解码/数学库线程数 ----
export FFMPEG_THREADS=1
export AV_LOG_FORCE_NOCOLOR=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
# ---- 自动检测 GPU 数量 ----
GPUS_PER_NODE=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Detected GPUs: $GPUS_PER_NODE"
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS_PER_NODE-1)))
echo "Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
time=$(date "+%Y%m%d-%H%M%S")
echo "time: $time"
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <train_config_name> [ckpt_name] [num_train_steps] [batch_size] [ema_decay] [save_interval]" >&2
  exit 1
fi
train_config_name=${1:-pi05_base_aloha_single_train_seed_0}
ckpt_name=${2:-default_ckpt}
seed=0
num_train_steps=${3:-10000}
batch_size=${4:-256}
ema_decay=${5:-None}
save_interval=${6:-2500}
train_stage="${OPENPI_TRAIN_STAGE:-auto}"

if [[ "${train_stage}" == "auto" ]]; then
  case "${train_config_name}" in
    msp_vae_stage1*|*stage1*)
      train_stage="stage1"
      ;;
    pi05_msp_stage2*|pi05_*|*stage2*)
      train_stage="stage2"
      ;;
    *)
      echo "Cannot infer train stage from train_config_name=${train_config_name}. Set OPENPI_TRAIN_STAGE=stage1 or stage2." >&2
      exit 1
      ;;
  esac
fi

case "${train_stage}" in
  stage1|vae_stage1|msp_vae_stage1)
    train_stage="stage1"
    decay_steps="${DECAY_STEPS:-$((num_train_steps))}"
    peak_lr="${PEAK_LR:-3e-4}"
    decay_lr="${DECAY_LR:-3e-5}"
    load_base_weights=false
    load_msp_vae_weights=false
    ;;
  stage2|msp_stage2|pi05_stage2)
    train_stage="stage2"
    decay_steps="${DECAY_STEPS:-$((num_train_steps / 2))}"
    peak_lr="${PEAK_LR:-5e-5}"
    decay_lr="${DECAY_LR:-1e-5}"
    load_base_weights=true
    load_msp_vae_weights=true
    ;;
  *)
    echo "Unsupported OPENPI_TRAIN_STAGE=${train_stage}" >&2
    exit 1
    ;;
esac

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ckpt_setting="${ckpt_name}-${seed}"
EXP_NAME="${ckpt_setting}_${time}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
log_dir="${ckpt_dir}/logs"
# ################## ----(2)数据集 & 权重变量（可通过环境变量覆盖）
LEROBOT_REPO_ID="${LEROBOT_REPO_ID:-stack_bowls}"
ASSETS_DIR="${ASSETS_DIR:-/mnt/vepfs/vbot/lzx/RoboDojo/data/norm_state/single_task}"
ASSET_ID="${ASSET_ID:-stack_bowls}"
PARAMS_PATH="${PARAMS_PATH:-/mnt/vepfs/vbot/RoboDojo_Checkpoint/down_cpkt/Pi_05/ckpt/RoboDojo/Pi_05/RoboDojo-sim-arx_x5-joint-0/59999/params}"
MSP_VAE_WEIGHT_PATH="${MSP_VAE_WEIGHT_PATH:-/mnt/vepfs/vbot/lzx/RoboDojo/XPolicyLab/policy/Pi_05/checkpoints/stack_bowls_stage_1-0/29999/params}"
mkdir -p "${ckpt_dir}" "${log_dir}"
echo "------------------------CONFIG_NAME: $train_config_name--------------------"
echo "------------------------TRAIN_STAGE: $train_stage--------------------"
echo "------------------------EXP_NAME: $EXP_NAME--------------------"
echo "-------num_train_steps: $num_train_steps, batch_size: $batch_size, decay_steps: $decay_steps, peak_lr: $peak_lr, decay_lr: $decay_lr, ema_decay: $ema_decay, save_interval: ${save_interval}------------------"
if [[ "${load_base_weights}" == "true" ]]; then
  echo "------------------------PARAMS_PATH: ${PARAMS_PATH}--------------------"
else
  echo "------------------------PARAMS_PATH: skipped for stage1--------------------"
fi
if [[ "${load_msp_vae_weights}" == "true" && -n "${MSP_VAE_WEIGHT_PATH}" ]]; then
  echo "------------------------MSP_VAE_WEIGHT_PATH: ${MSP_VAE_WEIGHT_PATH}--------------------"
elif [[ "${load_msp_vae_weights}" == "false" ]]; then
  echo "------------------------MSP_VAE_WEIGHT_PATH: skipped for stage1--------------------"
fi
# 数据加载 worker 数
num_workers=${NUM_WORKERS:-32}
# ---- taskset 绑核 ----
CPU_BIND=${CPU_BIND:-96}
TASKSET="taskset -c 0-$((CPU_BIND-1))"
echo "CPU bind: cores 0-$((CPU_BIND-1))"
# ---- JAX 编译缓存（避免每次重新编译） ----
# export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/openpi-jax-cache-$(hostname)}"
# mkdir -p "${JAX_COMPILATION_CACHE_DIR}"
echo ""
echo "========== 路径校验 =========="
_check_path() {
  local desc="$1"
  local path="$2"
  if [ -e "$path" ]; then
    echo "  [OK] $desc: $path"
  else
    echo "  [FAIL] $desc: $path (不存在)"
    exit 1
  fi
}
_check_path "代码目录"           "$POLICY_DIR"
_check_path "数据集根目录"       "$XDG_CACHE_HOME"
_check_path "LeRobot HF 缓存"    "${HF_LEROBOT_HOME}/${LEROBOT_REPO_ID}"
_check_path "norm state path"    "${ASSETS_DIR}/${ASSET_ID}"
if [[ "${load_base_weights}" == "true" ]]; then
  _check_path "Base params path" "$PARAMS_PATH"
fi
if [[ "${load_msp_vae_weights}" == "true" && -n "${MSP_VAE_WEIGHT_PATH}" ]]; then
  _check_path "MSP VAE params path" "${MSP_VAE_WEIGHT_PATH}"
fi
echo "==============================="
echo ""
# ---- 启动训练 ----
export PYTHONPATH="${POLICY_DIR}/openpi/src:${PYTHONPATH:-}"
cd "${POLICY_DIR}/openpi/"
train_args=(
  scripts/train_tb.py "${train_config_name}"
  --exp-name="${EXP_NAME}"
  --data.repo-id="${LEROBOT_REPO_ID}"
  --data.assets.assets-dir="${ASSETS_DIR}"
  --data.assets.asset-id="${ASSET_ID}"
  --save-interval="${save_interval}"
  --checkpoint-dir-override="${ckpt_dir}"
  --ema-decay="${ema_decay}"
  --lr-schedule.decay-steps="${decay_steps}"
  --lr-schedule.peak-lr="${peak_lr}"
  --lr-schedule.decay-lr="${decay_lr}"
  --num-train-steps="${num_train_steps}"
  --batch-size="${batch_size}"
  --fsdp-devices="${GPUS_PER_NODE}"
  --num-workers="${num_workers}"
  --seed="${seed}"
  --overwrite
)
if [[ "${load_base_weights}" == "true" ]]; then
  train_args+=(--weight-loader.params-path="${PARAMS_PATH}")
fi
if [[ "${load_msp_vae_weights}" == "true" && -n "${MSP_VAE_WEIGHT_PATH}" ]]; then
  train_args+=(--msp-vae-weight-path="${MSP_VAE_WEIGHT_PATH}")
fi
PYTHONUNBUFFERED=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_ALLOCATOR=platform $TASKSET \
  /mnt/vepfs/vbot/lzx/RoboDojo/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python -u "${train_args[@]}" 2>&1 | tee -a "${log_dir}/${EXP_NAME}.log"

# ===== 启动示例 =====
# Stage1: MSP VAE 预训练
# OPENPI_TRAIN_STAGE=stage1 \
# bash train_multi_H20.sh msp_vae_stage1_aloha_action_only_arx-x5 stack_bowls_stage_1 30000 512 None 10000
#
# Stage2: Pi0.5 + MSP 微调（使用脚本内默认的 MSP_VAE_WEIGHT_PATH）
# OPENPI_TRAIN_STAGE=stage2 \
# bash train_multi_H20.sh pi05_msp_stage2_aloha_arx-x5_seed_0 msp_stack_bowls_stage_2 30000 2 None 100
#
# Stage2: 覆盖默认 Stage1 VAE 权重路径
# OPENPI_TRAIN_STAGE=stage2 \
# MSP_VAE_WEIGHT_PATH=/your/stage1/params \
# bash train_multi_H20.sh pi05_msp_stage2_aloha_arx-x5_seed_0 msp_stack_bowls_stage_2 30000 2 None 100
