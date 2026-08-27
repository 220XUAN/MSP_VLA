#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 6 ]]; then
  echo "Usage: $0 <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>" >&2
  exit 1
fi

bench_name=$1
ckpt_name=$2
env_cfg_type=$3
action_type=$4
seed=$5
gpu_id=$6

POLICY_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# ckpt_setting is the run directory name; pass it verbatim as ckpt_name to eval.sh.
ckpt_setting="${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}-${seed}"
ckpt_dir="${POLICY_DIR}/checkpoints/${ckpt_setting}"
train_config_name="${OPENPI_TRAIN_CONFIG_NAME:-pi05_base_aloha_full_sim_arx-x5_seed_0}"
train_stage="${OPENPI_TRAIN_STAGE:-pi05_stage2}"
lerobot_repo_id="${OPENPI_LEROBOT_REPO_ID:-${bench_name}-${ckpt_name}-${env_cfg_type}-${action_type}}"
gpu_count=$(awk -F',' '{print NF}' <<<"${gpu_id}")
fsdp_devices="${OPENPI_FSDP_DEVICES:-$(( gpu_count < 2 ? 1 : 2 ))}"
msp_vae_weight_path="${OPENPI_MSP_VAE_WEIGHT_PATH:-}"

if [[ -z "${OPENPI_TRAIN_CONFIG_NAME:-}" ]]; then
  case "${train_stage}" in
    fake|vae_stage1_fake)
      train_config_name="msp_vae_stage1_fake"
      ;;
    vae_stage1|msp_vae_stage1|stage1)
      train_config_name="msp_vae_stage1_aloha_action_only_arx-x5"
      ;;
    msp_stage2|pi05_msp_stage2)
      train_config_name="pi05_msp_stage2_aloha_arx-x5_seed_0"
      ;;
    pi05_stage2|stage2|pi05)
      train_config_name="pi05_base_aloha_full_sim_arx-x5_seed_0"
      ;;
    *)
      echo "Unsupported OPENPI_TRAIN_STAGE=${train_stage}" >&2
      exit 1
      ;;
  esac
fi

mkdir -p "${ckpt_dir}"
export CUDA_VISIBLE_DEVICES="${gpu_id}"

# LeRobot loads parquet via HuggingFace datasets, which builds pyarrow mmap cache
# under HF_DATASETS_CACHE. Keep dataset on shared storage, but use per-host local
# cache to avoid NFS lock contention when multiple nodes train concurrently.
LOCAL_CACHE_ROOT="${OPENPI_LOCAL_CACHE_ROOT:-/tmp/openpi-cache-$(hostname)}"
mkdir -p "${LOCAL_CACHE_ROOT}/hf/datasets" "${LOCAL_CACHE_ROOT}/jax"
export HF_DATASETS_CACHE="${LOCAL_CACHE_ROOT}/hf/datasets"
export JAX_COMPILATION_CACHE_DIR="${LOCAL_CACHE_ROOT}/jax"

echo "[Pi_05] train_config_name=${train_config_name}"
echo "[Pi_05] train_stage=${train_stage}"
echo "[Pi_05] lerobot_repo_id=${lerobot_repo_id}"
echo "[Pi_05] fsdp_devices=${fsdp_devices}"
echo "[Pi_05] local_cache_root=${LOCAL_CACHE_ROOT}"
echo "[Pi_05] checkpoint_dir=${ckpt_dir}"
if [[ -n "${msp_vae_weight_path}" ]]; then
  echo "[Pi_05] msp_vae_weight_path=${msp_vae_weight_path}"
fi

cd "${POLICY_DIR}/openpi/"
train_args=(
  scripts/train.py "${train_config_name}"
  --exp-name="${ckpt_setting}"
  --data.repo-id="${lerobot_repo_id}"
  --fsdp-devices="${fsdp_devices}"
  --checkpoint-dir-override="${ckpt_dir}"
  --seed="${seed}"
  --overwrite
)
if [[ -n "${msp_vae_weight_path}" ]]; then
  train_args+=(--msp-vae-weight-path="${msp_vae_weight_path}")
fi

XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}" \
  uv run "${train_args[@]}"
