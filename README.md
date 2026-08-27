# Pi_05

**Contributor:** RoboDojo Team | **Paper:** Pi0.5 technical report | **arXiv:** TBD | **Original code:** https://github.com/Physical-Intelligence/openpi

`Pi_05` adapts Physical Intelligence's π0.5 policy to XPolicyLab/RoboDojo through the uv-managed OpenPI stack. Integration scripts live at this directory level; the vendored upstream implementation lives in `openpi/`.

Shared conventions — argument meanings, checkpoint naming, split-machine deployment, `EVAL_ENV_TYPE` — are documented in the [XPolicyLab README](../../README.md). Official results: [RoboDojo LeaderBoard](https://robodojo-benchmark.com/LeaderBoard).

## Installation

```bash
cd XPolicyLab/policy/Pi_05
bash install.sh
source openpi/.venv/bin/activate  # OpenPI is uv-managed; there is no policy conda env
```

`eval.sh` arg 9 is not a conda env: pass `uv` (uses `deploy.yml` `policy_uv_env_path`) or an explicit OpenPI project path.

## Data Processing

Converts RoboDojo demonstrations into the LeRobot repo consumed by training. The optional `expert_data_num` caps episodes for data conversion only (it is not part of checkpoint naming); the optional `raw_task_dirs` is a source task directory or comma-separated task list under `data/<bench_name>/` (defaults to `ckpt_name`). `raw_task_dirs` may also be passed directly as the 5th argument to write a differently named dataset from all of a task's demos, e.g. `bash process_data.sh RoboDojo stack_bowls_ablation arx_x5 joint stack_bowls`.

```bash
cd XPolicyLab/policy/Pi_05
bash process_data.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> [expert_data_num] [raw_task_dirs]

# Example: convert stack_bowls demos for arx_x5 joint control
bash process_data.sh RoboDojo stack_bowls arx_x5 joint

# Example: create a 50-episode ablation while reading from the original task data
bash process_data.sh RoboDojo stack_bowls_50ep arx_x5 joint 50 stack_bowls
```

## Training

```bash
cd XPolicyLab/policy/Pi_05
bash train.sh <bench_name> <ckpt_name> <env_cfg_type> <action_type> <seed> <gpu_id>

# Example: train a cotrain run on GPU 0 (comma-separated gpu_id for multi-GPU)
bash train.sh RoboDojo cotrain arx_x5 joint 0 0

# Example: MSP stage-1 action VAE training on GPU 0
OPENPI_TRAIN_STAGE=vae_stage1 bash train.sh RoboDojo cotrain arx_x5 joint 0 0

# Example: no-dataset smoke test for MSP stage-1
OPENPI_TRAIN_STAGE=fake bash train.sh RoboDojo smoke arx_x5 joint 0 0

# Example: MSP stage-2 pi0.5 training with multiscale latent action head
OPENPI_TRAIN_STAGE=msp_stage2 OPENPI_MSP_VAE_WEIGHT_PATH=/path/to/stage1/params \
  bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

Checkpoints land in `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`; at eval time `ckpt_name` may be the short run name (auto-combined into that directory name), the full run-directory name, or a path to a checkpoint directory. By default training reads the LeRobot repo produced by `process_data.sh` (`<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`); override with `OPENPI_LEROBOT_REPO_ID` when reusing an existing dataset. `train.sh` sets `fsdp_devices=1` for one visible GPU and `2` for multi-GPU by default (override with `OPENPI_FSDP_DEVICES`).
For MSP stage 1, `train.sh` selects the `msp_vae_stage1_aloha_action_only_arx-x5` config when `OPENPI_TRAIN_STAGE=vae_stage1`; that config uses an action-only LeRobot loader which reads raw parquet action rows and does not decode images. Without a dataset, use `OPENPI_TRAIN_STAGE=fake` to select `msp_vae_stage1_fake`, which only validates the training/model code path and does not validate real data semantics. For MSP stage 2, `OPENPI_TRAIN_STAGE=msp_stage2` selects `pi05_msp_stage2_aloha_arx-x5_seed_0`; set `OPENPI_MSP_VAE_WEIGHT_PATH=<stage1_params_dir>` so the latent encoder/decoder come from the trained stage-1 VAE.

### MSP Stage 2

第二阶段训练前，你至少需要准备好两样东西：

1. Stage-1 训练出来的 MSP VAE checkpoint 目录
2. Stage-2 要用的 LeRobot 数据集

最小启动命令：

```bash
cd XPolicyLab/policy/Pi_05
OPENPI_TRAIN_STAGE=msp_stage2 \
OPENPI_MSP_VAE_WEIGHT_PATH=/path/to/stage1/checkpoint_dir \
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

这个命令会做的事：
- `OPENPI_TRAIN_STAGE=msp_stage2`
  - 默认选择 config `pi05_msp_stage2_aloha_arx-x5_seed_0`
- `OPENPI_MSP_VAE_WEIGHT_PATH=...`
  - 把 Stage-1 的 VAE 参数并进 Stage-2 模型里的 `msp_action_vae`
- `bash train.sh RoboDojo cotrain arx_x5 joint 0 0`
  - 数据集默认读 `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`
  - 这里就是 `RoboDojo-cotrain-arx_x5-joint`

如果你的 Stage-2 不想读默认数据集，可以显式指定：

```bash
cd XPolicyLab/policy/Pi_05
OPENPI_TRAIN_STAGE=msp_stage2 \
OPENPI_LEROBOT_REPO_ID=your_lerobot_repo_id \
OPENPI_MSP_VAE_WEIGHT_PATH=/path/to/stage1/checkpoint_dir \
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

如果你要覆盖默认 config，也可以显式指定：

```bash
cd XPolicyLab/policy/Pi_05
OPENPI_TRAIN_STAGE=msp_stage2 \
OPENPI_TRAIN_CONFIG_NAME=pi05_msp_stage2_aloha_arx-x5_seed_0 \
OPENPI_MSP_VAE_WEIGHT_PATH=/path/to/stage1/checkpoint_dir \
bash train.sh RoboDojo cotrain arx_x5 joint 0 0
```

多卡时只需要把 `gpu_id` 改成逗号分隔，例如：

```bash
cd XPolicyLab/policy/Pi_05
OPENPI_TRAIN_STAGE=msp_stage2 \
OPENPI_MSP_VAE_WEIGHT_PATH=/path/to/stage1/checkpoint_dir \
bash train.sh RoboDojo cotrain arx_x5 joint 0 0,1
```

几个关键点：
- 第二阶段不是 action-only 训练，会走正常的 pi0.5 prefix/VLM 数据路径
- 第二阶段会继续加载 pi0.5 的预训练权重，同时额外合并 Stage-1 的 `msp_action_vae` 权重
- `OPENPI_MSP_VAE_WEIGHT_PATH` 应该指向 Stage-1 训练产出的参数目录，而不是随便一个上级目录
- 输出 checkpoint 目录仍然是：
  - `checkpoints/<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>-<seed>/`

## Evaluation

```bash
cd XPolicyLab/policy/Pi_05
bash eval.sh <bench_name> <task_name> <ckpt_name> <env_cfg_type> <action_type> <seed> \
  <policy_gpu_id> <env_gpu_id> <policy_uv_env> <eval_env_conda_env>

# Example: evaluate a trained cotrain checkpoint on stack_bowls
bash eval.sh RoboDojo stack_bowls RoboDojo-cotrain-arx_x5-joint-0 arx_x5 joint 0 0 0 uv <eval_env_conda_env>
```

`EVAL_ENV_TYPE=debug` runs the offline wiring check (no simulator); leave it unset or set `EVAL_ENV_TYPE=sim` for RoboDojo simulation. For split-machine deployment via `setup_eval_policy_server.sh` / `setup_eval_env_client.sh`, follow the [Deployment Flow](../../README.md#-deployment-flow).

## Configuration

`deploy.yml` keys to check before evaluation: `checkpoint_num`, `result_dir`, `obs_transform_pipeline`, `policy_uv_env_path`, `train_config_name` (must match the config used by `train.sh`), `repo_id`.

Environment variables used by the adapter scripts:

| Variable | Notes |
|---|---|
| `OPENPI_LEROBOT_REPO_ID` | Overrides the LeRobot repo id used by `train.sh`; defaults to `<bench_name>-<ckpt_name>-<env_cfg_type>-<action_type>`. |
| `OPENPI_FSDP_DEVICES` | Overrides the FSDP device count passed to OpenPI training. |
| `OPENPI_TRAIN_CONFIG_NAME` | Overrides the training config; defaults to `pi05_base_aloha_full_sim_arx-x5_seed_0`. |
| `OPENPI_TRAIN_STAGE` | Chooses the default config in `train.sh`; use `vae_stage1` for MSP VAE training, `fake` for no-dataset smoke validation, `msp_stage2` for MSP latent-head pi0.5 training, or `pi05_stage2` for the original pi0.5 training path. |
| `OPENPI_MSP_VAE_WEIGHT_PATH` | Stage-1 MSP VAE params directory to merge into the Stage-2 pi0.5 MSP model. |
| `OPENPI_DATA_MODE` | Data-processing mode passed to `openpi/scripts/process_data.py`; defaults to `image`. |
| `OPENPI_LOCAL_CACHE_ROOT` | Per-host local cache root for the HF datasets / JAX compilation caches; defaults to `/tmp/openpi-cache-$(hostname)`. |

`OPENPI_ROOT` and `OPENPI_SRC` are additional overrides consumed by the local scripts.
