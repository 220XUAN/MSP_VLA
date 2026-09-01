我先读完了本地 `Pi_05` 里的关键结构和 MSP 代码，也查了 MSP 项目页。OpenReview PDF 本身被浏览器校验挡住了，我目前用的是项目页摘要和本地 `MSP/` 代码作为依据：MSP 是两阶段，Stage 1 学动作 VAE，Stage 2 在 VAE latent 上做 coarse-to-fine scale-wise autoregressive 生成，最后用 VAE decoder 解码动作。来源：<https://220xuan.github.io/msp-policy/>

我的理解如下：

1. **Stage 1: Action VAE 接入 openpi JAX**
   - MSP 原始 VAE 在 [MSP/algos/vae/vae.py](/home/xuan/PycharmProjects/VLA/MSP_VLA/XPolicyLab/policy/Pi_05/MSP/algos/vae/vae.py:1)。
   - 结构是 `action_proj -> causal temporal conv downsample -> Transformer encoder -> Gaussian latent -> Transformer decoder -> action_head`。
   - loss 是 `L1(recon, action) + kl_weight * KL`。
   - 这一步应该新建 JAX/Flax NNX 版本，独立于 pi0.5 的 VLM，不加载图像和语言。

2. **Stage 1 数据只加载 action**
   - 当前 openpi 数据入口在 [data_loader.py](/home/xuan/PycharmProjects/VLA/MSP_VLA/XPolicyLab/policy/Pi_05/openpi/src/openpi/training/data_loader.py:137)，LeRobot loader 会默认构造完整 observation，再走 transforms。
   - 训练循环在 [train.py](/home/xuan/PycharmProjects/VLA/MSP_VLA/XPolicyLab/policy/Pi_05/openpi/scripts/train.py:139) 默认 batch 是 `(Observation, actions)`。
   - 所以需要加一个显式配置，比如 `train_stage="vae_stage1"` 或 `action_only=True`，让 dataset 创建和 transforms 都绕过图像 resize/tokenize/repack，只产出 actions。训练 step 也应允许 action-only model 不依赖 `Observation`。

3. **Stage 2: 替换 pi0.5 动作头为 MSP latent scale head**
   - pi0.5 主体在 [pi0.py](/home/xuan/PycharmProjects/VLA/MSP_VLA/XPolicyLab/policy/Pi_05/openpi/src/openpi/models/pi0.py:60)。
   - 当前 flow matching 逻辑在 `compute_loss()`：构造 `x_t = t*noise + (1-t)*actions`，然后 `embed_prefix(obs)` + `embed_suffix(obs, x_t, time)`，最后 `action_out_proj` 预测 velocity。
   - MSP scale AR 逻辑在 [MSP/algos/flow/flow_ar.py](/home/xuan/PycharmProjects/VLA/MSP_VLA/XPolicyLab/policy/Pi_05/MSP/algos/flow/flow_ar.py:164)：先把 finest latent 下采样成 `[1,2,4,8]` 等多尺度 latent，输入序列是 `cls/obs + upsample(previous scale)`，用 scale mask 限制只能看当前及以前尺度，推理时逐尺度生成，最后得到 finest latent。
   - 对 pi0.5 来说，VLM prefix 保持原样，动作 suffix 从 “noisy action tokens + time AdaRMS” 改为 “multi-scale latent tokens + scale mask/position/condition”，最后不直接输出 action，而是输出 finest latent，再走 Stage 1 decoder。

后面你按任务顺序贴代码时，我建议先给我这些：

- **任务 1**：你希望放 JAX VAE 的目标目录/命名；如果没有，我会建议放在 `openpi/src/openpi/models/msp_vae.py`，config 放进 `pi0_config.py` 或单独 `msp_config.py`。
- **任务 2**：你当前实际训练用的 config 名称和数据集字段样例，尤其 action 字段是 `action` 还是 `actions`，有没有必须保留的 delta/normalization 逻辑。
- **任务 3**：你希望 Stage 2 保留 MSP 的 MeanFlow head，还是只采用 “尺度自回归 Transformer + latent decoder” 思路并适配 pi0.5 现有 action expert。这个会决定改动范围。

## 2026-08-26 任务 1 开始：MSP Stage-1 Action VAE

新的约束：
- JAX 版 MSP Action VAE 放在 `openpi/src/openpi/models/msp_vae.py`。
- config 放进 `openpi/src/openpi/models/pi0_config.py`。
- 保留原来的 pi0.5 训练逻辑；Stage 1 不需要 pi0.5，也不加载 VLM/action expert。

本轮实现目标：
- 新增独立 `MspActionVAEConfig`，默认 `action_horizon=32`、`downsample_factor=4`、`latent_dim=16`，对应 MSP 原始 VAE 结构。
- 新增 `MspActionVAE` 模型，训练只使用 `actions`，loss 为 L1 reconstruction + KL。
- 暂不修改 pi0.5 的 flow matching 训练路径；Stage 2 后续再接 MSP scale autoregressive action head。

已修改文件：
- `openpi/src/openpi/models/msp_vae.py`
- `openpi/src/openpi/models/pi0_config.py`
- `openpi/src/openpi/models/model.py`
- `openpi/src/openpi/training/config.py`
- `task.md`

验证记录：
- `openpi/.venv/bin/python -m py_compile openpi/src/openpi/models/msp_vae.py openpi/src/openpi/models/pi0_config.py openpi/src/openpi/models/model.py openpi/src/openpi/training/config.py` 通过。
- `git diff --check -- ...` 通过。
- 真实 JAX/Flax 初始化测试未完成：`uv run` 需要下载大量 JAX/Torch/CUDA 依赖，下载重试后已中止；当前 `.venv` 里还缺 `jax`。

## 2026-08-26 任务 2：Stage-1 config 训练入口 + action-only dataloader

需求补充：
- Stage 1 也需要复用 `train.sh -> config` 的训练入口。
- Stage 1 数据加载只需要 action，不需要图像。
- 如果我在执行任务时需要安装环境，才临时使用阿里云镜像；不要把镜像默认写进用户训练脚本。

本轮实现：
- 新增 `ActionOnlyLeRobotDataConfig`，通过 `DataConfig.action_only=True` 显式进入 Stage-1 action-only 数据路径。
- 新增 `ActionOnlyLeRobotDataset`：基于 LeRobot 原始 parquet 行构建未来 `action_horizon` 的 action chunk，并用 `episode_index` 在轨迹末尾做 last-action padding；这条路径不走图像解码。
- `DataLoaderImpl` 在 action-only 模式下返回 dummy observation，保证训练循环接口仍然是 `(Observation, actions)`，不破坏原有 pi0.5 训练逻辑。
- `train.py` 在 batch 没有图像时跳过 wandb 相机视图日志。
- `train.sh` 新增 `OPENPI_TRAIN_STAGE`：
  - `fake` / `vae_stage1_fake` -> 默认 config `msp_vae_stage1_fake`
  - `vae_stage1` -> 默认 config `msp_vae_stage1_aloha_action_only_arx-x5`
  - `pi05_stage2` -> 默认 config `pi05_base_aloha_full_sim_arx-x5_seed_0`
- 新增 Stage-1 config：
  - `msp_vae_stage1_fake`
  - `msp_vae_stage1_aloha_action_only_arx-x5`

Stage-1 训练命令：
- `OPENPI_TRAIN_STAGE=vae_stage1 bash train.sh RoboDojo cotrain arx_x5 joint 0 0`
- `OPENPI_TRAIN_STAGE=fake bash train.sh RoboDojo smoke arx_x5 joint 0 0`

本轮修改文件：
- `train.sh`
- `README.md`
- `openpi/scripts/train.py`
- `openpi/src/openpi/transforms.py`
- `openpi/src/openpi/training/config.py`
- `openpi/src/openpi/training/data_loader.py`
- `task.md`

## 2026-08-26 验证策略修正：没有数据集时如何验证

澄清：
- 阿里云镜像只在我本地执行 `uv`/安装环境时临时使用，不应写死到用户训练脚本里。
- 没有真实数据集时，不能验证 Stage-1 action-only dataloader 的真实数据语义，只能验证代码路径。

补充实现：
- 从 `train.sh` 删除了 `UV_DEFAULT_INDEX` / `PIP_INDEX_URL` 默认导出。
- 新增 `msp_vae_stage1_fake` config，用 `FakeDataConfig()` 跑一个无数据集 smoke test。
- `train.sh` 新增 `OPENPI_TRAIN_STAGE=fake` 映射到 `msp_vae_stage1_fake`。

无数据集时可做的验证：
- 静态检查：`py_compile` / `git diff --check`
- fake smoke test：确认 `train.sh -> config -> dataloader -> model -> train_step` 能跑起来

无数据集时不能验证的内容：
- action-only LeRobot loader 是否正确读取你的真实 `action` 字段
- `episode_index` 边界 padding 是否符合你的实际数据
- norm stats / delta action 语义是否和真实数据一致

## 2026-08-26 设计判断：是否需要单独的 `train_stage1.py`

当前判断：
- 现有 `openpi/scripts/train.py` 会直接影响 Stage 1，因为 Stage 1 复用了同一套训练主循环、checkpoint、optimizer、FSDP、wandb 和 config CLI。
- 以目前的差异程度，不需要单独拆一个干净的 `train_stage1.py`。

原因：
- Stage 1 和 Stage 2 的公共部分很多：`TrainConfig`、checkpoint 管理、参数初始化、优化器、sharding、日志、CLI config 选择。
- 我已经把 `train.py` 里唯一会对 Stage 1 产生硬假设的地方收掉了：无图 batch 时跳过 `camera_views` 日志。
- Stage 1 现在虽然只用 action，但接口上仍然保持 `(Observation, actions)`，所以训练主循环不用分叉。

什么时候才值得拆：
- Stage 1 后面要引入独立验证集/重建可视化/latent 导出。
- Stage 1 想改成完全不同的 batch 结构，不再传 dummy observation。
- Stage 1 和 Stage 2 需要不同的 checkpoint 内容、恢复逻辑或训练调度。

现阶段建议：
- 继续共用 `openpi/scripts/train.py`。
- 把分歧限制在 `config`、`model`、`data_loader`。
- 等 Stage 2 动作头接完后，再判断训练脚本是否已经被分支逻辑污染到值得拆分。

## 2026-08-26 任务 3：MSP 多尺度 latent 动作头接入 pi0.5

确认的默认：
- VLM / prefix 保持现有 pi0.5 结构，不改。
- 动作头改成 MSP 风格的“尺度自回归 Transformer + latent decoder”。
- 不接 MSP 原始 MeanFlow head。
- Stage-2 默认冻结 Stage-1 VAE 子模块参数，只把它作为 latent encoder/decoder 使用。

本轮实现：
- 在 `Pi0Config` 里新增 MSP Stage-2 开关和超参：
  - `use_msp_action_head`
  - `freeze_msp_vae`
  - `msp_*` 一组 VAE/scales 配置
- 新增 `openpi/src/openpi/models/msp_scale_head.py`：
  - 多尺度长度构造
  - 线性 resize
  - scale-wise autoregressive mask
  - teacher-forced 多尺度 latent 输入/监督构造
- `openpi/src/openpi/models/msp_vae.py` 新增 `encode_mean()`，供 Stage-2 稳定地产生 finest latent 监督
- `openpi/src/openpi/models/pi0.py` 新增 MSP 分支：
  - 复用现有 prefix / PaliGemma / action expert
  - suffix 改为多尺度 latent token
  - 增加 scale embedding 和时间位置编码
  - 用 block-wise autoregressive mask 让不同尺度按 MSP 方式因果可见
  - 训练时在 latent 空间做 teacher-forced MSE
  - 推理时逐尺度自回归生成 finest latent，再走 Stage-1 decoder 解码动作
- `TrainConfig` 新增 `msp_vae_weight_path`
  - Stage-2 可额外加载 Stage-1 VAE params
- `weight_loaders.py` 新增 `merge_msp_vae_params()`
  - 把 Stage-1 checkpoint 中的 `action_vae/...` 参数重映射到 Stage-2 模型里的 `msp_action_vae/...`
- 新增 Stage-2 config：
  - `pi05_msp_stage2_aloha_arx-x5_seed_0`
- `train.sh` 新增：
  - `OPENPI_TRAIN_STAGE=msp_stage2`

Stage-2 训练命令：
- `OPENPI_TRAIN_STAGE=msp_stage2 bash train.sh RoboDojo cotrain arx_x5 joint 0 0`
- 真实使用时需要给 `train.sh` 额外传：
  - `OPENPI_MSP_VAE_WEIGHT_PATH=<stage1_params_dir>`

本轮修改文件：
- `openpi/src/openpi/models/msp_scale_head.py`
- `openpi/src/openpi/models/pi0.py`
- `openpi/src/openpi/models/pi0_config.py`
- `openpi/src/openpi/models/msp_vae.py`
- `openpi/src/openpi/training/config.py`
- `openpi/src/openpi/training/weight_loaders.py`
- `openpi/scripts/train.py`
- `train.sh`
- `README.md`
- `task.md`

### 任务 3 当前未完全对齐 MSP 原版的点

用户指出的 MSP 细节是成立的，当前实现还没有完全复现这三点：

1. `Attention.forward()` 里的 RoPE 用法
- MSP 训练时不是直接对整段序列做一次位置编码，而是按每个 scale block 分段做 `self.rope(...)`，相当于每个尺度内部的位置从 0 重新开始。
- 我当前在 `pi0.py` 里用的是统一的 `token_posemb_sincos + scale_embed`，没有实现“每个尺度单独重置的 RoPE”。

2. MSP 的 attention mask 形式
- 我当前的 `build_scale_ar_mask() + make_attn_mask()` 在可见性语义上接近 MSP 的 block-wise causal mask：当前尺度能看之前所有尺度和本尺度过去 token。
- 但实现形式不是 MSP 原版的显式二维 attention matrix / `0` 与 `-inf` buffer。

3. 推理阶段的 positional embedding 切片
- MSP 原版在推理阶段会对当前尺度使用 `decoder_pos_embed_learned[:, start:end]` 和 `diffusion_pos_embed_learned[:, start:end]`。
- 我当前实现没有这组 learned decoder/diffusion positional embedding，也没有按 `start:end` 切片；因为现在是每个尺度重跑 prefix + partial suffix，而不是严格按 MSP 原版 decoder 路径走。

结论：
- 当前任务 3 版本是“思路对齐版”，不是“细节逐项对齐版”。
- 如果要严格参考 MSP，需要下一步把 action head 再收敛到：
  - 分尺度重置的 RoPE
  - 显式 block attention matrix
  - 训练/推理分开的 learned positional embeddings，并在推理时按 `start:end` 切片

## 2026-08-26 补充：`MSP/algos/flow/flow_ar.py` 的 mask 与位置编码细节

这部分已经重新逐段核对，后续 Task 3 细化必须按这里实现，不再沿用当前简化版假设。

### 1. MSP 的 mask 不是普通 token-level causal mask，而是 block-wise scale mask

对应代码：
- [MSP/algos/flow/flow_ar.py](/home/xuan/PycharmProjects/VLA/MSP_VLA/XPolicyLab/policy/Pi_05/MSP/algos/flow/flow_ar.py:234)

核心逻辑：
- `self.scale=(1,2,4,8)` 时，总长度 `total_length=sum(scale)=15`
- 第 `i` 个尺度长度为 `pz`
- 该尺度里的每个 token 都能看到：
  - 所有更粗尺度的全部 token
  - 当前尺度内部的全部 token
  - 不能看到更细尺度 token
- 所以 mask 的每一行不是标准下三角，而是“按尺度块展开的前缀可见”

以 `(1,2,4,8)` 为例，拼出来的是：
- scale=1 的 1 行，看前 1 个 token
- scale=2 的 2 行，都看前 3 个 token
- scale=4 的 4 行，都看前 7 个 token
- scale=8 的 8 行，都看前 15 个 token

实现形式：
- 先拼 `1/0` 矩阵
- 再转成 attention bias：
  - `1 -> 0`
  - `0 -> -inf`
- 最终形状是 `[1, 1, total_length, total_length]`

这意味着：
- MSP 训练阶段**不是**尺度内逐 token 自回归
- 它是**尺度块级别**的 coarse-to-fine 自回归
- 同一尺度内 token 彼此全可见

我当前 `openpi` 实现的问题：
- 现在的 `build_scale_ar_mask()` 还是 token-level AR 语义，不是这个 block-wise 语义。
- 后续必须改成显式二维块掩码，而不是继续复用当前的布尔 `ar_mask` 近似。

### 2. MSP 训练态的 RoPE 是“每个尺度块内部重新从 0 开始”

对应代码：
- [MSP/algos/flow/flow_ar.py](/home/xuan/PycharmProjects/VLA/MSP_VLA/XPolicyLab/policy/Pi_05/MSP/algos/flow/flow_ar.py:127)

核心逻辑：
- 先根据 `self.scales` 构造各尺度在拼接序列里的切片范围
- 对 `q[:, :, sequence[i]:sequence[i+1]]` 单独做 `self.rope(...)`
- 对 `k[:, :, sequence[i]:sequence[i+1]]` 也单独做 `self.rope(...)`
- 再把各尺度块 `cat` 回去

这和“整段序列统一做一次位置编码”不同：
- 统一做一次时，后面尺度的 token 位置会持续递增
- MSP 原版是每个尺度块内部位置重新编号为 `0..scale_i-1`

我当前 `openpi` 实现的问题：
- 现在 `pi0.py` 用的是统一的 `token_posemb_sincos + scale_embed`
- 这不能等价复现 MSP 的分块重置 RoPE

### 3. MSP 推理态不是训练态那种整段输入，而是逐尺度增量生成

对应代码：
- [MSP/algos/flow/flow_ar.py](/home/xuan/PycharmProjects/VLA/MSP_VLA/XPolicyLab/policy/Pi_05/MSP/algos/flow/flow_ar.py:345)

推理流程：
- `x` 初始只有 `cls/obs`，长度 1
- 每一步只处理当前尺度对应的那一小段 token
- `start, end` 来自尺度前缀和
- 当前步 encoder/decoder 只接收当前段长度的输入
- attention 内部通过 `self.k/self.v` cache 把历史尺度记住

这意味着推理阶段的“自回归”是：
- 不是每次重喂完整多尺度序列
- 而是当前尺度增量 forward，历史靠 KV cache 保留

我当前 `openpi` 实现的问题：
- 现在 `_sample_msp_actions()` 是每个尺度重跑一次 prefix + partial suffix
- 语义上接近 coarse-to-fine，但执行路径不是 MSP 原版的增量 cache 机制

### 4. MSP 的 learned positional embedding 在训练态和推理态用法不同

对应代码：
- encoder:
  - [MSP/algos/flow/flow_ar.py](/home/xuan/PycharmProjects/VLA/MSP_VLA/XPolicyLab/policy/Pi_05/MSP/algos/flow/flow_ar.py:269)
- decoder:
  - [MSP/algos/flow/flow_ar.py](/home/xuan/PycharmProjects/VLA/MSP_VLA/XPolicyLab/policy/Pi_05/MSP/algos/flow/flow_ar.py:284)

三组 learned embedding：
- `encoder_pos_embed_learned`
- `decoder_pos_embed_learned`
- `diffusion_pos_embed_learned`

训练态：
- 直接使用全长 `[ :, :seq_len ]`

推理态：
- 只取当前尺度段：
  - `encoder_pos_embed_learned[:, start:end]`
  - `decoder_pos_embed_learned[:, start:end]`
  - `diffusion_pos_embed_learned[:, start:end]`

这里 `self.diffusion_pos_embed_learned[:, start:end]` 的意义是：
- flowed latent head 在每个尺度都只接收该尺度局部 token
- 但每个尺度局部 token 仍然有一套“属于自己这段”的 learned diffusion positional bias
- 不是共享整段 positional embedding，也不是简单 sinusoidal

我当前 `openpi` 实现的问题：
- 目前没有这三组 learned positional embedding
- 也没有推理态 `start:end` 切片
- 这也是为什么当前实现只能叫“思路对齐版”

### 5. 对 `openpi/pi0.py` 的后续实现约束

如果继续细化 Task 3，必须至少满足下面四点：

1. suffix attention mask 改成 MSP 同构的显式 block-wise 矩阵
2. 位置编码改成：
   - 训练态：全长 learned embedding
   - 推理态：当前尺度 `start:end` 切片
3. 如果要保留 MSP 的 attention 细节，就必须支持“每个尺度块内部重置位置”的 RoPE，而不是统一全局位置
4. 推理路径最好改成增量式当前尺度 forward；如果继续保留“每尺度重跑整段”的写法，必须明确这是和 MSP 原版不完全一致的近似实现

当前判断：
- 如果坚持“VLM 不动，只改 action head”，那就要评估 Gemma action expert 是否允许精确插入这套分块 RoPE + KV cache 逻辑。
- 如果 Gemma 内部不好改，下一步更稳妥的方案可能是：prefix 仍然用 pi0.5，MSP multi-scale decoder 独立成 suffix/action expert，而不是强行塞进现有 Gemma suffix 路径。

## 2026-08-26 下一步修改规划：Task 3 如何从“思路对齐版”收敛到“实现对齐版”

已经确认的实现边界：
- `openpi/src/openpi/models/gemma.py` 的 attention 内部直接用 `positions` 做全局 `_apply_rope(q/k, positions=positions)`。
- 这意味着当前 Gemma expert 的 rope 语义是“整段统一位置”，不是 MSP `flow_ar.py` 那种“每个 scale block 内部从 0 重置”的 rope。
- 同时，Gemma 的 KV cache 是标准 decoder cache；MSP `flow_ar.py` 的推理路径是按尺度段递增缓存，配套的是它自己的 encoder/decoder block 结构。

基于这个事实，Task 3 后续有两条路：

### 路线 A：继续复用 Gemma action expert，只做近似对齐

优点：
- 改动范围小
- 能保留现有 pi0.5 action expert 权重
- Stage-2 可直接从 pi0.5 预训练初始化

缺点：
- 不能严格复现 MSP 的 block-reset RoPE
- 推理也很难做成 `flow_ar.py` 那种逐尺度局部 forward + 内部 cache
- 最终只能做到“mask/多尺度监督/latent decode 对齐”，不是完整 MSP attention 机制对齐

如果走这条路，修改顺序是：
1. 把 `msp_scale_head.build_scale_ar_mask()` 改成显式二维 block-wise mask
2. `pi0.py` 中 MSP suffix 增加 learned positional embedding 参数
3. 训练态用全长 pos embedding，推理态用 `start:end` 切片
4. 保留 Gemma 原生 global RoPE，不再试图复现分块重置 RoPE
5. 在 `task.md` 和代码注释里明确这是“MSP-compatible approximation”

### 路线 B：VLM prefix 继续复用 pi0.5，MSP multi-scale decoder 独立实现

优点：
- 可以严格照着 `MSP/algos/flow/flow_ar.py` 做
- mask、RoPE、learned pos embedding、增量 cache 都能逐项对齐
- 不会被 Gemma 内部 attention 约束住

缺点：
- 改动更大
- Stage-2 动作头不再直接复用 pi0.5 的 action expert block 权重，只复用 prefix/VLM
- 需要在 JAX 里单独实现 MSP 的 `Attention/Block/forward_mae_encoder/forward_mae_decoder/sample_tokens`

如果走这条路，修改顺序是：
1. 新建独立 `openpi/src/openpi/models/msp_flow_ar.py`
2. 在里面实现：
   - `ActionRotaryEmbeddingFast` 的 JAX 版
   - MSP 的 block-wise attention mask builder
   - MSP `Attention`
   - MSP `Block`
   - `forward_mae_encoder`
   - `forward_mae_decoder`
   - 增量 `sample_tokens`
3. `pi0.py` 只负责：
   - 调 `embed_prefix(observation)` 得到 VLM prefix
   - 把 prefix 聚合成 MSP 需要的 condition / obs embedding
   - 调独立的 MSP latent decoder
   - 用 Stage-1 VAE decoder 解码动作
4. `TrainConfig` 和权重加载逻辑只保留 prefix/VLM 从 pi0.5 初始化，MSP decoder 走新参数

### 当前建议

从“严格参考 MSP 原实现”这个目标看，**更稳妥的是路线 B**。

原因：
- 你已经明确要求仔细参考 `flow_ar.py` 的 mask 和位置编码设计。
- 这两个点都不是外面补个 embedding 就能解决的，它们嵌在 attention 行为本身里。
- 如果继续硬塞进 Gemma action expert，最后大概率会得到一个行为近似但细节不一致的版本。

### 下一轮建议的最小可执行任务

先不要继续修改 `pi0.py` 的近似版细节，下一轮直接做下面两件事：

1. 新建 `openpi/src/openpi/models/msp_flow_ar.py`
   - 先只迁移 `flow_ar.py` 的 encoder/decoder/mask/rope/sample_tokens` 主体
   - 暂时不接 MSP 原始 `MPScalseFlowhead`
   - 输出 finest latent 即可

2. 再把 `pi0.py` 的 MSP 分支改成：
   - prefix 保持 pi0.5 不动
   - suffix 不再走 Gemma action expert
   - 改为调用独立的 `msp_flow_ar.py`

这样改完之后，Task 3 的结构会更干净：
- pi0.5 VLM prefix
- MSP latent autoregressive decoder
- Stage-1 VAE decoder

而不是把 MSP 的 attention 语义强行塞进 Gemma suffix。

## 2026-08-26 补充判断：`sample_tokens` 这种“VLM 只跑一次 + 尺度头增量 cache”能不能实现

结论：
- **能实现。**
- 但这里的前提是：`sample_tokens()` 这条推理路径要落在**独立的 MSP transformer 尺度头**里，而不是继续复用 pi0.5 当前的 Gemma suffix action expert。

原因：
- `MSP/algos/flow/flow_ar.py` 的 `sample_tokens()` 本质上就是：
  1. 先拿一次固定的 observation condition
  2. 然后按尺度段逐步生成 latent
  3. 每一步只输入当前尺度段 token
  4. 历史尺度通过 attention 内部 cache 保留
  5. 训练态 / 推理态的位置编码规则不同，并且推理态按 `start:end` 切片
- 这条路径本身就是标准 Transformer 可以做的事情，不是理论障碍。

真正的约束不在“Transformer 能不能做”，而在“当前 pi0.5 这个 action head 能不能原样承载这套语义”：
- Gemma suffix 现在的 rope 是全局 `positions`
- MSP 需要按尺度段重置位置
- Gemma 的 decode/cache 路径是普通 decoder token cache，不是 MSP 这种按尺度块递增的局部 forward 设计

所以更准确的判断是：
- **MSP 风格的增量 cache 推理路径可以做**
- **但应当由一个独立的 JAX MSP 尺度 Transformer 头来做**
- `pi0.py` 只保留：
  - VLM prefix 跑一次
  - 把 prefix/context 喂给 MSP 尺度头
  - MSP 尺度头逐尺度生成 finest latent
  - Stage-1 VAE decoder 解码动作

这也是后续 Task 3 推荐的落地结构。

## 2026-08-26 进一步澄清：Gemma 支持可变序列长度，不等于已经对齐 MSP 的位置机制

新的澄清点：
- `pi0.5` 的 Gemma suffix 确实可以输入任意长度序列。
- 所以如果只讨论“训练时全长、推理时按尺度段喂更短序列”，这在接口层面是可行的。

但这只解决了下面这类问题的一部分：
- 推理时当前尺度只跑 `start:end` 这段长度
- learned positional embedding 可以只取当前长度或当前切片

它**没有自动解决** MSP 原版里更关键的 attention 位置语义：
- MSP 训练态 `Attention.forward()` 对每个 scale block 单独做 `rope`
- 这等价于每个尺度块内部位置从 0 重新开始
- Gemma 的 `_apply_rope(q/k, positions=positions)` 是对整段位置一次性应用

所以要分开看：

1. **序列长度维度**
- Gemma 能处理可变长度输入
- 推理时也能只喂当前尺度长度的 token

2. **位置编码语义**
- 如果继续用 Gemma 原生 rope，那么位置仍然是“全局位置”
- 这和 MSP 的“每个尺度块内部重置位置”不是同一件事

因此：
- 如果你的目标是“VLM 跑一次，尺度头逐尺度细化，并且每一步只输入当前尺度段”，这个方向本身没问题。
- 但如果你的目标是“严格对齐 `flow_ar.py` 的位置编码细节”，光靠“推理时取想要的序列长度”还不够，训练态和推理态的 rope 语义仍然不同。

更精确的结论：
- **可变长度输入**：Gemma 可以做到
- **按尺度段增量推理**：理论上可以在 Gemma 外层组织出来
- **MSP 原版分块重置 RoPE**：Gemma 现成实现不能直接等价复现

## 2026-08-26 新方案评估：是否可以直接改动作头 Gemma，新增 block-local RoPE

结论：
- **可以改。**
- 这是一个可行方案，而且比“完全独立新写一个 MSP decoder”更保留 pi0.5 现有 action expert 权重。
- 但这不是简单加一个位置编码参数，而是要改 `openpi/src/openpi/models/gemma.py` 的 attention 行为，使它对 action expert 分支支持 MSP 需要的 **block-local RoPE**。

这个方案要解决的核心点：

1. **RoPE 不再只接受全局 `positions`**
- 现在 Gemma 的 `_apply_rope(q/k, positions=positions)` 假设整段 token 共用一条全局位置轴。
- 如果要支持 MSP，需要让 action expert 分支额外支持：
  - 输入 `rope_block_ids` 或 `rope_local_positions`
  - 对同一 block 内 token 使用 `0..len(block)-1`
  - 不同 block 之间位置重新开始

2. **只改动作头分支，不动 VLM/prefix 分支**
- prefix/VLM 部分继续使用现有全局位置，不改。
- action expert 分支在 `Pi0.compute_loss()` / `sample_actions()` 调 Gemma 时，额外传入 MSP 的局部位置元数据。
- 这样可以把改动限制在 suffix/action expert。

3. **mask 仍然要改成 MSP block-wise 语义**
- 即使 RoPE 改好了，也还要把 suffix self-attention mask 改成 MSP 那种“同尺度全可见、未来更细尺度不可见”的块掩码。

4. **推理路径仍然需要逐尺度增量调用**
- 训练时可以喂全长多尺度序列
- 推理时按尺度段调用 action expert
- 每一步给当前尺度段传对应的 local positions
- KV cache 保留历史尺度

### 相比“独立 MSP decoder”的利弊

优点：
- 能继续复用 pi0.5 action expert 权重
- 模型结构改动更集中，Stage-2 仍挂在 Gemma suffix 路径上
- 用户目标“只改动作头，不动 VLM”可以更严格满足

代价：
- 要侵入 `gemma.py` attention 接口
- 要给 Gemma 增加“仅 suffix/action expert 使用 block-local RoPE”的分支逻辑
- 实现和调试复杂度高于简单近似版

### 如果走这条路，建议的修改顺序

1. 改 `openpi/src/openpi/models/gemma.py`
- 给 attention 增加可选的 `rope_positions` 或 `rope_segments`
- 允许不同 expert 使用不同的 rope 位置语义

2. 改 `openpi/src/openpi/models/pi0.py`
- MSP suffix 构造：
  - block-wise attention mask
  - block-local rope positions
  - 训练态全长输入
  - 推理态按尺度 `start:end` 增量输入

3. 改 `openpi/src/openpi/models/msp_scale_head.py`
- 提供：
  - block-wise mask builder
  - 每个尺度块的 local positions builder
  - 推理阶段每一步 `start/end` 和当前 block positions

### 当前判断

这条路在工程上是可行的。

如果你的优先级是：
- 尽量保留 pi0.5 action expert
- 不单独再起一套 MSP decoder

那么下一步就应该按这个方向推进：**定制 Gemma action head，使 suffix 支持 block-local RoPE。**

## 2026-08-26 风险澄清：改 Gemma 会不会影响 VLM，预训练权重还能不能加载

结论：
- **可以做到不影响 VLM 部分。**
- **也可以继续加载 pi0.5 预训练权重。**
- 但前提是修改方式必须保持参数结构兼容，或者把新增能力做成可选分支，而不是直接改掉现有参数形状。

### 1. 为什么不一定会影响 VLM

`openpi/src/openpi/models/gemma.py` 现在同时服务两路 expert：
- 第一路是 PaliGemma / VLM prefix
- 第二路是 action expert suffix

如果修改方式是下面这种，就可以把影响限制在 action expert：
- 保留现有默认行为：全局 `positions -> _apply_rope`
- 只在 action expert 调用时，额外传一个可选的 `rope_positions`
- 当 `rope_positions is None` 时，完全走原逻辑
- 当 `rope_positions` 存在时，只对 suffix/action expert 使用 block-local positions

这样：
- VLM prefix 完全不需要改训练逻辑
- VLM 仍然走原来的 global positions
- 原有 pi0.5 非 MSP 路径也不受影响

### 2. 预训练权重为什么还能加载

只要不改下面这些已有参数的 shape，就还能直接加载：
- Q/K/V projection
- attention output projection
- MLP
- RMSNorm
- embedder
- 现有 adaRMS 相关层

如果只是：
- 改 `forward/__call__` 接口
- 新增一个可选 `rope_positions`
- 在 attention 里改 RoPE 的应用方式

那么**参数张量本身可以完全不变**，所以 pi0.5 预训练权重仍然可以直接加载。

### 3. 什么改法会破坏权重兼容

下面这些会让现有权重难以直接复用：
- 改 attention projection 的维度
- 改 block 宽度、head 数、层数
- 给现有线性层换 shape
- 把 action expert 整体替换成另一套不同结构的 decoder

所以如果目标是保留 pi0.5 预训练初始化，Gemma 的改法应该控制在：
- **改 attention 行为**
- **不改 attention 参数形状**

### 4. 当前推荐的实现原则

如果继续走“改 Gemma action head”这条路，应该遵守：

1. `gemma.py` 默认路径保持完全兼容
2. 新增 `rope_positions` 作为可选输入，而不是替换现有 `positions`
3. VLM expert 继续只用原始 `positions`
4. MSP suffix/action expert 才启用 block-local RoPE
5. 不改任何已有预训练参数 shape

这样可以同时满足：
- VLM 不动
- Stage-2 还能加载 pi0.5 预训练权重
- 动作头增加 MSP 需要的位置语义

## 2026-08-26 本轮实现：在 Gemma action expert 上接入 block-local RoPE

本轮目标：
- 不改 VLM expert 的行为
- 不改 Gemma 现有参数 shape
- 只给 MSP suffix / action expert 增加 block-local RoPE 能力
- 把 MSP 推理改成“prefix 只跑一次 + 按尺度段增量 cache”

本轮已修改文件：
- `openpi/src/openpi/models/gemma.py`
- `openpi/src/openpi/models/msp_scale_head.py`
- `openpi/src/openpi/models/pi0.py`
- `task.md`

### 1. `gemma.py` 的改动

改动位置：
- [openpi/src/openpi/models/gemma.py](/home/xuan/PycharmProjects/VLA/MSP_VLA/XPolicyLab/policy/Pi_05/openpi/src/openpi/models/gemma.py:157)

实现内容：
- 给 `Module.__call__()` 新增可选参数 `rope_positions`
- 给 `Block.__call__()` 和 `Attention.__call__()` 透传 `rope_positions`
- `rope_positions is None` 时，仍然完全走原来的 global `positions`
- `rope_positions` 提供时，对对应 expert 使用该局部位置，而不是整段全局位置

关键兼容性处理：
- attention 内部现在按 expert 的 token 长度切分 `positions`
- 所以 VLM prefix 和 action suffix 即使长度不同，也不会把位置轴串掉
- 参数 shape 没有变化，仍可加载原 pi0.5 预训练权重

### 2. `msp_scale_head.py` 的改动

新增工具函数：
- `build_block_local_positions(scales, batch_size)`
- `build_scale_segment_bounds(scales)`
- `build_suffix_block_attention_mask(scales, batch_size, input_mask=None)`
- `build_full_attention_mask(prefix_mask, suffix_mask, suffix_attention_mask)`
- `build_current_scale_inputs(generated_blocks, scales, ...)`

作用：
- 训练时构造 MSP 风格显式块掩码
- 训练/推理都构造 block-local positions
- 推理时按当前尺度生成输入 latent，而不是每次重构整段 partial sequence

### 3. `pi0.py` 的 MSP 路径改动

训练路径：
- `_embed_msp_suffix()` 不再叠加统一 temporal sin/cos
- 改为返回：
  - `suffix tokens`
  - `suffix input mask`
  - `adarms_cond`
  - `block-local rope_positions`
- `_compute_msp_loss()` 改为：
  - 用显式 `suffix block attention mask`
  - 再与 prefix mask 合成完整 attention mask
  - 调 Gemma 时传 `rope_positions=[None, rope_positions]`

推理路径：
- `_sample_msp_actions()` 先只跑一次 prefix，拿到 `kv_cache`
- 后续每个尺度：
  - 只构造当前尺度需要的 latent 输入
  - 只 forward 当前尺度长度的 suffix token
  - 使用当前尺度的 local rope positions
  - 复用并更新 `kv_cache`
- 最后把 finest latent 送入 Stage-1 VAE decoder 解码动作

### 4. 当前状态和剩余差距

这一轮之后，已经实现了：
- 保持 VLM expert 不变
- action expert 支持可选 block-local RoPE
- MSP suffix 使用显式块掩码
- 推理路径改成 prefix 一次 + 按尺度段增量 cache

还没有完全对齐 `flow_ar.py` 的点：
- 还没有补 MSP 原版那三组 learned positional embeddings：
  - `encoder_pos_embed_learned`
  - `decoder_pos_embed_learned`
  - `diffusion_pos_embed_learned`
- 目前是：
  - Gemma attention 内部用 block-local RoPE
  - suffix token 额外只保留 scale embedding

所以当前版本相对之前前进了一步：
- 已经不再是“整段统一位置 + 每尺度重跑整段”的近似版
- 但仍不是 `flow_ar.py` 的 100% 同构版

### 5. 本轮验证

已通过：
- `openpi/.venv/bin/python -m py_compile openpi/src/openpi/models/gemma.py openpi/src/openpi/models/msp_scale_head.py openpi/src/openpi/models/pi0.py`
- `git diff --check -- openpi/src/openpi/models/gemma.py openpi/src/openpi/models/msp_scale_head.py openpi/src/openpi/models/pi0.py task.md`

未做：
- 真实前向运行
- 真实 checkpoint 加载回归
- 真实 Stage-2 训练验证

## 2026-08-27 本轮实现：补 MSP learned positional embeddings 和推理 `start:end` 切片

本轮目标：
- 在现有 Gemma-action-expert 版 MSP 路径上补齐 MSP 原版的三组 learned positional embeddings
- 训练时走全长 embedding
- 推理时每个尺度只取当前段的 `start:end` 切片

本轮修改文件：
- `openpi/src/openpi/models/pi0.py`
- `task.md`

## 2026-08-28 MSP VAE 配置补充：`kl_weight` 默认改为 `1e-6`，并通过 `Pi0Config` 透传

需求：
- `openpi/src/openpi/models/pi0_config.py` 中：
  - `MspActionVAEConfig.kl_weight` 默认从 `1e-5` 改成 `1e-6`
- 并且在 `Pi0Config` 里把这个参数也挂出来，和 `msp_latent_dim` 一样可配置，再传入 Stage-2 内部构造的 `msp_action_vae`

本轮修改：
- 文件：`openpi/src/openpi/models/pi0_config.py`

1. `Pi0Config` 新增：
- `msp_kl_weight: float = 1e-6`

2. `make_msp_vae_config()` 透传：
- `kl_weight=self.msp_kl_weight`

3. `MspActionVAEConfig` 默认值修改：
- `kl_weight: float = 1e-6`

当前效果：
- Stage-1 直接使用 `MspActionVAEConfig(...)` 时，默认 `kl_weight` 已经是 `1e-6`
- Stage-2 通过 `Pi0Config.make_msp_vae_config()` 构造内部 `msp_action_vae` 时，也会带上：
  - `msp_kl_weight`

本轮验证：
- `openpi/.venv/bin/python -m py_compile openpi/src/openpi/models/pi0_config.py`
- `git diff --check -- openpi/src/openpi/models/pi0_config.py`

本轮修改文件：
- `openpi/src/openpi/models/pi0_config.py`
- `task.md`

## 2026-08-28 审计结论：当前 pi0.5 里的 MSP 还没有“完全移植”，而是“按 pi0.5 结构约束做了核心机制适配”

这次对照了：
- 原版：
  - `MSP/algos/MSP.py`
  - `MSP/algos/flow/flow_ar.py`
  - `MSP/algos/vae/vae.py`
- 当前实现：
  - `openpi/src/openpi/models/pi0.py`
  - `openpi/src/openpi/models/msp_vae.py`
  - `openpi/src/openpi/models/msp_scale_head.py`

结论先说：
- **Stage 1 VAE：大体移植完成，但不是逐层 1:1 复刻**
- **Stage 2 MSP 头：核心机制大体接上了，但不是原版 FlowAR 的完整移植**
- **整体上不是“完全移植”**
- 更准确地说，是：
  - **在保留 pi0.5 VLM / Gemma 主干不动的前提下，移植了 MSP 的多尺度 latent 生成思路**

### 一、已经对齐的部分

1. 两阶段训练拆分
- Stage 1：动作 VAE 单独训练
- Stage 2：观测 + latent scale autoregressive head
- 这一点和原版 MSP 设计一致

2. Stage 1 -> Stage 2 的 VAE 权重复用
- Stage 2 会把 Stage 1 的 `action_vae/*` remap 到 `msp_action_vae/*`
- 这一点已经打通

3. Stage 2 的 latent target 来源
- 现在已经从 `encode_mean` 改成 `get_sample`
- 和原版 `self.autoencoder.get_sample(action)` 对齐

4. 多尺度 teacher-forcing 训练思路
- finest latent -> 多尺度 target
- 训练时输入前一尺度上采样/重采样结果
- 推理时逐尺度生成 finest latent
- 这条 coarse-to-fine 主线已经接上

5. block-wise scale mask
- 同尺度全可见
- 可看更粗尺度
- 不可看更细尺度
- 训练和推理都已统一

6. 三组 learned positional embeddings
- `encoder_pos_embed`
- `decoder_pos_embed`
- `diffusion_pos_embed`
- 训练全长、推理按 `start:end` 对应 block 切片，这一点已经补上

7. block-local RoPE 语义
- suffix 已经通过 `rope_positions` 覆盖为“每个尺度段内部从 0 开始”
- 这点机制上已经成立

8. 原版的尺度加权 loss 汇总
- 已经按 `scale / max_scale` 汇总到 Stage 2 总 loss

### 二、没有完全移植的部分

1. **最大的差异：Stage 2 不是原版的 `FlowAR + MPScalseFlowhead`**

原版：
- `flow_ar.py` 里不是直接输出 latent 然后做 MSE
- 而是：
  - `forward_mae_encoder`
  - `forward_mae_decoder`
  - 再接 `MPScalseFlowhead`
- 每个尺度 block 的监督来自 `flownet(...)`

当前实现：
- `pi0.py` 里是：
  - 用 pi0.5 的 Gemma action expert 作为多尺度 latent token transformer
  - 直接预测 latent
  - loss 是 latent MSE，再加尺度权重

这意味着：
- **当前 Stage 2 还不是原版 MSP 的 flow/diffusion head**
- 只是保留了多尺度自回归 latent 框架

这是目前最本质的未完全移植点。

2. **原版 FlowAR 的双塔结构没有 1:1 复刻**

原版 `flow_ar.py`：
- 有显式的
  - `z_proj`
  - `z_proj_ln`
  - `forward_mae_encoder`
  - `decoder_embed`
  - `forward_mae_decoder`
- 也就是“encoder stack + decoder stack”两段式结构

当前 `pi0.py`：
- 没有单独复刻这两套 block
- 而是把 MSP suffix token 直接送进现有 Gemma action expert

所以：
- **结构思路相近**
- **但不是原版 FlowAR 模块逐层移植**

3. **原版的 `cls / obs token` 位置语义和当前实现不完全一样**

原版：
- `x_input` 是 `next_scale` 的输入块
- 再前面拼一个 `cls = fusion_obs(context).unsqueeze(1)`
- 所以第一个位置本质上是观察条件 token

当前实现：
- 观察条件来自 pi0.5 的 prefix（图像 token + 文本 token）
- suffix 里全是多尺度 latent token
- 没有把原版那个单独的 `cls` token 作为 suffix 第一位 1:1 塞进去

这点是**有意适配 pi0.5 结构**后的差异，不是 bug，但确实不等于原版。

4. **原版 block 内的调制方式和当前实现不同**

原版 `Block`：
- 每层用 `ada_lin(condition)` 生成
  - `gamma`
  - `scale`
  - `shift`
- 是显式条件调制

当前 `pi0.py`：
- suffix 主要通过：
  - prefix attention
  - learned pos/scale embeddings
  - Gemma 自身 attention
  来感知观测
- `adarms_cond` 目前是全零，不等于原版 `condition=cls`

这也是一个重要差异：
- **原版每层都有显式 condition modulation**
- **当前实现没有把 obs embedding 直接作为每层调制条件送进去**

5. **Stage 1 VAE 不是逐层 1:1 复刻**

虽然 Stage 1 已经比较接近，但仍有实现层面的差别：
- 原版用的是 PyTorch `TransformerEncoder/Decoder`
- 原版 encoder/decoder 的 norm/实现细节、position embedding 工具、层初始化都不是逐项照搬
- 当前 JAX 版是按同等结构重写，不是字节级复刻

不过这部分我认为是：
- **结构等价度较高**
- **足够称为“已移植”**
- 不像 Stage 2 那么有本质差异

### 三、哪些差异是“故意保留”的

1. VLM / prefix 不动
- 这是你一开始就明确要求的
- 所以 observation encoder 不会像原版 MSP 那样走 `FilmResNet + language proj + proprio concat`
- 而是保留 pi0.5 原始 VLM prefix

2. 动作维度与 pi0.5 兼容
- `action_dim=32` 继续给 pi0.5 主干
- `msp_action_dim=14` 单独给 MSP VAE
- 这是为了让 pi0.5 base checkpoint 可加载，不属于原版 MSP 设计

### 四、最终判断

如果问题是：
- **“核心 MSP 思路有没有接进 pi0.5？”**
答案是：**有，而且主线已经打通。**

如果问题是：
- **“是否已经把原版 MSP 完整移植到了 pi0.5？”**
答案是：**没有。**

最准确的说法是：
- 现在的实现是 **MSP-on-pi0.5**
- 不是原版 **MSP FlowAR 模块的完整 JAX 复刻**

### 五、离“更完整移植”还差什么

如果你要继续逼近原版，下一步优先级应该是：

1. **把 Stage 2 的 latent MSE 头换成原版 `MPScalseFlowhead` 风格**
- 这是最大差异

2. **把 obs condition 直接注入每层调制**
- 也就是更接近原版 `ada_lin(condition)` 的作用方式

3. **如果你要极致对齐，再考虑是否复刻独立的 encoder/decoder stack**
- 而不是继续复用 Gemma action expert

当前结论：
- **没有完全移植**
- **已经移植了 MSP 的关键训练范式、多尺度 mask/位置编码/逐尺度推理框架**
- **但 Stage 2 的核心预测头仍然不是原版 MSP 的 flow head**

## 2026-08-28 部署配置适配：`deploy.yml` 切到 `pi05+msp stage2`

判断结果：
- `deploy.py` 本身只是 rollout 逻辑，不依赖“原始 pi0.5 头”还是“MSP 头”
- 实际加载哪个模型，取决于：
  - `model.py`
  - `deploy.yml` 中的 `train_config_name` / checkpoint 信息

所以这一步只需要改 `deploy.yml`，不需要改 `deploy.py`

本轮修改：
- 文件：`deploy.yml`

1. `train_config_name`
- 从：
  - `pi05_base_aloha_full_sim_arx-x5_seed_0`
- 改为：
  - `pi05_msp_stage2_aloha_arx-x5_seed_0`

2. `checkpoint_num`
- 从：
  - `59999`
- 改为：
  - `29999`

3. `result_dir`
- 从：
  - `./results/pi05/`
- 改为：
  - `./results/pi05_msp_stage2/`

说明：
- 部署时不会再去单独加载 stage1/base 预训练权重
- 只会直接加载你保存好的 Stage-2 checkpoint
- 前提是你评估时指向的是正确的 Stage-2 checkpoint 目录

如果实际部署 checkpoint step 不是 `29999`，需要继续改：
- `deploy.yml: checkpoint_num`

本轮修改文件：
- `deploy.yml`
- `task.md`

## 2026-08-28 部署加载链判断：`model.py` 当前不需要为 `pi05+msp stage2` 额外改权重加载逻辑

这一步重新核对了部署时的真实加载链：

1. `model.py`
- 读取 `deploy.yml` 的：
  - `train_config_name`
  - `checkpoint_num`
  - `ckpt_name / model_path / checkpoint_path`

2. `model.py -> create_trained_policy(...)`
- 文件：`openpi/src/openpi/policies/policy_config.py`
- 这里直接做的是：
  - `train_config.model.load(_model.restore_params(checkpoint_dir / "params", ...))`

3. `BaseModelConfig.load(...)`
- 文件：`openpi/src/openpi/models/model.py`
- 这里是：
  - 先按 `train_config.model` 创建完整模型结构
  - 再把 checkpoint 里的 `params` 直接载入这个完整结构

4. 训练保存
- 文件：`openpi/src/openpi/training/checkpoints.py`
- `save_state(...)` 会把当前训练时的完整可推理参数单独保存到：
  - `params`

结论：
- **部署时不会再走训练时的 `weight_loader`**
- **也不会再走 `merge_msp_vae_params(...)`**
- 这些逻辑只发生在 Stage-2 训练初始化阶段

所以：
- 如果你部署的是一个**已经训练并保存完成的 Stage-2 checkpoint**
  - 它的 `params` 里已经包含：
    - pi0.5 base 主干权重
    - stage1 注入后的 `msp_action_vae/*`
    - stage2 学出来的 `msp_*` 头
  - 那么 `model.py` **不需要改**

- 只有在一种情况下才需要改 `model.py`：
  - 你不想加载完整 Stage-2 checkpoint
  - 而是想在部署时临时从：
    - pi0.5 base checkpoint
    - + stage1 VAE checkpoint
    - + 当前随机/部分 stage2 参数
    重新拼装模型
  - 这种做法当前部署链不支持，也不建议这么做

当前判断：
- **现在为了适配 `pi05+msp stage2`，改 `deploy.yml` 就够了**
- **`model.py` 不需要同步改权重加载逻辑**

补充：
- `deploy.py` 也不需要因为权重加载而改
- 它只负责 rollout，不负责 checkpoint 组合

## 2026-08-28 Stage-2 权重加载修复：先用完整模型形状合并 MSP VAE，再覆盖 base checkpoint

报错现象：
- Stage-2 在加载 `msp_vae_weight_path` 时触发：
  - `No remapped MSP VAE parameters matched the stage-2 model.`

根因分析：
- 这个判断是正确的。
- `init_train_state()` 里原本的顺序是：
  1. `_load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())`
  2. 返回 `partial_params`
  3. 把这个 `partial_params` 传给 `merge_msp_vae_params(...)`
- 问题在于：
  - `_load_weights_and_validate(...)` 会去掉所有未从 base checkpoint 实际加载到的 `ShapeDtypeStruct`
  - 因为原始 pi0.5 checkpoint 里没有 `msp_action_vae/*`
  - 所以返回的 `partial_params` 里也没有 `msp_action_vae/*` 这些 key
- 结果：
  - `merge_msp_vae_params(partial_params, ...)` 在 `flat_ref` 中找不到 `msp_action_vae/*`
  - `matched` 为空
  - 于是报错

修复策略：
- 不能用“已经裁掉 MSP key 的 base partial params”作为 stage1 VAE merge 的参考树
- 必须先用**完整模型参数形状**做参考，把 stage1 VAE 合并进去
- 然后再把 base checkpoint 的真实已加载参数覆盖上去

本轮修改：
1. `train.py` 和 `train_tb.py` 调整加载顺序
- 文件：
  - `openpi/scripts/train.py`
  - `openpi/scripts/train_tb.py`
- 新顺序：
  1. `full_shape = train_state_shape.params.to_pure_dict()`
  2. `base_params = _load_weights_and_validate(config.weight_loader, full_shape)`
  3. `stage1_params = merge_msp_vae_params(full_shape, config.msp_vae_weight_path)`
  4. `partial_params = merge(base_params over stage1_params)`

2. 新增 `_merge_partial_params(...)`
- 文件：
  - `openpi/scripts/train.py`
  - `openpi/scripts/train_tb.py`
- 用途：
  - 把 `base_params` 覆盖到 `stage1_params` 上
  - 同时去掉残留的 `jax.ShapeDtypeStruct`

当前效果：
- Stage-1 VAE merge 时，参考树里已经包含 `msp_action_vae/*`
- 所以 `merge_msp_vae_params(...)` 可以正确找到 remapped key
- 然后再用 base checkpoint 覆盖公共主干权重
- 最终得到的是：
  - pi0.5 base 权重
  - + stage1 MSP VAE 权重
  - + 其余 `msp_*` 新增头随机初始化

本轮验证：
- `openpi/.venv/bin/python -m py_compile openpi/scripts/train.py openpi/scripts/train_tb.py`
- `git diff --check -- openpi/scripts/train.py openpi/scripts/train_tb.py`

本轮修改文件：
- `openpi/scripts/train.py`
- `openpi/scripts/train_tb.py`
- `task.md`

## 2026-08-28 JAX tracer 修复：`build_scale_segment_bounds` 改成静态 numpy 边界

报错现象：
- Stage-2 训练在 `jit(train_step)` 内触发：
  - `jax.errors.ConcretizationTypeError`
- 具体点位是：
  - `start = int(scale_starts[block_idx])`

根因：
- 这个判断也是正确的。
- `build_scale_segment_bounds(scales)` 之前返回的是 `jnp.ndarray`
- 在 `jax.jit` 里，这些值会变成 tracer
- tracer 不能被 Python 的 `int()` 转成具体值

而这里的 `scales` 本质上是静态配置：
- 它来自 `tuple[int, ...]`
- 不是运行时动态张量

所以这组 segment bounds 没必要用 `jnp`

本轮修改：
- 文件：`openpi/src/openpi/models/msp_scale_head.py`
- 将：
  - `build_scale_segment_bounds(scales) -> tuple[jnp.ndarray, jnp.ndarray]`
- 改为：
  - `build_scale_segment_bounds(scales) -> tuple[np.ndarray, np.ndarray]`

实现方式：
- `ends = np.cumsum(scales, dtype=np.int32)`
- `starts = ends - np.asarray(scales, dtype=np.int32)`

效果：
- `starts/end` 现在是静态 numpy 数组
- 在 `jit` 内部不会变成 JAX tracer
- 所以可以安全地用于：
  - `int(starts[idx])`
  - `int(ends[idx])`

本轮验证：
- `openpi/.venv/bin/python -m py_compile openpi/src/openpi/models/msp_scale_head.py openpi/src/openpi/models/pi0.py openpi/scripts/train.py openpi/scripts/train_tb.py`
- `git diff --check -- openpi/src/openpi/models/msp_scale_head.py`

本轮修改文件：
- `openpi/src/openpi/models/msp_scale_head.py`
- `task.md`

## 2026-08-28 Stage-2 latent target 对齐原版 MSP：从 `encode_mean` 改为 `get_sample`

问题背景：
- 之前 Stage-2 在 `pi0.py` 里使用的是：
  - `msp_action_vae(..., method="encode_mean")`
- 但原版 `MSP/algos/MSP.py` 在 `compute_flowscale_loss()` 里用的是：
  - `self.autoencoder.get_sample(action)`

这两者的区别：
- `encode_mean`
  - 取 posterior 的均值
  - target 更稳定、更确定
- `get_sample`
  - 从 posterior 中采样
  - 更忠实原版 MSP 的训练方式

本轮修改：
- 文件：`openpi/src/openpi/models/pi0.py`
- Stage-2 的 `_compute_msp_loss_with_info(...)` 改为：
  1. 从训练 RNG 中切出：
     - `preprocess_rng`
     - `latent_rng`
  2. 用：
     - `msp_action_vae(..., latent_rng, method="get_sample", train=False)`
     生成 `finest_latent`

这意味着现在 Stage-2 的 latent supervision：
- 不再是固定的 posterior mean
- 而是和原版 MSP 一样，使用 VAE posterior sample

当前效果：
- Stage-2 更忠实原版 MSP
- latent target 会带有 VAE posterior sampling 的随机性
- 训练目标仍然是多尺度 latent 预测，只是 target 来源从 mean 改成了 sample

本轮验证：
- `openpi/.venv/bin/python -m py_compile openpi/src/openpi/models/pi0.py`
- `git diff --check -- openpi/src/openpi/models/pi0.py`

本轮修改文件：
- `openpi/src/openpi/models/pi0.py`
- `task.md`

## 2026-08-27 Stage-2 loss 对齐：接入原版 MSP 的尺度加权机制

你指出的这一点是对的。原版 MSP 在 `MSP/algos/flow/flow_ar.py` 里不是把所有尺度 token loss 直接平均，而是：

```python
max_scale = self.scale[-1]
loss = []
start = 0
for i in self.scale:
    l, l_dict = self.flownet(gt_latents[:, start:start + i], z[:, start:start + i, ...])
    start += i
    loss.append(l * i / max_scale)
flow_loss = sum(loss)
```

也就是：
- 先按尺度分段求每个 block 的 loss
- 再乘 `scale / max_scale`
- 最后把所有尺度加起来

本轮修改：
1. Stage-2 的训练总 loss 改成按尺度加权汇总
- 文件：`openpi/src/openpi/models/pi0.py`
- 在 `_compute_msp_loss_with_info(...)` 里：
  - 先得到每个尺度段的 `block_loss`
  - 再做：
    - `weighted_block_loss = block_loss * (scale / max_scale)`
  - 最后：
    - `weighted_total = sum(weighted_block_loss over scales)`
- 返回给训练主循环的 loss 改为：
  - `weighted_total[:, None]`

2. 保留 raw per-scale 监控，同时补 weighted 指标
- 文件：`openpi/src/openpi/models/pi0.py`
- 现在会同时输出：
  - `msp_loss_scale_<scale>`：该尺度原始平均 loss
  - `msp_loss_scale_<scale>_weighted`：该尺度乘权重后的 loss

3. 总指标语义更新
- `msp_loss_total`
  - 现在表示 **按原版 MSP 权重聚合后的总 loss**
- `msp_loss_finest`
  - 仍表示最细尺度的 raw loss
- `msp_loss_finest_weighted`
  - 表示最细尺度乘权重后的 loss

当前效果：
- Stage-2 的反向传播目标现在和原版 MSP 的尺度加权思路一致
- 粗尺度的贡献会按 `scale / max_scale` 参与总 loss
- 同时你在 TensorBoard / wandb 里还能区分：
  - 原始每尺度 loss
  - 加权后的每尺度 loss

当前会看到的 Stage-2 指标：
- `msp_loss_total`
- `msp_loss_finest`
- `msp_loss_finest_weighted`
- `msp_loss_scale_1`
- `msp_loss_scale_1_weighted`
- `msp_loss_scale_2`
- `msp_loss_scale_2_weighted`
- `msp_loss_scale_4`
- `msp_loss_scale_4_weighted`
- `msp_loss_scale_8`
- `msp_loss_scale_8_weighted`

本轮验证：
- `openpi/.venv/bin/python -m py_compile openpi/src/openpi/models/pi0.py`
- `git diff --check -- openpi/src/openpi/models/pi0.py`

本轮修改文件：
- `openpi/src/openpi/models/pi0.py`
- `task.md`

### 1. 新增三组 MSP learned positional embeddings

在 `Pi0.__init__()` 的 MSP 分支里新增了三个参数：
- `self.msp_encoder_pos_embed`
- `self.msp_decoder_pos_embed`
- `self.msp_diffusion_pos_embed`

实现方式：
- shape 都是 `(1, sum(msp_scales), action_expert_width)`
- 用 `std=0.02` 的正态分布初始化
- 只在 `use_msp_action_head=True` 时存在，不影响普通 pi0.5 路径

这一步对应 MSP `flow_ar.py` 里的：
- `encoder_pos_embed_learned`
- `decoder_pos_embed_learned`
- `diffusion_pos_embed_learned`

### 2. 新增统一的 positional slice helper

新增：
- `_msp_pos_slice(pos_embed, start=None, end=None, dtype=...)`

作用：
- 训练态可直接取全长
- 推理态按当前尺度段取 `[:, start:end, :]`
- 保持 dtype 和当前 token / hidden state 一致

### 3. 训练态：全长 positional embedding 接入

当前 `_compute_msp_loss()` 的 MSP 路径里：
- `suffix_tokens` 构造时接入：
  - `encoder_pos_embed`
  - `decoder_pos_embed`
- transformer 输出送到 latent projection 之前，再加：
  - `diffusion_pos_embed`

当前对应关系是：
- `encoder_pos + decoder_pos` 放在 suffix token 输入侧
- `diffusion_pos` 放在 suffix hidden output 侧

这不是 `flow_ar.py` 的双栈 encoder/decoder 完整同构，因为我们仍然复用的是单个 Gemma action expert。
但在现有结构约束下，三组 learned positional embedding 已经各自接到了最接近原语义的位置。

### 4. 推理态：按尺度段 `start:end` 切片

在 `_sample_msp_actions()` 里：
- 先用 `build_scale_segment_bounds(self.msp_scales)` 得到每个尺度在全长多尺度序列中的：
  - `start`
  - `end`
- 每个尺度增量生成时：
  - `encoder_pos_embed[:, start:end]`
  - `decoder_pos_embed[:, start:end]`
  - `diffusion_pos_embed[:, start:end]`

这样就对齐了 MSP 原版里：
- 推理时当前位置编码不是取全长
- 而是只取当前尺度局部段的 learned positional slice

### 5. 当前 MSP 路径的状态

相较上一轮，现在已经具备：
- block-local RoPE
- block-wise suffix mask
- prefix 只跑一次
- 按尺度段增量 cache 推理
- learned positional embeddings 的训练全长 / 推理切片双路径

仍然和 `flow_ar.py` 不完全同构的地方：
- 还不是独立的 `forward_mae_encoder + forward_mae_decoder` 双栈结构
- 当前是把这三组 learned positional embeddings 映射到现有单栈 Gemma action expert 的输入/输出侧

所以当前版本可以描述为：
- **已经补上 MSP 原版 positional embedding 的核心训练/推理切片机制**
- **但整体结构仍然是“MSP 位置机制 + Gemma action expert” 的适配版**

### 6. 本轮验证

已通过：
- `openpi/.venv/bin/python -m py_compile openpi/src/openpi/models/pi0.py openpi/src/openpi/models/gemma.py openpi/src/openpi/models/msp_scale_head.py`
- `git diff --check -- openpi/src/openpi/models/pi0.py openpi/src/openpi/models/gemma.py openpi/src/openpi/models/msp_scale_head.py task.md`

未做：
- 真实 MSP stage-2 前向验证
- 真实 pi0.5 checkpoint 加载回归
- 真实训练验证

## 2026-08-27 本轮实现：Stage-1 VAE 暴露 `recon_loss` 和加权 `kl_loss` 到训练日志

需求：
- `openpi/src/openpi/models/msp_vae.py` 里的 `MspActionVAE.compute_loss()` 目前只返回总 loss
- 现在还需要把两个关键指标暴露到 `openpi/scripts/train.py` 做可视化：
  - `recon_loss`
  - `self.kl_weight * kl_loss[:, None]`
- 这两个指标不参与梯度，只用于日志

本轮修改文件：
- `openpi/src/openpi/models/model.py`
- `openpi/src/openpi/models/msp_vae.py`
- `openpi/scripts/train.py`
- `task.md`

### 1. 在 `BaseModel` 增加兼容接口

新增：
- `BaseModel.compute_loss_with_info(...)`

默认行为：
- 返回 `(compute_loss(...), {})`

这样普通模型不用改：
- `Pi0`
- `Pi0Fast`
- 以及其他任何只关心总 loss 的模型

### 2. `MspActionVAE` 覆盖 `compute_loss_with_info`

实现：
- 保留原来的 `compute_loss()`，继续只返回总 loss，避免破坏现有接口
- 新增 `compute_loss_with_info()`：
  - 复用同一套前向
  - 计算：
    - `recon_loss`
    - `weighted_kl_loss = self.kl_weight * kl_loss[:, None]`
    - `total_loss = recon_loss + weighted_kl_loss`
  - 返回：
    - `total_loss`
    - `{"recon_loss": mean(recon_loss), "kl_loss": mean(weighted_kl_loss)}`

这里日志里记录的 `kl_loss` 是：
- **已经乘过 `kl_weight` 的 KL 项**
- 对应你提的 `self.kl_weight * kl_loss[:, None]`

### 3. `train.py` 改为通过 `has_aux=True` 接收额外指标

训练步现在做的是：
- `loss_fn()` 返回 `(mean(chunked_loss), metric_info)`
- `nnx.value_and_grad(..., has_aux=True)` 同时拿到：
  - 总 loss
  - 梯度
  - 附加 metrics
- 最终 `info` 里除了原来的：
  - `loss`
  - `grad_norm`
  - `param_norm`
- 还会在 `MspActionVAE` 路径下额外包含：
  - `recon_loss`
  - `kl_loss`

这样 wandb / 日志侧不需要额外改格式，按现有 `wandb.log(reduced_info, step=step)` 就会自动打出去。

### 4. 本轮验证

已通过：
- `openpi/.venv/bin/python -m py_compile openpi/src/openpi/models/model.py openpi/src/openpi/models/msp_vae.py openpi/scripts/train.py`
- `git diff --check -- openpi/src/openpi/models/model.py openpi/src/openpi/models/msp_vae.py openpi/scripts/train.py task.md`

## 2026-08-27 本轮实现：`train_tb.py` 去掉 wandb，改用 TensorBoard

需求：
- `openpi/scripts/train_tb.py` 是从 `train.py` 复制出来的训练脚本
- 现在要求删掉 `wandb`，改用 TensorBoard

本轮修改文件：
- `openpi/scripts/train_tb.py`
- `task.md`

实现内容：
- 删除 `import wandb`
- 改为：
  - `from torch.utils.tensorboard import SummaryWriter`
- 删除 `init_wandb(...)`
- 新增 `init_tensorboard(config)`：
  - 日志目录使用 `config.checkpoint_dir / "tensorboard"`
  - 启动时写一份 `config` 文本到 TensorBoard

日志改动：
- 标量日志：
  - 原来的 `wandb.log(reduced_info, step=step)`
  - 改成 `tb_writer.add_scalar(key, value, step)`
- 首批图像可视化：
  - 原来的 `wandb.Image(...)` + `wandb.log(...)`
  - 改成 `tb_writer.add_image(..., dataformats="HWC")`

恢复训练逻辑：
- 保留了原本的 `resuming` 路径
- `initialize_checkpoint_dir(...)` 返回 `resuming=True` 时，仍然会调用：
  - `_checkpoints.restore_state(...)`

收尾：
- 训练结束时调用 `tb_writer.close()`

本轮验证：
- `openpi/.venv/bin/python -m py_compile openpi/scripts/train_tb.py`
- `git diff --check -- openpi/scripts/train_tb.py task.md`

## 2026-08-27 本轮修正：`ActionOnlyLeRobotDataset` 的 `select_columns` 走 `hf_dataset`

需求：
- 用户指出 `openpi/src/openpi/training/data_loader.py` 里的：
  - `self._dataset = dataset.select_columns(column_names)`
- 更可能应该走：
  - `dataset.hf_dataset.select_columns(...)`

本轮修改文件：
- `openpi/src/openpi/training/data_loader.py`
- `task.md`

实现：
- 改成兼容写法：
  - 如果 `dataset` 有 `hf_dataset` 属性，则优先用：
    - `dataset.hf_dataset.select_columns(column_names)`
  - 否则 fallback 到旧写法：
    - `dataset.select_columns(column_names)`

这样做的原因：
- `LeRobotDataset` 往往是外层包装对象
- `select_columns` 更常见于其内部的 HuggingFace dataset
- 但为了兼容不同版本的 LeRobot，这里不把代码写死成单一路径

本轮验证：
- `openpi/.venv/bin/python -m py_compile openpi/src/openpi/training/data_loader.py`
- `git diff --check -- openpi/src/openpi/training/data_loader.py task.md`

## 2026-08-27 新理解：Stage1 14D / Stage2 32D 的“能训练”不等于维度契约正确

核对结果：
- `pi05_base_aloha_full_sim_arx-x5_seed_0` 的 `model=Pi0Config(pi05=True)`，默认 `action_dim=32`
- 训练时数据侧会经过 `PadStatesAndActions(model_config.action_dim)`
- 所以原始 14 维动作会被 pad 到 32 维，再送进 pi0.5

结论：
- “不显式传 `action_dim=14` 也能训练”是因为 **pi0.5 一直按 32 维建模，数据被 pad 到 32 维**
- 这不代表模型自动推断了 14 维

对 MSP 的影响：
- 如果 Stage2 也用 `action_dim=32`，那 pi0.5 base checkpoint 的动作相关层 shape 是兼容的
- 但如果 Stage1 VAE 是按 14 维训练出来的，而 Stage2 的 `msp_action_vae` 是按 32 维实例化的，那么两阶段的 VAE 输入输出 action 维度其实不一致

判断：
- 这种情况下“能启动训练”并不自动说明是合理的
- 更可能只是当前某条加载/校验路径没有把这个 mismatch 在初始化时拦下来
- 从模型语义上看，Stage1 和 Stage2 的 `msp_action_vae` action space 最好保持一致

## 2026-08-27 README 补充：第二阶段训练启动方式

需求：
- 把 MSP 第二阶段怎么启动训练写到 `README.md`

本轮修改文件：
- `README.md`
- `task.md`

补充内容：
- 在 Training 段落后新增 `MSP Stage 2` 小节
- 明确写了第二阶段训练前需要准备：
  - Stage-1 的 VAE checkpoint
  - Stage-2 用的 LeRobot 数据集
- 给出了最小启动命令
- 补了两个常见变体：
  - 显式指定 `OPENPI_LEROBOT_REPO_ID`
  - 显式指定 `OPENPI_TRAIN_CONFIG_NAME`
- 补了多卡启动示例
- 说明了第二阶段的关键行为：
  - 不是 action-only
  - 会加载 pi0.5 预训练权重
  - 会额外合并 Stage-1 的 `msp_action_vae` 权重

## 2026-08-27 Git 同步记录

当前用户要求：
- 把现在这版 `Pi_05` 代码同步到 GitHub

本次同步范围：
- 提交当前仓库里已修改的 `Pi_05` 代码
- 包括新建的 `openpi/scripts/train_tb.py`
- 不包含未跟踪的 `MSP/` 参考目录

本次同步结果：
- 上层仓库本地提交：
  - `fa0c909 Update Pi_05 training and logging`
- 由于远端 `msp_vla/main` 是 `Pi_05 subtree` 历史，普通 push 被拒绝为 non-fast-forward
- 已重新生成 subtree 分支：
  - `pi05_sync_20260827`
  - split head: `d87fff8`
- 已同步到 GitHub：
  - `msp_vla/main -> d87fff8`

## 2026-08-27 本轮优化：`train_multi_H20.sh` 增加 stage 开关

需求：
- 新增的 `train_multi_H20.sh` 用于服务器训练
- 需要增加一个开关判断 Stage 1 / Stage 2
- Stage 1 不需要加载 pi0.5 基础权重，也不需要加载 MSP VAE 权重
- 两个阶段的学习率和 decay 策略不同

本轮修改文件：
- `train_multi_H20.sh`
- `task.md`

实现内容：

1. 新增阶段选择：
- 使用环境变量：
  - `OPENPI_TRAIN_STAGE`
- 支持：
  - `stage1`
  - `stage2`
- 如果没有显式设置，就根据 `train_config_name` 自动推断

2. Stage 1 默认超参：
- `decay_steps=num_train_steps`
- `peak_lr=3e-4`
- `decay_lr=3e-5`
- `load_base_weights=false`
- `load_msp_vae_weights=false`

3. Stage 2 默认超参：
- `decay_steps=num_train_steps / 2`
- `peak_lr=5e-5`
- `decay_lr=1e-5`
- `load_base_weights=true`
- `load_msp_vae_weights=true`

4. 权重加载行为：
- Stage 1：
  - 不再传 `--weight-loader.params-path`
  - 不再传 `--msp-vae-weight-path`
  - 不再检查 `PARAMS_PATH`
- Stage 2：
  - 继续传 `--weight-loader.params-path="${PARAMS_PATH}"`
  - 如果给了 `MSP_VAE_WEIGHT_PATH`，再传 `--msp-vae-weight-path="${MSP_VAE_WEIGHT_PATH}"`

5. 日志和路径校验：
- 启动时打印 `TRAIN_STAGE`
- 打印当前生效的：
  - `decay_steps`
  - `peak_lr`
  - `decay_lr`
- Stage 1 会明确打印：
  - `PARAMS_PATH: skipped for stage1`
  - `MSP_VAE_WEIGHT_PATH: skipped for stage1`

6. 示例命令更新：
- Stage 1 示例改成显式：
  - `OPENPI_TRAIN_STAGE=stage1 ...`
- Stage 2 示例改成显式：
  - `OPENPI_TRAIN_STAGE=stage2 ...`

7. Stage 2 的 MSP VAE 默认路径：
- 脚本现在直接使用固定默认值：
  - `MSP_VAE_WEIGHT_PATH="${MSP_VAE_WEIGHT_PATH:-/mnt/vepfs/vbot/lzx/RoboDojo/XPolicyLab/policy/Pi_05/checkpoints/stack_bowls_stage_1-0/29999/params}"`
- 也就是说 Stage 2 不显式传时，会默认读取：
  - `/mnt/vepfs/vbot/lzx/RoboDojo/XPolicyLab/policy/Pi_05/checkpoints/stack_bowls_stage_1-0/29999/params`
- 如果默认路径不对，直接覆盖：
  - `MSP_VAE_WEIGHT_PATH=/your/stage1/params`

8. 启动示例位置：
- 已把 `train_multi_H20.sh` 的 Stage1 / Stage2 启动示例整理到脚本最底部
- 现在脚本尾部保留三种用法：
  - Stage1 预训练
  - Stage2 使用默认 `MSP_VAE_WEIGHT_PATH`
  - Stage2 手动覆盖 `MSP_VAE_WEIGHT_PATH`

本轮验证：
- `bash -n train_multi_H20.sh`
- `git diff --check -- train_multi_H20.sh task.md`

## 2026-08-27 Git 操作记录

当前用户要求：
- 先把现在这版代码推到 `220XUAN/MSP_VLA.git`

本次提交范围判断：
- 只提交 `policy/Pi_05` 下这次 MSP 接入相关修改
- 不把未跟踪的 `policy/Pi_05/MSP/` 参考目录一起提交，避免把本地参考实现混入本次适配提交

执行结果：
- 本地提交：
  - commit: `c9f81f9`
  - message: `Add MSP two-stage integration for Pi_05`
- 新增远端：
  - `msp_vla -> https://github.com/220XUAN/MSP_VLA.git`
- 已推送：
  - `main -> msp_vla/main`

修正说明：
- 上一次 push 的 commit 文件范围虽然只涉及 `policy/Pi_05`，但推送对象是上层 `XPolicyLab` 仓库的 `main`。
- 这会把整个上层仓库历史带到 `220XUAN/MSP_VLA.git`，不符合“只提交当前 `Pi_05`”的要求。
- 下一步需要把远端 `msp_vla/main` 改写成仅包含 `policy/Pi_05` 子目录内容的分支。

修正结果：
- 已通过 `git subtree split --prefix=policy/Pi_05 -b pi05_only` 生成只包含 `Pi_05` 内容的分支
- split 分支头：
  - `0627ca3 Add MSP two-stage integration for Pi_05`
- 已强制更新远端：
  - `pi05_only -> msp_vla/main`
- 现在 `220XUAN/MSP_VLA.git` 的 `main` 应该只包含 `Pi_05` 目录内容，不再包含整个上层 `XPolicyLab`

## 2026-08-27 Stage-2 权重加载修正

当前问题：
- Stage-2 模型在 pi0.5 基础上新增了 `msp_*` 参数。
- 直接加载原始 pi0.5 checkpoint 时，这些新增参数在源 checkpoint 中不存在，会触发结构不匹配。
- 同时需要在加载 Stage-1 VAE 权重后，明确输出成功/失败状态，方便服务器日志排查。

本轮修改：
1. `CheckpointWeightLoader` 支持可配置的 `missing_regex`
- 文件：`openpi/src/openpi/training/weight_loaders.py`
- 默认仍是：
  - `".*lora.*"`
- 现在可以在特定 config 里改成：
  - `".*lora.*|.*msp.*"`
- 这样加载原始 pi0.5 权重时，会自动跳过所有 `msp_*` 新增参数，并保留目标模型里的随机初始化值。

2. Stage-2 config 显式跳过 MSP 新增头
- 文件：`openpi/src/openpi/training/config.py`
- `pi05_msp_stage2_aloha_arx-x5_seed_0` 现在使用：
  - `CheckpointWeightLoader(..., missing_regex=".*lora.*|.*msp.*")`
- 含义：
  - 原始 pi0.5 checkpoint 只加载公共参数
  - 所有 MSP 相关新增参数不从 pi0.5 checkpoint 读取，改为随机初始化

3. Stage-1 VAE 权重合并增加校验和日志
- 文件：`openpi/src/openpi/training/weight_loaders.py`
- `merge_msp_vae_params()` 现在会：
  - 检查 checkpoint 中是否存在 `action_vae/...` 参数
  - 重映射到 Stage-2 模型中的 `msp_action_vae/...`
  - 校验 shape 是否一致
  - 如果没有匹配项或 shape 不一致，直接报错
  - 如果成功，输出成功日志，包含匹配到的参数数量

4. 训练入口增加显式状态日志
- 文件：
  - `openpi/scripts/train.py`
  - `openpi/scripts/train_tb.py`
- 现在初始化时会明确打印：
  - 开始加载 base model weights
  - base model weight load succeeded
  - 开始加载 MSP stage-1 weights
  - MSP stage-1 weight load succeeded

当前效果：
- 加载原始 pi0.5 checkpoint 时，不再因为 `msp_*` 新增参数缺失而报结构错
- Stage-1 VAE checkpoint 如果真正合并成功，日志里会有明确成功信息
- 如果 Stage-1 checkpoint 路径对了但内部结构/shape 不对，会直接报清楚，而不是静默失败

本轮修改文件：
- `openpi/src/openpi/training/weight_loaders.py`
- `openpi/src/openpi/training/config.py`
- `openpi/scripts/train.py`
- `openpi/scripts/train_tb.py`
- `task.md`

## 2026-08-27 维度解耦：Stage-2 保留 pi0.5 的 32 维接口，同时让 MSP VAE 只建模真实 14 维动作

新的设计决定：
- `Pi0Config.action_dim` 继续保留给 pi0.5 主干，用于兼容原始 base checkpoint
- 新增 `Pi0Config.msp_action_dim`
- Stage-2 的 MSP 分支不再默认复用 `action_dim`
- 对 Aloha 场景：
  - `action_dim=32`
  - `msp_action_dim=14`

原因：
- 原始 pi0.5 base checkpoint 的动作相关投影层是按 32 维训练的
- 但 MSP 的 Stage-1 / Stage-2 VAE 契约应该跟真实动作空间一致，即 14 维
- 所以需要把“pi0.5 兼容维度”和“MSP VAE 真实动作维度”拆开，不能继续混用

本轮实现：
1. `Pi0Config` 新增 `msp_action_dim`
- 文件：`openpi/src/openpi/models/pi0_config.py`
- `use_msp_action_head=True` 时：
  - 如果没传 `msp_action_dim`，默认回退到 `action_dim`
  - 增加校验：`msp_action_dim <= action_dim`
- `make_msp_vae_config()` 改为用 `msp_action_dim` 构造 Stage-2 里的 `msp_action_vae`

2. 数据 transforms 拆分 state/action padding
- 文件：
  - `openpi/src/openpi/transforms.py`
  - `openpi/src/openpi/training/config.py`
- 新增 `PadToDims(state_dim=..., action_dim=...)`
- 对 `Pi0Config(use_msp_action_head=True)`：
  - `state` 继续 pad 到 `action_dim=32`
  - `actions` 只 pad 到 `msp_action_dim=14`
- 这样 Stage-2 训练时送进 MSP VAE 的动作就是 14 维，不再被 pad 到 32

3. Stage-2 模型内部改为“MSP 用 14 维，外部接口仍保留 32 维”
- 文件：`openpi/src/openpi/models/pi0.py`
- `msp_action_vae.lazy_init(...)` 改成按 `msp_action_dim`
- `compute_msp_loss()` 里先裁剪：
  - `actions[..., :msp_action_dim]`
  - 再送入 `msp_action_vae.encode_mean(...)`
- 推理时：
  - `msp_action_vae.decode(...)` 先得到 14 维动作
  - 再 pad 回 `action_dim=32`

这一步的含义：
- MSP latent 建模只看真实动作维度
- 外部 `BaseModel.action_dim`、已有测试假设、以及部分下游接口仍能维持 32 维不变
- Aloha 输出端本来就只取前 14 维，所以推理链路仍兼容

4. Stage-2 config 显式使用 `msp_action_dim=14`
- 文件：`openpi/src/openpi/training/config.py`
- `pi05_msp_stage2_aloha_arx-x5_seed_0` 现在不再改 `action_dim`
- 改为：
  - `action_dim` 维持 pi0.5 默认值 32
  - `msp_action_dim=14`

当前效果：
- Stage-1 的 14D MSP VAE 与 Stage-2 的 `msp_action_vae` 维度契约终于一致
- 原始 pi0.5 checkpoint 仍可按 32D 主干结构加载
- Stage-2 的 MSP latent 监督不再吃 32D padded action，而是吃真实 14D action

本轮验证：
- `openpi/.venv/bin/python -m py_compile openpi/src/openpi/transforms.py openpi/src/openpi/models/pi0_config.py openpi/src/openpi/models/pi0.py openpi/src/openpi/training/config.py`
- `git diff --check -- openpi/src/openpi/transforms.py openpi/src/openpi/models/pi0_config.py openpi/src/openpi/models/pi0.py openpi/src/openpi/training/config.py`

本轮修改文件：
- `openpi/src/openpi/transforms.py`
- `openpi/src/openpi/models/pi0_config.py`
- `openpi/src/openpi/models/pi0.py`
- `openpi/src/openpi/training/config.py`
- `task.md`

## 2026-08-27 Stage-2 mask 对齐：先把 block-wise scale mask 收紧到 MSP 原版语义

这一步只处理第一优先级问题：attention mask。

原版 MSP 的关键语义：
- 同一尺度内 token 彼此全可见
- 当前尺度可以看到所有更粗尺度
- 当前尺度不能看到任何更细尺度
- 这不是 token-level causal mask，而是 block-wise scale mask

本轮修改：
1. 统一出显式的 MSP block-wise 可见性矩阵
- 文件：`openpi/src/openpi/models/msp_scale_head.py`
- 新增：
  - `build_blockwise_visibility(scales)`
- 该函数直接生成和 MSP 原版一致的 suffix 可见性矩阵

2. 训练阶段 suffix mask 改成复用这套显式矩阵
- 文件：`openpi/src/openpi/models/msp_scale_head.py`
- `build_suffix_block_attention_mask(...)` 现在不再现场拼接逻辑，而是直接基于：
  - `build_blockwise_visibility(scales)`
- 这样训练期 mask 语义被固定下来，更容易核对和回归

3. 推理阶段当前尺度的 mask 也改成从同一套矩阵切片
- 文件：
  - `openpi/src/openpi/models/msp_scale_head.py`
  - `openpi/src/openpi/models/pi0.py`
- 新增：
  - `build_current_block_attention_mask(scales, block_index, ...)`
- 推理时当前尺度 block 的 mask 不再手写 `all-ones`，而是直接取：
  - 全量训练 mask 中当前 block 对应的行
- 这样训练和推理使用的是同一套 block-wise 语义

4. 保留 `build_scale_ar_mask(...)`，但文档改清楚
- 文件：`openpi/src/openpi/models/msp_scale_head.py`
- 它现在被明确标注为：
  - big_vision 风格 block-wise AR 边界辅助
- 不是普通 token-level causal mask

5. 新增 mask 回归测试
- 文件：`openpi/src/openpi/models/msp_scale_head_test.py`
- 测了两件事：
  - `(1, 2, 4, 8)` 是否生成与 MSP 原版一致的 15x15 block-wise 可见性矩阵
  - 推理阶段 `build_current_block_attention_mask(...)` 是否严格等于训练期全量 mask 对应 block 的行切片

当前结果：
- Stage-2 的训练 suffix mask 和推理 suffix mask 现在都收敛到了同一套 MSP block-wise 语义
- 这一步没有动位置编码、RoPE 或损失，只收紧 mask

本轮验证：
- `openpi/.venv/bin/python -m py_compile openpi/src/openpi/models/msp_scale_head.py openpi/src/openpi/models/pi0.py openpi/src/openpi/models/msp_scale_head_test.py`
- `git diff --check -- openpi/src/openpi/models/msp_scale_head.py openpi/src/openpi/models/pi0.py openpi/src/openpi/models/msp_scale_head_test.py`
- 单测未执行：
  - `openpi/.venv/bin/python -m pytest openpi/src/openpi/models/msp_scale_head_test.py`
  - 失败原因：当前 `.venv` 里没有安装 `pytest`

本轮修改文件：
- `openpi/src/openpi/models/msp_scale_head.py`
- `openpi/src/openpi/models/pi0.py`
- `openpi/src/openpi/models/msp_scale_head_test.py`
- `task.md`

## 2026-08-31 多任务 stage2 推理统一倒臂姿态的诊断

现象：
- 多任务 stage2 训练后，推理时无论哪个任务，机械臂都会以近似相同姿态倒下

当前代码下的高概率原因，按优先级排序：

1. 部署侧如果没有提供 `instruction`，多任务条件会直接退化
- 文件：`model.py`
- `encode_obs()` 当前只从 `observation.get("instruction")` 取 prompt
- 如果环境侧没有这个字段，部署时传给 openpi 的 `prompt` 就是 `None`
- 而训练配置 `pi05_msp_stage2_aloha_arx-x5_seed_0` 是 `prompt_from_task=True`
- 这意味着训练时模型依赖任务文本区分多任务，但推理时如果没有 `instruction`，模型就只能靠视觉/状态，容易退化到统一动作

2. 当前 stage2 目标仍然是 latent MSE，不是 MSP 原版的完整 FlowAR 目标
- 文件：`openpi/src/openpi/models/pi0.py`
- 当前 `_compute_msp_loss_with_info()` 监督的是：
  - `pred_latents` 对 `target_latents` 的平方误差
- 即：
  - 第 294-316 行这一段本质是多尺度 latent 回归
- 这和 MSP 原版的 `FlowAR + flow loss` 仍有差别
- 在多任务下，这种目标更容易学成“条件无关的平均 latent”，再经 stage1 decoder 解码成一个固定坏姿态

3. 训练-推理存在 teacher forcing / exposure bias
- 训练：
  - `build_teacher_forced_inputs()` 用 GT finer/coarser latent 构造下一尺度输入
- 推理：
  - `build_current_scale_inputs()` 用模型上一尺度预测结果构造下一尺度输入
- 所以只要第一二个尺度预测偏掉，误差就会逐尺度放大，最后 finest latent 会塌到相似区域
- 这类问题在“所有任务都倒向同一姿态”时非常典型

4. 当前 MSP 头里 `adarms_cond` 是 0，不是原 pi0.5 flow head 那套时间条件
- 文件：`openpi/src/openpi/models/pi0.py`
- `_embed_msp_suffix()` 里：
  - `adarms_cond = jnp.zeros(...)`
- 这本身不一定是 bug，因为 stage2 已经不是 flow matching
- 但它说明现在动作头条件化能力比原 pi0.5 flow head 更弱，更多依赖 prefix 的视觉/语言 token

5. 如果多任务环境的初始观测相近，而文本条件又缺失，统一倒臂会更明显
- 这时 prefix 基本相同
- 自回归 MSP 头就很容易输出近似同一个 latent 序列

建议先做的排查，不先改结构：

1. 先确认部署时每个任务的 `obs["instruction"]` 是否真的非空
- 这是第一优先级
- 如果这里是空，当前多任务失败基本可以解释通

2. 打印推理时不同任务的：
- tokenized prompt 是否不同
- `generated_blocks[-1]` 的均值/方差是否几乎一样
- `decoded_actions[0, 0]` 是否在任务间几乎一致

3. 对比单任务 stage2 是否也会倒臂
- 如果单任务正常，多任务异常，优先怀疑任务条件没送进去或条件利用太弱
- 如果单任务也异常，优先怀疑 stage2 latent 头本身塌缩

当前判断：
- 不是单一“代码报错型”问题
- 更像“任务条件缺失 + latent 回归头塌缩 + teacher forcing 到自回归的分布偏移”叠加出来的行为

## 2026-08-31 动作从模型预测到仿真执行的链路审查

你这次重点怀疑的是：
- `norm_state / norm_action` 是否有问题
- pi0.5 原始 32 维动作头和当前 MSP 14 维动作头的维度适配是否错位
- 当前实现里 “先预测 14 维，再 pad 到 32 维” 会不会影响反归一化和执行

结论先说：
- 仅从当前代码链路看，`MSP 输出后 pad 到 32 再 Unnormalize` 这件事本身不是主要问题
- 这条链路在数值上是自洽的
- 真正更可疑的是：
  1. stage2 的 latent 头本身塌缩
  2. stage2 用的是 MSP 14 维动作统计，而不是原始 pi0.5 32 维 flow 头那套训练目标
  3. 如果部署/训练用的 norm stats 资产不一致，也会让 14 维输出被错误反归一化

### 当前推理链路的真实顺序

1. `model.py`
- `encode_obs()` 把环境观测打包成：
  - `state`
  - `images`
  - `prompt`
- 然后 `self.policy.infer(single_observation)`

2. `policy_config.create_trained_policy()`
- 输入变换顺序：
  - `repack_transforms.inputs`
  - `InjectDefaultPrompt`
  - `data_transforms.inputs`
  - `Normalize`
  - `model_transforms.inputs`
- 输出变换顺序：
  - `model_transforms.outputs`
  - `Unnormalize`
  - `data_transforms.outputs`
  - `repack_transforms.outputs`

3. Aloha 输入侧
- `LeRobotAlohaDataConfig.create()` 里接的是：
  - `AlohaInputs`
  - 可选 `DeltaActions`
- `AlohaInputs` 会先把 Aloha 原始状态/动作变换到 pi 训练空间
- 这里训练和推理都会生效

4. 归一化
- `Normalize` 发生在 `AlohaInputs` 之后、`PadToDims` 之前
- 也就是说：
  - 先在真实 14 维 Aloha/pi 动作空间上做归一化
  - 再做模型维度适配

5. MSP 配置下的维度适配
- `ModelTransformFactory` 对 `use_msp_action_head=True` 时，用的是：
  - `PadToDims(state_dim=32, action_dim=14)`
- 这意味着：
  - `state` pad 到 32 维
  - `actions` 保持 14 维，不 pad 到 32
- 所以 stage2 训练时，送进 `msp_action_vae` 的动作本来就是 14 维

6. MSP 模型输出
- `pi0.py::_sample_msp_actions()`
  - `msp_action_vae(..., method="get_action")` 先解码出 14 维动作
  - 然后 `_pad_msp_actions()` 才把它 pad 到 32 维

7. 反归一化
- `Unnormalize` 在输出侧先于 `AlohaOutputs`
- 它的实现对超出统计维度的 pad 部分采用：
  - `mean=0`
  - `std=1`
- 所以：
  - 前 14 维按真实动作统计反归一化
  - 后面补出来的 18 维保持原值不变，通常仍是 0
- 这一步本身没有把前 14 维搞乱

8. Aloha 输出侧
- `AlohaOutputs` 直接做：
  - `actions = data["actions"][..., :14]`
  - 然后 `_encode_actions(...)`
- 也就是：
  - 先截回前 14 维
  - 再做 Aloha 执行空间的关节符号/夹爪范围映射

9. XPolicyLab 执行侧
- `model.py::get_action_batch()`
  - `policy.infer()` 返回的已经是 14 维 Aloha 动作
  - 然后再喂给 `unpack_robot_state(..., source_type="obs")`
- `unpack_robot_state()` 只是把 14 维向量拆成左右臂字典，不做数值缩放

### 因此，关于你问的 “原版 pi0.5 小任务微调 14 维时有没有 pad”

原版 pi0.5 常规做法是：
- 输入侧：
  - 先在真实机器人动作维度上做数据变换和归一化
  - 再 pad 到模型动作维度
- 输出侧：
  - 先反归一化
  - 再截回真实机器人动作维度

你现在的 MSP 版本虽然不是完全一样的 32 维 flow 头，但在“norm 和 pad 的顺序”这件事上没有明显反了。

### 这次审查后的关键判断

1. `MSP 输出后 pad 到 32 再 Unnormalize`
- 这不会污染前 14 维
- 因为 `Unnormalize` 对新增 pad 维度用的是单位统计

2. 真正和原版 pi0.5 不同的是
- 原版 pi0.5 动作头本体就是 32 维 flow head
- 现在 MSP 模式下，动作建模主体已经换成了 14 维 VAE latent -> 14 维 decoder
- 32 维只剩下状态 token 对齐和接口兼容意义，不再是动作语义本体

3. 所以“统一倒臂”更像是动作头学坏了，而不是 pad/norm 顺序把动作毁了
- 尤其是当前 stage2 还是 latent MSE，不是原版 MSP FlowAR
- 多任务下更容易塌到平均 latent，再解码成固定姿态

4. 仍需单独核实的一点
- 当前 deploy 加载的 `norm_stats.json` 是否确实来自你这次 stage2 checkpoint 的 `assets/<repo_id>/`
- 如果误用了别的 checkpoint 的 norm stats，14 维动作会被系统性错误反归一化，这也会导致动作整体塌坏

下一步最值得做的不是继续猜，而是直接打印：
- `policy.infer()` 刚返回、尚未 `unpack_robot_state()` 前的 14 维 action
- `Unnormalize` 前后的 action
- `AlohaOutputs` 截断前后的 action
- 这样可以立刻判断问题是在 MSP 动作头、反归一化，还是 Aloha 输出映射

## 2026-08-31 参考 MINT 后的结构对比结论

参考代码：
- `/home/xuan/PycharmProjects/VLA/MSP_VLA/XPolicyLab/policy/Pi_05/MINT/policy/lerobot_policy_mint`

### MINT 这套方法的关键结构

1. 第一阶段不是连续 latent VAE，而是 `MultiScaleVQVAE`
- 文件：
  - `.../modeling_mint.py`
  - `.../mint_utils.py`
- 训练时先把动作编码成多尺度离散 code：
  - `idxBls_List = self.multi_scale_vqvae.inp_to_idxBl(actions)`

2. 第二阶段监督目标不是 latent MSE，而是离散 token 的交叉熵
- 文件：`modeling_mint.py:761-765`
- 直接做：
  - `F.cross_entropy(vq_logits.transpose(1, 2), gt_idxBls, reduction="none")`
- 这点非常关键
- 因为它不是让模型去回归“平均 latent”，而是让模型在 codebook 上做分类
- 在多任务下，这种目标通常比连续 latent 回归更不容易塌到统一姿态

3. 下一尺度输入不是简单 resize 上一尺度输出
- 训练时：
  - `quantizer.idxBl_to_next_scale_input(idxBls_List)`
- 推理时：
  - `quantizer.get_next_autoregressive_input(...)`
- 也就是说：
  - coarse -> fine 的过渡不是“上一尺度 token / latent 线性插值一下”
  - 而是通过 quantizer 内部的多尺度残差累计逻辑构造

4. 推理是标准的 prefix 一次编码 + suffix 分尺度 KV-cache 自回归
- 文件：`modeling_mint.py:934-1014`
- 先跑一次 prefix 拿 `past_key_values`
- 后面每个尺度只喂当前尺度 token
- 当前尺度可以看全部 prefix 和历史尺度
- 然后把当前尺度采样到的 code 更新到 `f_hat`
- 最后用 VQ-VAE decoder 解码出动作

### 它和我们当前 `pi0.5 + MSP` 的关键差异

1. 我们现在第二阶段还是连续 latent 回归
- 文件：`openpi/src/openpi/models/pi0.py`
- 当前 loss 是：
  - `pred_latents` vs `target_latents` 的 MSE
- MINT 是离散 code CE
- 这会直接影响多任务稳定性

2. 我们现在的 coarse -> fine 输入构造更弱
- 当前训练：
  - `build_teacher_forced_inputs()`
- 当前推理：
  - `build_current_scale_inputs()`
- 本质都是“上一尺度结果 resize 后作为下一尺度输入”
- MINT 这里是 quantizer 驱动的累计重建状态，不只是 resize

3. 我们现在最终 finest latent 直接送 stage1 decoder
- MINT 是把每一尺度采样结果逐步累计进 `f_hat`
- 最终 decode 的不是“最后一段 token 本身”
- 而是“累计后的完整多尺度量化表示”

4. 我们现在虽然也有 block-wise mask 和按尺度推理
- 但核心生成目标仍更像：
  - prefix-conditioned multi-scale latent regressor
- MINT 更像：
  - prefix-conditioned multi-scale discrete autoregressive generator

### 对你当前“所有任务统一倒臂”的启发

这份 MINT 参考很有价值，因为它说明：

1. 问题未必在 `norm/pad`
- 更可能在第二阶段目标过弱

2. 当前 `latent MSE + resize传递` 这条链路更容易塌缩
- 尤其是多任务
- 因为模型可以通过输出“平均 latent”获得一个不算太差的 MSE

3. MINT 这种离散 code CE 目标天然更抗“平均化”
- 分类任务里输出错误类别会被直接惩罚
- 不像连续回归那样容易往均值收缩

### 对我们当前实现的直接建议

不直接抄成 VQ-VAE，但可以借 MINT 的两个关键思想：

1. 第二阶段不要只做 finest latent 的连续 MSE
- 至少要增强为更强监督
- 例如：
  - 每尺度独立监督
  - coarse->fine residual supervision
  - 或者直接改成离散 latent/code 预测

2. 下一尺度输入不要只靠 resize 上一尺度预测
- 更应接近“累计重建状态”
- 也就是：
  - 当前尺度预测出来后，不只是把这一段 token 传下去
  - 而是要构造一个“到目前为止的多尺度重建表示”，再供下一尺度条件化

### 当前最重要的判断

如果你要找一个“和你现在做的最像、又能解释为什么你这版容易倒臂”的参考，
那 MINT 给出的核心答案是：

- 不是多尺度自回归这个大方向错了
- 是你当前第二阶段的“连续 latent MSE + 简单 resize 传递”太弱
- 它比 MINT 的“离散 code CE + quantizer累计状态传递”更容易塌缩成统一动作

## 2026-08-31 新判断：重点不是怀疑 MSP，而是学习 MINT 如何改 pi0.5 动作头

你这次纠正得对：
- `MSP` 这个方向本身没有问题
- 即使动作头里不带 flow，只保留“尺度 Transformer + decoder”，也可以正常工作
- 所以当前问题更应看成“从 pi0.5 迁到多尺度头时，动作头适配没有对齐好”

### 从 MINT 看，pi0.5 动作头到底改了什么

MINT 不是简单“在原动作头外面包一层多尺度逻辑”，它改的是动作头的输入、输出、训练目标、推理路径四件事。

1. 保留前缀 VLM，不动 prefix 编码
- 图像和语言 prefix 还是正常进 PaliGemma
- 这一点和我们当前思路一致

2. 原 pi0.5 的 suffix 输入被彻底替换了
- 原始 pi0.5：
  - suffix 输入是 `noisy_actions -> action_in_proj`
  - 再配合 timestep / adaRMS 做 flow matching
- MINT：
  - suffix 输入不再是连续动作
  - 而是 `多尺度 code embedding + level embedding + sos`
- 也就是说，动作 expert Gemma 还在，但它吃的 token 语义已经换了

3. 原 pi0.5 的输出头也被彻底替换了
- 原始 pi0.5：
  - `suffix_out -> action_out_proj`
  - 输出连续动作/速度场
- MINT：
  - `suffix_out -> vq_code_out_proj`
  - 输出每个尺度位置上的 codebook logits
- 所以 Gemma expert 不再回归动作值，而是在做多尺度 token 预测

4. 原 pi0.5 的训练目标被替换成多尺度自回归目标
- 原始 pi0.5：
  - flow matching / diffusion-style continuous regression
- MINT：
  - 多尺度 token CE
- 关键点不是“多尺度”三个字，而是：
  - 动作 expert 被改造成了一个真正的 AR decoder，而不再是 flow regressor

5. 推理路径也不是原 pi0.5 那套迭代 denoise
- 原始 pi0.5：
  - 多步 ODE / flow rollout
- MINT：
  - prefix 一次编码
  - suffix 逐尺度 KV-cache 自回归
  - 最后把最细尺度累计状态送 decoder 还原动作

### 这对我们当前 `pi0.5 + MSP` 的直接启发

我们现在虽然也做了“多尺度 suffix + block mask + 分尺度推理”，
但本质上还没有把 `pi0.5 action expert` 完整改造成“多尺度 AR 动作头”。

当前更像是：
- 用 Gemma expert 回归多尺度 latent

而 MINT 这种改法更像是：
- 用 Gemma expert 作为真正的多尺度自回归解码器

### 因此，后面改造动作头时应按这个映射来审查

需要逐项核对这 4 件事是否都改对了：

1. suffix token 语义有没有彻底替换
- 不能保留原 flow-head 的输入假设

2. action expert 的输出头有没有换成“尺度预测头”
- 不能还沿着原动作回归头思路改一半

3. 训练目标有没有和新头严格匹配
- 新头如果是 AR 头，loss 也要是 AR 对应的监督

4. 推理路径有没有和训练时的 token/state 传递一致
- 不能训练时一套、推理时另一套

### 当前下一步

下一步不先改代码，先专门对照：
- 原始 `pi0.5` 动作头
- `MINT` 的多尺度动作头
- 我们当前 `pi0.py` 的 MSP 动作头

把“输入 token 语义 / 输出头 / loss / 推理缓存路径”四栏做成一张差异表，
这样才能精确知道现在到底少改了哪一块。

## 2026-08-31 关于“直接拿 MINT 动作头来用”的判断

结论：
- 可以直接借用 `MINT` 动作头的整体骨架
- 不能把 `MINT` 头原样搬过来只改输入输出维度

原因不是简单的“离散/连续张量不同”，而是 `MINT` 头的定义里本身就包含了离散量化器语义：

1. `MINT` 的输入不是普通 latent
- 它输入的是：
  - `SOS`
  - `level embedding`
  - `quantizer.idxBl_to_next_scale_input(...)` 产生的下一尺度条件

2. `MINT` 的输出也不是普通 latent
- 它输出的是 codebook logits
- 然后通过采样得到离散 index

3. `MINT` 的尺度间状态传递依赖 quantizer
- 推理时核心不是“预测当前尺度 -> resize -> 下一尺度”
- 而是：
  - `quantizer.get_next_autoregressive_input(...)`
  - 维护累计重建状态 `f_hat`

4. `MINT` 的最终 decoder 输入也不是最后一层 token
- 而是累计后的量化重建表示

因此，真正可复用的是：
- `prefix 保留、suffix 改成多尺度 AR` 这个总体结构
- `SOS + level embedding + blockwise mask + prefix一次编码 + scale-wise kv-cache` 这套推理框架
- `动作 expert 从 flow regressor 改成 AR predictor` 这个改造方向

不能直接照搬的是：
- `vq_code_in_proj / vq_code_out_proj`
- `idxBl_to_next_scale_input`
- `get_next_autoregressive_input`
- `f_hat + codebook + CE loss`

如果迁到连续 latent MSP，应该做的是：

1. 保留 MINT 的 AR 骨架
- prefix 一次编码
- suffix 分尺度生成
- 每尺度有独立 level/scale embedding
- KV-cache 按尺度递推

2. 把离散 code 预测替换为连续 latent 预测
- `vq_code_out_proj` -> `latent_out_proj`

3. 把 quantizer 驱动的状态传递替换为连续 latent 的累计状态传递
- 这一步不能只写成“简单 resize”
- 需要设计成更接近“累计重建状态”的连续版本

4. decoder 仍走第一阶段 latent decoder
- 但输入最好是“累计后的 finest 表示”
- 不只是最后一次预测块的裸输出

## 2026-08-31 MINT 的尺度构造与 coarse-to-fine 状态传递

这一步只回答两个问题：

1. `MINT` 的多尺度是怎么构造的
2. `MINT` 的尺度为什么是从粗到细，以及尺度间状态怎么传

### 1. MINT 的多尺度怎么构造

核心配置是：
- `configuration_mint.py`
- `patch_nums`

它表示一组严格递增的尺度长度，例如概念上可以是：
- `(1, 2, 4, 8)`
- 或 `(2, 4, 8, 16)`

最后一个尺度必须等于 VQ-VAE latent 的最长时间长度。

在 quantizer 里，多尺度不是对原动作直接切块，而是对 VQ-VAE encoder 输出的 latent feature `f_BCH` 做多尺度残差量化：

1. 先得到最高分辨率 latent feature：
- `f_BCH`
- shape 类似 `(B, C, H)`
- 其中 `H = patch_nums[-1]`

2. 从最粗尺度开始，依次对当前残差 `f_rest` 做下采样：
- `F.interpolate(f_rest, size=pn, mode=self.downsample_mode)`

3. 在当前尺度上，为每个位置找到最近的 codebook embedding：
- 得到 `idx_BH`

4. 再把这个尺度的 embedding 上采样回最大长度 `H`
- 然后经过 `quant_resi`
- 得到这个尺度对应的重建分量 `h_BCH`

5. 累加到总重建：
- `f_hat = f_hat + h_BCH`

6. 同时更新剩余残差：
- `f_rest = f_rest - h_BCH`

所以 MINT 的多尺度本质上是：
- 对同一个最高分辨率 latent feature
- 做一串“从粗到细的残差量化分解”

不是简单把序列 resize 出几个版本完事。

### 2. 为什么是从粗到细

因为它每一层预测的不是完整最终表示，而是：
- 当前分辨率下，对“还没解释掉的残差”的一个补充

粗尺度先负责全局轮廓：
- 低分辨率
- 覆盖整段动作的大致趋势

细尺度再负责补细节：
- 更高分辨率
- 修正 coarse scale 没表达完的局部变化

这个逻辑直接体现在：
- `for si, pn in enumerate(self.patch_nums): # from small to large`

### 3. 训练时的尺度间状态传递

训练时用的是：
- `idxBl_to_next_scale_input(gt_ms_idx_Bl)`

它做的事不是“把上一尺度 token resize 一下”，而是：

1. 初始化累计状态：
- `f_hat = zeros(B, C, H_max)`

2. 对第 `si` 个 GT 尺度 token：
- 查 codebook embedding
- 上采样到最高尺度 `H_max`
- 过 `quant_resi`
- 累加进 `f_hat`

3. 把当前累计状态 `f_hat` 再下采样到下一尺度长度 `patch_nums[si+1]`
- 这个结果就是下一尺度 Transformer 的输入条件

4. 把所有“下一尺度输入”拼起来：
- 形成训练时 teacher-forcing 的 suffix 输入

所以训练时的 next-scale input 语义是：
- “截至当前尺度为止，已经累计重建出来的 latent 状态”

不是：
- “单独上一尺度的输出”

### 4. 推理时的尺度间状态传递

推理时用的是：
- `get_next_autoregressive_input(si, SN, f_hat, h_BCH)`

逻辑和训练时完全同源：

1. 当前尺度采样出 index
2. index -> embedding -> 当前尺度重建分量 `h_BCH`
3. 若不是最细尺度：
- 先上采样到最高尺度
- 过 `quant_resi`
- 累加进 `f_hat`
4. 再把累计后的 `f_hat` 下采样到下一尺度长度
5. 这个“下采样后的累计状态”作为下一尺度的输入 token map

所以推理时传给下一尺度的也不是“上一尺度预测值本身”，而是：
- 当前为止的累计重建状态

### 5. 这套机制最关键的抽象

MINT 跨尺度传递的不是 token，而是一个持续更新的隐藏状态：
- `f_hat`

每一层都在做：
- 当前尺度预测 -> 转成当前尺度重建分量 -> 更新全局累计状态 `f_hat`

然后下一层拿到的是：
- `downsample(f_hat)`

这比“上一尺度结果直接 resize 给下一层”强很多，因为：

1. 它保留了所有已生成尺度的累计信息
2. 它和最终 decoder 输入语义一致
3. 训练和推理完全对齐

### 6. 对替换成 MSP 连续 latent 的直接启发

如果我们要把这套逻辑替换成 MSP 连续 latent 版，最该保留的不是 VQ 离散化本身，而是这个抽象：

- 维护一个 coarse-to-fine 的累计隐藏状态
- 下一尺度条件输入 = 当前累计状态的尺度化版本

也就是说，MSP 连续 latent 版更合理的 next-scale input 应该是：
- `running_latent_state`
- 而不是简单的 `resize(prev_scale_latent)`

这是后面替换 `build_teacher_forced_inputs()` 和 `build_current_scale_inputs()` 时最重要的参考点。

## 2026-08-31 MINT 动作头骨架 + MSP 连续尺度语义的最终迁移方案

这里更正上一节最后一部分的推断：
- MINT 的确维护累计量化重建状态 `f_hat`
- 但原版 MSP `MSP/algos/flow/flow_ar.py` 没有照搬这套累计状态
- MSP 的尺度间传递明确是：
  - 当前/上一尺度 latent 上采样到最大尺度
  - 再下采样到下一尺度
  - 下一尺度直接使用这个 resize 结果
- 因此 pi0.5 + MSP 不应自行引入连续版 `f_hat`；尺度构造和传递必须以 MSP 为准

这次迁移分工如下：

1. 从 MINT 复用 pi0.5 动作头适配骨架
- VLM prefix 保持原样
- suffix 使用 learned SOS、scale embedding 和连续 latent projection
- 训练时一次输入完整多尺度 suffix
- suffix 使用尺度块 attention mask
- 推理时 VLM prefix 只运行一次
- action expert 按尺度增量更新 KV cache

2. 从 MSP 保留连续多尺度语义
- 最细 VAE latent 通过线性插值得到各尺度 target
- teacher forcing 的下一尺度输入来自上一 GT 尺度：先上采样到最大尺度，再下采样到下一尺度
- 推理的下一尺度输入来自上一预测尺度，使用完全相同的两次 resize
- 最终只把最细尺度连续 latent 送入 Stage-1 VAE decoder
- loss 继续按 `scale / max_scale` 加权

3. 不迁移 MINT 的离散部分
- 不使用 codebook/index
- 不使用 `idxBl_to_next_scale_input`
- 不使用 `get_next_autoregressive_input` 的 `f_hat` 累计逻辑
- 不使用 code logits、采样和交叉熵

对当前实现的关键审查结果：
- mask、block-local RoPE、三组 learned positional embeddings、prefix-once KV cache 已经存在
- MSP 的 resize 尺度传递也已经存在，方向正确
- 发现一个明确的训练/推理不一致：
  - 训练 `_embed_msp_suffix(..., self.msp_scales)` 会得到正确的尺度 ID `0,1,2,...`
  - 推理传入单元素 `current_scale=(scale,)`，`build_scale_ids(current_scale)` 每轮都会返回 0
  - 结果是所有推理尺度都错误地使用最粗尺度 embedding
  - MINT 推理使用 `level_embs(si)`，所以推理必须显式使用当前 `block_index`

分步修改顺序：
1. 修复推理 scale ID，并增加 MINT 式 learned SOS
2. 用测试锁定 MSP 的 target 构造、teacher forcing 和推理 resize 传递完全同源
3. 核对训练全序列与推理逐尺度 cache 的 mask/位置/尺度 embedding 一致性
4. 增加最小前向验证和逐尺度输出统计，重点排查多任务推理塌缩

本轮计划修改文件：
- `openpi/src/openpi/models/msp_scale_head.py`
- `openpi/src/openpi/models/msp_scale_head_test.py`
- `openpi/src/openpi/models/pi0.py`
- `task.md`

第一步实际修改：
1. 修复 scale embedding 的训练/推理不一致
- `build_scale_ids(..., block_index=None)`：训练返回完整的绝对尺度 ID
- `build_scale_ids((current_scale,), block_index=i)`：推理返回当前绝对尺度 ID `i`
- `_embed_msp_suffix()` 推理不再把所有尺度误当成第 0 层

2. 增加 MINT 式 learned SOS
- 新参数：`msp_sos_token`
- 训练完整 suffix 的第一尺度使用 SOS
- 推理第 0 个尺度使用同一个 SOS
- 后续尺度仍使用 MSP 的上一尺度 resize latent 经 `msp_latent_in_proj` 后的 token
- 参数名带 `msp`，加载原始 pi0.5 权重时由现有 `missing_regex=".*lora.*|.*msp.*"` 保持随机初始化

3. 显式 detach Stage-1 latent
- VAE `get_sample` 后增加 `jax.lax.stop_gradient`
- 对齐 MSP 原版 `act.detach()`
- Stage-2 的 target 和 teacher-forcing 输入不向 Stage-1 VAE 反向传播

4. 新增回归测试
- 训练完整尺度 ID 与推理绝对尺度 ID 一致
- 各尺度 target 是最细 latent 的线性 resize
- 训练 teacher forcing 与推理 next-scale input 都遵循 MSP 的：
  - 上一尺度 -> 最大尺度 -> 下一尺度

兼容性说明：
- 从原始 pi0.5 + Stage-1 VAE 开始训练 Stage-2：兼容，`msp_sos_token` 随机初始化
- 已经训练好的旧版 Stage-2 完整 checkpoint：缺少 `msp_sos_token`，不能当作新结构的完整 checkpoint 直接恢复；需要重新训练或单独提供旧 checkpoint 迁移逻辑

本轮验证：
- `py_compile` 通过：
  - `openpi/src/openpi/models/msp_scale_head.py`
  - `openpi/src/openpi/models/msp_scale_head_test.py`
  - `openpi/src/openpi/models/pi0.py`
- `git diff --check` 通过
- 当前本地 `openpi/.venv` 没有安装 `jax` 和 `pytest`，所以新增 JAX 单元测试未在本机执行
- 没有为此临时安装或改变训练环境；需要在服务器实际 openpi/JAX 环境运行：
  - `pytest openpi/src/openpi/models/msp_scale_head_test.py -q`

当前迁移完成度：
- MINT 提供的 pi0.5 suffix 改造骨架已经对齐：SOS、level embedding、block AR、prefix-once、scale-wise KV cache
- MSP 提供的连续 latent 语义已经保留：多尺度 target、两次 resize 传递、连续回归、最细 latent VAE decode
- VLM prefix 和原始 pi0.5 权重加载路径未修改
- 本轮没有引入 MINT 的 codebook、离散采样、累计 `f_hat` 或 CE loss

## 2026-08-31 再审查：当前单 Gemma action expert 与 MSP/MINT 的结构语义错配

用户指出的问题成立：当前代码把 MSP 原版 encoder/decoder 的 learned positional embeddings 同时加到单个 Gemma suffix 输入上，属于硬搬参数、没有搬对应计算阶段。

### 严重问题 1：三组 learned positional embeddings 的落点不成立

当前 `pi0.py`：
- `msp_encoder_pos_embed` 加在 Gemma action expert 输入前
- `msp_decoder_pos_embed` 也加在同一处
- `msp_diffusion_pos_embed` 加在 Gemma 输出后、continuous latent projection 前

MSP 原版真实结构：
1. `encoder_pos_embed_learned`
- 加在独立 MAE encoder 前
- 后面经过 `encoder_blocks`

2. `decoder_pos_embed_learned`
- encoder 输出先经过 `decoder_embed`
- 再加 decoder position
- 后面经过另一套独立 `decoder_blocks`

3. `diffusion_pos_embed_learned`
- 加在 decoder blocks 输出后
- 作为独立 flow head 的条件输入

当前 pi0.5 + MSP 只有一套 Gemma action expert：
- 没有 MSP encoder/decoder 两套 Transformer
- 没有 `decoder_embed + decoder_blocks`
- 没有 diffusion/flow head

所以不能把三张位置表按名字全部保留。尤其：
- `encoder_pos + decoder_pos` 在同一输入处相加没有结构依据
- `diffusion_pos` 在没有 diffusion head 时也没有对应语义

推荐修改：
- 删除 `msp_encoder_pos_embed`
- 删除 `msp_decoder_pos_embed`
- 删除 `msp_diffusion_pos_embed`
- 删除 `_msp_pos_slice()` 和训练/推理中的对应加法
- suffix token 保持 MINT 已验证的形式：
  - 第 0 尺度：`SOS + scale_embed`
  - 后续尺度：`latent_in_proj(resized_previous_latent) + scale_embed`
- 时间位置只交给 Gemma suffix 的 block-local RoPE

如果以后确实要保留 learned temporal position，最多只能为“这一套 Gemma stack”重新定义一张统一位置表；但这将是新设计，不是 MINT 或 MSP 的直接移植，因此当前不建议自行增加。

### 严重问题 2：当前 MSP action expert 仍使用 pi0.5 adaRMS，不等价于 MINT

当前初始化：
- `_gemma.Module(..., adarms=config.pi05)`
- `use_adarms=[False, True]`
- MSP forward 给 action expert 传一个全 0 `adarms_cond`

这不等于关闭 adaRMS：
- `cond=None` 才走普通 RMSNorm 和普通 residual
- `cond=zeros` 仍走 adaptive RMSNorm 的 Dense modulation 和 gated residual
- 预训练 adaRMS Dense 的 bias 可以使全 0 cond 产生非零 scale/shift/gate
- 因此当前 MSP 头仍带着原 flow timestep 调制结构，只是 timestep 被错误替换成常量 0 向量

MINT 的明确做法：
- action expert 初始化为 `use_adarms=[False, False]`
- forward 使用 `adarms_cond=[None, None]`
- 即多尺度 AR action expert 使用普通 RMSNorm，不保留 flow timestep modulation

推荐修改方向：
- MSP 模式下把 action expert 初始化成普通 RMSNorm
- MSP forward 传 `[None, None]`
- VLM expert 仍保持原结构和权重，不受影响

但这不能只改 forward：
- JAX Linen 的普通 RMSNorm 参数是 `scale`
- adaRMS 参数是 modulation Dense kernel/bias
- 两种初始化得到的参数树不同
- 直接把当前 `adarms_cond=zeros` 改成 `None` 会出现普通 RMSNorm `scale` 参数不存在的问题

所以必须同时修改：
1. Gemma action expert 初始化模式
2. pi0.5 checkpoint loader：继续加载 attention/MLP 等兼容参数
3. 跳过 action-expert adaRMS modulation 参数
4. 对新普通 RMSNorm `scale` 做确定的初始化并输出加载统计

在看到实际 pi0.5 checkpoint 展平 key 之前，不凭空编写 norm key regex。

### 高优先级问题 3：block-local RoPE 只实现了 reset，未复现 MSP 的频率缩放

MSP 原版：
- 每个尺度 block 的位置从 0 重新开始，这部分当前已实现
- 但 `ActionRotaryEmbeddingFast` 还执行：
  - `t = arange(seq_len) / pt_seq_len`
- `pt_seq_len=max(scale)`，默认 `(1,2,4,8)` 时为 8

当前 JAX Gemma：
- 自定义 suffix `rope_positions` 是整数 `0..scale-1`
- `_apply_rope()` 直接使用该值
- 没有除以 `max(msp_scales)`

因此当前 RoPE 相位比 MSP 原版大 `max_scale` 倍，并非完全一致。

推荐修改：
- 让 suffix 自定义 RoPE position 支持 float
- MSP suffix 使用：
  - `local_position / max(msp_scales)`
- prefix/VLM 继续使用原始 Gemma 全局整数 position，不改变
- 训练全尺度和推理分尺度必须调用同一 helper

### 中优先级问题 4：推理 global position offset 与训练/MINT 写法不一致

当前推理：
- `start = prefix_mask.shape[1] + block_start`

训练和 MINT：
- suffix 起点基于每个样本有效 prefix token 数：
  - `sum(prefix_mask)`

目前 suffix 已覆盖自定义 RoPE，所以这个差异通常不会改变 suffix Q/K 的旋转位置；但它仍是语义不一致，也会给后续取消自定义 RoPE或调整 cache 留下隐患。

推荐修改：
- 使用：
  - `sum(prefix_mask, axis=-1)[:, None] + block_start + arange(current_scale)`

### 已确认正确、无需重写的部分

1. 多尺度 target
- 最细 VAE latent 线性 resize 到 `(1,2,4,8)`
- 与 MSP 一致

2. 尺度间传递
- 上一尺度先上采样到最大尺度，再下采样到下一尺度
- 训练使用 GT 上一尺度，推理使用预测上一尺度
- 与 MSP 一致

3. 尺度 block mask
- 当前尺度可见 prefix、所有历史尺度和当前尺度全部 token
- 不可见未来尺度
- 与 MSP/MINT 一致

4. KV cache 生命周期
- prefix/VLM 只运行一次
- 每个尺度只输入当前 block
- cache 逐尺度追加
- 与 MINT 和 MSP 推理思想一致

5. scale embedding
- 训练使用完整绝对 scale ID
- 推理已修复为使用当前 `block_index`
- 与 MINT 一致

6. SOS
- 训练和推理第 0 尺度使用同一 learned SOS
- 与 MINT 一致

7. Stage-1 latent/decoder
- 训练使用 VAE posterior sample，并 stop-gradient
- 推理把最细尺度 continuous latent 送 Stage-1 decoder
- 与无 flow 的 MSP 连续 latent 路径一致

8. per-scale loss weighting
- 每尺度 loss 乘 `scale / max_scale` 后求和
- 与 MSP 原版一致

### 推荐的下一轮修改顺序

1. 先删除三组错误映射的 learned positional embeddings
2. 再把 MSP action expert 从 adaRMS 改成普通 RMSNorm，并同步 checkpoint 参数迁移
3. 然后补齐 MSP RoPE 的 `/ max_scale` 频率缩放
4. 最后统一推理 position offset，并做训练整段 forward 与增量 cache forward 的数值等价测试

这四步完成之前，不能认为当前动作头已经和 MINT 的 pi0.5 适配以及 MSP 的尺度细节完全一致。

## 2026-08-31 执行适配修正第 1 步：删除错误映射的三组位置表

本轮只处理上一节的严重问题 1，不同时修改 adaRMS 或 RoPE，保证每个结构变化可以独立检查。

已删除：
- `msp_encoder_pos_embed`
- `msp_decoder_pos_embed`
- `msp_diffusion_pos_embed`
- `_msp_pos_slice()`
- `msp_scale_head.slice_scale_positions()`
- 对应的位置切片单元测试

修改后的 suffix 输入：
- 第 0 尺度：
  - `msp_sos_token + msp_scale_embed(scale_id=0)`
- 后续尺度：
  - `msp_latent_in_proj(resized_previous_latent) + msp_scale_embed(scale_id=i)`
- 时间位置：
  - 只通过 Gemma action expert 的 block-local RoPE 注入

修改后的 suffix 输出：
- 训练：
  - `msp_latent_out_proj(suffix_out)`
- 推理：
  - `msp_latent_out_proj(current_scale_suffix_out)`
- 不再叠加没有 diffusion head 与 decoder stack 对应的 learned position

结构依据：
- 对齐 MINT 的单 action-expert Gemma 骨架
- 保留 MSP 的连续 latent、多尺度 resize、block-local RoPE 和 per-scale loss
- 不再把 MSP 双 Transformer + flow head 的位置参数硬映射到单 Gemma stack

验证：
- 已通过 `py_compile`：
  - `openpi/src/openpi/models/pi0.py`
  - `openpi/src/openpi/models/msp_scale_head.py`
  - `openpi/src/openpi/models/msp_scale_head_test.py`
- 已通过 `git diff --check`
- `rg` 确认 `openpi/src/openpi` 中没有三组旧位置参数及切片 helper 的残留引用
- 本地环境仍缺少 JAX/pytest，因此运行时单元测试留到服务器环境执行

checkpoint 兼容性：
- 从原始 pi0.5 权重 + Stage-1 VAE 权重启动新的 Stage-2：兼容
- 旧版 Stage-2 checkpoint 包含已经删除的位置参数，并且缺少/不同于当前结构，不应作为新结构的完整 checkpoint 直接 resume

下一步：
- 把 MSP action expert 从 pi0.5 adaRMS 常量零条件，改成 MINT 使用的普通 RMSNorm
- 必须同步处理 pi0.5 action-expert checkpoint 的 norm 参数迁移和加载日志

## 2026-08-31 执行适配修正第 2 步：MSP action expert 改为普通 RMSNorm

本轮完成 MINT 动作头适配中与 adaRMS 相关的结构修正。

### 1. MSP 模式关闭 action-expert adaRMS

`pi0.py` 现在显式计算：
- 普通 pi0.5 flow 模式：`use_action_expert_adarms=True`
- pi0.5 + MSP 模式：`use_action_expert_adarms=False`

Gemma 初始化同步使用：
- VLM expert：普通 RMSNorm，保持不变
- MSP action expert：普通 RMSNorm

MSP 训练和推理 forward 都改为：
- `adarms_cond=[None, None]`

已删除 MSP suffix 中原来的全 0 `adarms_cond`。

这与 MINT 的：
- `use_adarms=[False, False]`
- `adarms_cond=[None, None]`
保持一致。

### 2. MSP 模式不再创建 flow-only 参数

以下参数只在原 pi0/pi0.5 flow 路径创建：
- `action_in_proj`
- `action_out_proj`
- `time_mlp_in/time_mlp_out`
- `state_proj`
- `action_time_mlp_in/action_time_mlp_out`

MSP 模式只创建自己的：
- `msp_latent_in_proj`
- `msp_latent_out_proj`
- `msp_scale_embed`
- `msp_sos_token`
- `msp_action_vae`

这样 MSP action head 的参数树不再保留不会参与 forward 的 flow 参数。

### 3. SOS 初始化对齐 MINT

之前 `msp_sos_token` 使用 std=0.02 的随机正态初始化。

MINT 原版是：
- `nn.Parameter(torch.zeros(1, 1, width))`

现在 JAX 版改为全 0初始化，后续通过训练更新。

### 4. pi0.5 checkpoint 参数迁移

新增：
- `weight_loaders.MSP_ACTION_EXPERT_MISSING_REGEX`

它允许目标模型保留初始化值的参数包括：
- MSP 新动作头参数
- LoRA 参数
- action expert 普通 RMSNorm：
  - `pre_attention_norm_1/scale`
  - `pre_ffw_norm_1/scale`
  - `final_norm_1/scale`

checkpoint 合并行为：
- attention/MLP 等目标模型中存在且同名的 pi0.5 action-expert 权重正常加载
- checkpoint 中只属于 adaRMS modulation Dense 的参数因目标模型不存在而忽略
- 新普通 RMSNorm scale 从目标模型初始化值保留
- MSP projection/scale/SOS 从目标模型初始化值保留

加载日志新增三类统计：
- `loaded`
- `initialized_from_target`
- `ignored_checkpoint_only`

明细只打印前 20 个示例，避免 Stage-1 合并时刷屏。

### 5. Stage-1 VAE 合并语义收紧

复核时曾怀疑 base 随机 MSP 参数会覆盖 Stage-1 VAE；继续检查发现：
- `_load_weights_and_validate()` 会过滤 `ShapeDtypeStruct`
- 所以旧逻辑下 base partial params 通常并不包含随机 MSP 数组
- 不能把它定性为已经发生的覆盖 bug

但旧实现依赖这个隐含过滤行为，不够清晰。本轮改为：
- `merge_msp_vae_params()` 只返回真正匹配的 Stage-1 VAE 数组
- 不再返回一整棵带 shape fallback 的模型树
- `train.py/train_tb.py` 显式以 Stage-1 VAE 为 preferred，覆盖 base partial params

最终参数来源现在是明确的：
- pi0.5 兼容参数来自 base checkpoint
- MSP VAE 参数来自 Stage-1 checkpoint
- MSP 新头和普通 action-expert norm 来自新模型初始化

### 6. 新增测试

文件：
- `openpi/src/openpi/training/weight_loaders_test.py`

覆盖：
- MSP missing regex 只匹配 MSP/LoRA/新普通 action-expert norm
- action expert attention 权重从 checkpoint 加载
- adaRMS checkpoint-only Dense 参数不进入新参数树
- 普通 RMSNorm scale 和 MSP projection 保留目标初始化值

### 7. 验证

已通过：
- `py_compile`
- `git diff --check`
- regex 样例匹配检查
- `rg` 确认 MSP 路径没有零 adaRMS 条件和三组旧位置参数残留

本地环境缺少 JAX/Flax/pytest，新增单元测试需在服务器 openpi 环境运行：
- `pytest openpi/src/openpi/training/weight_loaders_test.py -q`

服务器首次启动 Stage-2 时应重点检查加载日志：
- 普通 action-expert norm 应出现在 `initialized_from_target` 示例中
- adaRMS Dense 和已删除 flow-only 参数应出现在 `ignored_checkpoint_only` 示例中
- `Base model weight load succeeded.`
- `MSP stage-1 weight load succeeded.`

下一步：
- 补齐 MSP block-local RoPE 的 `/ max(msp_scales)` 频率缩放
- 同时统一训练/推理的自定义 RoPE helper 和推理 global position offset

## 2026-08-31 执行适配修正第 3 步：对齐 MSP RoPE 相位与增量位置

本轮完成 MSP `flow_ar.py` 中 RoPE 细节的 JAX 适配，同时不改变 VLM 的原始 Gemma 位置编码。

### 1. MSP 原版 RoPE 的两个必要条件

原版 `ActionRotaryEmbeddingFast` 同时执行：
1. 每个尺度 block 内位置从 0 开始
2. 位置除以 `pt_seq_len=max(scale)`

默认尺度 `(1,2,4,8)` 时：
- 第 4-token 尺度使用：`[0/8, 1/8, 2/8, 3/8]`
- 第 8-token 尺度使用：`[0/8, ..., 7/8]`

之前 JAX 版只完成第 1 点，使用的是整数 `[0,1,2,...]`，相位比 MSP 原版大 8 倍。

### 2. 新增统一 MSP RoPE helper

文件：
- `openpi/src/openpi/models/msp_scale_head.py`

新增：
- `build_msp_rope_positions(scales, batch_size, normalization_length)`

语义：
- 每个尺度 block 独立生成 `arange(scale)`
- 统一除以完整 MSP 配置的最细尺度 `msp_scales[-1]`
- 返回 float32 position

训练：
- 输入完整 scales `(1,2,4,8)`
- normalization length 为 8

推理：
- 每次只输入当前 `(scale,)`
- normalization length 仍显式使用完整配置的 8
- 不会错误地按当前 scale 自己归一化

因此训练完整序列中任意 block 的 RoPE position，与推理单独生成该 block 时逐值相同。

### 3. Gemma 只放宽自定义 RoPE position 类型

文件：
- `openpi/src/openpi/models/gemma.py`

修改：
- `rope_positions` 从只接受整数改为接受 `at.Real`

未修改：
- 默认 `positions` 仍然是整数
- prefix/VLM 不传自定义 RoPE，继续使用原始 Gemma 全局 position
- 只有 MSP suffix 使用归一化 float position

所以这一步不会改变 VLM 的位置编码和预训练权重语义。

### 4. 推理 global position offset 对齐训练/MINT

新增：
- `build_incremental_positions(prefix_mask, block_start, block_length)`

之前推理使用：
- 固定 padded 长度 `prefix_mask.shape[1] + block_start`

现在使用：
- 每个样本的 `sum(prefix_mask) + block_start + arange(block_length)`

这与：
- 训练 `cumsum(concat(prefix_mask, suffix_mask)) - 1`
- MINT `prefix_offsets = sum(prefix_pad_masks)`
保持一致。

对于 batch 内 prompt 长度不同的样本，各自 suffix global position 不再被统一按最大 padding 长度偏移。

### 5. 新增/更新测试

文件：
- `openpi/src/openpi/models/msp_scale_head_test.py`

覆盖：
1. 每个尺度 block 的位置归零
2. 所有 block 都除以完整最细尺度
3. 训练全序列 block slice 与推理单 block position 完全一致
4. batch 内 prefix 有效长度不同时，incremental global position 分别使用各自有效长度

### 6. 验证

已通过：
- `py_compile`
- `git diff --check`
- `rg` 确认旧 `build_block_local_positions`、固定 padded prefix offset 和未使用的 `block_end` 没有残留

本地缺少 JAX/pytest，服务器环境需运行：
- `pytest openpi/src/openpi/models/msp_scale_head_test.py -q`
- `pytest openpi/src/openpi/training/weight_loaders_test.py -q`

下一步：
- 做训练 full-suffix forward 与推理 scale-wise KV-cache forward 的逐尺度数值等价测试
- 该测试需要固定同一组 teacher-forcing latent，使推理每层输入使用 GT 前一尺度，排除模型预测误差，只比较 full forward 和 cache forward 的 hidden/pred latent

## 2026-09-01 第 4 步：full suffix 与 scale-wise KV cache 数值等价测试

本轮没有继续修改网络结构，而是增加一个隔离测试，验证当前 mask、position、RoPE、普通 RMSNorm 和 KV cache 的组合是否真正做到训练/推理同构。

新增文件：
- `openpi/src/openpi/models/msp_gemma_cache_test.py`

### 测试设计

使用一个轻量双 expert Gemma：
- prefix/VLM expert
- suffix/action expert
- width=8
- depth=2
- 普通 RMSNorm
- float32
- dropout=0

不加载：
- 图像编码器
- 动作 VAE
- 真实 checkpoint
- 数据集

输入固定为：
- 同一组随机 prefix embedding
- 同一组随机完整多尺度 suffix embedding
- scales `(1,2,4)`
- batch 内两条样本具有不同有效 prefix 长度，用于覆盖 padding 情况

### 路径 A：训练式 full forward

一次输入：
- 完整 prefix
- 完整多尺度 suffix
- 完整 blockwise attention mask
- 完整 block-local `/ max_scale` RoPE
- `adarms_cond=[None,None]`

保存完整 `full_suffix_out`。

### 路径 B：推理式 incremental forward

1. prefix 单独 forward 一次，获得 KV cache
2. 按 `(1,2,4)` 逐尺度输入与路径 A 完全相同的 suffix token slice
3. 每层使用：
- 当前尺度 attention rows
- 每样本有效 prefix global offset
- 当前尺度 block-local `/ max_scale` RoPE
- 上一轮 KV cache
4. 拼接所有尺度输出为 `cached_suffix_out`

最终断言：
- `cached_suffix_out` 与 `full_suffix_out` 在 `atol=1e-5, rtol=1e-5` 下完全接近

### 该测试覆盖的关键风险

1. full mask 与 incremental mask 的当前尺度 rows 是否一致
2. prefix padding 是否在两条路径中同样被屏蔽
3. 每样本 suffix global position offset 是否一致
4. full block slice 与 incremental block 的 MSP RoPE 是否一致
5. prefix cache 和历史尺度 cache 的追加顺序是否一致
6. action expert 普通 RMSNorm 在 full/cache 路径中是否一致
7. final norm 后的每尺度 hidden state 是否一致

### 与已有 helper 测试的关系

已有测试已经分别锁定：
- MSP resize teacher forcing 与推理输入规则
- scale ID
- block mask rows
- RoPE block slice
- global offset

新测试把这些组件放进真实 Gemma attention/cache forward 中做端到端数值比较。

### 当前验证状态

已通过：
- `py_compile`
- `git diff --check`
- 文件行宽检查

尚未执行数值测试：
- 本地 `openpi/.venv` 没有 JAX/pytest
- 未为此临时下载 CUDA JAX 或改变用户训练环境

服务器 openpi 环境执行：
- `pytest openpi/src/openpi/models/msp_gemma_cache_test.py -q`

同时建议一次执行本轮全部相关测试：
- `pytest openpi/src/openpi/models/msp_scale_head_test.py openpi/src/openpi/models/msp_gemma_cache_test.py openpi/src/openpi/training/weight_loaders_test.py -q`

只有该数值测试通过后，才能确认 action expert 在固定同一输入时 full training forward 与 incremental inference forward 等价；它不解决 teacher forcing 下 GT 输入与真实推理预测输入之间的 exposure bias。

## 2026-08-27 结论收口：Gemma action head 的 block-local RoPE 已经具备，不需要再重写 attention

这一步先核实了 `Gemma` 是否真的还缺 block-local RoPE。

核对结果：
- `openpi/src/openpi/models/gemma.py` 里的 `Attention.__call__(...)` 已经支持：
  - 默认使用全局 `positions`
  - 但如果传了 `rope_positions`，就优先用 `rope_positions`
- 当前 MSP Stage-2 路径在 `pi0.py` 里对 suffix 已经传入：
  - `rope_positions=[None, rope_positions]`
- 也就是说：
  - prefix / VLM 仍走原始全局 RoPE
  - suffix / MSP action head 已经可以走自定义 RoPE 位置

而当前 suffix 提供的 `rope_positions` 是：
- `build_block_local_positions(scales, batch_size=...)`
- 它的语义正是：
  - 每个尺度 block 内部位置从 `0` 重新开始
  - 例如 `(1,2,4,8)` 会得到：
    - `[0 | 0,1 | 0,1,2,3 | 0,1,2,3,4,5,6,7]`

所以结论是：
- 从机制上说，当前实现已经具备你要求的 “新增一个 block-local RoPE”
- 而且不需要改 VLM 部分，也不需要重写 Gemma attention
- 关键只在于：
  - suffix 要稳定传入 block-local `rope_positions`
  - 训练和推理都要保持这个约定

本轮做的事情：
1. 在 `pi0.py` 里补明确注释
- 文件：`openpi/src/openpi/models/pi0.py`
- 明确写清：
  - 全局 `positions` 继续服务于 sequence / KV cache 排布
  - `rope_positions` 单独覆盖为 per-scale local positions

2. 新增 block-local RoPE 回归测试
- 文件：`openpi/src/openpi/models/msp_scale_head_test.py`
- 新增测试：
  - `build_block_local_positions_resets_inside_each_scale_block()`
- 锁定 `(1,2,4,8)` 时的位置为：
  - `[0, 0,1, 0,1,2,3, 0,1,2,3,4,5,6,7]`

当前结果：
- 这一步没有再去改 `Gemma` 模块结构
- 结论是：现有 `rope_positions` 覆盖路径已经足够实现 MSP 所需的 block-local RoPE
- 现在 Stage-2 在 mask、三组 learned positional embeddings、以及 block-local RoPE 这三块都已经有了明确实现路径

本轮验证：
- `openpi/.venv/bin/python -m py_compile openpi/src/openpi/models/pi0.py openpi/src/openpi/models/msp_scale_head_test.py`
- `git diff --check -- openpi/src/openpi/models/pi0.py openpi/src/openpi/models/msp_scale_head_test.py`

本轮修改文件：
- `openpi/src/openpi/models/pi0.py`
- `openpi/src/openpi/models/msp_scale_head_test.py`
- `task.md`

## 2026-08-27 Stage-2 指标补充：输出 per-scale loss 到 TensorBoard / wandb

这一步的目标不是改训练目标，而是把 MSP Stage-2 的 coarse-to-fine 学习状态直接暴露出来。

现状：
- `train.py` 和 `train_tb.py` 本来就支持记录 `model.compute_loss_with_info(...)` 返回的 `metric_info`
- 之前 Stage-2 的 `Pi0(use_msp_action_head=True)` 只返回总的 latent token loss，没有把各尺度拆出来

本轮修改：
1. `Pi0` 新增 MSP 专用的带指标损失路径
- 文件：`openpi/src/openpi/models/pi0.py`
- 新增：
  - `_compute_msp_loss_with_info(...)`
- 逻辑：
  - 先保持原有 Stage-2 latent MSE 不变
  - 再把 `token_loss` 按 `msp_scales` 切成多个 block
  - 对每个 block 求均值，作为单独指标输出

2. `compute_loss()` 保持原训练目标不变
- 文件：`openpi/src/openpi/models/pi0.py`
- `use_msp_action_head=True` 时：
  - `compute_loss()` 只是调用 `_compute_msp_loss_with_info(...)[0]`
- 所以反向传播使用的仍然是原来的：
  - multi-scale latent prediction MSE

3. `compute_loss_with_info()` 接入 Stage-2 指标
- 文件：`openpi/src/openpi/models/pi0.py`
- `use_msp_action_head=True` 时，返回：
  - `token_loss`
  - `metric_info`

新增指标：
- `msp_loss_total`
  - 所有尺度 token loss 的整体均值
- `msp_loss_finest`
  - 最细尺度 block 的均值
- `msp_loss_scale_<scale>`
  - 每个尺度 block 的均值
  - 例如 `(1,2,4,8)` 时会有：
    - `msp_loss_scale_1`
    - `msp_loss_scale_2`
    - `msp_loss_scale_4`
    - `msp_loss_scale_8`

这意味着现在训练时你能直接观察：
- 最粗尺度是否先收敛
- 中间尺度是否跟上
- 最细尺度是否迟迟不降

日志链路说明：
- `train_tb.py` 已经会把 `metric_info` 里的每一项写入 TensorBoard
- `train.py` 也会把同样的指标写入 wandb
- 所以这一步不需要改训练脚本，只改模型输出即可

本轮验证：
- `openpi/.venv/bin/python -m py_compile openpi/src/openpi/models/pi0.py`
- `git diff --check -- openpi/src/openpi/models/pi0.py`

本轮修改文件：
- `openpi/src/openpi/models/pi0.py`
- `task.md`

## 2026-08-27 Stage-2 位置编码语义收紧：训练全长，推理按尺度段切片

这一步处理的是你前面反复强调的那组 MSP 细节：
- `encoder_pos_embed_learned`
- `decoder_pos_embed_learned`
- `diffusion_pos_embed_learned`
- 以及推理时的 `start:end` 切片语义

核对结果：
- 这三组 learned positional embeddings 之前其实已经接进 `pi0.py`
- 但训练路径是通过 `0:total_len` 取全长，功能上没问题，语义上不够明确
- 推理路径虽然也在切 `block_start:block_end`，但没有被抽象成统一规则，后续容易被改乱

本轮修改：
1. 统一出“训练全表 / 推理切当前 block” 的位置切片 helper
- 文件：`openpi/src/openpi/models/msp_scale_head.py`
- 新增：
  - `slice_scale_positions(pos_embed, scales, block_index)`
- 规则：
  - `block_index=None` -> 返回整张位置表，对应训练
  - `block_index=i` -> 返回第 `i` 个尺度 block 的位置切片，对应推理

2. `pi0.py` 改成显式按 `block_index` 取三组 learned pos embed
- 文件：`openpi/src/openpi/models/pi0.py`
- `_msp_pos_slice(...)` 不再接 `start/end`
- 改为通过：
  - `block_index=None` 表示训练全长
  - `block_index=block_idx` 表示推理当前尺度段

3. 训练路径语义改清楚
- 文件：`openpi/src/openpi/models/pi0.py`
- `_compute_msp_loss()` 中：
  - `encoder_pos_embed`
  - `decoder_pos_embed`
  - `diffusion_pos_embed`
  都显式走 `block_index=None`
- 这对应 MSP 原版里的 `training=True` 使用整张 learned position table

4. 推理路径语义改清楚
- 文件：`openpi/src/openpi/models/pi0.py`
- `_sample_msp_actions()` 中每个尺度 block：
  - `encoder_pos_embed`
  - `decoder_pos_embed`
  - `diffusion_pos_embed`
  都显式走 `block_index=block_idx`
- 这对应 MSP 原版里的：
  - `[:, start:end]`

5. 增加 shape 断言
- 文件：`openpi/src/openpi/models/pi0.py`
- 训练时断言：
  - `diffusion_pos.shape[1] == total_suffix_len`
- 推理时断言：
  - `diffusion_pos.shape[1] == current_scale_len`
- 这样位置切片一旦和 block 长度不一致，会立即暴露

6. 新增位置切片回归测试
- 文件：`openpi/src/openpi/models/msp_scale_head_test.py`
- 新增测试覆盖：
  - `block_index=None` 时返回完整位置表
  - `block_index=i` 时返回全表的对应 `start:end` 切片

当前结果：
- 这一步没有改变 Stage-2 的总体结构
- 但把 MSP 原版“训练全长 / 推理分段切片”的位置编码语义固定下来了
- 后面做 block-local RoPE 时，可以直接在这套分段语义上继续，不需要再返工这三组 learned pos embed

本轮验证：
- `openpi/.venv/bin/python -m py_compile openpi/src/openpi/models/msp_scale_head.py openpi/src/openpi/models/pi0.py openpi/src/openpi/models/msp_scale_head_test.py`
- `git diff --check -- openpi/src/openpi/models/msp_scale_head.py openpi/src/openpi/models/pi0.py openpi/src/openpi/models/msp_scale_head_test.py`

本轮修改文件：
- `openpi/src/openpi/models/msp_scale_head.py`
- `openpi/src/openpi/models/pi0.py`
- `openpi/src/openpi/models/msp_scale_head_test.py`
- `task.md`
## 2026-09-01 任务 3 扩展：移植 MSP Stage-2 MeanFlow head

新的理解：
- `MSP/algos/flow/flow_ar.py` 的 Transformer 不直接回归 latent；它输出每个尺度、每个 token 的 condition。
- condition 和带噪 latent 一起送入 `MPScalseFlowhead`。该 head 实际采用 MeanFlow 训练：log-normal 采样 `t/r`、JVP 构造目标、adaptive L2 优化。
- 推理不是多步 ODE，而是原版 `generate()` 的单步更新：从高斯噪声 `sample` 得到 `sample - u(sample, t=1, r=0, condition)`。
- 每个尺度分别计算 flow loss，再按原版 `scale / max_scale` 加权求和。
- 逐尺度推理时，当前尺度生成的连续 latent 仍按 MSP 的“两次 resize”规则构造下一尺度 Transformer 输入；最终 finest latent 送 Stage-1 VAE decoder。

本轮任务：
1. 新增 JAX/Flax 版 `MpScaleMlpResNet` 和 `MPScalseFlowhead`。
2. 用 MeanFlow loss 替换当前 Stage-2 的直接 latent MSE。
3. 用单步 MeanFlow generation 替换当前 `msp_latent_out_proj` 直接输出 latent。
4. 保持已有 Gemma 多尺度 mask、block-local RoPE、KV cache、MINT 风格 SOS/scale embedding 不变。
5. 修复当前 `_compute_msp_loss_with_info()` 漏接 `scale_ends` 的问题。

计划修改文件：
- `openpi/src/openpi/models/msp_flow_head.py`（新增）
- `openpi/src/openpi/models/msp_flow_head_test.py`（新增）
- `openpi/src/openpi/models/pi0.py`
- `openpi/src/openpi/models/pi0_config.py`
- `task.md`

完成情况：
1. 新增 JAX `MspScaleFlowHead`
- `LearnedPosEmb`：`randn(std=0.2)` learned Fourier embedding，输出顺序与 PyTorch 一致为 `cos/sin`。
- `TimeEncoder`：同一组参数分别编码 `t` 和 `r`，再相加。
- `MlpResNetBlock`：`Dropout -> LayerNorm -> Dense(4H) -> GELU -> Dense(H) -> residual`。
- head 默认保持原版参数：3 blocks、hidden 256、time dim 32、time hidden 256、dropout 0.1。
- pi0.5 action expert 的 hidden width 只作为 condition dimension，不擅自放大 flow hidden width。

2. Stage-2 训练改为原版 MeanFlow
- Stage-1 VAE sampled latent 继续 `stop_gradient`。
- Gemma action expert 输出作为 flow condition，不再经 `msp_latent_out_proj` 直接回归 latent。
- 每个尺度独立采样高斯 noise 和 log-normal `t/r`。
- 50% batch 样本设置 `r=t`。
- `v_hat = 2 * v - u_t`，其中 guide `u_t` stop-gradient。
- 用 `jax.jvp` 计算方向导数，构造 `u_tgt = v_hat - (t-r) * dudt`。
- 使用 MSP adaptive L2，并继续按 `scale / max_scale` 加权。

3. Stage-2 推理改为原版单步 Flow
- 每个尺度独立采样 `[B, scale, latent_dim]` 高斯噪声。
- Gemma 当前尺度 hidden 作为 condition。
- 执行 `sample - model(sample, t=1, r=0, condition)`。
- 生成 latent 按现有 MSP resize 规则传给下一尺度，finest latent 由 Stage-1 decoder 解码。
- `sample_actions()` 现在实际使用传入的 RNG；不再产生固定 latent。

4. 指标
- 新增 `msp_flow_loss_scale_<scale>`、对应 weighted 指标和 `msp_flow_mse_scale_<scale>`。
- 新增 `msp_flow_loss_total`、`msp_flow_loss_finest`。
- 保留原 `msp_loss_*` 名称作为 MeanFlow loss 的兼容别名，旧 TensorBoard 面板无需修改。

5. 配置和权重
- `Pi0Config` 新增 `msp_flow_*` 参数，默认值对齐 MSP 原版。
- 新 flow 参数路径属于 `msp_flow_head/...`，现有 `.*msp.*` missing regex 会在加载 pi0.5 时跳过并随机初始化。
- Stage-1 checkpoint 仍只覆盖 `msp_action_vae/...`；flow head 按 Stage-2 新参数训练。
- 旧的“直接 latent MSE” Stage-2 checkpoint 与新 flow 结构不是严格 resume-compatible，应从 pi0.5 base + Stage-1 VAE 权重重新启动新 Stage-2 实验。

6. 修复
- 修复 `_compute_msp_loss_with_info()` 只接收 `scale_starts`、却引用 `scale_ends` 的错误。

验证：
- `py_compile` 通过。
- `git diff --check` 通过。
- 新增 `msp_flow_head_test.py`，覆盖 `t/r` 约束、adaptive L2、模块/JVP shape 与有限值、单步生成公式。
- 项目 `.venv` 缺少 JAX 和 pytest，因此使用阿里云 PyPI 镜像在 `/tmp/msp-flow-test` 创建了隔离 CPU 环境，没有改项目环境或锁文件。
- 已运行 `msp_flow_head_test.py + msp_scale_head_test.py`：`11 passed`。
- 测试额外覆盖 `ToNNX` bridge 下的 MeanFlow JVP、参数梯度存在且有限。
- 完整 Pi0.5 + 图像/VLM 训练 smoke test 仍需在服务器正式 openpi 环境和真实数据上运行。

本轮实际修改文件：
- `openpi/src/openpi/models/msp_flow_head.py`
- `openpi/src/openpi/models/msp_flow_head_test.py`
- `openpi/src/openpi/models/pi0.py`
- `openpi/src/openpi/models/pi0_config.py`
- `task.md`
## 2026-09-01 Stage-1 VAE 与原版 MSP 逐层复核

复核基准：
- `MSP/algos/vae/vae.py`
- `MSP/config/algo/action_vae.yaml`
- `MSP/config/algo/MSP.yaml`

确认无需修改的超参：
- 原版实际 YAML（不是 `ActionVAE.__init__` 的未覆盖默认值）使用：
  - encoder/decoder dim = 128
  - downsample factor = 4
  - encoder heads/layers = 2/2
  - decoder heads/layers = 4/4
  - latent dim = 16
  - KL weight = 1e-6
- 当前 Stage-1 和 Stage-2 VAE 配置与这些值一致。

确认需要修正的实现差异：
1. 正弦位置编码
- 原版 `positional_encodings.PositionalEncoding1D` 按频率交错排列：
  `[sin0, cos0, sin1, cos1, ...]`。
- 当前 JAX 版是先拼全部 sin、再拼全部 cos，语义不一致。

2. Transformer residual dropout
- PyTorch `TransformerEncoderLayer/DecoderLayer(norm_first=True)` 在每个 attention 和 FF block 输出后都有 dropout，再与 residual 相加。
- 当前 JAX 版只有 attention-weight dropout 和 FF 中间 dropout，缺少 residual branch 输出 dropout。

3. Attention dropout mask
- PyTorch attention dropout 对 batch/head/query/key 独立采样。
- Flax MHA 默认 `broadcast_dropout=True` 会跨 batch/head 共享，需要设为 `False`。

4. GELU
- PyTorch `activation='gelu'` 使用 exact GELU。
- Flax `nn.gelu` 默认 approximate=True，需要显式 `approximate=False`。

5. Norm epsilon
- PyTorch Transformer LayerNorm 和 GroupNorm 默认 epsilon 都是 `1e-5`。
- Flax 两者默认 `1e-6`，需要显式对齐。

6. 参数初始化
- 原版普通 Linear/Conv 使用 PyTorch default Kaiming-uniform 等价范围 `[-1/sqrt(fan_in), 1/sqrt(fan_in)]`，bias 同范围。
- 原版 MHA Q/K/V 使用 combined in-projection Xavier uniform；out projection 使用 Linear default，attention biases 在 reset 时为 0。
- 当前 Flax 默认是 LeCun normal + zero bias，需要为新训练显式对齐；参数 shape/name 不变，已有 checkpoint 加载不受影响。

7. Stage-2 冻结 VAE 的 train/eval 状态
- 原版只执行 `autoencoder.requires_grad_(False)`，外层每轮仍执行 `model.train()`，所以 Stage-2 编码 latent 时 VAE dropout 仍启用。
- 当前 Stage-2 强制 `train=False`，需要改为跟随 Stage-2 的 `train`，同时继续 stop-gradient 和冻结 VAE 参数。

计划修改文件：
- `openpi/src/openpi/models/msp_vae.py`
- `openpi/src/openpi/models/pi0.py`
- `openpi/src/openpi/models/msp_vae_test.py`（新增）
- `task.md`

完成情况：
1. 已修正位置编码为 MSP 原版交错 sin/cos，并支持原版对奇数 channel 的截断行为。
2. Encoder 每层已对齐：
- pre-LayerNorm epsilon 1e-5
- MHA attention dropout 不跨 batch/head 广播
- attention 输出 dropout 后再 residual
- exact GELU
- FF 中间 dropout + FF 输出 dropout
3. Decoder 每层已对齐：
- self-attention、cross-attention、FF 三个 residual branch 都增加原版输出 dropout
- Norm epsilon、attention dropout 和 GELU 同 Encoder
4. Temporal Conv 已对齐：
- causal 路径保持与 PyTorch“对称 padding 后裁右侧”等价的左 padding 实现
- non-causal 路径改成显式 `kernel//2` 双侧 padding，保留偶数 kernel 时 PyTorch 输出 `L+1` 的行为
- GroupNorm epsilon 改为 1e-5
5. 初始化已对齐：
- Linear/Conv kernel 和 bias 使用 PyTorch default uniform bounds
- MHA Q/K/V 使用原版 combined in-projection Xavier uniform 的边缘分布
- MHA out projection 使用 Linear default，attention bias 为 0
6. Stage-2 VAE 状态已对齐：
- VAE 参数仍由 freeze filter 冻结，latent 仍 stop-gradient
- Stage-2 train 时 VAE encoder dropout 改为启用；eval/inference 时关闭
7. 没有修改原版 YAML 已明确覆盖的 VAE 超参。

新增测试：
- `msp_vae_test.py`
- 覆盖交错位置编码、downsample/latent/decode shape、train/eval dropout、PyTorch 初始化范围、非因果偶数卷积长度。

验证结果：
- `msp_vae_test.py + msp_flow_head_test.py + msp_scale_head_test.py`：`16 passed`
- `py_compile` 通过
- `git diff --check` 通过
- 本地数值测试使用 `/tmp/msp-flow-test` 隔离 CPU JAX/Flax 环境；由于该环境不含完整 openpi/PyTorch 依赖，仅对 import 依赖使用测试进程内模块桩，不影响被测试的 Linen VAE 实现。

checkpoint 结论：
- 本轮没有改变 VAE 参数 key/shape，旧 Stage-1 checkpoint 可以加载。
- 但旧 checkpoint 是在旧位置编码/dropout/epsilon 语义下训练的，不应视为“原版对齐权重”。
- 建议重新训练 Stage-1，并用新 Stage-1 权重重新训练当前 MeanFlow Stage-2。

本轮实际修改文件：
- `openpi/src/openpi/models/msp_vae.py`
- `openpi/src/openpi/models/pi0.py`
- `openpi/src/openpi/models/msp_vae_test.py`
- `task.md`

## 2026-09-01 Stage-2 Flow 推理链路复核与接口补全

新的理解：
- Flow 推理发生在 `Pi0.sample_actions()` 内部：VLM prefix 只编码一次，每个尺度由 Gemma action expert 给出 condition，MeanFlow 从该尺度高斯 latent noise 单步生成 latent，最细尺度再由 Stage-1 VAE decoder 解码动作。
- `deploy.py -> model.py -> Policy.infer()` 是通用部署包装，不应复制 Flow 算法。部署使用 MSP Stage-2 train config 创建模型并恢复完整 checkpoint 后，会自动进入上述分支。
- `Policy.infer()` 每次调用都会 split JAX RNG；未显式传 noise 时，各次推理和各尺度均使用不同高斯噪声。
- VAE 解码先得到归一化动作，模型补齐 openpi 的 32 维接口后，output transform 再反归一化；机器人适配层最后按实际动作维度解包。

发现并修复的问题：
- MSP 分支此前直接丢弃公开推理接口的 `noise` 参数，无法固定噪声复现实验。
- 现在 MSP 模式允许 `noise` 使用 `[B, sum(msp_scales), msp_latent_dim]`，并严格按尺度段切分后分别送入 Flow head。
- 不传 `noise` 时行为不变：使用传入 RNG 为每个尺度独立采样高斯 latent。
- 如果误传原 pi0.5 的 action-space noise（例如 `[B, 32, 14]`），现在会给出明确 shape 错误，不再静默忽略。

本轮修改文件：
- `openpi/src/openpi/models/pi0.py`
- `openpi/src/openpi/models/msp_scale_head.py`
- `openpi/src/openpi/models/msp_scale_head_test.py`
- `task.md`

验证结果：
- `msp_scale_head_test.py + msp_flow_head_test.py`：`13 passed`。
- `py_compile` 和 `git diff --check` 通过。
- 没有真实 Stage-2 checkpoint 和仿真环境，因此本地未执行完整 VLM + Flow + VAE decoder 部署推理；服务器评估必须使用包含 `msp_flow_head/*` 的新 Stage-2 checkpoint，旧的直接 latent-MSE checkpoint 不兼容。

## 2026-09-01 按原版 `FlowAR.sample_tokens()` 收紧推理逻辑

逐行对齐基准：
- `MSP/algos/flow/flow_ar.py::FlowAR.sample_tokens()`
- `MSP/algos/flow/flow_ar.py::forward_mae_decoder()`

确认已对齐的行为：
1. VLM prefix 只前向一次并建立 KV cache，对应原版进入尺度循环前清理并重新建立 attention cache。
2. 每次循环只输入当前尺度 block；当前 block 能看 prefix、全部旧尺度 cache 和当前尺度全部 token。
3. 每个尺度独立采样高斯 latent，并调用原版单步 MeanFlow：`sample - u(sample, t=1, r=0)`。
4. 非最终尺度的生成结果先 resize 到 finest scale，再 resize 到下一尺度，作为下一轮 Transformer 输入。
5. 最终只取 finest latent，送 Stage-1 VAE decoder 得到动作。

发现并修复的差异：
- 原版 decoder 输出在进入 Flow head 前会加 `diffusion_pos_embed_learned`，推理使用 `[:, start:end]`；当前实现此前缺少这一步。
- 新增 `msp_flow_pos_embed`，初始化为原版 `normal(std=0.02)`。
- Stage-2 训练时给整段 Gemma condition 加完整位置表。
- 推理时按当前 `block_index` 对应的累计尺度边界严格切 `start:end`，再送 Flow head。
- 上一尺度生成 latent 在传给下一尺度前显式 `stop_gradient`，对应原版 `z_sample.detach()`；推理数值不变，但语义与原版一致。

没有硬搬的结构：
- 原版有独立 MAE encoder 和 decoder，因此各自需要一组 learned position；pi0.5 适配后只有一个 Gemma action expert，并已使用 block-local RoPE、global cache position 和 MINT level embedding。
- 因此这里只保留与 Flow condition 一一对应的 `msp_flow_pos_embed`，不在同一个 Gemma 输入上重复叠加 encoder/decoder 两组位置参数。

checkpoint 影响：
- 新参数路径为 `msp_flow_pos_embed`，加载原始 pi0.5 base 时由现有 `.*msp.*` 规则跳过并随机初始化。
- 该参数参与 Stage-2 训练，必须使用本次修改后重新训练的 Stage-2 checkpoint；旧 Stage-2 完整 checkpoint 缺少该参数，不能严格恢复为当前结构。

本轮修改文件：
- `openpi/src/openpi/models/pi0.py`
- `openpi/src/openpi/models/msp_scale_head.py`
- `openpi/src/openpi/models/msp_scale_head_test.py`
- `task.md`

验证结果：
- `msp_scale_head_test.py + msp_flow_head_test.py`：`14 passed`。
- `py_compile` 和 `git diff --check` 通过。
- `msp_gemma_cache_test.py` 在临时 CPU 环境收集阶段缺少 openpi 完整依赖；已按阿里云镜像补充 `einops`，随后仍缺少 `beartype/jaxtyping/torch`。本轮未修改 Gemma/cache 路径，因此没有继续安装整套 Torch；完整集成验证留给服务器现有 openpi 环境。

## 2026-09-01 关于正在运行的旧版 Stage-2 训练

结论：
- 最近一次修改不只是推理修改。
- `noise` 分尺度切片和推理尺度循环属于纯推理接口修改，不要求重新训练。
- 但新增的 `msp_flow_pos_embed` 同时接入了 Stage-2 训练的 Flow condition，并新增了可训练参数，因此旧版正在训练的 checkpoint 不包含该参数。
- 已经启动的旧训练进程加载的是启动时的旧 Python 程序，不受工作区后续代码修改影响，可以继续完成；但它产出的 checkpoint 应使用对应旧版代码推理。
- 如果要使用当前包含 `msp_flow_pos_embed[:, start:end]` 的严格对齐版本，Stage-2 应从 pi0.5 base + Stage-1 VAE 重新训练。
- 不建议在加载旧 checkpoint 时随机补 `msp_flow_pos_embed` 后直接推理，因为该位置参数没有参与旧版训练，会给 Flow condition 引入未训练扰动。

## 2026-09-01 暂时关闭 `msp_flow_pos_embed`

需求：
- 先使用当前代码加载并评估上一版正在训练的 Stage-2 checkpoint。
- 等旧版实验完成后，再选择是否启用原版 MSP 风格的 Flow condition learned position。

实现：
- `Pi0Config` 新增 `msp_use_flow_pos_embed: bool = False`，默认关闭。
- `pi05_msp_stage2_aloha_arx-x5_seed_0` 中也显式设置为 `False`，后续实验只需在该配置改成 `True`。
- 关闭时不创建 `msp_flow_pos_embed` 参数，训练和推理的 Flow condition 都保持上一版的原始 `suffix_out`。
- 因为参数树中不存在该新增参数，上一版 Stage-2 checkpoint 可以按原结构严格加载。
- 设置 `msp_use_flow_pos_embed=True` 时，才创建 `msp_flow_pos_embed`，训练使用完整位置表，推理使用当前尺度 `start:end` 切片。

使用约束：
- 旧版 checkpoint 推理必须保持 `msp_use_flow_pos_embed=False`。
- 后续设置为 `True` 后需要重新训练 Stage-2，不能在旧 checkpoint 推理时临时打开，因为旧权重中没有训练过该参数。

本轮修改文件：
- `openpi/src/openpi/models/pi0_config.py`
- `openpi/src/openpi/models/pi0.py`
- `openpi/src/openpi/training/config.py`
- `task.md`

## 2026-09-01 LeRobot 时间戳容差

需求与修改：
- 在 `create_torch_dataset()` 构造普通 `LeRobotDataset` 时增加 `tolerance_s=0.041`。
- fake dataset 和 Stage-1 action-only dataset 仍在 metadata/video dataset 创建前提前返回，不受该参数影响。
- 其他 `delta_timestamps`、video backend 和 prompt transform 逻辑保持不变。

本轮修改文件：
- `openpi/src/openpi/training/data_loader.py`
- `task.md`

## 2026-09-01 Stage-1 KL 指标语义调整（用户修改）

用户已将 `MspActionVAE.compute_loss_with_info()` 返回的指标从：
- `kl_loss = mean(kl_weight * KL)`

改为：
- `kl_loss = mean(KL)`

影响：
- 只改变 TensorBoard/W&B 中 `kl_loss` 曲线的数值和含义。
- `total_loss` 仍为 `recon_loss + kl_weight * KL`，训练梯度和优化目标没有变化。
- 当前 `kl_loss` 表示未加权原始 KL，不能直接当作 KL 对总损失的实际贡献。

用户修改文件：
- `openpi/src/openpi/models/msp_vae.py`

## 2026-09-01 TensorBoard 学习率曲线

修改：
- 在 `train_tb.py::train_step()` 的 `info` 中新增 `lr`。
- 数值使用 `config.lr_schedule.create()(state.step)`，与当前 optimizer update 使用的 schedule 和 step 对齐。
- 现有 TensorBoard 通用指标写入循环会自动生成 `lr` 曲线，不需要额外修改 writer 逻辑。

本轮修改文件：
- `openpi/scripts/train_tb.py`
- `task.md`
