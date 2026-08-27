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

## 2026-08-27 Git 操作记录

当前用户要求：
- 先把现在这版代码推到 `220XUAN/MSP_VLA.git`

本次提交范围判断：
- 只提交 `policy/Pi_05` 下这次 MSP 接入相关修改
- 不把未跟踪的 `policy/Pi_05/MSP/` 参考目录一起提交，避免把本地参考实现混入本次适配提交
