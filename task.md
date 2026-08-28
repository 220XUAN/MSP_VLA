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
