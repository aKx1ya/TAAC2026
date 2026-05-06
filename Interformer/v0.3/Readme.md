# v0.3 — PCVRInterFormer 超参调优实验报告

## 一、项目概述

v0.3 在 v0.1 基线基础上进行了一系列超参数调整实验，旨在缓解 v0.1 中观察到的**过早收敛**与**训练震荡**问题。主要调整方向包括：切换损失函数、调整序列编码器、增大 Batch Size、启用高基数特征重置策略以及限制训练轮数。

### 代码结构

```
v0.3/
├── dataset.py       # Parquet 数据集加载（与 v0.1 一致）
├── model.py         # PCVRInterFormer 模型定义（⚠️ 存在 seq_encoder_type 参数未生效的 Bug）
├── trainer.py       # 训练器（与 v0.1 一致）
├── train.py         # 训练入口（多个默认参数调整）
├── utils.py         # 工具函数（与 v0.1 一致）
├── ns_groups.json   # 非序列特征分组配置（与 v0.1 一致）
├── run.sh           # 启动脚本（num_workers、batch_size、buffer_batches 调整）
├── Loss.png         # 训练集 Loss 曲线
├── AUC.png          # 验证集 AUC 曲线
├── LogLoss.png      # 验证集 LogLoss 曲线
└── 数据记录.txt      # 实验记录
```

---

## 二、v0.1 → v0.3 变更总览

### 2.1 超参数对比表

| 参数 | v0.1 | v0.3 | 变更说明 |
|------|------|------|----------|
| `loss_type` | `bce` | **`focal`** | 切换为 Focal Loss 以应对正负样本不平衡 |
| `focal_alpha` | 0.1 | 0.1 | 未调整 |
| `focal_gamma` | 2.0 | 2.0 | 未调整 |
| `seq_encoder_type` | `transformer` | **`swiglu`** | 尝试切换序列编码器（⚠️ 未生效，见下方 Bug 说明） |
| `batch_size` | 256 | **1024** | 扩大 4 倍 |
| `num_epochs` | 999 | **3** | 严格限制最大训练轮数 |
| `num_workers` | 8 | **4**（run.sh） | DataLoader 线程数减半 |
| `buffer_batches` | 20（默认） | **20**（run.sh 显式指定） | 与 v0.1 实际值一致 |
| `reinit_cardinality_threshold` | 0（关闭） | **10000** | 启用高基数特征冷重启 |
| `dropout_rate` | 0.01 | 0.01（train.py 默认改为 0.1，但 run.sh 未显式传入，实际取值需确认） | -- |

> **注意**：`train.py` 中 `--dropout_rate` 的默认值从 0.01 改为 **0.1**，但 `run.sh` 中未显式传入该参数。由于 `run.sh` 通过 `"$@"` 传递额外参数，若实际运行时未通过命令行覆盖，则使用 train.py 的默认值 0.1。

---

## 三、各项变更详解

### 3.1 损失函数切换：BCE → Focal Loss

| 方面 | v0.1 (BCEWithLogits) | v0.3 (Focal Loss) |
|------|----------------------|-------------------|
| 动机 | 标准二分类损失 | 应对正负样本极度不平衡 |
| 公式 | $\mathcal{L}_{\text{BCE}} = -\log(p_t)$ | $\mathcal{L}_{\text{Focal}} = -\alpha_t (1-p_t)^\gamma \log(p_t)$ |
| 参数 | -- | $\alpha=0.1, \gamma=2.0$ |

**潜在问题**：若 `focal_alpha` 和 `focal_gamma` 未经过精细调参，Focal Loss 的调制因子 $(1-p_t)^\gamma$ 会大幅压缩易分样本的梯度，使整体梯度变小，配合 `sparse_lr=0.05` 可能导致模型更难收敛。

### 3.2 序列编码器尝试切换：⚠️ 代码 Bug（未生效）

v0.3 将 `train.py` 中 `--seq_encoder_type` 的默认值从 `transformer` 改为 `swiglu`，试图使用 SwiGLU（无注意力机制）替代标准自注意力编码器。

**Bug 说明**：`model.py` 中 `PCVRInterFormer.__init__` 的第一行即执行：

```python
del num_queries, seq_encoder_type, seq_top_k, seq_causal, rank_mixer_mode, use_rope, rope_base, seq_id_threshold
```

这意味着传入的 `seq_encoder_type='swiglu'` 参数在模型构造时被直接删除，**模型实际运行的依然是默认的 Transformer 注意力机制**。这一修改并未真正改变底层模型结构。

### 3.3 Batch Size 大幅调整：256 → 1024

| 方面 | v0.1 | v0.3 |
|------|------|------|
| Batch Size | 256 | 1024 |
| 每 Epoch 更新次数 | ~N/256 | **~N/1024（减少为 1/4）** |
| num_workers | 8 | 4（配合大 Batch 避免 OOM） |
| buffer_batches | 默认 20 | 显式指定 20 |

**影响分析**：Batch Size 扩大 4 倍意味着每个 Epoch 的参数更新次数减少为原来的 1/4。在 `num_epochs=3` 的限制下，总更新步数大幅减少。若学习率未等比例调整（如线性缩放法则 $lr \propto \sqrt{\text{batch\_size}}$），模型可能处于欠拟合状态。

### 3.4 启用高基数特征冷重启

| 参数 | v0.1 | v0.3 |
|------|------|------|
| `reinit_cardinality_threshold` | 0（关闭） | **10000** |
| 触发条件 | 永不触发 | 每 Epoch 结束时，vocab_size > 10000 的 Embedding 会被重新 Xavier 初始化 |
| 策略来源 | -- | [MultiEpoch 冷重启](https://arxiv.org/pdf/2305.19531) |

**潜在问题**：在仅训练 3 个 Epoch 的限制下（第 1、2、3 个 epoch 结束后各触发 1 次），频繁重置 Embedding 可能破坏模型刚刚学到的特征记忆，尤其对高频 ID 特征的表示学习造成干扰。

### 3.5 限制最大训练轮数

| 参数 | v0.1 | v0.3 |
|------|------|------|
| `num_epochs` | 999（完全依赖 Early Stopping） | **3**（硬限制） |
| 实际运行 | 多轮训练至收敛 | 最多 3 轮即停止 |

---

## 四、训练结果

### 4.1 训练时间线

| 事件 | 时间 |
|------|------|
| 训练开始 | 2026-05-04 17:42:24 |
| 最优 Checkpoint | 2026-05-05 15:39:27 |
| 最优 Checkpoint 路径 | `global_step61608.layer=2.head=4.hidden=64.best_model` |

### 4.2 核心评估指标

| 指标 | v0.1 | v0.3 | 变化 |
|------|------|------|------|
| **Evaluation AUC** | 0.806701 | **0.803814** | ↓ -0.0029 |
| **推理耗时** | 226.89s | **351.47s** | ↑ +124.58s |

---

## 五、结果退化分析

v0.3 的一系列超参调整**并未带来正向收益**，AUC 从 0.8067 下降至 0.8038，推理耗时反而增加了约 55%。核心原因分析如下：

### 5.1 Focal Loss 参数未调优

- Focal Loss 的调制因子 $(1-p_t)^\gamma$ 会使模型"聚焦"于难分样本，同时压低易分样本的梯度贡献
- 在 `sparse_lr=0.05` 仍然偏高的情况下，整体梯度信号的减弱使得模型更难在参数空间中稳定移动
- `focal_alpha=0.1` 对正类赋予的权重较小，可能进一步加剧了正样本学习不足

### 5.2 Batch Size 扩大导致欠拟合

- 每 Epoch 更新次数减少为 1/4，而学习率未按缩放法则调整（$lr$ 应随 batch size 增大而增大）
- 在 `num_epochs=3` 的限制下，总更新步数极低，模型远未达到收敛状态

### 5.3 冷重启策略破坏已学表示

- `reinit_cardinality_threshold=10000` 会在每个 Epoch 结束时重置高基数 Embedding
- 3 轮训练中的频繁重置破坏了模型对长尾特征的记忆，导致指标直接掉点

### 5.4 seq_encoder_type 修改因 Bug 未生效

- 本想用 SwiGLU 替代 Transformer 编码器，但由于 `del` 语句的存在，模型结构未发生任何变化
- 该修改对实验结果无实际影响（既不正向也不负向）

---

## 六、关键发现与警示

### 🔴 Bug 警示：`seq_encoder_type` 参数被 `del` 删除

`model.py` 第 480 行的 `del seq_encoder_type, ...` 语句导致该参数被静默丢弃，序列编码器始终使用 `InterFormerBlock` 中的默认自注意力机制。**在 v0.4 及后续版本中，如需使用 SwiGLU/Longer 等非标准编码器，必须修复此 Bug。**

### 🟡 推理耗时增加的可能原因

推理时间从 226.89s 增加到 351.47s（+55%），可能与以下因素有关：
- 更大的 Dropout 率（0.1 vs 0.01）导致模型在推理时仍需更多计算（尽管 Dropout 在 eval 模式下关闭，但模型权重分布可能不同）
- 环境差异（GPU 负载、显存状态）

---

## 七、v0.4 优化建议

基于 v0.3 的实验结论，后续版本应重点关注：

### 🔴 高优先级

1. **修复 `del` Bug**：在 `model.py` 中保留 `seq_encoder_type` 参数，使其能够真正控制序列编码器类型
2. **精细调参 Focal Loss**：对 `focal_alpha` 和 `focal_gamma` 进行网格搜索或贝叶斯优化
3. **降低稀疏学习率**：将 `sparse_lr` 从 0.05 降至 0.01 或 0.005，配合 Focal Loss 的梯度缩放效应
4. **引入 LR Scheduler**：添加 Linear Warmup + Cosine Annealing

### 🟡 中优先级

5. **合理设置冷重启参数**：适当提高 `reinit_cardinality_threshold`（如 50000+）或仅在首轮后触发，减少对已学表示的破坏
6. **Batch Size 与 LR 联动**：若继续使用大 Batch Size，按缩放法则同步调整学习率

### 🟢 低优先级

7. **特征消融实验**：单独评估各序列域和非序列特征组对 AUC 的贡献
8. **尝试混合精度训练**：加速训练并节省显存

---

## 八、快速复现

```bash
# 使用 run.sh 启动训练（v0.3 配置）
bash run.sh

# 或直接调用 train.py
python train.py \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --emb_skip_threshold 1000000 \
    --num_workers 4 \
    --buffer_batches 20 \
    --batch_size 1024
```

训练数据目录和输出路径通过以下环境变量配置：
- `TRAIN_DATA_PATH`：训练数据目录（含 `*.parquet` 和 `schema.json`）
- `TRAIN_CKPT_PATH`：Checkpoint 输出目录
- `TRAIN_LOG_PATH`：日志输出目录

---

*PCVRInterFormer v0.3 — 超参调优实验（KDD 2026 TAAC）*
