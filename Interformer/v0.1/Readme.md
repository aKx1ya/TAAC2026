# v0.1 — PCVRInterFormer 基线训练与评估报告

## 一、项目概述

本项目基于 **InterFormer** 架构实现 PCVR（Post-Click Conversion Rate）预估模型，面向二分类/排序任务。v0.1 为基线版本，旨在验证模型基础能力并定位训练过程中的潜在问题，为后续版本迭代提供方向。

### 代码结构

```
v0.1/
├── dataset.py       # Parquet 数据集加载、Schema 解析、预处理
├── model.py         # PCVRInterFormer 模型定义（三大模块）
├── trainer.py       # 训练器（双优化器、EarlyStopping、Checkpoint管理）
├── train.py         # 训练入口（参数解析、组件装配）
├── utils.py         # 工具函数（日志、早停、Focal Loss）
├── ns_groups.json   # 非序列特征分组配置
├── run.sh           # 启动脚本（RankMixer/Group 两种模式切换）
├── eva/             # 评估模块（独立推理 & 指标计算）
│   ├── dataset.py
│   ├── infer.py
│   └── model.py
├── Loss.png         # 训练集 Loss 曲线
├── AUC.png          # 验证集 AUC 曲线
├── LogLoss.png      # 验证集 LogLoss 曲线
└── 数据记录.txt      # 实验记录
```

---

## 二、模型架构：PCVRInterFormer

模型遵循 InterFormer 论文的三路拆分设计，由三大核心模块组成：

| 模块 | 名称 | 功能 |
|------|------|------|
| **Interaction Arch** | 非序列交互模块 | 对用户/物品的 ID 类 & Dense 特征做分词（NS Tokenizer），通过自注意力进行跨特征交互 |
| **Sequence Arch** | 序列建模模块 | 编码用户的 4 条行为序列（seq_a ~ seq_d），使用 Transformer/SwiGLU/Longer 编码器，支持时间桶嵌入和正弦位置编码 |
| **Cross Arch** | 跨域融合模块 | 通过 `CrossSummary` 双向汇总序列信息与 NS 信息，经 `InterFormerBlock` 堆叠后由门控机制融合，最终送入分类器输出点击概率 |

### 关键设计细节

- **NS Tokenizer**：v0.1 使用 `rankmixer` 模式，将所有非序列特征 Embedding 拼接后通过线性投影分割为等长 Token（`user_ns_tokens=5`, `item_ns_tokens=2`），无需手动指定特征分组。
- **Query Token**：每个序列域独立生成 2 个 Query Token（`num_queries=2`），用于 `CrossSummary` 的注意力汇聚。
- **PersonalizedFFN**：序列特征通过上下文条件化的前馈网络（由 NS 特征生成 Gate/Bias），实现个性化的序列表示更新。
- **Sinusoidal Position Encoding**：序列位置编码采用标准正弦位置编码。

---

## 三、实验配置

### 3.1 超参数一览

| 参数 | 值 | 说明 |
|------|-----|------|
| `d_model` | 64 | 主干隐藏维度 |
| `emb_dim` | 64 | Embedding 表维度 |
| `num_hyformer_blocks` | 2 | InterFormerBlock 堆叠层数 |
| `num_heads` | 4 | 多头注意力头数 |
| `hidden_mult` | 4 | FFN 内层扩展倍数 |
| `dropout_rate` | 0.01 | 主干 Dropout 率 |
| `num_queries` | 2 | 每序列域的 Query 数量 |
| `user_ns_tokens` | 5 | 用户非序列 Token 数（RankMixer） |
| `item_ns_tokens` | 2 | 物品非序列 Token 数（RankMixer） |
| `seq_max_lens` | seq_a:256, seq_b:256, seq_c:512, seq_d:512 | 各序列域截断长度 |
| `batch_size` | 256 | 训练/验证批大小 |
| `seq_id_threshold` | 10000 | 高基数 ID 特征阈值（超阈值额外 Dropout） |
| `emb_skip_threshold` | 1000000 | Embedding 跳过阈值（超阈值不建表） |

### 3.2 优化器策略（双优化器）

| 参数类型 | 优化器 | 学习率 | 说明 |
|----------|--------|--------|------|
| 稀疏参数（Embedding 层） | **Adagrad** | `sparse_lr=0.05` | 所有 `nn.Embedding` 权重 |
| 稠密参数（其余所有） | **AdamW** | `lr=1e-4` | betas=(0.9, 0.98) |

### 3.3 损失函数

- 类型：**BCEWithLogitsLoss**（`loss_type=bce`）
- 备选：Focal Loss（可通过 `--loss_type focal` 切换）

### 3.4 其他策略

- **Early Stopping**：patience=5，监控指标为验证集 AUC（无 delta 容差）
- **MultiEpoch 冷重启**：从第 1 个 epoch 起，每 epoch 结束后对高基数 Embedding（vocab > `reinit_cardinality_threshold`）重新 Xavier 初始化并重建 Adagrad 状态
- **数据加载**：IterableDataset + Parquet 直读，16 workers，20 batch 的 shuffle buffer
- **序列编码器**：默认 `transformer`（标准自注意力）模式

---

## 四、训练结果

### 4.1 训练时间线

| 事件 | 时间 |
|------|------|
| 训练开始 | 2026-05-04 17:42:24 |
| 最优 Checkpoint | 2026-05-05 15:39:27 |
| 最优 Checkpoint 路径 | `global_step61608.layer=2.head=4.hidden=64.best_model` |

> 训练总耗时约 **22 小时**，共经历 61608+ 步。

### 4.2 核心评估指标

| 指标 | 值 |
|------|-----|
| **Evaluation AUC** | **0.806701** |
| **推理耗时** | 226.89s |

---

## 五、训练过程诊断

结合 TensorBoard 训练曲线（见 `Loss.png`、`AUC.png`、`LogLoss.png`），模型表现出以下关键特征：

### 5.1 训练集 Loss（`Loss/train`）

- **剧烈震荡**：Loss 主体区间在 0.15–0.35，但频繁出现飙升至 0.6–0.8+ 的突刺
- **未形成平滑下降梯度**：表明网络更新过程极不稳定

### 5.2 验证集 AUC（`AUC/valid`）与 LogLoss

- **过早收敛**：AUC 在约 3624 步即达到平台期（~0.85–0.86），此后至 76104 步几乎完全水平
- **LogLoss 同趋势**：极早期微降后彻底走平

### 5.3 核心问题定位

| 问题 | 推测原因 |
|------|----------|
| **稀疏参数学习率过激** | `sparse_lr=0.05` 对海量稀疏特征偏大，Embedding 向量在局部最优附近剧烈震荡 |
| **强特征主导梯度（Shortcut Learning）** | 高频 ID 特征过早拟合目标，深层模块梯度消失或无效学习 |
| **缺乏学习率调度** | 无 Warmup / Cosine Annealing，震荡无法在中后期收敛 |

---

## 六、v0.2 优化规划

基于 v0.1 的诊断结论，建议在 v0.2 中开展以下实验：

### 🔴 高优先级

- **降低稀疏学习率**：将 `sparse_lr`（Adagrad）从 0.05 下调至 **0.01** 或 **0.005**，观察 Loss 毛刺是否缓解
- **引入 LR Scheduler**：添加 Linear Warmup + Cosine Annealing，早期稳定特征表示，中后期平滑收敛

### 🟡 中优先级

- **增强正则化**：适当调高 `dropout_rate`（当前 0.01），或对序列 ID Embedding 施加更强的 Dropout
- **高基数特征惩罚**：降低 `seq_id_threshold`（当前 10000），对更多特征施加额外 Dropout

### 🟢 低优先级

- **特征消融实验**：单独关闭 Sequence Tokenizer 或禁用特定强特征 NS Group，验证复杂网络模块是否实际提供增益
- **尝试 Focal Loss**：将 `loss_type` 从 `bce` 切换为 `focal`，缓解正负样本不均衡

---

## 七、快速复现

```bash
# 使用 run.sh 启动训练（RankMixer 模式）
bash run.sh

# 或直接调用 train.py
python train.py \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --emb_skip_threshold 1000000 \
    --num_workers 8
```

训练数据目录和输出路径通过以下环境变量配置：
- `TRAIN_DATA_PATH`：训练数据目录（含 `*.parquet` 和 `schema.json`）
- `TRAIN_CKPT_PATH`：Checkpoint 输出目录
- `TRAIN_LOG_PATH`：日志输出目录

---

*PCVRInterFormer v0.1 — KDD 2026 TAAC 基线实验*
