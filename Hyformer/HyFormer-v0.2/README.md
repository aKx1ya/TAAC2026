# HyFormer v0.2 — TAAC 2026 优化版

> 基于 HyFormer (Hybrid Transformer) 架构的 Post-Click Conversion Rate (PCVR) 预测模型
>
> 针对 **KDD Cup 2026 TAAC** 竞赛 ~100 万级稀疏数据的专项优化

---

## 📁 文件结构

```
HyFormer-v0.2/
├── dataset.py          # Parquet 数据集加载（时间戳排序 + item_dense 支持）
├── model.py            # PCVRHyFormer 模型（含 ItemFeatureInteraction）
├── train.py            # 训练入口（支持时间戳划分、Focal Loss 等新参数）
├── trainer.py          # 训练器（双优化器：AdamW + Adagrad，冷重启策略）
├── utils.py            # 工具函数（EarlyStopping, Focal Loss, 日志等）
├── run.sh              # 启动脚本（推荐超参数配置）
├── generate_schema.py  # Schema 自动生成（含 item_dense 列扫描）
└── ns_groups.json      # NS 特征分组配置（备选 GroupNSTokenizer 方案）
```

---

## 🚀 快速启动

### 1. 生成 Schema

```bash
python generate_schema.py /path/to/data.parquet ./data/schema.json
```

### 2. 启动训练

```bash
bash run.sh \
    --data_dir /path/to/training_data \
    --schema_path ./data/schema.json \
    --ckpt_dir ./checkpoints \
    --log_dir ./logs \
    --tf_events_dir ./tf_logs
```

### 3. 自定义参数示例

```bash
python train.py \
    --data_dir /path/to/data \
    --ckpt_dir ./ckpt \
    --log_dir ./log \
    --batch_size 512 \
    --d_model 128 \
    --loss_type focal \
    --sort_by_timestamp \
    --valid_time_ratio 0.1
```

---

## ✨ v0.2 核心优化

### 1. 时间戳排序与划分

| 特性 | 说明 |
|---|---|
| **元数据级排序** | 读取 Parquet Row Group 的列统计信息（min/max timestamp），纯元数据操作，不扫描数据行 |
| **时间复杂度** | $O(N_{rg} \log N_{rg})$，对百万级数据（约数千个 RG）几乎零开销 |
| **时间维度划分** | `--valid_time_ratio 0.1` → 最近 10% 时间的数据作为验证集，模拟「历史→未来」的真实预测场景 |
| **向后兼容** | `--no_sort_by_timestamp` 回退到原始文件顺序划分 |

**关键参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--sort_by_timestamp` | bool | `True` | 启用时间戳排序 |
| `--valid_time_ratio` | float | `None` | 时间维度的验证集比例 |

### 2. 稀疏场景参数优化

TAAC 2026 PCVR 数据特点：**百万级样本、标签极度稀疏（正样本率 < 1%）、特征高基数**。

| 优化策略 | 参数设置 | 作用 |
|---|---|---|
| **Focal Loss** | `--loss_type focal --focal_alpha 0.25 --focal_gamma 2.0` | 自动聚焦难分样本，缓解正负样本严重不均衡 |
| **LongerEncoder** | `--seq_encoder_type longer --seq_top_k 50 --seq_causal` | Top-K 压缩长序列 + 因果掩码，高效时序建模 |
| **RoPE** | `--use_rope --rope_base 10000.0` | 旋转位置编码，增强序列位置感知 |
| **高基数 ID Dropout** | `--seq_id_threshold 5000` | vocab > 5K 的特征额外 dropout（×2），防止记忆化 |
| **冷重启** | `--reinit_sparse_after_epoch 1 --reinit_cardinality_threshold 10000` | 每 epoch 结束后重置高基数 Embedding，参考 MultiEpoch 论文 |
| **双优化器** | 稀疏参数 Adagrad (`lr=0.05`) + 稠密参数 AdamW (`lr=1e-4`) | 适配 Embedding 的稀疏梯度特性 |

**完整推荐超参数（见 `run.sh`）：**

```
d_model=128          # 主干维度（↑64→128，更大容量）
batch_size=512       # 批大小（↑256→512，稀疏标签下梯度更稳定）
dropout_rate=0.1     # Dropout（↑0.01→0.1，防过拟合）
emb_dim=64           # Embedding 维度
num_hyformer_blocks=2
num_heads=4          # 注意力头数（d_model/num_heads = 32）
hidden_mult=4        # FFN 扩展倍数
```

**RankMixer 约束验证：**

$$T = N_q \times S + N_{ns} = 1 \times 4 + 12 = 16, \quad d_{model} \bmod T = 128 \bmod 16 = 0 \quad \checkmark$$

其中 $N_{ns} = 7(\text{user}) + 1(\text{user\_dense}) + 4(\text{item}) = 12$

### 3. Item Feature 增强

#### a) Item Dense 特征支持

- `dataset.py`：新增 `item_dense_feats_*` 列的解析、缓冲区和处理逻辑
- `generate_schema.py`：自动扫描 `item_dense_feats_*` 列
- `model.py`：已有 `item_dense_proj` 模块，将 item 数值特征投影为 NS token
- **向后兼容**：旧 schema（无 `item_dense` 字段）自动 fallback 到空 tensor

#### b) ItemFeatureInteraction 模块

新增 `ItemFeatureInteraction` 类，在 item NS tokens 进入 HyFormer 块之前进行**门控双线性特征交叉**：

```
EnhancedItem = ItemTokens + Gate(ItemTokens) ⊙ Σⱼ Wᵢⱼ · ItemTokenⱼ
```

- `W ∈ R^{N×N}`：可学习的 token 间交互权重矩阵（softmax 归一化）
- `Gate(·)`：Sigmoid 门控，自适应控制交互强度
- 当 `item_ns_tokens ≥ 2` 时自动启用

#### c) Item NS Token 数量

| 版本 | `item_ns_tokens` | 说明 |
|---|---|---|
| v0.1/v0.2 原版 | 2 | item 侧表达能力有限 |
| **v0.2 优化版** | **4** | 更细粒度的 item 特征解耦 |

---

## 🔧 全部 CLI 参数

### 数据路径

| 参数 | 默认值 | 环境变量 |
|---|---|---|
| `--data_dir` | — | `TRAIN_DATA_PATH` |
| `--schema_path` | `<data_dir>/schema.json` | — |
| `--ckpt_dir` | — | `TRAIN_CKPT_PATH` |
| `--log_dir` | — | `TRAIN_LOG_PATH` |
| `--tf_events_dir` | — | `TRAIN_TF_EVENTS_PATH` |

### 数据管道

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--batch_size` | 256 | 批大小 |
| `--num_workers` | 16 | DataLoader 工作进程数 |
| `--buffer_batches` | 20 | 随机打乱缓冲区（单位：batch） |
| `--train_ratio` | 1.0 | 使用训练数据的前 N% |
| `--valid_ratio` | 0.1 | Row Group 维度的验证比例 |
| `--valid_time_ratio` | None | ⭐ 时间维度的验证比例 |
| `--sort_by_timestamp` | True | ⭐ 按时间戳排序 Row Group |
| `--seq_max_lens` | `seq_a:256,...` | 各序列域的截断长度 |

### 模型结构

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `--d_model` | **128** | 主干隐藏维度 |
| `--emb_dim` | 64 | 单特征 Embedding 维度 |
| `--num_queries` | **1** | 每序列生成的 Query token 数 |
| `--num_hyformer_blocks` | 2 | HyFormer 块堆叠层数 |
| `--num_heads` | 4 | 注意力头数 |
| `--hidden_mult` | 4 | FFN 隐层倍数 |
| `--dropout_rate` | **0.1** | Dropout 率 |
| `--seq_encoder_type` | **longer** | 序列编码器：swiglu / transformer / longer |
| `--seq_top_k` | 50 | LongerEncoder 保留的最近 token 数 |
| `--seq_causal` | **True** | LongerEncoder 因果掩码 |
| `--use_rope` | **True** | 启用 RoPE 位置编码 |
| `--rank_mixer_mode` | full | RankMixer 模式：full / ffn_only / none |

### NS Tokenizer

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `--ns_tokenizer_type` | rankmixer | NS token 生成方式：group / rankmixer |
| `--user_ns_tokens` | **7** | User NS token 数量 |
| `--item_ns_tokens` | **4** | ⭐ Item NS token 数量 |
| `--ns_groups_json` | `""` | NS 分组 JSON（RankMixer 模式下忽略） |
| `--emb_skip_threshold` | 1000000 | 超过此词表大小的特征跳过 Embedding |

### 损失函数

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `--loss_type` | **focal** | 损失类型：bce / focal |
| `--focal_alpha` | **0.25** | Focal Loss 正类权重 |
| `--focal_gamma` | **2.0** | Focal Loss 聚焦参数 |

### 优化器 & 正则化

| 参数 | 推荐值 | 说明 |
|---|---|---|
| `--lr` | 1e-4 | 稠密参数学习率（AdamW） |
| `--sparse_lr` | 0.05 | 稀疏参数学习率（Adagrad） |
| `--sparse_weight_decay` | **1e-6** | 稀疏参数权重衰减 |
| `--reinit_sparse_after_epoch` | **1** | 从第 N epoch 开始冷重启 |
| `--reinit_cardinality_threshold` | **10000** | 冷重启的基数阈值 |
| `--seq_id_threshold` | **5000** | ID 特征识别阈值 |
| `--patience` | 5 | Early Stopping 耐心值 |

---

## 📊 预期性能

| 指标 | v0.1/v0.2 原版 | v0.2 优化版 | 提升来源 |
|---|---|---|---|
| 数据划分 | 文件顺序 | ⭐ 时间戳排序 | 真实预测场景评估 |
| 训练速度 | 基准 | ≈持平 | 元数据级排序零开销 |
| 稀疏标签处理 | BCE Loss | ⭐ Focal Loss | 正负样本自适应加权 |
| Item 建模 | 2 tokens | ⭐ 4 tokens + 特征交叉 | ItemFeatureInteraction |
| 过拟合控制 | Dropout 0.01 | ⭐ Dropout 0.1 + ID Dropout + 冷重启 | 多重正则化 |
| 序列建模 | Transformer | ⭐ LongerEncoder + RoPE + Causal | Top-K 压缩 + 位置编码 |
| Item 数值特征 | ❌ 不支持 | ⭐ item_dense 支持 | 利用 item 数值属性 |

---

## ⚠️ 注意事项

1. **Schema 兼容性**：新版 `schema.json` 可包含 `item_dense` 字段；旧版 schema 自动兼容（视为空）
2. **RankMixer 约束**：使用 `rank_mixer_mode=full` 时，须确保 `d_model % T == 0`，否则自动报错并提示合法 T 值
3. **时间戳列**：Parquet 数据须包含 `timestamp` 列（int64 类型），否则自动回退到文件顺序划分
4. **内存估计**：`d_model=128, batch_size=512` 约需 12-16 GB GPU 显存；若 OOM 可降低 `batch_size` 至 256

---

## 📚 参考资料

- HyFormer 架构：基于 RankMixer 的混合 Transformer 推荐模型
- MultiEpoch 冷重启：[MultiEpoch: Reusing Training Data for CTR Prediction](https://arxiv.org/pdf/2305.19531)
- Focal Loss：[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- RoPE：[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
