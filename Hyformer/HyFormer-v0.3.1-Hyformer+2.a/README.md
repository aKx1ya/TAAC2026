# HyFormer-v0.3.1 — Hyformer + 1.a + 1.b + 2.a
> 当前版本：**v0.3.1 · 1.a + 1.b + 2.a** | 在 1.a+1.b 基础上叠加 Step 2a（平铺型 KV 加权 Embedding）
>
> 🔬 用于消融实验：对比 1.a+1.b vs 1.a+1.b+2.a 的效果

---

## 项目文件结构

| 文件 | 用途 |
|------|------|
| `model.py` | 模型定义（PCVRInterFormer、Tokenizer、InterFormerBlock 等） |
| `train.py` | 训练入口（参数解析、数据加载、模型构建、训练启动） |
| `trainer.py` | 训练循环（双优化器、early stopping、checkpoint 管理） |
| `dataset.py` | 数据加载（Parquet 读取、特征 schema、时间分桶） |
| `utils.py` | 工具函数（日志、EarlyStopping、Focal Loss） |
| `run.sh` | 一键训练脚本 |
| `schema.json` | 全量数据特征 schema（线上训练日志导出） |
| `ns_groups.json` | 非序列特征分组配置 |
| `dataset-analysis.md` | 对 schema.json 的深度分析文档（含 7 个已验证洞察 + 7 个优化方向） |

---

### 评估指标

- **主指标**：Binary AUROC（越高越好）
- **辅助指标**：Binary LogLoss（越低越好）
- 训练时通过 `EarlyStopping` 监控验证集 AUC，patience=5

---

## 🔬 逐步优化清单

> **核心原则**：每次只改一处 → 跑全量训练 + 评估 → 确认提升后再改下一处。
>
> 每个步骤的格式：**改动内容** | **涉及文件** | **预期影响** | **验证方式**。

---

### 📋 Step 0：Baseline 基准
**目标**：记录优化前的 baseline 性能，作为后续所有对比的参照系。

### 📋 Step 1：Item 统计特征构造 🔴 最高优先级

**目标**：填补 `item_dense_dim = 0` 的空白，让模型感知商品的"热度"和"历史转化倾向"。

#### 1a. Item 对数频次 ✅ 已实现（方案 B：在线计算）

**原理**：统计每个 item_id（fid=16）在训练集中的出现次数，取 `log(1+count)` 作为新的 item dense 特征。热门商品与长尾商品的区分本身就是强信号。

**实现方式**：**完全在线**——`dataset.py` 初始化时自动扫描训练 Row Group 统计 item 频次，构建内存查找表。验证集自动复用训练集的统计结果（防标签泄露）。无需任何预处理脚本或外部文件。

**涉及文件**：
| 文件 | 改动 |
|------|------|
| `schema.json` | 新增 `"item_dense": [[200, 1]]`（定义 1 维 log1p item freq 特征） |
| `dataset.py` | `_build_item_freq_map()` 在线扫描训练集 → 构建 freq map → 注入 item_dense |
| `train.py` | **无需改动** |
| `run.sh` | **无需改动** |

**扫描结果**：  
- 训练集：扫描了 907,381 行数据，发现了 20,898 个独立的商品 (Unique items)。这告诉你训练集中有交互的商品池大约是 2 万个。  
- 验证集：扫描了 102,619 行数据，发现了 11,361 个独立的商品。  
- 总参数量：239,158,977  
- Dense 参数：1,710,913  

#### 1b. 贝叶斯平滑 CTR ✅ 已实现（与 1a 融合）

**原理**：统计每个 item 的历史转化率（label_type==2 的比例），用全局 CTR 做贝叶斯平滑。

$$ \text{smooth\_ctr} = \frac{\text{clicks} + \alpha \cdot \text{global\_ctr}}{\text{impressions} + \alpha} $$

**实现方式**：同 1.b 版本。`dataset.py` 初始化时自动扫描训练集统计 impressions 和 clicks，计算贝叶斯平滑 CTR，作为 `item_dense` 第二列（fid=201）注入。

**涉及文件**：
| 文件 | 改动 |
|------|------|
| `schema.json` | 新增 `"item_dense": [[200, 1], [201, 1]]`（两列：对数频次 + 贝叶斯CTR） |
| `dataset.py` | `_build_item_freq_map()` + `_build_item_ctr_map()` 在线扫描 → 双特征向量化注入 |
| `train.py` | **无需改动** |
| `run.sh` | **无需改动** |

**item_dense 结构**：
| 列 | fid | 含义 | 来源 |
|:--:|:---:|------|:----:|
| 0 | 200 | $\log(1+\text{count})$ 对数频次 | Step 1a |
| 1 | 201 | $\text{smooth\_ctr}$ 贝叶斯平滑 CTR | Step 1b |

**预期影响**：
- `item_dense_dim`：`0 → 2`
- AUC 预期提升：**+0.003~0.008**（两者叠加）

**验证方式**：对比 Baseline / 1.a / 1.b 的 AUC / LogLoss。

---

### 📋 Step 2：(Key, Value) 加权 Embedding 🔴 最高优先级

**目标**：利用 8 对 (int, dense) 对齐特征的结构信息，用 dense 值（统计量）作为 attention 权重对 int 的 embedding 做加权聚合，替代 baseline 中完全分离的处理方式。

**背景**：`dataset-analysis.md` 发现 2 — 8 对 fid 的 int.length 与 dense.dim 完全一致，形成 (key=实体ID, value=统计量) 结构。Baseline 将两者完全分离处理，浪费了最强的结构先验。

**涉及文件**：`model.py`（主要）、`dataset.py`（可能需要预处理 dense 值）

#### 2a. 平铺型对齐（fid 89/90/91）✅ 已实现

**原理**：3 对均匀 10×10 特征（fid 89/90/91），用 user_dense 值作为 softmax 权重对 user_int embedding 做加权聚合，替代原来的平均 pooling。

**实现方式**：
- 修改 `RankMixerNSTokenizer` 和 `GroupNSTokenizer`：新增 `weighted_fid_config` 参数和 `dense_feats` 可选入参
- 当 fid 在配置中且有 dense 值时，使用 `softmax(log1p(dense_vals))` 加权 pooling
- 否则回退到原来的 mean pooling（**向后兼容**）

**核心代码**：
```python
# 对每对 (int_ids, dense_vals):
emb = embedding(int_ids)                    # [batch, 10, emb_dim]
w = softmax(log1p(dense_vals))              # [batch, 10]  —— log1p 防极端值
weighted = (emb * w.unsqueeze(-1)).sum(1)   # [batch, emb_dim]
```

**涉及文件**：
| 文件 | 改动 |
|------|------|
| `model.py` | `RankMixerNSTokenizer` + `GroupNSTokenizer` 新增加权 pooling 分支；`PCVRHyFormer` 构建 `_kv_weighted_config` 并传递 `dense_feats` |
| `train.py` | **无需改动** |
| `dataset.py` | **无需改动** |

**预期影响**：平铺型 3 对的特征表示质量提升，AUC 预期 +0.001~0.003

**验证方式**：对比 1.a+1.b（无 2.a）vs 当前版本（有 2.a）的 AUC / LogLoss。

#### 2b. 金字塔型层次化融合（fid 62→66）

**原理**：5 层从粗到细的兴趣层次。粗粒度（fid 62, vocab=11）的加权结果作为细粒度（fid 66, vocab=1403）的 attention bias。

```
coarse (fid=62) → mid (fid=63) → mid-fine (fid=64) → fine (fid=65) → finest (fid=66)
```

**涉及文件**：`model.py` — 需要在 NS tokenizer 中新增层次化融合逻辑

**预期影响**：
- 论文级别的创新点
- AUC 预期提升：**+0.003~0.008**

**验证方式**：
```bash
bash run.sh
# 对比 AUC / LogLoss；确认参数量增加可控
```

---

### 📋 Step 3：超大 Seq 特征频率截断 🟡 中优先级

**目标**：对 vocab > 100 万的 seq 特征做 Top-K 频率截断，释放 VRAM 瓶颈，同时将"完全丢弃"的信息部分恢复。

**背景**：当前 `emb_skip_threshold=1000000` 导致以下特征的 embedding 被零向量替代：

| 域 | fid | vocab | 状态 | 参数量 |
|----|-----|-------|------|--------|
| seq_c | 47 | 86,335,515 | ❌ 跳过（零向量） | ~8,633 万 |
| seq_b | 69 | 64,710,562 | ❌ 跳过（零向量） | ~6,471 万 |
| seq_c | 29 | 5,764,358 | ❌ 跳过（零向量） | ~576 万 |
| seq_c | 34 | 1,031,305 | ❌ 跳过（零向量） | ~103 万 |

**做法**：
1. 在 `dataset.py` 中新增频率统计阶段：扫描训练集 Row Group，统计每个超大 vocab 特征的 ID 频次分布
2. 只保留 Top-K（K=100,000）高频 ID，其余映射到统一的 OOV 桶（index=1）
3. 将频次映射表写入文件（如 `freq_map.json`），推理时复用

**涉及文件**：`dataset.py`（频率统计 + ID 映射）、`model.py`（调整 emb_skip_threshold 逻辑）

**预期影响**：
- 总参数量：**2.4 亿 → ~8000 万**（压缩 67%）
- VRAM 使用显著下降，为 Step 1/2 的新增参数腾出空间
- 同时恢复了部分原本被零向量替代的特征信息
- AUC 预期：**不降或微升**（比零向量好）

**验证方式**：
```bash
bash run.sh
# 重点监控：参数量变化、单 step 速度、AUC 是否下降
```

---

### 📋 Step 4：绝对时间差特征 🟡 中优先级

**目标**：让模型感知"这条行为距离现在有多久"（行为新鲜度），增强序列时序建模。

**背景**：4 个 seq 域的时间戳 fid（39/67/27/26）vocab 均为 0，说明绝对时间在线上不可用。必须基于相对时间差构建特征。

**做法**：
1. 在 `dataset.py` 中：计算每个序列位置距当前预测时刻的时间差，用 `BUCKET_BOUNDARIES` 做分桶
2. 在 `model.py` 中：新增一个 `abs_time_emb`（nn.Embedding），加到序列 token 上与现有 inter_time_emb 并列

**涉及文件**：`dataset.py`、`model.py`

**预期影响**：
- 参数量增加：~65 × d_model × 4 域 ≈ 16KB（可忽略）
- AUC 预期提升：**+0.001~0.003**

**线上兼容性**：仅使用相对时间差，不依赖绝对时间戳 ✅

---

### 📋 Step 5：多域序列独立投影 🟡 中优先级

**目标**：为不同 seq 域分配独立的投影层，避免高基数域（B/C）的噪声污染低基数域（A/D）的有效信号。

**背景**：当前 4 个 seq 域共享同一套 Transformer 层权重，但 seq_b（14 特征，含 6471 万 vocab）和 seq_a（9 特征，含 75 万 vocab）的数据分布差异极大。

**做法**：
- 检查 `model.py` 中 `_seq_proj` 是否已按 domain 独立
- 如果已独立（代码中 `self.sequence_tokenizers` 是 `nn.ModuleDict`，每个 domain 已有独立 tokenizer），则确认无误
- 若内部仍有共享层，拆分为独立的投影层

**涉及文件**：`model.py`

**预期影响**：
- 参数量少量增加（4 套投影层 vs 1 套）
- AUC 预期：**+0.001~0.002**

---

### 📋 Step 6：消融实验 & 组合测试 🔵 必做

**目标**：为论文准备完整的 ablation study 表格，验证每个改动的独立贡献和组合效果。

**操作**：

| 实验编号 | 配置 | 说明 |
|----------|------|------|
| E0 | Baseline（Step 0） | 基准线 |
| E1 | + Item 频次 CTR（Step 1a+1b） | 统计特征 |
| E2 | + KV 加权 Embedding（Step 2a+2b） | 结构先验 |
| E3 | + 频率截断（Step 3） | 参数压缩 |
| E4 | + 绝对时间差（Step 4） | 时序增强 |
| E5 | + 独立投影（Step 5） | 域隔离 |
| E6 | ALL（1+2+3+4+5） | 全量组合 |

### 📋 Step 7：User Dense 预训练向量利用 🟢 加分项（可选）

**目标**：优化 user_dense 中占比最大的预训练向量（fid 61: 256 维 + fid 87: 320 维），用瓶颈压缩或替代序列注意力的初始 query。

**涉及文件**：`model.py`

**预期影响**：
- 训练速度提升（减少 dense 投影计算量）
- AUC 影响不确定，需实验验证

---



## 📊 结果追踪表

| Step | 改动 | Best Val AUC | Best Val LogLoss | Eval AUC | Infer Time | 参数量 | vs Baseline |
|------|------|-------------|-----------------|----------|------------|--------|-------------|
| 0 | Baseline | 0.862227 | 0.224119 | 0.806701 | 226.89s | ~2.40亿 | — |
| 1a | +Item 频次 | | | | | | |
| 1b | +贝叶斯 CTR | | | | | | |
| 2a | +KV 平铺型 | | | | | | |
| 2b | +KV 金字塔型 | | | | | | |
| 3 | +频率截断 | | | | | | |
| 4 | +绝对时间差 | | | | | | |
| 5 | +独立投影 | | | | | | |
| 6 | ALL 组合 | | | | | | |

> **列说明**：
> - **Best Val AUC / LogLoss**：训练过程中验证集上的最佳指标（EarlyStopping 选出的 best checkpoint）
> - **Eval AUC**：加载 best checkpoint 后在独立评估集上跑的 AUC（最终评测指标）
> - **Infer Time**：评估集全量推理耗时（ms/样本 或 总秒数），衡量线上部署效率
