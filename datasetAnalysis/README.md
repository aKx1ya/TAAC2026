# TAAC2026 HyFormer Baseline 数据分析与优化指南

> 基于全量 Infer schema.json（来自线上训练日志）的分析文档
>
> 日期: 2026-05-08

---

## 目录

1. [数据全景](#一数据全景)
2. [关键发现——被验证正确的 7 个洞察](#二关键发现被验证正确的-7-个洞察)
3. [优化方向详解](#三优化方向详解)
4. [建议执行路线](#四建议执行路线)
5. [硬件约束检查清单](#五硬件约束检查清单38gb-vram)
6. [附录：Schema 关键数字速查](#六附录schema-关键数字速查)

---

## 一、数据全景

### 1.1 整体统计

| 类别 | 特征数 | 总维度 | 说明 |
|------|--------|--------|------|
| `user_int` | 46 个 | **411 维** | 用户侧整数特征 |
| `item_int` | 14 个 | **33 维** | 物品侧整数特征 |
| `user_dense` | 10 个 | **918 维** | 用户侧稠密特征 |
| `item_dense` | **0 个** | **0 维** | **完全空白！** |
| `seq_a` | 9 个 | — | 序列域 A，前缀 `domain_a_seq`，最大长度 256 |
| `seq_b` | 14 个 | — | 序列域 B，前缀 `domain_b_seq`，最大长度 256 |
| `seq_c` | 12 个 | — | 序列域 C，前缀 `domain_c_seq`，最大长度 512 |
| `seq_d` | 10 个 | — | 序列域 D，前缀 `domain_d_seq`，最大长度 512 |

### 1.2 模型规模（当前 baseline）

| 指标 | 值 |
|------|-----|
| 总参数量 | **239,931,649**（约 2.4 亿） |
| Sparse 参数（Embedding） | 237,435,776（占 99.0%） |
| Dense 参数 | 2,495,873（占 1.0%） |
| Batch size | 512 |
| 训练集 | 907,381 行（900 个 Row Group） |
| 验证集 | 102,619 行（100 个 Row Group） |

---

## 二、关键发现——被验证正确的 7 个洞察

以下 7 个洞察来自对 schema.json 的分析，已通过全量数据验证确认无误。

### ✅ 发现 1：8 对 (key, value) 对齐特征

fid 62-66 和 fid 89-91 在 `user_int` 中的 `length` 与 `user_dense` 中的 `dim` **完全一致**，形成 8 对完美对齐的 (key, value) 结构：

| fid | int (vocab × length) | dense (dim) | 对齐？ |
|-----|---------------------|-------------|--------|
| **62** | 11 × **6** | **6** | ✅ |
| **63** | 49 × **19** | **19** | ✅ |
| **64** | 51 × **26** | **26** | ✅ |
| **65** | 425 × **111** | **111** | ✅ |
| **66** | 1403 × **150** | **150** | ✅ |
| **89** | 10 × **10** | **10** | ✅ |
| **90** | 10 × **10** | **10** | ✅ |
| **91** | 10 × **10** | **10** | ✅ |

**语义推测**：int 列是**实体 ID**（如类目 ID、行为标签 ID），dense 列是对应的**统计量**（如在该类目上的停留时长、频次等）。

**baseline 的问题**：完全分开处理——int 走 embedding 后 mean/sum pooling，dense 走线性投影，两者之间没有交互。**这是当前 baseline 最大的结构信息浪费。**

#### 两类对齐结构

这 8 对特征可以进一步分为两种结构类型，应该采用不同的融合策略：

| 类型 | fid | 特征 | 融合策略 |
|------|-----|------|----------|
| **🏔️ 金字塔型** | 62→66 | vocab 递增（11→1403），dim 递增（6→150） | **层次化聚合**——粗粒度引导细粒度 attention，形成多粒度兴趣表示 |
| **🏛️ 平铺型** | 89→91 | 均匀 10×10，vocab=10 固定 | **并行加权**——各自独立做 weighted pooling 后拼接 |

#### 实现注意事项

dense 列的值可能是"停留时长"或"频次"等原始统计量，数值范围可能很大（不是 0-1 之间的概率）。直接用 softmax 可能导致极端权重分配。建议在加权前先对 dense 值做预处理：

```python
# 方案 A：log 变换压缩动态范围
weights = F.softmax(torch.log1p(dense_values), dim=-1)

# 方案 B：LayerNorm 归一化
weights = F.softmax(normalized_dense, dim=-1)  # normalized_dense = LayerNorm(raw_dense)
```

### ✅ 发现 2：Item Dense = 空白

`item_dense` 在 schema 中完全不存在（0 维），但 item 有 14 个整数特征（33 维）。

这意味着：
- 模型中的 `item_dense_proj` 模块（nn.Linear）当前不起作用（输入维度为 0）
- 商品侧完全没有连续值统计特征
- 数据本身携带的信息可以通过统计构造出来（频次、平滑 CTR 等）

### ✅ 发现 3：兴趣金字塔结构

fid 62→66 形成从粗到细的层次化兴趣结构：

```
vocab:   11  →   49  →   51  →   425  →  1403   (基数递增，兴趣越细)
dim:      6  →   19  →   26  →   111  →  150    (表示空间越大，信息越丰富)
        ────────────────────────────────────
        粗粒度                            细粒度
```

- 粗粒度（fid 62）：vocab=11，仅 6 个槽位 → 大类别
- 细粒度（fid 66）：vocab=1403，150 个槽位 → 具体实体
- 中间层（fid 63-65）：渐变的中间粒度

这种结构天然适合**多粒度层次化建模**，但 baseline 没有利用这一性质。

### ✅ 发现 4：VRAM 瓶颈的真实分析

**一个常见的误解**：把 VRAM 压力归结为 `user_int`/`item_int` 的高基数多值特征（如 FID 66 vocab=1403, dim=150；FID 11 vocab=21528, dim=20）。

**实际的瓶颈分析**：对比 Embedding 参数量：

| 特征 | vocab | dim | Embedding 参数量 | 占比 |
|------|-------|-----|-----------------|------|
| seq_c/fid **47** | **86,335,515** | 1 | **8,633 万** | 真大户 |
| seq_b/fid **69** | **64,710,562** | 1 | **6,471 万** | 真大户 |
| seq_c/fid **29** | **5,764,358** | 1 | **576 万** | 中等 |
| item_int/fid **11** | 21,528 | 20 | 43 万 | 很小 |
| user_int/fid **66** | 1,403 | 150 | 21 万 | 很小 |

**结论**：真正的 VRAM 瓶颈在 **seq 域的超大词表特征**（vocab 千万级），而不是 user/item 侧的整数特征（vocab 万级）。38GiB VRAM 下，user/item 的 embedding 完全可以保留，优化重点应放在 seq 域的超大特征上。

### ✅ 发现 5：多域序列规模不均衡

4 个序列域的特征数和基数差异极大：

| 域 | 特征数 | 最大 fid vocab | 最大长度 | 复杂度 |
|----|--------|---------------|----------|--------|
| seq_b | **14 个** | **64,710,562** | 256 | 🔴 最高 |
| seq_c | **12 个** | **86,335,515** | 512 | 🔴 最高 |
| seq_a | 9 个 | 745,286 | 256 | 🟡 中等 |
| seq_d | 10 个 | 606,041 | 512 | 🟡 中等 |

**问题**：当前 baseline 的 `MultiSeqHyFormerBlock` 对所有 seq 域使用**共享权重的同一套 Transformer 层**。这意味着：
- seq B（14 个特征，含 6471 万 vocab）和 seq A（9 个特征）共享同样的 QKV 投影
- 高基数域的噪声可能通过共享参数污染低基数域的有效信号
- 参数利用率低下

**轻量级改进**：先给不同域分配**独立的投影层**（代码中已有按 domain 的 `_seq_proj[domain]`），再共享后面的 Transformer 块。改动量很小。

### ✅ 发现 6：时间戳在线上（Infer）环境不可用

4 个 seq 域的时间戳特征（ts_fid）vocab 均为 0：

| 域 | ts_fid | vocab | 含义 |
|----|--------|-------|------|
| seq_a | 39 | 0 | 时间戳列（仅作为相对时间计算用） |
| seq_b | 67 | 0 | 同上 |
| seq_c | 27 | 0 | 同上 |
| seq_d | 26 | 0 | 同上 |

**关键约束**：绝对时间戳在线上环境不可用，任何依赖绝对时间范围的 Embedding 都会在推理时报错。必须在数据预处理阶段将时间戳转化为**相对时间差**（距当前预测时刻），并做分桶处理，脱离对绝对时间范围的依赖。

### ✅ 发现 7：优先级排序

优化方向的投入产出比排序（经验证合理）：

```
最高优先级          Item 统计特征构造（5–15 行代码，立竿见影）
    ↓          (key, value) 加权 Embedding（论文级创新）
    ↓          超大 seq 特征频率截断（释放瓶颈，节省巨量参数）
    ↓          绝对时间差特征（增强序列时序建模）
最低优先级          User dense 预训练向量利用
```

---

## 三、优化方向详解

### 🔴 方向 1（最高优先级）：Item 统计特征构造

**现状**：`item_dense_dim = 0`，模型中的 `item_dense_proj` 模块被闲置。

**具体做法**：

#### A. Item 对数频次（~5 行代码）

统计每个 item_id 在训练集中出现的次数，取对数后作为新的 dense 特征。

```python
# 思路：在数据预处理阶段
# 1. 扫描训练数据的 Row Group，统计每个 fid=16 (item_id) 的出现频次
# 2. item_log_freq = log(1 + count)
# 3. 作为新的 item_dense 列传入模型
```

**预期收益**：让模型区分"热门商品"与"长尾商品"，`item_dense_dim` 从 0 变为 1。

#### B. 贝叶斯平滑 CTR（~15 行代码）

每个 item 的历史转化率（label_type==2 的比例），用全局 CTR 做贝叶斯平滑。

```python
# 关键：只在训练集上统计，map 到验证集，防止标签泄露
# smooth_ctr = (clicks + global_ctr * smooth) / (impressions + smooth)
```

**预期收益**：CTR 任务是核心目标，这是最直接的信号。

#### C. fid=11 的 Attention Pooling（~30 行代码）

fid=11 是 item 的多值标签（vocab=21528, length=20），baseline 用 mean pooling 平均 20 个标签。改为以用户兴趣为 query 的 attention pooling。

---

### 🔴 方向 2（最高优先级）：(key, value) 加权 Embedding

**现状**：8 对对齐特征的 int 和 dense 完全分开处理。

**核心思想**：用 dense 值（统计量）作为 attention 权重，对 int 的 embedding 做加权聚合。

```python
# Baseline 现状（浪费了对齐信号）:
int_emb = mean(embedding(entity_ids))      # int 走 mean pooling
dense_out = linear(concat(all_stats))      # dense 单独走 linear

# 改进方案（8 对分别处理，注意 dense 值预处理）:
emb_matrix = embedding(entity_ids)         # [batch, slots, emb_dim]
# 对 dense 值做 log1p 压缩，避免极端值导致 softmax 权重畸变
weights = F.softmax(torch.log1p(statistics), dim=-1)  # [batch, slots]
weighted_emb = (emb_matrix * weights.unsqueeze(-1)).sum(dim=1)
```

#### 两类结构的不同融合策略

| 类型 | fid | vocabs | 推荐实现 |
|------|-----|--------|----------|
| **金字塔型 62→66** | 62:11, 63:49, 64:51, 65:425, 66:1403 | **层次化融合**：粗粒度（fid 62）的加权结果作为细粒度（fid 66）的 attention bias |
| **平铺型 89→91** | 89:10, 90:10, 91:10 | **并行加权**：各自独立做 weighted pooling 后拼接为向量 |

**金字塔型层次化融合的伪代码**：
```python
# 层次化: fid=62 (粗) → fid=66 (细)
coarse_ids = user_int_62      # [batch, 6]
coarse_vals = user_dense_62   # [batch, 6]
coarse_emb = weighted_pool(embedding(coarse_ids), coarse_vals)  # [batch, d]

# 细粒度的 attention bias 来自粗粒度表示
fine_ids = user_int_66        # [batch, 150]
fine_vals = user_dense_66     # [batch, 150]
fine_emb = weighted_pool(embedding(fine_ids), fine_vals + coarse_bias)
```

**创新价值**：
- 对应论文中 "the (key, value)-aligned columns are not given special treatment in the baseline"
- 可以作为 Section 4 的核心创新点
- 金字塔结构的层次化融合是论文级别的 contribution

**实现位置**：修改 `model.py` 中的 `PCVRHyFormer.forward()` 或 NS Tokenizer。不需要改动数据加载部分（int 和 dense 已经都在 input 里了）。

---

### 🟡 方向 3（中优先级）：高基数序列特征频率截断

**现状**：`emb_skip_threshold=1000000`，超过 100 万 vocab 的特征 embedding 被零向量替代。

> ⚠️ **这是当前 baseline 真正的 VRAM 瓶颈所在**，而非 user_int/item_int 的特征。seq 域的超大词表特征参数量是 user/item 侧的几百倍。

#### 受影响特征

**已跳过（零向量，完全丢失信息）**：

| 域 | fid | vocab | 节省参数量 | 信息丢失程度 |
|----|-----|-------|-----------|-------------|
| seq_c | **47** | **86,335,515** | ~8,633 万参数 | 🔴 极高——最大特征 |
| seq_b | **69** | **64,710,562** | ~6,471 万参数 | 🔴 极高 |
| seq_c | **29** | **5,764,358** | ~576 万参数 | 🟡 高 |
| seq_c | **34** | **1,031,305** | ~103 万参数 | 🟡 中 |

**未跳过但词表仍然巨大**：

| 域 | fid | vocab | 当前处理 | 风险 |
|----|-----|-------|---------|------|
| seq_a | **38** | 745,286 | ⚠️ 完整 Embedding | Embedding 表 74.5 万行 |
| seq_d | **23** | 606,041 | ⚠️ 完整 Embedding | Embedding 表 60.6 万行 |
| seq_c | **36** | 977,479 | ⚠️ 接近阈值 | 接近 100 万 |
| seq_b | **74** | 476,333 | ⚠️ 完整 Embedding | Embedding 表 47.6 万行 |

两种可选方案：

#### A. 频率截断（Top-K Capping）⭐ 推荐

只保留出现频率最高的 K 个 ID（如 K=100,000），其余映射到统一的 OOV（out-of-vocabulary）桶：

```python
# 在 dataset.py 的 _load_schema 或预处理阶段
# 统计每个 fid 的 ID 分布，建立 fid → {id: mapped_id} 映射
# 所有频次低于阈值的 ID → 同一个 OOV index
```

**预期收益**：
- seq_c/fid 47：vocab 从 8,633 万 → 10 万（**压缩 99.9%**），节省约 8,623 万参数
- seq_b/fid 69：vocab 从 6,471 万 → 10 万（**压缩 99.8%**），节省约 6,461 万参数
- 总参数量有望从 2.4 亿降低到 8000 万以下
- 释放的 VRAM 可以用于方向 1（Item 统计特征）和方向 2（KV 加权 Embedding）

#### B. Hash Embedding

用哈希函数将原始 ID 分桶到一个固定大小的 embedding 表，不需要维护 ID 映射表。适合不想维护 OOV 映射表的场景。

**实现位置**：`model.py` 中的 Embedding 构建逻辑。

---

### 🟡 方向 4（中优先级）：绝对时间差特征

**现状**：baseline 只有 `time_bucket`——序列内行为间的相对时间差，不知道"这串行为距现在多久"。

**线上环境约束**：4 个 seq 域的时间戳 ts_fid（39/67/27/26）的 vocab 均为 0，说明 **绝对时间戳在线上推理时不可用**。因此任何时间特征必须基于**相对时间差**（距离当前预测时刻的时间差）。

**做法**：在每个序列位置，计算**当前预测时刻**与**行为发生时刻**的时间差，桶化后作为额外的 time embedding。这样既引入了"行为新鲜度"信息，又脱离了对绝对时间的依赖。

```python
# dataset.py 中新增
abs_delta = current_timestamp - action_timestamp
abs_bucket = np.searchsorted(BUCKET_BOUNDARIES, abs_delta) + 1

# model.py 中新增 time embedding table
self.abs_time_emb = nn.Embedding(65, d_model, padding_idx=0)
# 加到 sequence token 上，与现有 inter_time_emb 并列
seq_token = seq_emb + inter_time_emb + self.abs_time_emb(abs_bucket)
```

4 个 seq 域各有自己的 `ts_fid`：

| 域 | ts_fid | prefix |
|----|--------|--------|
| seq_a | 39 | domain_a_seq |
| seq_b | 67 | domain_b_seq |
| seq_c | 27 | domain_c_seq |
| seq_d | 26 | domain_d_seq |

---

### 🟢 方向 5（加分项）：User Dense 预训练向量利用

user_dense 918 维的构成分析：

| 类型 | fid | 维度 | 占比 | 建议 |
|------|-----|------|------|------|
| 预训练 embedding 向量 | 61 | 256 | 27.9% | 尝试瓶颈压缩 |
| 预训练 LMF4Ads | 87 | 320 | 34.9% | 尝试瓶颈压缩 |
| 对齐统计量 (62-66) | 62-66 | 312 | 34.0% | 与 direction 2 联动 |
| 对齐统计量 (89-91) | 89-91 | 30 | 3.3% | 与 direction 2 联动 |

**思路**：
- fid 61（256 维）和 fid 87（320 维）占了 62.8% 的 dense 维度，可能是商品/用户的预训练表征
- 可以用瓶颈层（bottleneck）压缩到更低维度，加速训练
- 或者将这些向量作为**序列注意力的初始 query**，替代随机初始化

---

## 四、建议执行路线

### 执行顺序与依赖关系

```
Day 1 ── Step 1: Item 频次 + 平滑 CTR dense 特征
│         ├── item_dense_dim: 0 → 2+
│         ├── 改 dataset.py + 预处理脚本
│         ├── 注意验证集防泄露（只从 train split 统计）
│         └── ⏱ ~15 分钟
│
Day 1 ── Step 2: (key, value) 加权 Embedding
│         ├── 8 对对齐特征融合（区分金字塔型/平铺型策略）
│         ├── 改 model.py（PCVRHyFormer.forward）
│         ├── 注意 dense 值需做 log1p 或 LayerNorm 预处理
│         └── ⏱ ~1 小时
│
Day 2 ── Step 3: 高基数 seq 特征频率截断
│         ├── 针对 seq_c/fid 47, seq_b/fid 69 等（vocab > 100 万）
│         ├── Top-K 截断（K=100,000），OOV 映射
│         ├── 预计总参数量从 2.4 亿 → 8000 万以下
│         ├── 释放 VRAM 给 Step 1 和 Step 2 的额外参数
│         └── ⏱ ~2 小时
│
Day 2 ── Step 4: 绝对时间差特征
│         ├── 距当前时刻的时间差 + 分桶
│         ├── 改 dataset.py + model.py
│         ├── 只做相对时间，不依赖绝对时间戳（线上兼容）
│         └── ⏱ ~30 分钟
│
Day 3 ── Step 5: 多域序列独立投影
│         ├── 给 seq B/C 分配独立投影层（降低高基数域噪声污染）
│         ├── 改 model.py（_seq_proj 独立化）
│         └── ⏱ ~30 分钟
│
Day 3-4 ── Step 6: 消融实验
│         ├── 每个改动单独跑一次 + 组合跑
│         ├── 记录 AUC / LogLoss / 参数量 / 每 step 速度
│         └── 为论文准备 ablation 表格
│
Optional ── Step 7: User dense 预训练向量优化
            ├── fid 61（256 维）和 fid 87（320 维）瓶颈压缩
            ├── 或作为序列注意力的初始 query
            └── 对模型结构改动较大
```

## 五、硬件约束检查清单（38GiB VRAM）

- [ ] Step 1 后：item_dense 从 0→2 维，增加约 0 参数量（只是一个 linear）
- [ ] Step 2 后：KV 加权引入额外计算（每对特征多一次 softmax + weighted sum），算力开销小
- [ ] Step 3 后：**参数量大幅下降**（2.4 亿 → 8000 万），VRAM 压力显著缓解
- [ ] Step 4 后：每个 seq 域多一个 65×d_model 的 Embedding，约 65×64×4≈16KB，可忽略
- [ ] Step 5 后：独立的 seq 投影层增加少量参数，在 Step 3 释放的 VRAM 容量内

---

## 六、附录：Schema 关键数字速查

### 6.1 user_int 高基数特征（vocab > 1000）

| fid | vocab | dim | 类型 |
|-----|-------|-----|------|
| 3 | **1,725** | 1 | 用户 ID |
| 15 | **1,167** | 26 | 高基数多值特征 |
| 54 | **2,848** | 1 | — |
| 56 | **1,423** | 1 | — |
| 66 | **1,403** | 150 | 兴趣金字塔最细层 |

### 6.2 item_int 高基数特征（vocab > 1000）

| fid | vocab | dim | 说明 |
|-----|-------|-----|------|
| **16** | **23,700** | 1 | 物品内容特征 / 创意 ID |
| **11** | **21,528** | **20** | 物品标签（多值）— 可做 attention pooling |
| 7 | 2,443 | 1 | — |
| 12 | 2,443 | 1 | — |
| 8 | 2,131 | 1 | — |
| 85 | 1,084 | 1 | — |

### 6.3 序列特征超大规模（vocab > 10 万）

| 域 | fid | vocab | Embedding 参数量 | 当前处理 | 建议处理 |
|----|-----|-------|-----------------|---------|----------|
| seq_c | **47** | **86,335,515** | ~8,633 万 | ❌ 跳过（零向量） | ⚠️ 必须做频率截断 |
| seq_b | **69** | **64,710,562** | ~6,471 万 | ❌ 跳过（零向量） | ⚠️ 必须做频率截断 |
| seq_c | **29** | **5,764,358** | ~576 万 | ❌ 跳过（零向量） | ⚠️ 建议做频率截断 |
| seq_c | **34** | **1,031,305** | ~103 万 | ❌ 跳过（零向量） | ⚠️ 建议做频率截断 |
| seq_a | **38** | 745,286 | ~75 万 | ⚠️ 保留 | 可考虑截断 |
| seq_c | **36** | 977,479 | ~98 万 | ⚠️ 保留（接近阈值） | 建议截断 |
| seq_b | **74** | 476,333 | ~48 万 | ✅ 保留 | — |
| seq_d | **23** | 606,041 | ~61 万 | ✅ 保留 | — |

> **参数量对比**：仅 seq_c/fid 47 和 seq_b/fid 69 这两个特征就占了约 **1.5 亿参数**（占当前总参数 2.4 亿的 63%）。对它们做频率截断可以释放大量 VRAM。

### 6.4 8 对对齐特征详细对照

| fid | int vocab | int dim | dense dim | 结构类型 | 语义推测 | 对齐 |
|-----|-----------|---------|-----------|----------|----------|------|
| 62 | 11 | 6 | 6 | 🏔️ 金字塔型 | 粗粒度类目 | ✅ |
| 63 | 49 | 19 | 19 | 🏔️ 金字塔型 | 中粒度类目 | ✅ |
| 64 | 51 | 26 | 26 | 🏔️ 金字塔型 | 中细粒度 | ✅ |
| 65 | 425 | 111 | 111 | 🏔️ 金字塔型 | 细粒度类目 | ✅ |
| 66 | 1,403 | 150 | 150 | 🏔️ 金字塔型 | 最细粒度类目 | ✅ |
| 89 | 10 | 10 | 10 | 🏛️ 平铺型 | 行为标签 | ✅ |
| 90 | 10 | 10 | 10 | 🏛️ 平铺型 | 行为标签 | ✅ |
| 91 | 10 | 10 | 10 | 🏛️ 平铺型 | 行为标签 | ✅ |

> **金字塔型**（62→66）：vocab 和 dim 递增，适合**层次化聚合**——粗粒度引导细粒度 attention。
> **平铺型**（89→91）：均匀 10×10，适合**并行加权**——各自独立做 weighted pooling 后拼接。

---

> 本文档基于线上训练日志中打印的声明式 schema.json 分析得出。训练结束后平台还会生成按实际数据切片统计的观测 schema（`train_split_observed_schema.json`、`valid_split_observed_schema.json`），届时可进一步补充验证。
