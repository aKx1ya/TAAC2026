# TAAC2026 Research Log

> **文件用途：** 项目唯一的"事实源"（Single Source of Truth）。每次更新训练版本、形成新想法、做新实验时，都来这里更新。
> **使用方法：** 和 AI 聊天时，把这份 markdown 作为上下文丢进去。AI 立刻知道项目全貌。
> **更新频率：** 每个版本训练完后立即更新（不要拖延，记忆会失真）。
> **当前版本：** v2.0（2026-05-14 大重构，整合 HyFormer 全系列扫描 + InterFormer 全系列扫描 + datasetAnalysis 扫描）

---

# 第一部分：项目档案

## 1.1 项目身份

- **比赛名称：** TAAC 2026 × KDD Cup 2026 Academic Track
- **任务类型：** CTR 预测（二分类）
- **评估指标：** AUC（主），LogLoss（辅）
- **数据集：** TencentGR
  - 训练集：907,381 行（900 个 Row Group）
  - 验证集：102,619 行（100 个 Row Group）
  - 总计 ~100 万条曝光记录，120 列，脱敏 Parquet 格式
- **训练平台：** Tencent Angel/Taiji (taiji.algo.qq.com)，2 GPU
- **官方 baseline：** HyFormer（基于 Tencent 同名论文）
- **关键 deadline：**
  - Phase 2 截止：2026-05-23
  - 比赛终点：2026 年 6 月底
  - 论文投稿目标：KDD Workshop（DLP-KDD 等）

## 1.2 团队

| 角色 | 姓名 | 主要负责 |
|------|------|---------|
| 整体规划 + 数学/理论 | 徐子健（你） | 研究方向、论文写作、决策 |
| 代码主力 | Zesong Qiu | model.py / trainer.py 修改、训练运行、所有 HyFormer/InterFormer 版本的实际实现 |
| 辅助 1 | _待填_ | _待填_ |
| 辅助 2 | _待填_ | _待填_ |
| 指导老师 | _待填_ | 关键节点 review，技术兜底 |

## 1.3 目标（双轨）

**轨道 A：发表论文（核心目标）**
- 投稿目标：KDD Workshop（DLP-KDD 或 Industrial AI Track）
- 备选：CIKM 短文、RecSys Late-Breaking Result
- 时间：6 月底前完成投稿

**轨道 B：比赛名次（副产品）**
- 现实目标：前 20 名（Honorable Mentions 范围）
- 拉伸目标：前 10 名
- 不追求前 3（不现实）

## 1.4 数据集结构（120 列详解）

| 类别 | 特征数 | 总维度 | 备注 |
|------|--------|--------|------|
| user_int | 46 | 411 | 用户整数特征 |
| item_int | 14 | 33 | 物品整数特征 |
| user_dense | 10 | 918 | 用户稠密特征 |
| **item_dense** | **0** | **0** | **空——无物品稠密特征（关键缺陷 = 关键机会）** |
| seq_a | 9 | — | Domain A 序列 (max_len 256) |
| seq_b | 14 | — | Domain B 序列 (max_len 256) |
| seq_c | 12 | — | Domain C 序列 (max_len 512) |
| seq_d | 10 | — | Domain D 序列 (max_len 512) |

**模型规模：** ~240M 总参数（99% 在稀疏 Embedding，1% 为稠密参数）

**关键观察：**
- 序列 Domain c/d 的 max_len 是 a/b 的 2 倍（512 vs 256），暗示长期行为对 CTR 预测更重要
- Item Dense = 0 是脱敏数据的固有缺陷，可以通过统计特征（log frequency、贝叶斯 CTR）填补
- 4 个 vocab > 1M 的序列特征当前被替换为零向量（VRAM 瓶颈）

---

# 第二部分：研究方向与决策

## 2.1 当前选定方向

**方向：** 架构改进（InterFormer 系列）

**Research Question:** 在保持参数效率的前提下，能否通过双向序列-特征交互（InterFormer）+ 训练稳定性优化（AMP + GradScaler + warmup），在 TencentGR CTR 预测任务上超越官方 HyFormer baseline，并且修复影响性能的关键 bug 后保持稳定收敛？

**Hypothesis:** HyFormer 作为官方 baseline 已经在调参空间被充分优化，继续在它上面做超参数搜索的边际收益有限；架构创新（特别是双向 NS-Seq 交互 + 显式特征交叉如 DHEN/DCN）是更有潜力的方向。

**为什么选这个方向：**
- 实证依据 A：HyFormer v0.2 一次性堆叠 5 个改进（focal/RoPE/LongerEncoder/d_model 翻倍/ItemFeatureInteraction）反而退步（AUC 0.7952 < baseline 0.8622）
- 实证依据 B：InterFormer v0.1 用一半的 d_model（64 vs 128）、零调参，跑赢了 HyFormer v0.2 的 0.7952（Eval AUC 0.8067）
- 主观判断：团队认为 baseline 已被充分优化，需要从架构维度寻找突破

## 2.2 文献综述

### 三篇核心参考论文（赛事提供）

| 论文 | 年份 | 核心思想 | 在本项目中的角色 |
|------|------|---------|-----------------|
| **HyFormer** | 2024 | NS-Seq 单向 cross-attention，Query Decoding 提取序列精华 | 官方 baseline 来源，作为对照基线 |
| **InterFormer** | 2024 (CIKM 2025) | 双向 NS-Seq 交互，可学习的 interaction tokens | **当前主方向**（v0.1 / v0.3 / v0.4 / v0.5） |
| **OneTrans** | 2024 | 单 Transformer 统一序列+特征交互，金字塔压缩 | 激进方案参考（未实现） |

### 待读论文（在比赛过程中持续补充）
- [ ] DHEN 原始论文（Deep & Heterogeneous Ensemble Networks）— v0.4 引入
- [ ] DCN v2 论文（Deep & Cross Network）— v0.4 引入
- [ ] MTGR 论文（Multi-Task Generative Recommendation）— OneTrans 前身
- [ ] Wukong / RankMixer 相关论文 — 特征交互扩展性
- [ ] 搜索关键词：`sparse dense optimizer recommendation`
- [ ] 搜索关键词：`item dense feature engineering deidentified`
- [ ] 搜索关键词：`AMP GradScaler recommendation training`

## 2.3 关键决策记录

> **格式：** 日期 + 决策 + 理由 + 决策人

- **2026-05-11** — 决定先吃透 baseline 再改进，避免盲目调参 — 团队共识
- **2026-05-14** — 目标定为"发刊为主 + 名次为辅" — 子健决策
- **2026-05-04** ⭐ — **项目主方向从 HyFormer 切换到 InterFormer** — 子健 + Zesong 二人共识
  - **理由：** HyFormer 作为官方 baseline 已经被充分优化；InterFormer 架构上更先进（双向交互）；团队主观判断架构创新比超参调优更有潜力
  - **触发因素（可能）：** HyFormer v0.2 激进调参失败（AUC 0.7952 退步），表明 HyFormer 调参空间有限
  - **决策性质：** 主观判断（基于团队对架构演进的理解），并非严格的对比实验结论
  - **影响：** 此后所有正式版本（InterFormer v0.3 / v0.4 / v0.5）均围绕 InterFormer 架构展开；HyFormer v0.3.1 ablation 系列变为"已走过的路"，不再是主战场

---

# 第三部分：环境与基础设施

## 3.1 代码仓库

- **GitHub：** github.com/aKx1ya/TAAC2026
- **本地路径：** /Users/xuzijian/Desktop/Tencent_Project/TAAC2026/
- **目录结构：**
  ```
  TAAC2026/
  ├── Hyformer/                       # HyFormer 实验家族（11 个不同实验）
  │   ├── HyFormer-v0.0/              # ⭐ 官方 baseline（== 毛坯baseline）
  │   ├── HyfFormer-v0.1/             # ⚠️ typo 目录，疑似旧备份
  │   ├── HyFormer-v0.1/              # 工程化打包（路径默认值 + eva/ 推理容器）
  │   ├── HyFormer-v0.2/              # ⭐ 激进调参失败（AUC 0.7952 ↓）
  │   ├── HyFormer-v0.3.0-基础HyFormer/   # 重启分支，回滚到 v0.0 + 真实 schema
  │   ├── HyFormer-v0.3.1-Hyformer+1.a/   # 消融：item log freq
  │   ├── HyFormer-v0.3.1-Hyformer+1.b/   # 消融：贝叶斯 CTR
  │   ├── HyFormer-v0.3.1-Hyformer+1.a+1.b/   # 消融：1a+1b 融合
  │   ├── HyFormer-v0.3.1-Hyformer+2.a/        # 消融：KV 加权 embedding
  │   ├── HyFormer-v0.3.1-Hyformer+2.a-update/ # 2.a 修正版：加 softmax 温度
  │   ├── HyFormer-v0.3.1-Hyformer+2.b/        # 消融：金字塔层次 KV 融合
  │   ├── HyFormer-v0.3.1-Interformer+1.a（对照试验）/   # 架构对照试验
  │   └── 毛坯baseline/               # == HyFormer-v0.0（重复副本）
  │
  ├── Interformer/                    # ⭐ 当前主方向（4 个实验）
  │   ├── v0.1/                       # ⭐ InterFormer 起点（Eval AUC 0.8067）
  │   ├── v0.3/                       # 调参失败（AUC 0.8038 ↓）
  │   ├── v0.4/                       # 大改架构 + NaN 崩溃
  │   ├── v0.5/                       # ⭐ 峰值 AUC 0.852（被 bug 拖崩）
  │   ├── Hyformer_baseline/          # HyFormer-v0.1 本地工作副本（不在演进链上）
  │   ├── Interformer_Pytorch_reference/   # 空目录（占位）
  │   └── data/                       # 共用 1000 行 demo 数据
  │
  ├── datasetAnalysis/                # ⭐ 项目"侦察兵"沙箱 + 研究纲领（7 条发现）
  ├── docs/Report Versions/          # 论文阶段性产出
  └── 参考论文/                       # 三篇核心论文 PDF
  ```

## 3.2 工作流工具栈

- **代码理解：** VSCode + Claude Code (Opus 4.7)
- **快速问答：** Terminal + DeepSeek API
- **宏观规划：** claude.ai 主对话
- **训练：** Tencent Angel Platform（网页操作）
- **风格规范：** ~/Desktop/markdown/HTML_Style_Guide.md
- **本对话工具：** 维护本 Research Log + 论文叙事

## 3.3 关键超参数（baseline 默认）

```
模型架构：
  d_model:        64
  num_layers:     2 (num_hyformer_blocks)
  num_heads:      4
  num_queries:    2
  user_ns_tokens: 5
  item_ns_tokens: 2
  ns_tokenizer:   rankmixer
  hidden_mult:    4

序列：
  seq_max_lens:   {a:256, b:256, c:512, d:512}
  num_time_buckets: 65
  emb_skip_threshold: 1,000,000

训练：
  batch_size:     256 (train.py default) / 512 (run.sh)
  loss_type:      bce
  sparse_lr:      0.05   (Adagrad，管 Embedding ~230M 参数)
  dense_lr:       1e-4   (AdamW，管其他 ~10M 参数)
  grad_clip:      1.0
  patience:       5      (EarlyStopping)
  num_epochs:     999    (实际由 EarlyStopping 决定)
  num_workers:    8
  buffer_batches: 20     (IterableDataset shuffle buffer)
```

---

# 第四部分：版本日志（核心区）

> **每个版本一个 entry。包括失败的版本——失败也是 contribution，会出现在 paper 的 Ablation 里。**
>
> **版本号说明：** 数字版本号 ≠ 时间顺序。`毛坯baseline` ≡ `HyFormer-v0.0`，`HyFormer-v0.3.0` 的代码 ≡ `HyFormer-v0.0`。版本号更像"实验分支命名"，不是线性递增。

## 4.A HyFormer 家族（已走过的路，11 个不同实验）

> 本系列在 2026-05-04 之后停止主力推进，转向 InterFormer。但 v0.3.1+2.b（金字塔 KV）仍可能作为论文创新点回收。

---

### v0.0 · HyFormer 官方 baseline ⭐

- **日期：** 2026-05-11 / 2026-05-12（子健亲跑）
- **运行者：** 子健
- **改动：** 无（完全默认配置）
- **目的：** 建立 baseline，作为后续所有改动的对照
- **训练时长：** ~16h，47000 steps，EarlyStopping 终止
- **结果：**
  - Best checkpoint: `global_step28992.layer=2.head=4.hidden=64.best_model`
  - **Val AUC: ~0.85**（plateau 早期出现在 step 3624）
  - Val LogLoss: ~0.22-0.23（后期微微上翘）
  - Train Loss: 0.15-0.3 主体，频繁 spike 到 0.5-0.75
  - Eval AUC（提交线上）：[❓ 待补，子健亲跑时未提交]
- **观察 / 诊断：**
  - Loss 频繁 spike → 推测 `sparse_lr=0.05` 过大
  - AUC 极早 plateau → 推测缺少 warmup + cosine schedule
  - LogLoss 后期上翘 → 过拟合开始，EarlyStopping 正确刹车
- **结论：** Baseline 跑通，三个改进方向已识别（降 sparse_lr / 加 warmup / 加 cosine）
- **下一步：** 已被后续版本超越
- **等价目录：** `毛坯baseline/`（diff -r 无输出）

---

### HyfFormer-v0.1 · ⚠️ 疑似 typo 备份目录

- **日期：** [❓ 未知]
- **运行者：** [❓ Zesong 确认]
- **状态：** 目录名拼写疑似 typo（HyfFormer 多了一个 f）
- **改动：** [❓ 待 Zesong 确认这是 v0.0 的备份还是另一个独立尝试]
- **结果：** [❓ 待补]
- **结论：** [❓ 待确认是否是历史遗留目录，可考虑删除或归档]

---

### v0.1 · 工程化打包（无算法改动）

- **日期：** [❓ Zesong 确认]
- **运行者：** Zesong
- **改动总结：** **零算法改动，纯工程打包**
  - `train.py`：4 个 CLI 默认路径从 `None` 改为本地相对路径
    - `--data_dir` default → `'./data'`
    - `--ckpt_dir` default → `'./checkpoints'`
    - `--log_dir` default → `'./logs'`
    - `tf_events_dir` 兜底 → `'./logs/tensorboard'`
  - 新增 `generate_schema.py`：扫 Parquet 自动生成 schema.json（本地小数据集用）
  - 新增 `eva/` 目录：评测容器自包含推理代码（`infer.py` + 复制的 `dataset.py` / `model.py`）
  - 新增三张 PNG（AUC / LogLoss / Loss）
- **目的：** 让 `bash run.sh` 在本地零参数直接跑 + 评测容器能独立打包
- **结果：** 三张训练曲线 PNG [❓ 是 v0.0 训练曲线还是 v0.1 重跑？待 Zesong 确认]
- **结论：** 工程基础设施，不影响算法表现
- **下一步：** v0.2 开始大幅修改

---

### v0.2 · ⭐ 激进调参失败（关键负面结果）

- **日期：** [❓ Zesong 确认]
- **运行者：** Zesong
- **改动总结：** 一次性堆叠 5 个改进
  - **架构层：** 新增 `ItemFeatureInteraction`（gated bilinear cross 模块）
  - **超参层：** `d_model 64→128`，`batch_size 256→512`，`dropout 0.01→0.1`，`user_ns_tokens 5→7`，`item_ns_tokens 2→4`，`num_queries 2→1`
  - **训练层：** `--loss_type focal --focal_alpha 0.25 --focal_gamma 2.0`，`--seq_encoder_type longer --seq_top_k 50 --seq_causal`，`--use_rope`，`--reinit_sparse_after_epoch 1`
  - **数据层：** 新增 `sort_by_timestamp + valid_time_ratio 0.1`（按时间戳排序的训练/验证划分）
  - **删除：** `eva/` 目录
- **目的：** 一次性把所有看起来合理的改进都加上，希望大幅提升 AUC
- **结果：**
  - **AUC = 0.7952** ⚠️（比 baseline 的 ~0.8622 val 下降）
  - 训练 8h27m，best_step = 9165
  - 三张 PNG 留档
- **观察 / 诊断：** [❓ Zesong 是否记得当时哪个改动是主要拖累？是 Focal 还是 LongerEncoder？]
- **结论：** **一次性堆叠多个改进会相互干扰，违反"单变量消融"原则**
- **后果：** 触发 v0.3.0 完整回滚 → 重启分支

---

### v0.3.0-基础HyFormer · 重启分支（代码 ≡ v0.0）

- **日期：** [❓ Zesong 确认]
- **运行者：** Zesong
- **改动总结：**
  - **代码层面回滚到 v0.0**：`dataset.py` / `model.py` / `train.py` / `trainer.py` / `utils.py` 全部回退到官方 baseline 一字不差
  - **撤销 v0.2 的所有改动**（focal / RoPE / LongerEncoder / ItemFeatureInteraction / d_model=128 全部去掉）
  - 新增 `schema.json`（生产平台导出的真实 schema）
  - 新增 `dataset-analysis.md`（数据分析文档）
  - 删除 `generate_schema.py`、三张 PNG、`README.md`
  - `run.sh` 退回到简版 5 个参数，但加 `--schema_path` 指向新 schema.json
- **目的：** v0.2 失败后，承认激进改动方向有问题，**重新拉一条干净分支**做规范的消融实验
- **结果：**
  - 后续 v0.3.1 系列的 README 引用 "baseline AUC = 0.8622"（**Val AUC**）作为参照
  - Eval AUC = 0.8067（与 InterFormer v0.1 相同）
- **结论：** 这次回滚是非常重要的研究决策——承认 v0.2 失败、回到原点、采用"每次只改一个变量"的消融原则
- **下一步：** 启动 v0.3.1 系列 ablation

---

### v0.3.1+1.a · 消融：item log frequency

- **日期：** [❓ Zesong 确认]
- **运行者：** Zesong
- **改动总结：** **只改 dataset.py**（不改 model.py）
  - `dataset.py:476` 新增 `_build_item_freq_map()`：
    - 扫训练 Row Group 统计每个 item_id 出现次数
    - 取 `log(1+count)` 作为新特征
    - 在线注入为 `item_dense` 第 0 列（fid=200）
    - 验证集复用训练统计（防标签泄露）
  - `schema.json` 新增 `"item_dense": [[200, 1]]`
  - `utils.py` +6 行配套修改
  - `run.sh`: `--user_ns_tokens 5→4`，`--batch_size 512→128`
- **目的：** 填补 `item_dense_dim=0` 空白（来自 datasetAnalysis 发现 2），用 item 出现频次（热度）作为新特征
- **结果：** **Val AUC = 0.8589**（比 baseline 0.8622 略降 -0.0033）
- **结论：** 单独使用 item log freq 略微拖累性能；可能与其他特征组合后受益
- **三张 PNG 留档**

---

### v0.3.1+1.b · 消融：贝叶斯平滑 CTR

- **日期：** [❓ Zesong 确认]
- **运行者：** Zesong
- **改动总结：** **只改 dataset.py**（不改 model.py）
  - `dataset.py:374` 新增 `_build_item_ctr_map()`：
    - 扫训练集统计 `impressions` + `clicks` (label==2)
    - 算全局 CTR
    - `smooth_ctr = (clicks + α·global_ctr) / (impressions + α)`，α=100
    - 注入 `item_dense` 第 0 列（fid=201）
  - `schema.json` 同步更新
  - `run.sh` 同 1.a（user_ns 4, batch_size 128）
- **目的：** 用 item 历史转化率做新特征，贝叶斯平滑避免低曝光 item 抖动
- **结果：** **Val AUC = 0.8622**（持平 baseline）
- **结论：** Bayesian CTR 没拖累也没明显提升，但提供了一个稳定的新信号
- **三张 PNG 留档**
- **注：** 1.b 没有改 `utils.py`（1.a 改了），说明两个分支是独立平行实现

---

### v0.3.1+1.a+1.b · 消融：两个特征融合

- **日期：** [❓ Zesong 确认]
- **运行者：** Zesong
- **改动总结：** 同时实现 `_build_item_freq_map` + `_build_item_ctr_map`
  - 注入 2 列 item_dense（fid=200 对数频次 + fid=201 贝叶斯 CTR）
  - dataset.py 从 763 行扩到 1034 行
- **目的：** 两个统计特征叠加，让模型同时拿到"热度"和"历史转化率"两个信号
- **结果：** **无 PNG / 无 AUC 数据** ⚠️
- **结论：** [❓ 是没跑完、跑崩了，还是结果丢失？待 Zesong 确认]
- **后续：** 必须补跑这个版本，因为 1a+1b 组合是消融实验的关键一环

---

### v0.3.1+2.a · 消融：KV 加权 Embedding（扁平型）

- **日期：** [❓ Zesong 确认]
- **运行者：** Zesong
- **改动总结：** 基于 1.a+1.b 之上，进一步改 model.py
  - `RankMixerNSTokenizer` 和 `GroupNSTokenizer` 新增 `weighted_fid_config` 参数
  - 对 **fid 89/90/91**（3 对 10×10 (int, dense) 对齐特征），用 `softmax(log1p(dense_vals))` 作为权重对 int embedding 做加权 pooling（替代原 mean pooling）
  - `PCVRHyFormer.__init__` 自动构建 `_kv_weighted_config` 并把 dense_feats 传给 tokenizer
- **理论依据：** 来自 datasetAnalysis 发现 1 —— 这 3 对 fid 是天然的 (Key, Value) 对齐特征，应该用加权聚合而非 mean pool
- **目的：** 让 dense 值告诉模型"哪个 int embedding 重要"，比 mean pool 信息量更高
- **结果：** **无 PNG / 无 AUC 数据** ⚠️
- **结论：** [❓ 待 Zesong 确认结果情况]

---

### v0.3.1+2.a-update · 2.a 修正版（softmax 温度系数）

- **日期：** [❓ Zesong 确认]
- **运行者：** Zesong
- **改动总结：**
  - 引入 `--kv_softmax_temperature` 超参（默认 T=2.0）
  - 原 2.a 的 `softmax(log1p(x))` 过于尖锐 → 加温度系数：`softmax(log1p(x) / T)`，T=2 让分布更平滑，泛化更好
  - `run.sh`: `batch_size 128 → 512` 改回（适配新增容量）；新增 `--kv_softmax_temperature 2.0`
- **目的：** 修正 2.a softmax 过尖锐导致泛化差的问题
- **结果：** **🖼️ 三张 PNG**（v0.3.1 系列 2.x 中唯一跑出可视化结果的版本）
- **关键 AUC：** [❓ Zesong 确认具体数字]
- **结论：** 软化 softmax 后效果如何？需要数字才能下结论

---

### v0.3.1+2.b · ⭐ 消融：金字塔层次 KV 融合（潜在论文创新点）

- **日期：** [❓ Zesong 确认]
- **运行者：** Zesong
- **改动总结：** 在 2.a 基础上，新增 `HierarchicalKVFusion` 类（model.py +404 行，从 1714 → 2118）
  - 针对 **fid 62→63→64→65→66**（5 级粗→细层次特征，vocab 11→1403）构造金字塔
  - 第 l 级用前一级输出 `C_{l-1}` 做 query 关注当前级 embedding
  - 注意力分数与 dense 值偏置联合决定 softmax 权重
  - MLP 融合上下文：`C_l = MLP([C_{l-1}, level_out])`
  - 新增参数量 ~67K
  - `PCVRHyFormer.__init__` 自动检测 fid 62-66 存在就启用
  - `run.sh` 新增 `--kv_softmax_temperature 2.0`（继承 2.a-update）
- **理论依据：** 直接来自 datasetAnalysis README 发现 3 ——
  > "兴趣金字塔结构（fids 62→66），词表大小从 11 增长到 1403，形成从粗到细的兴趣层级，适合层级注意力建模"
- **目的：** **把树状粗→细层次结构显式编码进 attention**，对应论文级别创新点
- **结果：** **无 PNG / 无 AUC 数据** ⚠️⚠️⚠️
- **结论：** **这是最关键、最值得追问的版本** —— 如果跑出来效果好，可以作为论文核心创新点；如果没跑，需要立刻安排
- **下一步：** 必须找 Zesong 确认这个版本的状态

---

### v0.3.1+Interformer+1.a · 架构对照试验（关键决策依据）

- **日期：** [❓ Zesong 确认，应该在 2026-05-04 决策前后]
- **运行者：** Zesong
- **改动总结：** **完全换架构**
  - `model.py` 整个重写：1714 → 569 行
  - 类名全换：`PCVRHyFormer` → `PCVRInterFormer`
  - 新类层级：`RMSNorm` / `FeatureEmbeddingBank` / `NonSequentialTokenizer` / `DenseTokenProjector` / `SequenceTokenizer` / `EmbeddingParameterMixin` / `PersonalizedFeedForward` / `CrossSummary` / `InterFormerBlock`
  - 同时也加了 Step 1.a 的 item_freq
  - **`run.sh` 完全不变** —— 所有差异都在代码里
- **理论依据：** 验证"换架构不换 pipeline"的可行性，这是 2026-05-04 切换决策的实证基础
- **目的：** 在同一份特征工程上对比 HyFormer vs InterFormer 架构本身的贡献
- **结果：** **无 PNG / 无 AUC 数据** ⚠️
- **结论：** [❓ Zesong 是否跑过这个对照试验？如果跑过，结果是论文 Motivation 章节的关键数据]
- **重要性：** 如果这个实验有数据，可以直接支撑论文的"为什么选 InterFormer"叙事

---

### 毛坯baseline · 官方原始 baseline 副本（== v0.0）

- **状态：** 与 `HyFormer-v0.0` **逐字节完全一致**（diff -r 无任何输出）
- **作用推测：** 命名快照，可能是项目早期的备份目录
- **下一步：** 可考虑归档或在 README 里标注其等价关系

---

## 4.B InterFormer 家族（当前主战场，4 个版本 + 1 个参考工作区）

> 2026-05-04 之后的所有主力开发都在这条线上。包含项目最重要的负面结果（v0.4 NaN）和最高峰值（v0.5 AUC 0.852）。

---

### InterFormer v0.1 · ⭐ PCVRInterFormer 起点（基线参照）

- **日期：** [❓ Zesong 确认，应该在 2026-05-04 之前作为决策前置验证]
- **运行者：** Zesong
- **架构：** PCVRInterFormer（569 行 model.py）
  - 三模块：Interaction Arch + Sequence Arch + Cross Arch
  - NS Tokenizer：RankMixer (user_ns=5, item_ns=2)
  - num_queries = 2
  - 位置编码：标准 sinusoidal（无 RoPE）
- **改动总结：** 相对 HyFormer 完全是架构层面重写
- **目的：** InterFormer 架构的"第一个可跑通版本"，作为后续优化的起点
- **结果：**
  - **Eval AUC = 0.8067** ⭐（这是完整训练后离线 Evaluation 指标）
  - 推理时长 226.89s
  - 训练时长 ~22h，best_step = 61608
  - 🖼️ AUC.png / LogLoss.png / Loss.png 留档
- **README 关键点：** "用一半的 d_model（64 vs 128），零调参跑赢 HyFormer-v0.2 的 0.7952"
- **三个已知 bug：**
  - [ ] `model.py:480` `del seq_encoder_type` 静默丢弃参数
  - [ ] `reinit_high_cardinality_params` 逻辑反转（cardinality_threshold=0 时反而全部重置）
  - [ ] GAUC 计算硬编码为 0（trainer.py 未实现）
- **结论：** 架构本身比 HyFormer v0.2 优秀；这是决策切换到 InterFormer 的实证依据之一
- **架构同源：** model.py 与 `Hyformer/HyFormer-v0.3.1-Interformer+1.a` 同一份

---

### InterFormer v0.3 · 负面调参实验（"应该一次只改一个变量"）

- **日期：** [❓ Zesong 确认]
- **运行者：** Zesong
- **改动总结：** **模型代码几乎一行不改**（model.py / dataset.py / trainer.py / utils.py 全部不变），改动几乎全在 CLI 默认值
  - `loss_type: bce → focal`（focal_alpha 0.1，**过低，把正样本权重压得过小**）
  - `batch_size: 256 → 1024`（**每 epoch 梯度步数砍到 1/4**）
  - `num_epochs: 999 → 3`（**硬上限**）
  - `reinit_cardinality_threshold: 0 → 10000`（**embedding 冷启重置**）
  - `seq_encoder_type: transformer → swiglu`（**但 model.py:480 `del seq_encoder_type` bug 让这个改动静默失效！**）
  - 删除 `eva/` 目录
  - `run.sh` 新增 `--num_workers 4` / `--buffer_batches 20` / `--batch_size 1024`
- **目的：** 同时尝试 focal loss + 大 batch + epoch 限制 + embedding 冷启 + 新 encoder，希望快速 +AUC
- **结果：**
  - **Eval AUC = 0.8038** ⚠️（比 v0.1 的 0.8067 **退步 0.0029**）
  - best_step = 61608
  - 🖼️ 三张 PNG + `数据记录.txt`
- **README 自我反思（重要！）：** **"应该一次只改一个变量"**
- **失败的三个原因（README 已剖析）：**
  1. `batch_size 1024 + epoch=3` → 每 epoch 梯度步数砍到 1/4，又只跑 3 epoch，**总梯度步数严重不足**
  2. `reinit_cardinality_threshold=10000` + 只跑 3 epoch → 高频 embedding 被冷启**摧毁 2 次**，刚学会就被清零
  3. `seq_encoder_type=swiglu` 实际无效（被 `del seq_encoder_type` bug 静默丢弃）；同时 `focal_alpha=0.1` 过低
- **结论：** **这是一个有教学价值的负面结果**，反映了"多变量同改"的危险；写论文时可作为方法论说明
- **后续影响：** 直接导致 v0.4 转向"专注架构改进 + 工程修复"

---

### InterFormer v0.4 · ⭐ 大改架构 + NaN 崩溃（关键诊断案例）

- **日期：** [❓ Zesong 确认]
- **运行者：** Zesong
- **改动总结：** **大改造**（model.py +958 行，从 569 → 1527）
  - **架构新增 17 个类：**
    - DHEN 系列：`DHENLayer`（行 703）, `DHENInteraction`（行 737）
    - DCN 系列：`CrossLayer`（行 651）, `DCNInteraction`（行 670）, `DCNInteractionStack`（行 780）
    - Interaction 框架：`DotProductInteraction`, `DotInteractionStack`, `InteractionArch`
    - 序列架构：`RotaryMultiHeadAttention`（行 884）, `SequenceArch`
    - Item 交互：`ItemInteractionArch`, `ItemAwareGating`
    - 压缩/门控：`LinearCompressedEmbedding`, `SelfGating`
    - 汇总器：`NonSequenceSummarizer`, `PoolingMultiHeadAttention`, `SequenceSummarizer`
    - 删除：`CrossSummary`
  - **dataset.py 改动：**
    - 新增 `time_order_split`（按时间戳排 Row Group）
    - `log1p` 变换（fid 62-66）
    - `sort_seq_domains`（seq_d 域内按值排序）
  - **trainer.py 改动：** 引入 AMP `torch.amp.autocast`（**但没加 GradScaler——直接 NaN 元凶**）
  - **run.sh 改动：** `item_ns_tokens 2→5`，`batch_size 1024→384`，`buffer_batches 20→50`，新增 `--interaction_backbone dhen --interaction_layers 2 --dcn_layers 2 --cross_low_rank 32 --num_cls_tokens 4 --num_pma_tokens 2 --num_recent_tokens 2 --use_rope`，`loss_type` 切回 `bce`
- **目的：** 首次引入 DHEN+DCN+RoPE 大幅扩容模型容量
- **结果：** 💥 **NaN 崩溃**
  - 训练时间线：
    ```
    Step 1794  loss=0.2159  ← 一切正常
    Step 1795  loss=0.2463  ← 最后一个正常 step
    Step 1796  loss=nan     ← 从此万劫不复
    ...此后所有 step 均为 NaN...
    ```
  - 三张 PNG（但 AUC=0、LogLoss=inf，因为权重已被 NaN 污染）
  - `记录.txt` 完整训练日志
  - `Evaluation/` 目录含独立 `infer.py`
- **NaN 根因（README.md 第四节"崩溃根因分析"已完整诊断）：**

  **唯一根因：AMP autocast 启用了 FP16，但缺少 GradScaler**

  ```python
  # 错误代码：
  with torch.amp.autocast('cuda', enabled=use_amp):
      logits = self.model(model_input)
      loss = F.binary_cross_entropy_with_logits(logits, label)
  loss.backward()   # ← FP16 梯度未经缩放
  ```

  - FP16 表示范围只有 ~6e-8 到 ~65504
  - 跑了 1795 步都恰好在 FP16 范围内
  - 第 1796 步某个 batch 激活值偏大 → 梯度超过 65504 → **上溢为 inf**
  - `optimizer.step()` 把 inf 写进权重 → **权重变 NaN**
  - 之后所有 forward 都吐 NaN → **永久污染，无法自愈**

- **排除的假设（README 表格）：**
  | 假设 | 判断依据 |
  |---|---|
  | 稀疏 LR (0.05) 过高 | ❌ 若是，训练早期就该发散，不是 1795 步后突变 |
  | log(1+x) 变换产生 NaN | ❌ 否则 step 1 即崩 |
  | BCE 公式不稳定 | ❌ `binary_cross_entropy_with_logits` 内含 log-sum-exp，FP32 下稳定 |
  | 梯度裁剪缺失 | ❌ 代码本就有 `clip_grad_norm_(max_norm=1.0)` |
- **结论：** **教科书级的"诊断-修复"案例**。247 行 README 完整剖析 + 排除假设表 + 修复建议清单 → 直接驱动了 v0.5 的实现
- **论文价值：** 这个故事可以写进 Methodology 章节作为"训练稳定性教训"

---

### InterFormer v0.5 · ⭐⭐⭐ 修复 NaN + 峰值 AUC 0.852（项目目前最高）

- **日期：** [❓ Zesong 确认，应该是 v0.4 之后不久]
- **运行者：** Zesong
- **改动总结：** **model.py 一行不改！**（架构和 v0.4 完全一致，仍是 1527 行 + 17 个新类）
  - **trainer.py（+73 行，核心改动）：**
    1. 新增 `self.scaler = torch.amp.GradScaler('cuda')`（第 78 行）
    2. 重写 `_train_step` 末尾（第 327-336 行）：
       ```python
       scaler.scale(loss).backward()
       scaler.unscale_(optimizer)
       # clip_grad_norm here
       scaler.step(optimizer)
       scaler.update()
       ```
    3. 新增 `warmup_steps=1000`（第 61 行）和线性 warmup 实现（第 313-318 行）：
       ```python
       if total_step < self.warmup_steps:
           pg['lr'] = base_lr * (total_step / warmup_steps)
       ```
    4. 新增 `gradient_accumulation_steps=1`（第 62 行）的框架
  - **dataset.py（+3 行）：** 在 log1p 前加 `np.maximum(x, -0.999)` 边界保护
  - **run.sh 新增：** `--loss_type focal --focal_alpha 0.25 --focal_gamma 2.0`，`--sparse_lr 0.05→0.01`，`--eval_every_n_steps 0→1000`
- **关键澄清：** ⚠️ **DHEN / DCN / RoPE 是 v0.4 加的，不是 v0.5**。v0.5 真正新加的只有 GradScaler + warmup + Focal Loss + log1p 保护
- **目的：** 修复 v0.4 的 NaN crash，让大架构能稳定训练
- **结果：**
  - **峰值 Val AUC = 0.852** ⭐⭐⭐（step 2396）
  - 🖼️ 6 张 PNG：AUC.png / LogLoss.png / Loss.png / Gauc.png / pos_rate.png / precision_at_1pct.png
  - `Evaluation/` 目录
  - **但峰值之后出现 AUC 悬崖式 decay**（从 ~0.852 跌至 ~0.760）
- **峰值之后 decay 的根因（README Known Issues 第一条）：**

  > **Bug 1: Embedding Reinitialization Logic Inverted**
  >
  > In `model.py`, `reinit_high_cardinality_params()` uses the condition `num_embeddings - 1 <= cardinality_threshold` to skip reinit. When `cardinality_threshold=0` (the default, documented as 'never reset'), this evaluates to `num_embeddings <= 1`, meaning **all non-trivial embeddings are reset** — the exact opposite of the intended behavior.
  >
  > Starting from epoch 3, all learned embedding weights are periodically wiped to Xavier random initialization, causing the metric cliff observed in the AUC curve.

  简单说：
  - `cardinality_threshold=0` 本意是"永不重置"
  - 实际效果是"全部重置"
  - 从 epoch 3 开始每 epoch 重置已学好的 embedding
  - 造成 AUC 悬崖式下跌
- **另两个 Known Issues：**
  - Bug 2 (Medium)：**GAUC 硬编码为 0** → `Gauc.png` 全是 0，不是真实的分组 AUC
  - Correction：**Positive Rate 是 ~9.6%，不是 0.1%**（中等不平衡，不是极端稀疏）
- **重要数字方法论说明：** v0.5 报告的 **0.852 是"训练时验证集峰值 AUC"**，与 v0.1 报告的 **0.8067 "完整训练后离线 Evaluation AUC"** 不是同一回事。两个数字方法论不同，**不能直接相减**
- **结论：**
  - **正面：** GradScaler + warmup + Focal 是正确的修复方向；峰值证明 v0.4 的大架构有潜力
  - **负面：** reinit bug 让"潜力"无法转化为"稳态部署效果"；真实部署 AUC 应该低于 0.852
- **下一步（关键）：** 修 reinit bug，重跑稳定版本，看真实 ceiling 是多少

---

### Hyformer_baseline · 不在演进链上的参考工作区

- **本质：** HyFormer-v0.1 的本地工作副本
- **代码差异：**
  - `model.py` / `train.py` 与 `Hyformer/HyFormer-v0.1/` 一字不差
  - `dataset.py` 多了 `time_order_split` 参数（按 Row Group timestamp 排序切分训练/验证集），70 行 diff
- **多出的目录：** `data/`（1000 行 demo）、`logs/`（含 tensorboard 和 train.log）、`.vscode/`、`__pycache__/`、`eva/`
- **run.sh：** 与 HyFormer-v0.1 完全一致
- **训练结果：** `logs/train.log` 和 `logs/tensorboard/`，但无 PNG / AUC 报告
- **作用推测：** 作者在 Interformer 目录下保留的"参考用 HyFormer baseline 工作区"——开发 InterFormer 时方便对照 HyFormer，并且本地能直接跑通。`time_order_split` 的早期尝试可能从这里萌发，后来在 InterFormer v0.4 正式启用

### Interformer_Pytorch_reference · 空目录占位

- **状态：** 空（只有 `.` 和 `..`）
- **作用推测：** 预留给 InterFormer 官方 PyTorch 参考实现的占位目录，至今未填充

---

## 4.C datasetAnalysis · 项目"侦察兵"沙箱 + 研究纲领（不是版本）

> **重要性：** 这不是普通的 EDA 工具，而是**整个项目的研究纲领**。HyFormer v0.3.1 系列和 InterFormer v0.5 的所有特征工程改进都源自这里的 7 条发现。

### 文件清单

```
datasetAnalysis/
├── README.md              # 7 条关键发现 + 优化优先级排序
├── data_analysis.py       # Jaccard 相似度全量扫描脚本
├── schema.json            # 全量推理 schema（120 列）
├── demo_1000.parquet      # 1000 行 demo 数据（实际仅 133 字节，header-only）
├── demo_1000_copy.parquet # 上者的副本
├── dataset.py / model.py / utils.py    # == HyFormer-v0.0（完全一致）
├── ns_groups.json         # == HyFormer-v0.3.0-基础HyFormer（完全一致）
├── train.py               # HyFormer-v0.0 + AMP + compile 支持
├── trainer.py             # HyFormer-v0.0 + 完整 AMP 训练流程（正确实现，含 GradScaler）
└── run.sh                 # RankMixer NS tokenizer + AMP 加速
```

### 7 条关键发现（来自 README.md）

**发现 1：8 对 (Key, Value) 对齐的特征对**
- fids 62-66 和 89-91 在 user_int（length）和 user_dense（dim）中共享相同维度
- 形成自然的（实体 ID, 统计值）配对
- baseline 模型将它们独立处理 → **结构性信息浪费**
- → 催生了 v0.3.1+2.a（KV 加权 embedding）

| fid | int (vocab × length) | dense dim | 结构 |
|-----|---------------------|-----------|------|
| 62 | 11 × 6 | 6 | 金字塔（粗） |
| 63 | 49 × 19 | 19 | 金字塔 |
| 64 | 51 × 26 | 26 | 金字塔 |
| 65 | 425 × 111 | 111 | 金字塔 |
| 66 | 1403 × 150 | 150 | 金字塔（细） |
| 89 | 10 × 10 | 10 | 扁平 |
| 90 | 10 × 10 | 10 | 扁平 |
| 91 | 10 × 10 | 10 | 扁平 |

**发现 2：Item Dense 为空（0 维）**
- baseline 的 `item_dense_proj` 模块完全闲置
- → 催生了 v0.3.1+1.a（log freq）和 +1.b（贝叶斯 CTR）

**发现 3：兴趣金字塔结构（fids 62→66）**
- 词表从 11 增长到 1403，从粗到细的兴趣层级
- 适合层级注意力建模
- → **直接催生了 v0.3.1+2.b（HierarchicalKVFusion）** —— 论文级创新点

**发现 4：真正的 VRAM 瓶颈 = 序列 Embedding**
- 内存压力来自超大序列词表（86M, 64M）
- 而非 user/item 特征（最大 23K）
- 4 个 vocab > 1M 的序列特征当前被替换为零向量

**发现 5：序列 Domain 不平衡**
- seq_b（14 特征, 64M max vocab）和 seq_c（12 特征, 86M max vocab）远比 seq_a 和 seq_d 复杂
- 但它们共享相同的 Transformer 权重

**发现 6：推理时时间戳不可用**
- 所有 4 个序列时间戳特征在推理 schema 中 vocab=0
- 时间特征必须使用相对时间差 + 分桶

**发现 7：优化优先级排序**
```
最高优先   物品统计特征（5-15 行代码，即刻见效）   ← 已实现 v0.3.1+1.a/1.b
  ↓        (Key, Value) 加权 Embedding（论文级创新）  ← 已实现 v0.3.1+2.a/2.b
  ↓        高基数序列特征截断（释放 VRAM）
  ↓        绝对时间差特征（增强时序建模）
最低优先   用户 Dense 预训练向量利用
```

### data_analysis.py 核心逻辑

```python
def calc_jaccard(list1, list2):
    """计算两个列表的 Jaccard 相似度"""
    s1, s2 = set(list1), set(list2)
    if len(s1) == 0 or len(s2) == 0:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))
```

- **扫描范围：** 11 个 user 多值特征 × 45 个 sequence 特征 = 495 个特征对
- **统计指标：** Non_Zero_Overlap_Ratio、Mean_Jaccard
- **产出 3 张可视化：** top_features_heatmap.png / scalar_vs_seqlen.png / length_correlation.png

### datasetAnalysis 的 AMP 实现（v0.4 NaN 故事的关键参考）

datasetAnalysis 的 `trainer.py` 实现了**完整正确**的 AMP 训练流程：

```python
# __init__
self.use_amp = use_amp and device.startswith('cuda')
self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

# _train_step
with torch.amp.autocast('cuda', enabled=self.use_amp):
    logits = self.model(model_input)
    loss = F.binary_cross_entropy_with_logits(logits, label)

self.scaler.scale(loss).backward()
self.scaler.unscale_(self.dense_optimizer)
torch.nn.utils.clip_grad_norm_(...)
self.scaler.step(self.dense_optimizer)
self.scaler.update()
```

**关键观察：v0.4 在引入 AMP 时只借走了 `autocast`，遗漏了 `GradScaler`**——这就是 step 1796 NaN crash 的直接根因。datasetAnalysis 提供了正确的实现示范，但 v0.4 没有完整借鉴。v0.5 的修复实质上就是回到 datasetAnalysis 的完整实现。

---

# 第五部分：累积发现（Insights）

> **每次实验后，把通用结论提炼到这里。**
> **paper 的 Discussion / Conclusion 章节就从这里来。**

## 5.1 关于训练动力学

- **AMP 的"半截借鉴"陷阱：** v0.4 引入 `autocast` 但漏掉 `GradScaler` 导致 step 1796 NaN crash。AMP 是一套完整流程（autocast + scaler.scale + unscale + step + update），任何一环缺失都可能导致 FP16 上溢污染权重
- **NaN 的不可逆性：** 一旦权重出现 NaN，所有后续 forward 都吐 NaN，**无法自愈**。必须从 checkpoint 重启
- **warmup 的必要性：** v0.5 引入 1000 步线性 warmup 后，配合 GradScaler 让大模型能稳定起步。HyFormer baseline 缺少 warmup 是 AUC 早 plateau 的原因之一
- **Adagrad 学习率衰减问题：** Adagrad 的累积量 s 只增不减，训练久了高基数 embedding 学习率衰减到接近 0。`reinit_high_cardinality_params` 本意是解决这个问题，但 v0.5 发现其逻辑反转——这是一个值得论文记录的"工程教训"
- **sparse_lr 过大导致 loss spike：** baseline 默认 sparse_lr=0.05 会让 loss 频繁 spike 到 0.5-0.75（正常 0.15-0.3）。v0.5 把 sparse_lr 降到 0.01 是稳定训练的关键

## 5.2 关于架构设计

- **架构 > 调参：** InterFormer v0.1 用 d_model=64、零调参、Eval AUC 0.8067，跑赢了 HyFormer v0.2 大幅调参的 0.7952。**架构选择的天花板高于超参数调优**
- **双向 NS-Seq 交互优于单向 Query Decoding：** HyFormer 是 NS → Seq 单向（Query 从 Seq 取信息），InterFormer 引入双向 interaction tokens，让 NS 和 Seq 能互相影响。从 v0.1 的实证看，双向收益显著
- **DHEN + DCN 的容量扩展：** v0.4 在 InterFormer 上加 DHEN（深度异构集成）+ DCN（Deep & Cross Network v2）+ RoPE，model.py 从 569 → 1527 行（+958 行）。v0.5 证明这种容量扩展能爬到 0.852 峰值，但训练稳定性需要配套保障
- **金字塔注意力来自数据特性：** v0.3.1+2.b 的 HierarchicalKVFusion 不是凭空设计，而是直接源自 datasetAnalysis 发现 3——fids 62→66 的词表从 11→1403 的天然层级。**数据洞察 → 算法设计**是论文最有说服力的叙事
- **HyFormer 的天花板：** v0.2 一次性堆叠 5 个改进反而退步，v0.3.1 ablation 系列大多与 baseline 持平或略降——这暗示 HyFormer 架构的优化空间确实有限

## 5.3 关于特征工程

- **Item Dense 空缺是机会：** baseline 的 `item_dense_proj` 完全闲置（item_dense_dim=0）。通过统计特征（log frequency、贝叶斯 CTR）填补可以提供新信号
- **统计特征要防标签泄露：** v0.3.1+1.a 的 `_build_item_freq_map` 验证集复用训练统计 —— 这是标准防泄露做法
- **贝叶斯平滑解决低曝光抖动：** v0.3.1+1.b 用 `smooth_ctr = (clicks + α·global_ctr) / (impressions + α)`，α=100 让低曝光 item 倾向于全局均值，避免高方差噪声
- **(Key, Value) 对齐特征需要专门处理：** datasetAnalysis 发现 8 对天然对齐特征（fid 62-66, 89-91），baseline 用 mean pool 是信息浪费。v0.3.1+2.a 用 dense 值做 attention 权重对 int embedding 加权 —— softmax 温度系数（T=2）防止权重过于尖锐
- **数据洞察驱动特征工程：** 整个 v0.3.1 系列的设计都源自 datasetAnalysis 的 7 条发现，这是规范的"先 EDA 再建模"流程

## 5.4 关于数据本身

- **120 列结构：** 46 user_int + 14 item_int + 10 user_dense + 0 item_dense + 4 seq domains（共 45 个 sequence 特征）
- **序列长度的工程妥协：** seq_max_lens {a:256, b:256, c:512, d:512}，c/d 域是 a/b 的 2 倍，暗示长期行为对 CTR 预测更重要
- **Truncation vs Padding：** 截断丢真实信息（重度用户早期行为），Padding 引入假信息（模型需要学 mask），两者都是对 GPU 矩阵乘法的妥协
- **正样本率 ~9.6%：** 中等不平衡，**不是极端稀疏**（v0.5 早期叙事误标为 0.1%）。这意味着 BCE 默认是稳定的选择，Focal Loss 的收益需要谨慎评估
- **VRAM 瓶颈不在 user/item 而在序列：** 4 个 vocab > 1M 的序列特征被替换为零向量是 baseline 的隐式压力点

## 5.5 关于实验方法论（来自团队真实教训）

- **"一次只改一个变量" 原则（InterFormer v0.3 README 自反思）：** v0.3 同时改 4 个变量（focal+bs 1024+epoch 3+冷启）导致退步，无法归因。HyFormer v0.2 同时堆 5 个改动也是同样的失败模式
- **诊断 → 文档 → 修复的工作流（InterFormer v0.4 → v0.5）：** v0.4 NaN crash 后，团队没有立即猜测原因，而是写了 247 行 README 完整剖析+排除假设表+修复建议。v0.5 直接按这份诊断清单落地。**这是教科书级的工程实践**
- **失败也是 contribution：** v0.2、v0.3 都是负面结果，但它们的诊断（"HyFormer 调参空间有限"、"一次只改一个"）成为后续决策（切换 InterFormer、规范消融）的依据
- **统一指标定义的重要性：** v0.1 报"Eval AUC"，v0.5 报"峰值 Val AUC"，两个数字方法论不同不能直接相减。**写论文前必须统一指标口径**

---

# 第六部分：阻塞与风险

## 6.1 当前阻塞项

- [ ] **关键 Bug：v0.5 的 embedding reinitialization 逻辑反转** —— 必须修复后重跑，才能知道 v0.5 架构的真实 ceiling（不是被 bug 偷走的 0.852）
- [ ] **关键 Bug：`model.py:480 del seq_encoder_type`** —— 让 `--seq_encoder_type` 参数静默失效，所有想换 encoder 的实验都受影响
- [ ] **GAUC 硬编码为 0** —— 比赛主指标可能用 GAUC，必须实现真实的分组 AUC 计算
- [ ] **多个版本无训练结果：** v0.3.1+1.a+1.b / +2.a / +2.b / +Interformer+1.a 都没有 AUC 数据。**需要找 Zesong 确认状态**
- [ ] **v0.5 之后没有新版本** —— 需要规划 v0.6（基于 v0.5 修 bug 后的稳定版）

## 6.2 已识别风险

| 风险 | 概率 | 影响 | 缓解策略 |
|------|------|------|---------|
| 算力不够，跑不完所有对照实验 | 中 | 高 | 优先级排序，只做核心实验（特别是 v0.5 修 bug + v0.3.1+2.b 金字塔 KV） |
| 研究方向和现有 paper 撞车 | 中 | 高 | 投稿前再做一次文献检索；金字塔 KV 是相对新颖的方向 |
| 英文写作时间不够 | 低 | 中 | 提前 2 周开始写作；先用本 Research Log 作为初稿骨架 |
| v0.5 修 bug 后效果不如预期 | 中 | 高 | 准备 backup 叙事：以"训练稳定性诊断与修复"为 main contribution |
| v0.3.1+2.b（金字塔 KV）跑不出来 | 中 | 中 | 退而求其次，用 v0.3.1+2.a-update 的数据撑论文 |
| InterFormer 架构在论文中不够新颖 | 中 | 中 | 强调"训练稳定性 + 工程修复 + 数据洞察驱动特征工程"组合贡献 |

## 6.3 给 Zesong 的问题清单（需要补充的信息）

> 这些是基于扫描报告无法独立回答的问题，需要找 Zesong 确认。

**版本时间线：**
1. HyFormer v0.1 / v0.2 / v0.3.0 / v0.3.1 系列各自的训练日期？
2. InterFormer v0.1 / v0.3 / v0.4 / v0.5 各自的训练日期？
3. `HyfFormer-v0.1`（typo 目录）是什么？

**缺失的训练结果：**
4. HyFormer v0.3.1+1.a+1.b（融合版）有跑过吗？为什么没有 PNG？
5. HyFormer v0.3.1+2.a（KV 加权扁平版）有跑过吗？为什么没有 PNG？
6. HyFormer v0.3.1+2.b（**金字塔 KV，潜在论文创新点**）有跑过吗？AUC 是多少？
7. HyFormer v0.3.1+Interformer+1.a（**架构对照试验**）有跑过吗？AUC 是多少？
8. HyFormer v0.3.1+2.a-update 的具体 AUC 是多少？

**关键技术问题：**
9. v0.2 失败的主因，你的直觉是哪个改动？（focal? LongerEncoder? d_model 翻倍?）
10. v0.5 的"峰值之后 decay 到 0.760"这个数字是从 PNG 视觉读图还是有具体 log？
11. v0.5 之后是否计划过 v0.6？规划是什么？

**v0.0 baseline 是否提交过线上 Eval？**
12. 子健亲跑的 HyFormer v0.0（Val AUC ~0.85）是否提交过腾讯平台，得到了线上 Eval AUC？

---

# 第七部分：参考资料

## 7.1 已生成的学习文档（项目知识库）

- `Phase_1.html` — dataset.py 完全理解（数据管道三步本质：分类 → 对齐 → 打包）
- `Phase_2.html` — model.py 完全理解（NS Tokenizer → 序列编码 → Query 三件套 → HyFormer Block → 输出）
- `Phase_3.html` — trainer.py 完全理解（双优化器、梯度裁剪、EarlyStopping、loss 函数）
- `Structure.html` — HyFormer 全架构五文件关系图谱
- `Baseline_Guide_Structure.html` — Baseline 五阶段学习路线图
- `HyFormer_全系列扫描.md` — HyFormer 11 个不同实验的完整 diff 扫描
- `InterFormer_全系列扫描.md` — InterFormer 4 个版本 + Hyformer_baseline 的完整 diff 扫描
- `datasetAnalysis_扫描.md` — datasetAnalysis 沙箱的完整剖析

## 7.2 关键论文（项目知识库）

- `HyFormer__Revisiting_the_Roles_of_Sequence_Modeling_and_Feature_Interaction_in_CTR_Prediction.pdf` — Tencent 2024
- `InterFormer_Effective_Heterogeneous_Interaction_Learning_for_ClickThrough_Rate_Prediction.pdf` — CIKM 2025
- `OneTrans__Unified_Feature_Interaction_and_Sequence_Modeling_with_One_Transformer_in_Industrial_Recommender.pdf` — 2024

## 7.3 比赛官方

- 官网：algo.qq.com
- Competition Introduction PDF：在项目知识库
- 训练平台：taiji.algo.qq.com

## 7.4 项目内部分析文档

- `Hyformer/HyFormer-v0.3.0-基础HyFormer/dataset-analysis.md` — 数据分析文档（在 v0.3.0 目录内）
- `Interformer/v0.4/Readme.md` — NaN crash 完整诊断报告（247 行）⭐ 必读
- `Interformer/v0.5/README.md` — 峰值 0.852 + Known Issues
- `datasetAnalysis/README.md` — 7 条关键发现 + 优化优先级 ⭐ 必读

---

# 第八部分：论文叙事预备（草案）

> 这一节是基于已有实验事实的潜在论文故事框架，**不是结论**。等所有实验数据补齐后再正式定型。

## 8.1 潜在论文主线

**标题草案：** "Training Stability and Hierarchical Feature Interaction in Industrial CTR Prediction: A Diagnostic Study on InterFormer"

**核心叙事弧：**

1. **Motivation：** HyFormer 作为 baseline 调参空间有限（v0.2 失败实证）；InterFormer 双向交互更有潜力
2. **架构选择：** 对比 HyFormer vs InterFormer（v0.1 数据 + v0.3.1+Interformer+1.a 对照试验）
3. **训练稳定性：** v0.4 NaN crash 完整诊断 → v0.5 GradScaler + warmup + Focal 修复 → 峰值 AUC 0.852
4. **数据洞察驱动的特征工程：** datasetAnalysis 7 条发现 → 金字塔层次 KV 融合（v0.3.1+2.b）+ KV 加权 embedding（v0.3.1+2.a-update）
5. **Known Issues 与 future work：** Embedding reinit bug 揭示了工程实现层面的隐患

**Contribution 列表（草案）：**
- C1: 实证对比 HyFormer vs InterFormer 在工业 CTR 任务上的架构效率差异
- C2: AMP 训练在大规模推荐模型上的故障模式诊断（FP16 上溢 → 永久权重污染）
- C3: 数据驱动的金字塔层次注意力机制（基于天然的粗→细兴趣层级）
- C4: 工程实现陷阱的揭示（reinit 逻辑反转、del seq_encoder_type 静默失效、GAUC 硬编码）

## 8.2 论文的"王牌"（独特卖点）

- **真实工业数据 + 真实工业故障：** 不是合成数据上的玩具实验，而是腾讯真实日志 + 真实 NaN crash + 真实 bug 暴露
- **诊断方法论：** v0.4 → v0.5 的"出问题→写文档→分类讨论→修复"工作流可以作为方法论贡献
- **数据洞察 → 算法设计的完整链条：** datasetAnalysis 7 条发现 → 具体模块设计（HierarchicalKVFusion）的因果链清晰
- **诚实的负面结果：** v0.2 / v0.3 失败 + 一次只改一个变量的反思，比单纯报最高分更有说服力

---

# 更新日志

| 日期 | 更新人 | 改动概要 |
|------|--------|---------|
| 2026-05-14 | 子健 | 初始化研究日志，填入 v0.0 信息 |
| 2026-05-14 | 子健 + Claude | **v2.0 大重构**：整合 HyFormer 全系列扫描 + InterFormer 全系列扫描 + datasetAnalysis 扫描；补全所有 17 个版本的事实档案；提炼 5 大维度的累积发现；列出给 Zesong 的 12 个待确认问题；起草潜在论文叙事 |
| _[待补]_ | | |

---

*Last updated: 2026-05-14*
*Maintained by: 徐子健（TAAC2026 项目规划）*
*Use this file as context when chatting with any AI tool (Claude, DeepSeek, GPT, etc.)*

---

# 附：使用建议

**对当前的子健：**
- 这份 Research Log 是项目的"事实档案"，事实部分（4.A / 4.B / 4.C）由扫描报告支撑，准确度高
- 标了 `[❓]` 的地方是需要找 Zesong 确认的，不影响日常使用
- 第五部分（Insights）是潜在论文素材，每次新实验跑完后回来更新
- 第六部分（阻塞与风险）是行动清单，按优先级处理

**对未来接手的人/AI：**
- 项目从 HyFormer 转向 InterFormer 的关键时间点：2026-05-04
- 当前最高峰值：InterFormer v0.5 的 Val AUC 0.852（被 bug 拖崩）
- 最值得追的版本：HyFormer v0.3.1+2.b（金字塔 KV，潜在论文创新点）
- 必读的项目内部文档：`Interformer/v0.4/Readme.md`（NaN 诊断）+ `datasetAnalysis/README.md`（7 条发现）
