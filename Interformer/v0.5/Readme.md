# v0.5 — PCVRInterFormer 增强版（NaN 修复 + 极端稀疏样本适配）

## 一、版本定位

v0.5 在 v0.4 基础上完成了两类工作：
1. **🔴 致命修复**：解决 v0.4 训练中 Loss 在 step 1795 突变为 NaN 且永不恢复的崩溃
2. **🟡 稀疏样本适配**：针对 ~0.1% 正样本率的极端类别不平衡场景进行全面增强

> 崩溃根因分析详见 [v0.4/Readme.md](../v0.4/Readme.md) 第四节。

---

## 二、v0.4 → v0.5 完整变更清单

### 2.1 超参数变化

| 参数 | v0.4 | v0.5 | 理由 |
|------|------|------|------|
| `loss_type` | `bce` | **`focal`** | BCE 在 0.1% 正样本率下会被负样本梯度淹没 |
| `focal_alpha` | -- | **0.25** | 正样本权重 0.25 → 相对负样本提升 ~250× |
| `focal_gamma` | -- | **2.0** | 标准配置，抑制易分类负样本的梯度 |
| `sparse_lr` | 0.05 | **0.01** | 降低稀疏 Embedding 的 Adagrad LR |
| `eval_every_n_steps` | 0（关闭） | **1000** | 每 1000 步验证 + 存 checkpoint |
| `warmup_steps` | 无 | **1000**（内置） | 线性 LR warmup，防止冷启动梯度爆炸 |
| `gradient_accumulation_steps` | 无 | **1**（内置，可配） | 预留梯度累积，设为 4 可模拟 batch≈1536 |

### 2.2 关键修复

| 文件 | 修改内容 | 优先级 |
|------|----------|--------|
| `trainer.py` | **添加 `torch.amp.GradScaler`** | 🔴 致命 |
| `trainer.py` | **线性 LR Warmup**（前 1000 步逐步提升稠密参数 LR） | 🟡 核心 |
| `trainer.py` | **梯度累积框架**（`_train_step` 只 backward，`train()` 控制 step 频率） | 🟡 核心 |
| `trainer.py` | **评估指标增强**：`evaluate()` 返回 dict，新增 `precision_at_1pct`、`pos_rate` | 🟡 核心 |
| `run.sh` | **`--loss_type focal --focal_alpha 0.25 --focal_gamma 2.0`** | 🔴 核心 |
| `run.sh` | `--sparse_lr 0.05` → **`0.01`** | 🟡 防御 |
| `dataset.py` | **log1p 前加 `np.maximum(x, -0.999)` 保护** | 🟡 防御 |

### 2.3 代码级变更亮点

#### ① Focal Loss（解决 0.1% 正样本率下 BCE 失效）
```python
# run.sh
--loss_type focal --focal_alpha 0.25 --focal_gamma 2.0
```
Focal Loss 对"容易分类"的负样本自动降权 $(1-p_t)^\gamma$，逼迫模型关注稀缺的正样本。

#### ② 线性 LR Warmup
```python
if total_step < self.warmup_steps:
    progress = total_step / self.warmup_steps
    warmup_lr = self._base_dense_lr * progress
    for pg in self.dense_optimizer.param_groups:
        pg['lr'] = warmup_lr
```

#### ③ 梯度累积
```python
# _train_step: 只 backward，不 step
if self.scaler is not None:
    self.scaler.scale(loss).backward()
else:
    loss.backward()

# train() 循环：每 gradient_accumulation_steps 步才 unscale → clip → step
```

#### ④ 增强评估指标
`evaluate()` 返回 `Dict[str, float]`，包含：

| 指标 | 说明 |
|------|------|
| `auc` | 全量 AUC |
| `logloss` | 二分类对数损失 |
| `precision_at_1pct` | Top-1% 预测的精确率——衡量极稀疏正样本的排序质量 |
| `pos_rate` | 验证集正样本率——诊断数据分布问题的快速指标 |

---

## 三、代码结构

```
v0.5/
├── dataset.py       # Parquet 数据集加载（log1p 防护）
├── model.py         # PCVRInterFormer 模型定义
├── trainer.py       # 训练器（GradScaler + Warmup + 梯度累积 + 增强评估）
├── train.py         # 训练入口
├── utils.py         # 工具函数（sigmoid_focal_loss, EarlyStopping）
├── ns_groups.json   # 非序列特征分组配置（备用）
├── run.sh           # 启动脚本（Focal Loss + sparse_lr=0.01）
├── Readme.md        # 本文件
└── Evaluation/      # 离线评估脚本
```

---

## 四、如何启动

### 4.1 全新训练

```bash
cd v0.5
bash run.sh
```

### 4.2 启用梯度累积（模拟 batch_size ≈ 1536，提升稀疏正样本梯度稳定性）

```bash
bash run.sh --gradient_accumulation_steps 4
```

### 4.3 从 v0.4 checkpoint 恢复

```bash
bash run.sh --resume_from global_step2396.layer=2.head=4.hidden=64.best_model
```

### 4.3 额外调参（按需覆盖）

```bash
bash run.sh \
    --sparse_lr 0.005 \
    --eval_every_n_steps 500 \
    --batch_size 256 \
    --dropout_rate 0.05
```

---

## 五、NaN 崩溃机制复盘

```
AMP autocast (FP16 前向)
        │
        ▼
  某 batch 激活值偏大
        │
        ▼
  梯度超出 FP16 最大值 65504 → inf
        │
        ▼
  optimizer.step() 将 inf 写入权重 → 权重 = NaN
        │
        ▼
  后续所有 forward 输出 NaN（永续传染）
```

**修复后的保护链**：

```
GradScaler 将 loss 乘以动态缩放因子
        │
        ▼
  FP16 下梯度保持健康范围（不再溢出）
        │
        ▼
  unscale_() 恢复真实梯度
        │
        ▼
  clip_grad_norm_(max_norm=1.0) 二次截断
        │
        ▼
  scaler.step() 安全更新权重
```

---

## 六、预期效果

| 指标 | v0.4（有 Bug） | v0.5（修复后预期） |
|------|----------------|---------------------|
| 训练稳定性 | Step ~1795 突然 NaN | 全程无 NaN（GradScaler） |
| 冷启动安全性 | 无 warmup，第一步就全 LR | 前 1000 步线性提升 LR |
| 正样本学习 | BCE，负样本梯度淹没正样本 | Focal Loss，自动聚焦难分样本 |
| 评估说服力 | 仅 AUC + LogLoss | AUC + LogLoss + P@1% + pos_rate |
| 稀疏参数 LR | 0.05（偏高） | 0.01（适中） |
| 验证频率 | 仅在 epoch 结束时 | 每 1000 步一次 |
| log1p 安全性 | 无保护 | 有 clip 保护 |
| 梯度累积 | 无 | 框架已就绪，改参数即用 |

---

## 七、与 v0.4 的关系

- **v0.5 不包含新的模型架构变更**，纯粹是 bug 修复
- v0.4 的 checkpoint (`global_step2396.*.best_model`) 可在 v0.5 中直接加载恢复训练
- 如果你在 v0.4 基础上已经手动改了 `trainer.py` 并运行成功，可以跳过 v0.5 直接使用 v0.4
