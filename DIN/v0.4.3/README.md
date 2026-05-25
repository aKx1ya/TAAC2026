# � v9.0.3 — 方案A：消融实验（Focal → BCE+Pairwise）

> 目标：验证 v9.0.2 掉分是否源于 Focal Loss 替换了 Pairwise Ranking Loss

### v9.0.2 → v9.0.3 唯一改动

| 参数 | v9.0.2 | v9.0.3 | 原因 |
|------|--------|--------|------|
| `loss_type` | focal | **bce_pairwise** | 回退到 v9.0 的 pairwise loss，验证是否是掉分元凶 |
| `pairwise_lambda` | — | **0.05** | 与 v9.0 保持一致 |

其他所有参数（dropout=0.05, label_smoothing=0.05, batch_size=128 等）**完全不变**。

### 消融逻辑

```
v9.0 ──(改4个参数)──▶ v9.0.2 (掉分)
                        │
                        └──(只把loss切回pairwise)──▶ v9.0.3 ← 当前版本
```

- 如果 v9.0.3 涨回来了 → **Focal Loss 是罪魁祸首**
- 如果 v9.0.3 还是掉分 → 问题在 dropout/label_smoothing，需继续消融