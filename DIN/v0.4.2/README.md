# 🔧 v9.0 → v9.0.2 调参优化

> 基线 AUC: 0.8267 | 目标：不改架构，只调参提分

### 改动一览

| 参数 | v9.0 | v9.0.2 | 原因 |
|------|------|--------|------|
| `d_model` | 64 | **128** | 容量翻倍，表达空间 ↑ |
| `num_heads` | 4 | **8** | 头数随 d_model 等比放大 |
| `use_se_net` | False | **True** | NS Token 自适应加权 |
| `use_ns_self_attn` | False | **True** | Token 间特征交叉 |
| `dropout_rate` | 0.01 | **0.05** | 配合大模型防过拟合 |
| `loss_type` | bce | **focal** | 处理 CVR 正负不平衡 |
| `focal_alpha` | 0.1 | **0.25** | 正样本权重 ↑ |
| `label_smoothing` | 0.0 | **0.05** | 标签软化防过拟合 |
| `batch_size` (run.sh) | 256 | **128** | d_model 翻倍后显存不够 |

### run.sh 精简

去掉已变默认值的 flags：`--use_se_net` `--use_ns_self_attn` `--use_target_attention` `--label_smoothing`

`loss_type` 从 `bce_pairwise` 切回默认 `focal`，如需 pairwise 见 run.sh 注释。