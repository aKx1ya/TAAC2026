#TAAC-KDD Cup 2026 (ACM SIGKDD) Tencent Advertising Algorithm Competition

#"The score is currently 0.823689"#
![Score](top_rank.png)
---

## 🚀 PCVRHyFormer v9 改进内容（自己懒得写了，deepseek写一下吧，涨分经验不多，掉分经验挺多）（主要在架构/item特征/泛化手段/数据增强里面涨分，xhs很多佬的提的时间特征/din等，我加的太过了，基本都是掉点，没有达到他们那种级别的涨分设计）

相比 baseline，v9 版本引入了以下核心改进：

---

### 1. 模型架构增强

| 模块 | 修改内容 |
|------|----------|
| **HashEmbedding** | 新增多哈希碰撞缓解的嵌入层，支持超高频特征的哈希分桶 |
| **DINTargetAttention** | 新增 DIN 风格目标注意力机制，通过 MLP(concat(q, k, q-k, q*k)) 计算加权序列表示 |
| **UserSparseDensePairResidual** | 新增用户稀疏 ID 与稠密值的配对残差模块，加权融合多值统计特征 |
| **NSSelfAttention** | 新增 NS 令牌自注意力层，在 HyFormerBlock 内部进行特征交叉 |
| **CrossNet (DCN-v2)** | 新增 DCN-v2 交叉网络，支持低秩分解和输出门控，可叠加在 NS 令牌上 |
| **SENetGating** | 新增 SE-Net 特征门控，对 NS 令牌进行 squeeze-and-excitation 重加权 |
| **TemporalBias** | 新增可学习的时间衰减偏置，每头独立 alpha 参数，用于注意力 logits 的加法偏置 |

---

### 2. 时间特征工程

| 特征 | 修改内容 |
|------|----------|
| **seq_time_deltas** | 新增 log1p(dt/3600) 连续时间差，用于 TemporalBias 计算 |
| **seq_time_gaps** | 新增 log1p(gap_minutes) 相邻事件时间间隔 |
| **seq_time_hours** | 新增事件发生的小时 (1..24)，用于绝对时间建模 |
| **seq_time_weekdays** | 新增事件发生的星期几 (1..7)，用于周期模式捕捉 |
| **seq_time_span_buckets** | 新增离散的时间跨度桶（15 个边界），互补连续 gap 特征 |
| **cyclical_time_proj** | 新增 sin/cos 循环时间编码（小时/星期），经可学习投影后加性融合 |

---

### 3. 训练优化技巧

| 技巧 | 修改内容 |
|------|----------|
| **BF16 混合精度** | 新增 `--precision` 参数（auto/fp32/bf16），支持 BF16 自动混合精度训练 |
| **EMA (指数移动平均)** | 新增 `--ema_decay` 参数，维护影子权重，验证时使用平滑参数 |
| **学习率调度** | 新增 `--lr_schedule`（cosine）+ `--warmup_steps`，支持 warmup + 余弦退火 |
| **标签平滑** | 新增 `--label_smoothing` 参数，`y' = y*(1-ε) + 0.5*ε` |
| **Pairwise Ranking Loss** | 新增 `--loss_type bce_pairwise`，`loss = BCE + λ * pairwise_rank_loss` |
| **权重衰减** | 新增 `--weight_decay` 参数，AdamW 密集参数正则化 |

---

### 4. NS 令牌处理增强

| 组件 | 修改内容 |
|------|----------|
| **RankMixerNSTokenizer** | 新增 `hash_bucket_size` 支持，对超高基数特征使用哈希嵌入而非零向量 |
| **GroupNSTokenizer** | 同步增加哈希嵌入支持 |
| **QueryGenerator** | 将 `target_emb` 加入全局信息拼接，实现 item-conditioned 查询生成 |
| **NS 输出融合** | 新增 `--use_ns_output_fusion`，将池化的最终 NS 令牌融合到输出 |
| **跨层交叉网络** | 新增 `--num_cross_layers` / `--cross_low_rank`，可堆叠 DCN-v2 层 |

---

### 5. 序列编码器增强

| 组件 | 修改内容 |
|------|----------|
| **TransformerEncoder** | 新增 `time_delta` + `temporal_bias_module` 参数，支持时间感知自注意力 |
| **LongerEncoder** | 新增 `temporal_bias_module` 支持，压缩时同步传递 `time_delta` |
| **MultiSeqHyFormerBlock** | 新增 `seq_time_deltas` 传播、`query_temporal_bias` / `seq_temporal_bias` |
| **返回值元组统一** | 所有 encoder 返回 `(output, mask, time_delta)`，保持接口一致 |

---

### 6. 数据集增强

| 字段 | 修改内容 |
|------|----------|
| **seq_time_delta** | 新增 `log1p(时间差/3600)` 连续值 |
| **seq_time_gap** | 新增 `log1p(事件间隔/60)` 连续值 |
| **seq_time_hour** | 新增事件小时 (1..24) |
| **seq_time_weekday** | 新增事件星期几 (1..7) |
| **seq_time_span_bucket** | 新增离散时间桶，边界 `[30,60,180,300,600,900,1800,3600,10800,21600,43200,86400,172800,345600,604800]` 秒 |
| **用户稠密对数变换** | 对 fid ∈ {62,63,64,65,66} 应用 `log1p(abs(val)) * sign(val)` |
| **TIME_SPAN_BUCKETS** | 新增 `NUM_TIME_SPAN_BUCKETS = len(boundaries) + 2` 常量 |

---

### 7. 命令行参数新增

```bash
# 精度与优化
--precision {auto,fp32,bf16}
--ema_decay 0.999
--warmup_steps 500
--lr_schedule {none,cosine}
--label_smoothing 0.01
--weight_decay 0.02
--pairwise_lambda 0.05

# 模型组件开关
--hash_bucket_size 100000
--num_cross_layers 2
--cross_low_rank 64
--cross_dropout 0.1
--use_se_net
--use_target_attention (默认开启)
--use_ns_self_attn
--use_ns_output_fusion
--use_temporal_bias
--use_time_gap

# 时间特征
--use_calendar_time (默认开启)
--use_time_span_buckets (默认开启)
--no_time_buckets (禁用)
```

---

### 8. 代码质量改进

| 类型 | 修改内容 |
|------|----------|
| **类型注解** | 新增 `pairwise_log_time_delta` 等辅助函数的返回类型 |
| **文档字符串** | 补充 `_make_user_dense_token`、`_seq_mean_last_pool` 等方法的说明 |
| **常量定义** | 新增 `USER_SPARSE_DENSE_PAIR_FIDS`、`TIME_SPAN_BUCKET_BOUNDARIES` |

---

### 9. 关键设计决策

- **NS 自注意力**：在 HyFormerBlock 内部的 NS 令牌上增加自注意力，不跨 block 传递，位置中性（无 RoPE）
- **时间偏置公式**：`bias = -α_head * log1p(|expm1(dt_q) - expm1(dt_k)|)`，可微且数值稳定
- **配对残差**：`embedding × (sign(x) * log1p(|x|)) / mask_count`，平衡稀疏/稠密信号
- **EMA + 重初始化**：高基数特征重置后调用 `ema.resync()`，保持影子权重同步

---

**总结**：v9 在 baseline 的基础上引入了工业级推荐系统的多项成熟技术（DIN、DCN-v2、SE-Net、EMA、时间感知注意力），并通过 BF16 + 余弦调度 + 标签平滑等训练技巧提升训练效率与收敛稳定性。
<img src="share.png" alt="网友分享的图片" height="900" />
