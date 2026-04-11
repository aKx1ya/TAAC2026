# InterFormer: 有效的 CTR 预测异构交互学习
# (Effective Heterogeneous Interaction Learning for Click-Through Rate Prediction)

## 1. 核心概述
* **解决痛点**：解决推荐系统中“动态行为”与“静态画像”的融合问题 。
* **背景与验证**：Meta 基于真实广告系统中 700 亿级样本和长度为 1000 的序列数据进行了实验，证实了 InterFormer 的有效性 。

---

## 2. 三大核心架构设计

### 1. Interaction Arch（特征交叉网络）
* **目标**：让静态画像学习“感知行为”的非序列特征 。
* **机制**：
    * 不仅计算用户画像、商品属性之间的交叉，还会引入**序列摘要 (Sequence Summarization)** 共同进行特征交叉 。
    * **逻辑比喻**：不仅参考用户的长期偏好，还结合其即时的点击行为（刚刚点击了什么），动态调整对该用户的画像认知 。

### 2. Sequence Arch（序列建模网络）
* **目标**：学习“感知上下文”的序列特征 。
* **机制**：
    * 使用了**个性化前馈网络 (PFFN)** 和**多头注意力 (MHA)** 。
    * **CLS Token 引导**：在处理用户行为序列前，将**非序列摘要 (Non-sequence Summarization)** 作为 CLS Token 插入序列最前面，作为 Query 引导整条序列的注意力分配 。

### 3. Cross Arch（信息桥）
* **目标**：在不破坏原始高维信息的前提下，进行有效的信息筛选和浓缩 。
* **机制**：
    * **自门控 (Self-gating)**：由于直接交换原始特征矩阵算力开销巨大且噪音多，使用该技术对非序列特征进行“脱水提纯”，过滤无效信息，提取高密度向量 。
    * **信息提炼**：使用 **CLS Token**、**PMA Token** 以及 **Recent** 精准概括序列信息 。
    * **双边交换**：提纯后的信息跨界交换，在保留完整度的同时实现高效沟通 。

---

## 4. 关键术语解析

* **序列摘要 (Sequence Summarization)**：用于过滤误触行为 。
* **非序列摘要 (Non-sequence Summarization)**：高度概括的标签信息 。
* **CLS Token (Classification)**：直接将非序列摘要插在用户序列最前面，辅助过滤信息 。
* **PMA Token (Pooling by Multi-Head Attention)**：不依赖静态画像，在训练中随机初始化并自学习出几个独立的 **Learnable Queries（无偏见万能探针）**，纯粹从序列本身出发 。

---

## 5. 实践建议与优化策略

### 序列提炼
* 对于超长用户行为序列，避免使用简单的 Average Pooling 将序列“拍扁”，建议借鉴 InterFormer 采用 **CLS + PMA + Recent** 结合的方式进行序列提炼 。

### 解决特征早衰 (PFFN)
* 在序列送入 Transformer 提取特征前，利用一个线性层将用户静态特征映射为权重矩阵，并乘到序列特征上（参考论文中的 $f(X_{sum}^{(l)})S^{(l)}$ 机制），利用 **PFFN 个性化前馈网络** 解决特征早衰问题 。

### 显存与工程优化 (Self-Gating)
* 利用 **自控门 (Self-Gating)** 机制对非序列特征进行提纯后再进行交互。通过降低显存占用换取更大的 **Batch Size**，提升训练效率 。

### 学习风格重构 (Interleaving Learning Style)
* **交替式 Block 结构**：不要将序列建模和特征交叉做成上下级的流水线，而应设计成交替式的 Block 结构 。
* **迭代思路**：
    1. **Block 1**：静态特征去序列里搜索信息，随后通过 RankMixer 处理 。
    2. **Block 2**：使用上一步更新后的静态特征获取深层信息，再次通过 RankMixer 处理 。
* **结论**：这种迭代结构已被 Meta 证明有效，建议直接采用该重构思路 。