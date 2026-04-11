# [cite_start]OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender [cite: 65]

## 1. 核心痛点与解决思路对比
* [cite_start]**核心痛点**：OneTrans 解决的核心痛点与 HyFormer 相同，即如何实现“序列建模”和“特征交叉”的有效融合 [cite: 66][cite_start]。但是两者的解法完全不同 [cite: 67]。
* [cite_start]**OneTrans 的解法**：将静态特征和动态序列强行拼接成一条完整的序列，用一个纯粹的单向 Transformer 一步到位 [cite: 69][cite_start]。这是一种带有“万物皆可 Token”暴力解决色彩的思路 [cite: 69]。
* [cite_start]**HyFormer 的局限性指出与对比**：HyFormer 指出了 OneTrans 架构的局限性 [cite: 68][cite_start]。相比之下，HyFormer 采用了“交替”的架构，先用 Query Decoding 交叉注意力机制，让全局特征去阅读长序列提取信息，然后再用 Query Boosting 让这些特征互相进行交叉，然后一层一层交替下去 [cite: 70]。

---

## 2. 核心架构对立点

### [cite_start]特征交叉的底层算子区别 [cite: 71]
* [cite_start]**OneTrans**：完全依赖 Transformer 的 Self-Attention（自注意力机制）来计算各种商品属性、用户画像之间的特征交叉 [cite: 72]。
* [cite_start]**HyFormer**：其论文中指出，对于非序列特征的交叉，使用 Self-Attention 不仅会导致 AUC 下降，计算效率也低 [cite: 73][cite_start]。因此，其使用了专门针对工业化优化的 MLP-Mixer 技术（类似 RankMixer 的技术） [cite: 73]。

### [cite_start]多序列的处理方式 [cite: 74]
* [cite_start]**OneTrans**：面对用户不同的行为序列，其做法是插入 `[SEP]` 分隔符，强行把不同语义的序列连成一条超长线进行联合建模 [cite: 75]。
* [cite_start]**HyFormer**：强烈批判了 OneTrans 的上述做法，认为强行合并不同序列的特征维度会抹杀不同序列的独特性，从而导致性能下降 [cite: 76][cite_start]。HyFormer 的做法是保持不同序列完全独立，为每个序列分配专属的全局 Token 分别读取，互不干扰 [cite: 76]。

---

## 3. 基于 Baseline 与论文的实践思考
[cite_start]官方提供的 Baseline 模板已包含 Transformer 架构、RankMixer 风格的分词和 SwiGLU 激活函数 [cite: 77][cite_start]。基于此 Baseline 和上述论文，我们可以得出以下几点深度思考 [cite: 77]：

### [cite_start]思考一：Mixed Parameterization (混合参数化) [cite: 78]
* [cite_start]**输入层设计** [cite: 79][cite_start]：我们不能用一个简单的全连接层把所有的特征一视同仁地映射 [cite: 82][cite_start]。需要修改输入层，提升 AUC [cite: 82]。
* [cite_start]**权重分配**：对于同质化高的序列特征，可以共享一套 Q/K/V 和 FNN 权重 [cite: 80][cite_start]。而对于用户画像、当前上下文等同质化低的静态特征，则需要为每一个 Token 分配专属的参数（即保留独立的特征变换矩阵） [cite: 81, 82]。
* [cite_start]**解耦处理**：因此，我们需要对多行为序列进行解耦处理，解耦方式可以参考 HyFormer 的全局 Query [cite: 83]。

### [cite_start]思考二：Pyramid Stack (金字塔序列截断) [cite: 84]
* [cite_start]**序列层截断机制** [cite: 85][cite_start]：模型越往深层走，实际需要的序列长度越短 [cite: 86][cite_start]。OneTrans 采用了金字塔结构，在每一层逐步减少作为 Query 的序列 Token 数量，以此来降低训练的内存开销 [cite: 86]。
* [cite_start]**实践建议**：如果正式数据集中用户的行为序列极长，可以尝试采用类似机制：在前两层完整捕捉序列，深层则仅捕捉 20% 的序列 Query [cite: 86][cite_start]。省下来的内存可以用来开启更大的 Batch Size 以跑并行，或者用于增加模型深度等尝试 [cite: 86]。
* [cite_start]**关于 Batch Size**：Batch Size 太小会有问题，因为用户平时误触广告的行为也会被记录在序列里（大 Batch 有助于减小此类噪音影响） [cite: 87]。

### [cite_start]思考三：结合 HyFormer 的 MLP-Mixer 架构 [cite: 88]
* [cite_start]**深度交叉阶段引入 Mixer**：HyFormer 论文中已经证实，在特征深层交叉阶段，引入基于 MLP-Mixer 的轻量级 Token 混合机制会更好 [cite: 89]。
* [cite_start]**架构借鉴**：既然 Baseline 已经提供了类似 RankMixer 的分词与交叉思路，我们应该借鉴 HyFormer 的交叉架构：先用 Cross-Attention 让目标广告去提取用户长序列中的兴趣，拿到结果后再送入 MLP-Mixer 中与用户的静态画像做深度交叉 [cite: 89]。
* [cite_start]**循环堆叠**：采用“提取-交叉-再提取-再交叉”的循环堆叠方式，是一个不错的思路 [cite: 90]。
* [cite_start]**提前评估**：特征的稠密性和稀疏性需要提前计算一下，这样才知道在模型的哪些部分需要限制 Self-Attention 的使用 [cite: 91]。