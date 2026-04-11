# HyFormer: 重新审视序列建模与特征交互在点击率预测中的作用
# (Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction) 

## 1. 核心架构概述
HyFormer 提出了一个**统一混合 Transformer 架构**，旨在融合以下两个步骤： 
* **序列建模 (Sequence Modeling)**：用于处理用户历史行为。 
* **特征交叉 (Feature Interaction)**：用于处理用户画像、上下文等非序列特征。 

HyFormer 是由多个 **HyFormer Layer** 堆叠而成，每一个 Layer 包含两个关键模块： 
1. **查询解码 (Query Decoding)** 
2. **查询增强 (Query Boosting)** 

---

## 2. 核心执行步骤

### 第一步：查询生成 (Query Generation) 
* **分词 (Tokenization)**：模型首先会对输入进行分词。 
* **生成全局 Token (Global Tokens)**：将用户的静态属性、上下文等“非序列特征 (NS Tokens)”以及全局序列汇总信息，通过一个轻量级的全连接层 (MLP) 转化为一组 Global Tokens，作为后续的 Query。 
* **核心组件说明**：
    * **MLP 全连接层**：输入拼接好的非序列特征（用户的静态属性）以及从用户行为序列中提取的“全局序列汇总信息”，输出全局 Token。 
    * **Global Tokens**：即 Sequence Queries（序列查询），融合了用户全局画像，是用来去长序列中精准提取历史行为的“超级探针”。 

### 第二步：查询解码 (Query Decoding) —— 提取序列信息 
* **KV 编码**：对于超长的用户行为序列，模型支持采用标准 Transformer、LONGER 或极简的 SwiGLU 进行编码，生成每一层的 Key 和 Value。 
* **多头交叉注意力计算 (Cross-Attention)**：使用第一步生成的全局上下文探针 Query，去和长序列的 KV 对进行计算： 
    > $$Q_{decoded} = CrossAttn(Q, K, V)$$ 
* **模块功能**：
    * **交叉注意力**：模型将全局 Query 作为查询词，与 KV 对进行注意力分数计算。相当于让全局特征直接去关注长序列里的每一个历史行为，识别最有价值的行为以注入并更新 Query。 
    * **多头**：注意力计算分成多个平行的“头”同时进行。 
    * **迭代过程**：探针会把关键历史信号吸收进自己的向量中（根据搜索到的东西迭代自己的全局 Token）。 
    * **意义**：让全局上下文信息带着“目的”去长序列里反照有用的历史行为，实现全局信息 (Global Context/Information) 对序列信息的介入。 

### 第三步：查询增强 (Query Boosting) —— 深度特征交叉 
* **Token-mixing (跨 Token 混合)**：融合序列信息和非序列特征。对刚刚解码得到的 Query 和非序列 Tokens 进行混合。 
* **作用**：让各个维度的特征在内部进行充分的“交流讨论”，从而丰富 Query 的语义表示。 
* **后续**：增强后的 Query 会被送入下一层，带着更丰富的知识去继续提取长序列。 

### 第四步：多序列建模 (Multi-Sequence Modeling) 
* **应用场景**：处理用户不同类型的序列（如“看过的视频”和“买过的商品”）。 
* **处理方式**：为每一个独立的序列分配专属的 Query Tokens，各自独立进行 Query Decoding，互不干扰。 
* **融合机制**：既保留了序列的独特性，又能在随后的 Query Boosting 模块中，通过 Token-mixing 实现跨序列的深度交互与全局融合。 

---

## 3. 系统层面工程优化 

### 长序列 GPU 池化 (GPU Pooling)
* **解决瓶颈**：解决数据搬运的瓶颈。 
* **实现机制**：由于长序列中存在大量重复 ID，模型在底层去重后压缩传输，大幅降低数据在 GPU 和 CPU 之间的传输成本和内存占用。 
* **技术流程**：在数据传进 GPU 之前建立去重后的压缩特征表；数据传到 GPU 后，通过高性能前向算子进行解压并重构原始长序列。 

### 异步 AllReduce
* **解决瓶颈**：解决多卡通信等候瓶颈。 
* **实现机制**：在分布式训练时，让前向/反向传播和梯度同步异步进行，消除了通信气泡，提升 GPU 利用率。 

---

## 4. 与 TAAC 的联系及实践建议 

* **共享语义**：可以参考利用全局 Token 作为共享语义接口的思路来构建模型。 
* **Tokenizer 构建**：借鉴 HyFormer 的语义分组策略，按内在含义（用户、上下文、行为）对输入特征进行分区整合。 
* **辅助信息处理**：处理长序列时，参考将时间戳、行为类型等辅助信息拼接到序列的稀疏维度 (Sparse Dim) 中。 
* **推理性能优化**：
    * 比赛有严格的推理时延 (Latency) 限制。HyFormer 在取得最高 AUC (0.6489) 时，计算开销 FLOPs 仅为 3.9T。 
    * **编码策略替换**：算力充足用 Full Transformer；需要平衡则用 LONGER 机制；严重超时则用 SwiGLU 激活层进行无注意力映射。 
* **核心收益**：在 HyFormer 这种双向流动架构下，序列输入特征越丰富，模型获得的 AUC 收益比传统模型更大。 
* **开发策略**：不要急着推翻 Baseline（已包含 SwiGLU 和 RankMixer 等）。建议以 Baseline 为底座，局部改写并引入“Global Tokens 交替进行 Decoding 和 Boosting”及“多序列独立建模”的代码。 