# v0.3 — UE Token + UE×Item 显式交互
新增用户身份 token 与显式 user-item 匹配信号。   
把UE独立出来作为一个token，然后与useritem做pair。  

**EVA**  
auc: 0.825249  
inference time：130.62s  
目前最牛逼的版本了  


---

## 改动概要

### model.py

| 改动 | 说明 |
|------|------|
| **新增 `USER_DENSE_UE_END = 256`** | 标记 fid 61 在 user_dense_feats 中的结束偏移 |
| **新增 `UETokenModule`** | 从 fid 61 (256-dim dense) + fid 97 (vocab≈4 int) 融合出独立 UE token。fid 61 经 `Linear(256→d_model)+SiLU`，fid 97 经 `Embedding→d_model`，二者 concat 后经 `Linear(2d→d)+SiLU` 融合 |
| **新增 `UEItemInteraction`** | UE token ⊙ item_emb（hadamard 积）→ `Linear+SiLU`，经可学习 gate（初始 ~0.05）控幅后作为残差加到 output 上 |
| **`__init__` 新增参数** | `ue_int_offset` (fid 97 在 user_int_feats 中的偏移)、`ue_int_vocab` (fid 97 词表大小)，二者均 >0 时启用 UE token |
| **`_make_user_dense_proj_input` 修改** | 启用 UE token 时同时剥离 fid 61 [0:256] 和 fid 62-66 [256:568]，避免双重计数 |
| **`forward` / `predict` 修改** | 在 NS token 构建前提取 fid 61/97 生成 UE token；在 classifier 前注入 `ue_item_interaction` 残差 |
| **`_init_params` / `reinit_high_cardinality_params` 修改** | 适配新增的 `ue_int_emb` 初始化和重初始化逻辑 |

### train.py

| 改动 | 说明 |
|------|------|
| **新增 `UE_INT_FID = 97`** | UE token 的 int 侧特征 fid |
| **新增 `build_ue_spec()`** | 从 schema 中解析 fid 97 的 int_offset 和 vocab_size |
| **新增 `--no_ue_token` 开关** | 默认启用 UE token，`--no_ue_token` 可回退到 v9.0 行为 |
| **`model_args` 新增** | `ue_int_offset`、`ue_int_vocab` 传入模型 |

---

## 架构变化

```
v9.0:                               v0.3:
                                    
user_int ─→ NS Tokenizer            user_int ─→ NS Tokenizer
user_dense ─→ Dense Token           user_dense ─→ Dense Token (fid 61 已剥离)
  + pair residual                      + pair residual
                                    
                                    
                                     ┌─ fid 61 (256d) ─→ Linear ─┐
                                     │                            ├→ UE Token
                                     └─ fid 97 (int)  ─→ Emb ───┘
                                                              │
                                                   UE ⊙ item_emb
                                                   (hadamard积)
                                                       │
                                                   gate × feat ─→ + output
```

## 设计原则

- **独立路径**：UE token 不走 NS tokenizer / ns_tokens，避免 RankMixer 的 token mixing 稀释信号
- **残差注入**：UE×item 交互以 gated residual 形式加到 output，初始 gate≈0，让模型逐步学会使用
- **双重剥离**：fid 61 和 fid 62-66 从 dense_proj 中剥离，各自走独立的专门路径，避免信息重复

