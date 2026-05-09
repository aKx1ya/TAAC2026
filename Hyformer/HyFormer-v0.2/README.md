# HyFormer v0.2 — Optimized PCVRHyFormer Baseline

> **This is our modified version of the official PCVRHyFormer baseline** provided by the TAAC 2026 competition organizers. All architectural and training modifications below are our contributions; the original baseline code is preserved in `HyFormer-v0.0-raw-baseline/`.

---

## Training & Evaluation Summary

| Item | Detail |
|------|--------|
| **Model** | HyFormer v0.2 (PCVRHyFormer with our optimizations) |
| **Training start** | 2026-05-07 14:24:36 |
| **Training end** | 2026-05-07 22:51:52 |
| **Total training time** | ~8 hours 27 minutes |
| **Best checkpoint** | `global_step9165.layer=2.head=4.hidden=128.best_model` |
| **Configuration** | layers=2, heads=4, d_model=128 |

### Evaluation Results

| Metric | Value |
|--------|-------|
| **AUC** | **0.7952** |
| **Inference time** | **155.55 s** |

### Training Curves

| Chart | File | Description |
|-------|------|-------------|
| AUC | `AUC.png` | Validation AUC over training steps |
| LogLoss | `LogLoss.png` | Validation log-loss over training steps |
| Loss | `Loss.png` | Training loss over training steps |

---

## Changes from Official Baseline (v0.0)

| Component | Official Baseline | v0.2 (Ours) |
|-----------|------------------|-------------|
| Model dimension | d_model=64 | **d_model=128** |
| Batch size | 256 | **512** |
| Dropout | 0.01 | **0.1** |
| Loss function | BCE | **Focal Loss** (alpha=0.25, gamma=2.0) |
| Sequence encoder | Transformer | **LongerEncoder** (Top-K=50, causal) |
| Position encoding | None | **RoPE** (base=10000) |
| User NS tokens | 5 | **7** |
| Item NS tokens | 2 | **4** |
| Num queries | 2 | **1** |
| Item dense features | Not supported | **Supported** |
| ItemFeatureInteraction | None | **Gated bilinear cross** |
| Embedding reinitialization | None | **Cold restart after epoch 1** |
| High-cardinality ID dropout | None | **2x dropout for vocab > 5K** |

---

## Model Architecture

- **Model**: PCVRHyFormer (Post-Click Conversion Rate Hybrid Transformer)
- **Position encoding**: Rotary Position Embedding (RoPE)
- **Activation**: SwiGLU
- **Attention**: RoPE-enhanced Multi-Head Self-Attention
- **Task**: CTR / PCVR prediction (binary classification)
- **Loss**: Focal Loss (alpha=0.25, gamma=2.0)

---

## File Structure

```
HyFormer-v0.2/
├── README.md           # This file
├── dataset.py          # Parquet dataset loader (timestamp sorting + item_dense support)
├── model.py            # PCVRHyFormer model (with ItemFeatureInteraction)
├── train.py            # Training entry point (Focal Loss, timestamp-based split)
├── trainer.py          # Trainer (dual optimizer: AdamW + Adagrad, cold restart)
├── utils.py            # Utilities (EarlyStopping, Focal Loss, logging)
├── run.sh              # Launch script (recommended hyperparameters)
├── generate_schema.py  # Schema generator (with item_dense column scanning)
├── ns_groups.json      # NS feature grouping config
├── AUC.png             # AUC training curve
├── LogLoss.png         # LogLoss training curve
└── Loss.png            # Loss training curve
```

---

## Quick Start

### 1. Generate Schema

```bash
python generate_schema.py /path/to/data.parquet ./data/schema.json
```

### 2. Launch Training

```bash
bash run.sh \
    --data_dir /path/to/training_data \
    --schema_path ./data/schema.json \
    --ckpt_dir ./checkpoints \
    --log_dir ./logs \
    --tf_events_dir ./tf_logs
```

> **Note**: The TAAC 2026 dataset should be downloaded from the official competition portal. Place the parquet files in `data_dir`. If using the 1000-row demo dataset, it is available at `Interformer/data/demo_1000.parquet`.

### 3. Custom Parameters

```bash
python train.py \
    --data_dir /path/to/data \
    --ckpt_dir ./ckpt \
    --log_dir ./log \
    --batch_size 512 \
    --d_model 128 \
    --loss_type focal \
    --sort_by_timestamp \
    --valid_time_ratio 0.1
```

---

## Key Optimizations in v0.2

### 1. Timestamp-Based Data Splitting

| Feature | Description |
|---------|-------------|
| **Metadata-level sorting** | Reads Parquet Row Group column statistics (min/max timestamp); pure metadata operation, no data scanning |
| **Time complexity** | O(N_rg log N_rg); near-zero overhead for ~1M samples |
| **Temporal split** | `--valid_time_ratio 0.1` uses the most recent 10% of data as validation, simulating real-world train-on-past / evaluate-on-future |
| **Backward compatible** | `--no_sort_by_timestamp` reverts to original file-order split |

### 2. Sparse-Label Optimization

The TAAC 2026 dataset is characterized by ~1M samples with a low positive rate and high-cardinality features.

| Strategy | Parameters | Purpose |
|----------|-----------|---------|
| **Focal Loss** | `--loss_type focal --focal_alpha 0.25 --focal_gamma 2.0` | Down-weights easy negatives, focuses on hard samples |
| **LongerEncoder** | `--seq_encoder_type longer --seq_top_k 50 --seq_causal` | Top-K compression for long sequences + causal masking |
| **RoPE** | `--use_rope --rope_base 10000.0` | Rotary position encoding for better position awareness |
| **High-cardinality ID dropout** | `--seq_id_threshold 5000` | 2x dropout for features with vocab > 5K |
| **Cold restart** | `--reinit_sparse_after_epoch 1 --reinit_cardinality_threshold 10000` | Reinitializes high-cardinality embeddings after each epoch |
| **Dual optimizer** | Sparse: Adagrad (lr=0.05), Dense: AdamW (lr=1e-4) | Adapts to the sparse gradient characteristics of embeddings |

**Recommended hyperparameters (see `run.sh`):**

```
d_model=128          # Backbone dimension (64 -> 128)
batch_size=512       # Batch size (256 -> 512)
dropout_rate=0.1     # Dropout (0.01 -> 0.1)
emb_dim=64           # Embedding dimension
num_hyformer_blocks=2
num_heads=4          # Attention heads (d_model / num_heads = 32)
hidden_mult=4        # FFN expansion factor
```

**RankMixer constraint check:**

T = N_q x S + N_ns = 1 x 4 + 12 = 16, and d_model mod T = 128 mod 16 = 0. (Valid.)

Where N_ns = 7 (user) + 1 (user_dense) + 4 (item) = 12.

### 3. Item Feature Enhancement

#### a) Item Dense Feature Support

- `dataset.py`: Added parsing, buffering, and processing for `item_dense_feats_*` columns
- `generate_schema.py`: Automatically scans for `item_dense_feats_*` columns
- `model.py`: `item_dense_proj` module projects item numerical features into NS tokens
- **Backward compatible**: Old schemas without `item_dense` field automatically fall back to empty tensors

#### b) ItemFeatureInteraction Module

A new `ItemFeatureInteraction` class performs **gated bilinear feature crossing** on item NS tokens before they enter the HyFormer blocks:

```
EnhancedItem = ItemTokens + Gate(ItemTokens) * sum_j(W_ij * ItemToken_j)
```

- W is a learnable N x N token interaction weight matrix (softmax-normalized)
- Gate is a sigmoid gate that adaptively controls interaction strength
- Automatically enabled when `item_ns_tokens >= 2`

#### c) Item NS Token Count

| Version | item_ns_tokens | Notes |
|---------|---------------|-------|
| Official baseline | 2 | Limited item-side expressiveness |
| **v0.2 (ours)** | **4** | Finer-grained item feature decomposition |

---

## Full CLI Reference

### Data Paths

| Parameter | Default | Env Variable |
|-----------|---------|-------------|
| `--data_dir` | — | `TRAIN_DATA_PATH` |
| `--schema_path` | `<data_dir>/schema.json` | — |
| `--ckpt_dir` | — | `TRAIN_CKPT_PATH` |
| `--log_dir` | — | `TRAIN_LOG_PATH` |
| `--tf_events_dir` | — | `TRAIN_TF_EVENTS_PATH` |

### Data Pipeline

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--batch_size` | 256 | Batch size |
| `--num_workers` | 16 | DataLoader worker processes |
| `--buffer_batches` | 20 | Shuffle buffer size (in batches) |
| `--train_ratio` | 1.0 | Use first N% of training data |
| `--valid_ratio` | 0.1 | Validation ratio (Row Group level) |
| `--valid_time_ratio` | None | Temporal validation ratio |
| `--sort_by_timestamp` | True | Sort Row Groups by timestamp |
| `--seq_max_lens` | `seq_a:256,...` | Truncation length per sequence domain |

### Model Architecture

| Parameter | Recommended | Description |
|-----------|------------|-------------|
| `--d_model` | **128** | Hidden dimension |
| `--emb_dim` | 64 | Per-feature embedding dimension |
| `--num_queries` | **1** | Query tokens per sequence domain |
| `--num_hyformer_blocks` | 2 | Number of stacked HyFormer blocks |
| `--num_heads` | 4 | Attention heads |
| `--hidden_mult` | 4 | FFN expansion factor |
| `--dropout_rate` | **0.1** | Dropout rate |
| `--seq_encoder_type` | **longer** | Sequence encoder: swiglu / transformer / longer |
| `--seq_top_k` | 50 | LongerEncoder: number of retained tokens |
| `--seq_causal` | **True** | LongerEncoder: causal masking |
| `--use_rope` | **True** | Enable RoPE position encoding |
| `--rank_mixer_mode` | full | RankMixer mode: full / ffn_only / none |

### NS Tokenizer

| Parameter | Recommended | Description |
|-----------|------------|-------------|
| `--ns_tokenizer_type` | rankmixer | NS token generation: group / rankmixer |
| `--user_ns_tokens` | **7** | Number of user NS tokens |
| `--item_ns_tokens` | **4** | Number of item NS tokens |
| `--emb_skip_threshold` | 1000000 | Skip embedding for features with vocab > this |

### Loss Function

| Parameter | Recommended | Description |
|-----------|------------|-------------|
| `--loss_type` | **focal** | Loss type: bce / focal |
| `--focal_alpha` | **0.25** | Focal Loss positive-class weight |
| `--focal_gamma` | **2.0** | Focal Loss focusing parameter |

### Optimizer & Regularization

| Parameter | Recommended | Description |
|-----------|------------|-------------|
| `--lr` | 1e-4 | Dense parameter learning rate (AdamW) |
| `--sparse_lr` | 0.05 | Sparse parameter learning rate (Adagrad) |
| `--sparse_weight_decay` | 1e-6 | Sparse parameter weight decay |
| `--reinit_sparse_after_epoch` | 1 | Cold restart starting from epoch N |
| `--reinit_cardinality_threshold` | 10000 | Cold restart cardinality threshold |
| `--seq_id_threshold` | 5000 | High-cardinality ID feature threshold |
| `--patience` | 5 | Early stopping patience |

---

## Cautions

1. **Schema compatibility**: The updated `schema.json` may include an `item_dense` field; older schemas without it are automatically handled (treated as empty).
2. **RankMixer constraint**: When using `rank_mixer_mode=full`, ensure `d_model % T == 0`. The code will raise an error with valid T values if violated.
3. **Timestamp column**: The Parquet data must contain a `timestamp` column (int64). If absent, the system falls back to file-order splitting.
4. **Memory estimate**: `d_model=128, batch_size=512` requires approximately 12–16 GB GPU memory. Reduce `batch_size` to 256 if OOM occurs.

---

## References

- [HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction](https://arxiv.org/abs/2601.12681)
- [MultiEpoch: Reusing Training Data for CTR Prediction](https://arxiv.org/abs/2305.19531)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
