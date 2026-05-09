# InterFormer v0.1 — PCVRInterFormer Baseline

> **Baseline version** of our InterFormer implementation for the TAAC 2026 Academic Track.
> This is the first training run with no hyperparameter tuning, serving as the reference
> point for all subsequent InterFormer iterations.
>
> **Key result: AUC 0.8067** — outperforms the heavily optimized HyFormer v0.2 (AUC 0.7952)
> at half the model dimension (d=64 vs d=128), with zero tuning.

---

## Evaluation Results

| Metric | Value |
|--------|-------|
| **Evaluation AUC** | **0.8067** |
| **Inference time** | 226.89 s |
| **Training time** | ~22 hours |
| **Best checkpoint** | `global_step61608.layer=2.head=4.hidden=64.best_model` |
| **Total parameters** | ~240M |

### Training Curves

| Chart | File | Description |
|-------|------|-------------|
| Loss | `Loss.png` | Training loss (exhibits spike instability) |
| AUC | `AUC.png` | Validation AUC (plateaus at ~0.85–0.86 after step 3624) |
| LogLoss | `LogLoss.png` | Validation LogLoss |

---

## Model Architecture: PCVRInterFormer

The model follows the InterFormer paper's three-module decomposition:

| Module | Name | Function |
|--------|------|----------|
| **Interaction Arch** | NS interaction module | Tokenizes user/item ID and dense features via NS Tokenizer, performs cross-feature self-attention |
| **Sequence Arch** | Sequence modeling module | Encodes 4 behavioral domains (seq_a–seq_d) with Transformer encoder, time-bucket embeddings, and sinusoidal position encoding |
| **Cross Arch** | Cross-modal fusion module | Bidirectional summarization via `CrossSummary`, stacked `InterFormerBlock`s, gated output fusion → classifier |

### Key Design Details

- **NS Tokenizer:** RankMixer mode — concatenates all non-sequential feature embeddings, splits into equal-length tokens via linear projection (`user_ns_tokens=5`, `item_ns_tokens=2`)
- **Query Tokens:** 2 per sequence domain (`num_queries=2`), used for CrossSummary attention pooling
- **PersonalizedFFN:** Context-conditioned FFN where NS features generate gate and bias for sequence token processing
- **Position Encoding:** Standard sinusoidal (fixed, not learned)

---

## Configuration

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 64 | Hidden dimension |
| `emb_dim` | 64 | Embedding dimension |
| `num_hyformer_blocks` | 2 | Number of stacked InterFormerBlocks |
| `num_heads` | 4 | Attention heads |
| `hidden_mult` | 4 | FFN expansion factor |
| `dropout_rate` | 0.01 | Dropout rate |
| `num_queries` | 2 | Query tokens per sequence domain |
| `user_ns_tokens` | 5 | User NS tokens (RankMixer) |
| `item_ns_tokens` | 2 | Item NS tokens (RankMixer) |
| `seq_max_lens` | a:256, b:256, c:512, d:512 | Truncation lengths |
| `batch_size` | 256 | Batch size |
| `emb_skip_threshold` | 1,000,000 | Skip embedding for vocab > this |

### Optimizer (Dual)

| Parameter type | Optimizer | Learning rate |
|---------------|-----------|---------------|
| Sparse (all `nn.Embedding`) | Adagrad | 0.05 |
| Dense (everything else) | AdamW | 1e-4, betas=(0.9, 0.98) |

### Loss Function

BCE with logits (`loss_type=bce`). No Focal Loss in this version.

---

## Training Diagnostics

Analysis of the training curves revealed three issues that informed subsequent versions:

**1. Severe loss oscillation.** Training loss fluctuated between 0.15–0.35 with frequent
spikes to 0.6–0.8+, indicating unstable optimization. Root cause: `sparse_lr=0.05` is
too aggressive for high-cardinality embeddings.

**2. Premature convergence.** Validation AUC plateaued at ~0.85–0.86 by step 3624 and
remained flat through step 76104. The model stopped learning meaningful patterns early,
likely due to shortcut learning on high-frequency ID features.

**3. No learning rate schedule.** Without warmup or cosine annealing, the constant learning
rate prevents the oscillation from damping in later training stages.

These findings directly motivated the changes in v0.3 (Focal Loss, larger batch) and
v0.5 (GradScaler, warmup, DHEN/DCN integration).

---

## File Structure

```
v0.1/
├── README.md          # This file
├── dataset.py         # Parquet dataset loader with schema parsing
├── model.py           # PCVRInterFormer model definition
├── train.py           # Training entry point
├── trainer.py         # Training loop (dual optimizer, early stopping)
├── utils.py           # Utilities (logging, EarlyStopping, Focal Loss)
├── ns_groups.json     # Non-sequential feature grouping config
├── run.sh             # Launch script
├── eva/               # Evaluation module
│   ├── dataset.py     # Eval data loader
│   ├── infer.py       # Inference script
│   └── model.py       # Model copy for evaluation
├── AUC.png            # Validation AUC curve
├── LogLoss.png        # Validation LogLoss curve
└── Loss.png           # Training loss curve
```

---

## Quick Start

```bash
bash run.sh
```

Or with explicit parameters:

```bash
python train.py \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 5 \
    --item_ns_tokens 2 \
    --num_queries 2 \
    --emb_skip_threshold 1000000
```

Data paths are configured via environment variables:
`TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH`, `TRAIN_LOG_PATH`.

---

## References

- [InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction](https://dl.acm.org/doi/10.1145/3627673.3680088) (CIKM 2025)
- [HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction](https://arxiv.org/abs/2601.12681)
