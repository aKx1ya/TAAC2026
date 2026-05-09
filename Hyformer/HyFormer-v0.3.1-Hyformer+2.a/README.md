# HyFormer v0.3.1 — Feature Engineering Ablation (Step 1a + 1b + 2a)

> **Ablation experiment** building on the v0.3 baseline.
> This version stacks three feature engineering improvements:
> **Step 1a** — Item log-frequency · **Step 1b** — Bayesian-smoothed CTR · **Step 2a** — Flat KV-weighted embedding
>
> Purpose: Isolate the contribution of Step 2a on top of the 1a+1b combination.
> Compare with `HyFormer-v0.3.1-Hyformer+1.a+1.b/` to measure 2a's independent effect.

---

## Experiment Results

| Step | Modification | Best Val AUC | Best Val LogLoss | Eval AUC | Infer Time | Params |
|------|-------------|-------------|-----------------|----------|------------|--------|
| 0 | Baseline (v0.3) | 0.8622 | 0.2241 | 0.8067 | 226.89 s | ~240M |
| 1a | + Item log-frequency | — | — | — | — | ~240M |
| 1b | + Bayesian-smoothed CTR | — | — | — | — | ~240M |
| **1a+1b+2a** | **+ Flat KV-weighted embedding** | — | — | — | — | ~240M |

> Results for Steps 1a, 1b, and 2a were not recorded in this experimental run.
> See `HyFormer-v0.3.1-Hyformer+1.a+1.b/` for the 1a and 1b individual results.

---

## What Was Implemented

### Step 1a — Item Log-Frequency
Statistical hot/cold popularity signal. Each item's training-set count is transformed as
`log(1 + count)` and injected as the first `item_dense` column (fid=200).
See `HyFormer-v0.3.1-Hyformer+1.a+1.b/` for full implementation details.

### Step 1b — Bayesian-Smoothed CTR
Item-level historical conversion rate smoothed by global CTR, injected as the second
`item_dense` column (fid=201).
See `HyFormer-v0.3.1-Hyformer+1.a+1.b/` for full implementation details.

**Combined `item_dense` structure (inherited from 1a+1b):**

| Column | fid | Meaning |
|--------|-----|---------|
| 0 | 200 | `log(1 + impression_count)` |
| 1 | 201 | Bayesian-smoothed CTR |

### Step 2a — Flat KV-Weighted Embedding (New in this version)

**Motivation:** The schema contains 3 aligned (int, dense) pairs at fids 89/90/91,
each with 10 entities and 10 corresponding statistics. The baseline discards this
alignment and mean-pools all embeddings equally. Step 2a uses the dense values as
softmax attention weights over the integer embeddings.

**Implementation:** Modified `RankMixerNSTokenizer` and `GroupNSTokenizer` to accept
an optional `dense_feats` input and a `weighted_fid_config` parameter. When a fid
is in the config and dense values are present, weighted pooling is applied; otherwise
it falls back to mean pooling (fully backward-compatible).

**Core logic:**
```python
# For each aligned (int_ids, dense_vals) pair:
emb = embedding(int_ids)               # [batch, 10, emb_dim]
w = softmax(log1p(dense_vals))         # [batch, 10]  — log1p prevents extremes
weighted = (emb * w.unsqueeze(-1)).sum(1)  # [batch, emb_dim]
```

**Files modified:**

| File | Change |
|------|--------|
| `model.py` | `RankMixerNSTokenizer` + `GroupNSTokenizer`: added weighted pooling branch; `PCVRHyFormer`: builds `_kv_weighted_config` and passes `dense_feats` |
| `train.py` | No change |
| `dataset.py` | No change |

**Expected impact:** +0.001~0.003 AUC from improved user-side feature representation
at fids 89/90/91.

---

## File Structure

```
HyFormer-v0.3.1-Hyformer+2.a/
├── README.md              # This file
├── dataset-analysis.md    # In-depth schema analysis (7 insights + 7 optimization directions)
├── dataset.py             # Data loader with online item freq/CTR map construction
├── model.py               # Model with KV-weighted embedding (Step 2a)
├── train.py               # Training entry point
├── trainer.py             # Training loop (dual optimizer, early stopping, checkpointing)
├── utils.py               # Utilities (logging, EarlyStopping, Focal Loss)
├── run.sh                 # Launch script
├── schema.json            # Full dataset schema (120 columns)
└── ns_groups.json         # Non-sequential feature grouping config
```

---

## Evaluation Protocol

- **Primary metric:** Binary AUROC (higher is better)
- **Secondary metric:** Binary LogLoss (lower is better)
- **Validation strategy:** Temporal split — most recent 10% of data as validation
- **Early stopping:** patience = 5 on validation AUC

---

## How to Run

```bash
bash run.sh \
    --data_dir /path/to/taac2026_data \
    --schema_path ./schema.json \
    --ckpt_dir ./checkpoints \
    --log_dir ./logs
```

---

## Ablation Context

This folder is part of a sequential ablation series:

| Folder | Modifications |
|--------|--------------|
| `HyFormer-v0.3.0-base/` | Baseline (v0.3, no feature engineering) |
| `HyFormer-v0.3.1-Hyformer+1.a+1.b/` | + Item log-freq + Bayesian CTR |
| **`HyFormer-v0.3.1-Hyformer+2.a/`** | **+ Flat KV-weighted embedding (this folder)** |
| `HyFormer-v0.3.1-Interformer+1.a(control)/` | Control experiment (InterFormer architecture) |

---

## References

- [HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction](https://arxiv.org/abs/2601.12681)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [MultiEpoch: Reusing Training Data for CTR Prediction](https://arxiv.org/abs/2305.19531)
