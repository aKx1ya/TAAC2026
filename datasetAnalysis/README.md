# Dataset Analysis — TAAC 2026 Schema Exploration

> Analysis of the full-scale inference schema (`schema.json`) exported from the
> TAAC 2026 training platform. Contains 7 verified insights and prioritized
> optimization directions that guided our feature engineering in HyFormer v0.3.1
> and InterFormer v0.5.

---

## Dataset Overview

| Category | Features | Total Dims | Notes |
|----------|----------|-----------|-------|
| `user_int` | 46 | 411 | User-side integer features |
| `item_int` | 14 | 33 | Item-side integer features |
| `user_dense` | 10 | 918 | User-side dense features |
| `item_dense` | **0** | **0** | **Empty — no item dense features provided** |
| `seq_a` | 9 | — | Domain A sequences (max len 256) |
| `seq_b` | 14 | — | Domain B sequences (max len 256) |
| `seq_c` | 12 | — | Domain C sequences (max len 512) |
| `seq_d` | 10 | — | Domain D sequences (max len 512) |

**Model scale:** ~240M total parameters (99% in sparse embeddings, 1% dense).
Training set: 907,381 rows (900 Row Groups). Validation: 102,619 rows (100 RGs).

---

## Key Findings

### 1. Eight (Key, Value) Aligned Feature Pairs

Fids 62–66 and 89–91 share identical dimensions between `user_int` (length) and
`user_dense` (dim), forming natural (entity ID, statistic) pairs. The baseline
treats them independently — the largest structural information waste.

| fid | int (vocab × length) | dense dim | Alignment | Structure |
|-----|---------------------|-----------|-----------|-----------|
| 62 | 11 × 6 | 6 | ✅ | Pyramid (coarse) |
| 63 | 49 × 19 | 19 | ✅ | Pyramid |
| 64 | 51 × 26 | 26 | ✅ | Pyramid |
| 65 | 425 × 111 | 111 | ✅ | Pyramid |
| 66 | 1403 × 150 | 150 | ✅ | Pyramid (fine) |
| 89 | 10 × 10 | 10 | ✅ | Flat |
| 90 | 10 × 10 | 10 | ✅ | Flat |
| 91 | 10 × 10 | 10 | ✅ | Flat |

### 2. Item Dense = Empty (0 dimensions)

The `item_dense_proj` module in the baseline is idle. We constructed two item-level
features from training statistics: log-frequency and Bayesian-smoothed CTR
(see `HyFormer-v0.3.1-Hyformer+1.a+1.b/`).

### 3. Interest Pyramid Structure (fids 62→66)

Vocabulary grows from 11→1403, forming a coarse-to-fine interest hierarchy
suitable for hierarchical attention modeling.

### 4. True VRAM Bottleneck = Sequence Embeddings

The real memory pressure comes from ultra-large sequence vocabularies (86M, 64M),
not user/item features (max 23K). Four sequence features with vocab > 1M are
currently replaced with zero vectors.

### 5. Unbalanced Sequence Domains

seq_b (14 features, 64M max vocab) and seq_c (12 features, 86M max vocab) are
far more complex than seq_a and seq_d, yet share the same Transformer weights.

### 6. Timestamps Unavailable at Inference

All 4 sequence timestamp features have vocab=0 in the inference schema, meaning
absolute timestamps are not available online. Time features must use relative
deltas with bucketing.

### 7. Optimization Priority Ranking

```
Highest    Item statistical features (5–15 lines, immediate impact)
  ↓        (Key, Value) weighted embedding (paper-level innovation)
  ↓        High-cardinality seq feature truncation (release VRAM)
  ↓        Absolute time-delta features (enhance temporal modeling)
Lowest     User dense pretrained vector utilization
```

---

## File Structure

```
datasetAnalysis/
├── README.md              # This file
├── data_analysis.py       # EDA script (Jaccard similarity, feature distributions)
├── schema.json            # Full dataset schema (120 columns, from production logs)
├── demo_1000.parquet      # 1000-row demo dataset
├── demo_1000_copy.parquet # Copy of demo dataset
├── dataset.py             # Parquet data loader
├── model.py               # Model definition (for local testing)
├── train.py               # Training entry point
├── trainer.py             # Training loop
├── utils.py               # Utilities
├── ns_groups.json         # NS feature grouping config
└── run.sh                 # Launch script
```

---

## How to Run the Analysis

```bash
cd datasetAnalysis
python data_analysis.py
```

Outputs: `top_features_heatmap.png`, `scalar_vs_seqlen.png`, `length_correlation.png`.

---

## References

- [HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction](https://arxiv.org/abs/2601.12681)
- [InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction](https://dl.acm.org/doi/10.1145/3627673.3680088)
