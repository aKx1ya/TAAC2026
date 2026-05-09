# HyFormer v0.3.1 — InterFormer + 1.a (Control Experiment)

> **Control experiment** using the **InterFormer (PCVRInterFormer)** architecture with Step 1a (item log-frequency).
>
> Purpose: Compare InterFormer vs HyFormer under the same feature engineering
> (Step 1a), isolating the effect of architecture choice from feature improvements.
> This is the InterFormer counterpart to `HyFormer-v0.3.1-Hyformer+1.a+1.b/`.

---

## Experiment Results

| Step | Modification | Architecture | Best Val AUC | Best Val LogLoss | Eval AUC | Infer Time | Params |
|------|-------------|-------------|-------------|-----------------|----------|------------|--------|
| 0 | Baseline (v0.3) | InterFormer | 0.8622 | 0.2241 | 0.8067 | 226.89 s | ~240M |
| **1a** | **+ Item log-frequency** | **InterFormer** | — | — | — | — | ~240M |

> This experiment tests whether the InterFormer architecture benefits from
> the same item statistical features that were applied to HyFormer.

---

## What Was Implemented

### Step 1a — Item Log-Frequency

Same implementation as in the HyFormer 1a+1b variant: each item's training-set
appearance count is transformed as `log(1 + count)` and injected as
`item_dense` column (fid=200). Fully online computation in `dataset.py` with
no label leakage.

**Scan results:**
- Training set: 907,381 rows, 20,898 unique items
- Validation set: 102,619 rows, 11,361 unique items

**Schema change:** Added `"item_dense": [[200, 1]]`.

---

## File Structure

```
HyFormer-v0.3.1-Interformer+1.a(control)/
├── README.md              # This file
├── dataset-analysis.md    # Schema analysis (7 insights + 7 optimization directions)
├── dataset.py             # Data loader with online item freq map
├── model.py               # PCVRInterFormer (InterFormer architecture)
├── train.py               # Training entry point
├── trainer.py             # Training loop (dual optimizer, early stopping)
├── utils.py               # Utilities (logging, EarlyStopping, Focal Loss)
├── run.sh                 # Launch script
├── schema.json            # Full dataset schema (120 columns)
└── ns_groups.json         # Non-sequential feature grouping config
```

---

## Ablation Context

This folder is the **InterFormer control** in a cross-architecture ablation:

| Folder | Architecture | Feature Engineering |
|--------|-------------|-------------------|
| `HyFormer-v0.3.0-base/` | HyFormer | None (baseline) |
| `HyFormer-v0.3.1-Hyformer+1.a+1.b/` | HyFormer | + Item freq + Bayesian CTR |
| `HyFormer-v0.3.1-Hyformer+2.a/` | HyFormer | + Item freq + CTR + KV-weighted emb |
| **This folder** | **InterFormer** | **+ Item freq only** |

By comparing this folder's results against `HyFormer-v0.3.1-Hyformer+1.a+1.b/`,
we can measure whether the AUC difference between architectures persists after
applying the same feature engineering.

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

## References

- [InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction](https://dl.acm.org/doi/10.1145/3627673.3680088) (CIKM 2025)
- [HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction](https://arxiv.org/abs/2601.12681)
