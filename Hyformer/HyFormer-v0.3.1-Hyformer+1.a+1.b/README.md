# HyFormer v0.3.1 — Feature Engineering Ablation (Step 1a + 1b)

> **Ablation experiment** building on the v0.3 baseline.
> This version adds two item-side statistical features constructed from the training set:
> **Step 1a** — Item log-frequency · **Step 1b** — Bayesian-smoothed CTR
>
> Purpose: Measure the independent and combined contribution of each feature to AUC,
> and fill the `item_dense_dim = 0` gap in the original schema.

---

## Experiment Results

| Step | Modification | Best Val AUC | Best Val LogLoss | Eval AUC | Infer Time | Params |
|------|-------------|-------------|-----------------|----------|------------|--------|
| 0 | Baseline (v0.3) | 0.8622 | 0.2241 | 0.8067 | 226.89 s | ~240M |
| 1a | + Item log-frequency | 0.8589 | 0.2269 | — | — | ~240M |
| 1b | + Bayesian-smoothed CTR | 0.8622 | 0.2242 | — | — | ~240M |
| **1a+1b** | **+ Both combined** | — | — | — | — | ~240M |

> **Finding:** Step 1b (Bayesian CTR) matches baseline AUC while Step 1a alone slightly
> regresses. The combined effect of 1a+1b requires further evaluation.
> See `dataset-analysis.md` for the full schema analysis that motivated these features.

---

## What We Built

### Step 1a — Item Log-Frequency

**Motivation:** Fill the empty `item_dense` field with a hot/cold popularity signal.
Each item's appearance count in the training set is transformed as `log(1 + count)`.

**Implementation:** Fully online — `dataset.py` scans training Row Groups at initialization
to build an in-memory frequency map. The validation set reuses the training statistics
(no label leakage). No preprocessing scripts or external files required.

**Scan results:**
- Training set: 907,381 rows · 20,898 unique items
- Validation set: 102,619 rows · 11,361 unique items

**Schema change:** Added `"item_dense": [[200, 1]]` (1-dimensional log1p item frequency).

### Step 1b — Bayesian-Smoothed CTR

**Motivation:** Give the model a prior on each item's historical conversion tendency,
smoothed by the global CTR to handle low-impression items.

**Formula:**

```
smooth_ctr = (clicks + alpha * global_ctr) / (impressions + alpha)
```

**Implementation:** `dataset.py` additionally builds a CTR map at initialization,
injected as the second column of `item_dense` (fid=201).

**Schema change:** Updated to `"item_dense": [[200, 1], [201, 1]]`.

**Combined `item_dense` structure:**

| Column | fid | Meaning | Source |
|--------|-----|---------|--------|
| 0 | 200 | `log(1 + impression_count)` | Step 1a |
| 1 | 201 | Bayesian-smoothed CTR | Step 1b |

---

## File Structure

```
HyFormer-v0.3.1-Hyformer+1.a+1.b/
├── README.md              # This file
├── dataset-analysis.md    # In-depth schema analysis (7 insights + 7 optimization directions)
├── dataset.py             # Data loader with online item freq/CTR map construction
├── model.py               # Model definition (PCVRInterFormer + Tokenizers + Blocks)
├── train.py               # Training entry point
├── trainer.py             # Training loop (dual optimizer, early stopping, checkpointing)
├── utils.py               # Utilities (logging, EarlyStopping, Focal Loss)
├── run.sh                 # Launch script
├── schema.json            # Full dataset schema (120 columns, exported from production logs)
└── ns_groups.json         # Non-sequential feature grouping config
```

---

## Evaluation Protocol

- **Primary metric:** Binary AUROC (higher is better)
- **Secondary metric:** Binary LogLoss (lower is better)
- **Validation strategy:** Temporal split — most recent 10% of data as validation set,
  simulating train-on-past / evaluate-on-future
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

## Planned Next Steps (Not Implemented in This Version)

The following optimizations were designed during this experiment but not yet executed.
They are documented here for reference and future work.

| Step | Description | Expected AUC gain |
|------|-------------|------------------|
| 2a | (Key, Value) weighted embedding — flat pairs (fid 89/90/91) | +0.002~0.005 |
| 2b | (Key, Value) weighted embedding — pyramid hierarchy (fid 62→66) | +0.003~0.008 |
| 3 | Top-K frequency truncation for ultra-large vocab features | neutral or slight gain |
| 4 | Absolute time-delta feature (recency encoding) | +0.001~0.003 |
| 5 | Per-domain independent sequence projection | +0.001~0.002 |

For detailed motivation and implementation notes for each step,
see the original `dataset-analysis.md` in this folder.

---

## References

- [HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction](https://arxiv.org/abs/2601.12681)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [MultiEpoch: Reusing Training Data for CTR Prediction](https://arxiv.org/abs/2305.19531)
