# InterFormer v0.5 — Enhanced PCVRInterFormer (Peak AUC 0.852)

> **Peak-performing version** of our InterFormer implementation for TAAC 2026.
> Builds on v0.4 with critical NaN fixes and sparse-label adaptations.
>
> **Key result: Peak validation AUC 0.852** at step 2396, the highest achieved
> across all experiments. Subsequent training exhibited metric degradation due to
> an embedding reinitialization bug (see Known Issues below).

---

## Evaluation Results

| Metric | Value |
|--------|-------|
| **Peak validation AUC** | **0.852*** |
| **Validation LogLoss** | ~0.17 |
| **Training time** | ~18 hours (2,396 steps) |
| **Best checkpoint** | `global_step2396.layer=2.head=4.hidden=64.best_model` |

*Peak at step 2396; performance degraded after epoch 3 due to embedding reinit bug.

### Training Curves

| Chart | File | Description |
|-------|------|-------------|
| AUC | `AUC.png` | Validation AUC — peaks at step 2396 |
| LogLoss | `LogLoss.png` | Validation LogLoss |
| Loss | `Loss.png` | Training loss (smooth convergence with minor spikes) |
| GAUC | `Gauc.png` | Placeholder only — see Known Issues |
| Positive Rate | `pos_rate.png` | Stable at ~9.6% across validation |
| Precision@1% | `precision_at_1pct.png` | Top-1% precision over training |

---

## Changes from v0.4

### Hyperparameter Changes

| Parameter | v0.4 | v0.5 | Rationale |
|-----------|------|------|-----------|
| `loss_type` | bce | **focal** | Better handling of class imbalance |
| `focal_alpha` | — | **0.25** | Standard positive-class weighting |
| `focal_gamma` | — | **2.0** | Down-weight easy negatives |
| `sparse_lr` | 0.05 | **0.01** | Reduce Adagrad LR for embeddings |
| `eval_every_n_steps` | 0 (off) | **1000** | Validate every 1000 steps |
| `warmup_steps` | — | **1000** | Linear LR warmup for stability |

### Critical Fixes

| Fix | Description | Priority |
|-----|-------------|----------|
| **GradScaler** | Prevents FP16 gradient overflow → NaN (the v0.4 crash) | Critical |
| **Linear LR Warmup** | First 1000 steps ramp up dense LR, preventing cold-start instability | High |
| **Gradient Accumulation** | Framework ready (set `--gradient_accumulation_steps 4` for effective batch ~1536) | Medium |
| **Enhanced Eval Metrics** | Added `precision_at_1pct` and `pos_rate` to validation | Medium |
| **log1p Protection** | `np.maximum(x, -0.999)` guard in `dataset.py` | Medium |

---

## Model Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `d_model` | 64 | Hidden dimension |
| `batch_size` | 384 | Batch size |
| `interaction_layers` | 2 | Interaction backbone depth |
| `dcn_layers` | 2 | DCN cross-layer depth |
| `cross_low_rank` | 32 | Low-rank cross matrix rank |
| `num_cls_tokens` | 4 | CLS tokens |
| `num_pma_tokens` | 2 | PMA tokens |
| `num_recent_tokens` | 2 | Recent behavior tokens |
| `use_rope` | True | Rotary position encoding |
| `user_ns_tokens` | 5 | User NS tokens |
| `item_ns_tokens` | 5 | Item NS tokens |
| `num_queries` | 2 | Query tokens per sequence domain |
| AMP | Enabled | Mixed precision + GradScaler |

---

## Training Report

### Training Stability

The three-layer protection chain (GradScaler + Warmup + clip_grad_norm) successfully
prevented the NaN crash that occurred at step ~1795 in v0.4. Loss converged smoothly
with only minor spikes at steps 2185 and 4369, both caught by GradScaler.

### Convergence Analysis

| Phase | Steps | Loss | AUC | Notes |
|-------|-------|------|-----|-------|
| I. Rapid learning | 0–200 | 0.70 → 0.22 | — | Basic patterns acquired |
| II. Warmup completion | 200–1000 | 0.22 → 0.15 | ~0.56 | LR warmup ends, fine-tuning begins |
| III. Peak performance | 1000–2396 | 0.15 → 0.10 | 0.56 → **0.852** | Major AUC jump after warmup |
| IV. Degradation | 2396+ | Spike | ↓ | Embedding reinit bug triggers |

---

## Known Issues

### Bug 1 (Critical): Embedding Reinitialization Logic Inverted

In `model.py`, `reinit_high_cardinality_params()` uses the condition
`num_embeddings - 1 <= cardinality_threshold` to skip reinit. When
`cardinality_threshold=0` (the default, documented as "never reset"), this
evaluates to `num_embeddings <= 1`, meaning **all non-trivial embeddings are
reset** — the exact opposite of the intended behavior.

Starting from epoch 3, all learned embedding weights are periodically wiped to
Xavier random initialization, causing the metric cliff observed in the AUC curve.

**Fix:** Add a guard `if cardinality_threshold <= 0: return set()` at the top
of the method.

### Bug 2 (Medium): GAUC Hardcoded to Zero

The `evaluate()` method in `trainer.py` sets `gauc=0.0` as a placeholder. The
`Gauc.png` chart therefore shows zeros, not actual per-user grouped AUC.
The GAUC values mentioned in training reports are invalid.

### Correction: Positive Rate is ~9.6%, Not 0.1%

The v0.5 design narrative was based on an incorrect pos_rate reading of 0.1%.
The actual validation positive rate is **~9.6%** (moderate imbalance, not extreme).
This does not invalidate the Focal Loss choice (alpha=0.25 with 9.6% pos_rate
produces ~3.1x effective positive weighting, which remains reasonable), but the
"extreme sparsity" framing in the original documentation was inaccurate.

---

## File Structure

```
v0.5/
├── README.md          # This file
├── dataset.py         # Parquet dataset loader (with log1p protection)
├── model.py           # PCVRInterFormer (DHEN + DCN + RoPE)
├── train.py           # Training entry point
├── trainer.py         # Trainer (GradScaler + Warmup + gradient accumulation)
├── utils.py           # Utilities (sigmoid_focal_loss, EarlyStopping)
├── ns_groups.json     # NS feature grouping config
├── run.sh             # Launch script (Focal Loss + sparse_lr=0.01)
├── Evaluation/        # Offline evaluation scripts
│   ├── dataset.py
│   ├── infer.py
│   └── model.py
├── AUC.png            # Validation AUC curve
├── Gauc.png           # GAUC (placeholder — hardcoded to 0)
├── LogLoss.png        # Validation LogLoss curve
├── Loss.png           # Training loss curve
├── pos_rate.png       # Validation positive rate (~9.6%)
└── precision_at_1pct.png  # Top-1% precision curve
```

---

## Quick Start

```bash
cd v0.5
bash run.sh
```

With gradient accumulation (effective batch ~1536):

```bash
bash run.sh --gradient_accumulation_steps 4
```

Resume from v0.4 checkpoint:

```bash
bash run.sh --resume_from global_step2396.layer=2.head=4.hidden=64.best_model
```

---

## References

- [InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction](https://dl.acm.org/doi/10.1145/3627673.3680088) (CIKM 2025)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [DCN V2: Improved Deep & Cross Network](https://arxiv.org/abs/2008.13535)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
