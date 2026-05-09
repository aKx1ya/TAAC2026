# InterFormer v0.3 — Hyperparameter Tuning Experiment (Negative Result)

> **Hyperparameter sweep** on top of the v0.1 baseline, attempting to address the
> premature convergence and training oscillation observed in v0.1.
>
> **Result: AUC regressed from 0.8067 → 0.8038 (−0.0029).** This version demonstrates
> that naive training modifications — oversized batches, aggressive embedding reinitialization,
> and under-tuned Focal Loss — can negate the inherent architectural advantage.

---

## Evaluation Results

| Metric | v0.1 | v0.3 | Change |
|--------|------|------|--------|
| **Evaluation AUC** | 0.8067 | **0.8038** | **−0.0029** |
| **Inference time** | 226.89 s | 351.47 s | +124.58 s |

### Training Curves

| Chart | File | Description |
|-------|------|-------------|
| AUC | `AUC.png` | Validation AUC over training steps |
| LogLoss | `LogLoss.png` | Validation LogLoss |
| Loss | `Loss.png` | Training loss |

---

## Changes from v0.1

| Parameter | v0.1 | v0.3 | Rationale |
|-----------|------|------|-----------|
| `loss_type` | bce | **focal** | Address class imbalance |
| `focal_alpha` | — | 0.1 | Positive-class weight |
| `focal_gamma` | — | 2.0 | Focusing parameter |
| `seq_encoder_type` | transformer | **swiglu** | Attempt lighter encoder (⚠️ bug: did not take effect) |
| `batch_size` | 256 | **1024** | Stabilize gradients |
| `num_epochs` | 999 | **3** | Hard cap on training |
| `reinit_cardinality_threshold` | 0 (off) | **10000** | Cold restart high-cardinality embeddings |
| `dropout_rate` | 0.01 | **0.1** (via train.py default) | Stronger regularization |

---

## Failure Analysis

The v0.3 modifications produced a net regression of −0.0029 AUC. Post-hoc analysis identified three contributing factors:

### 1. Under-training from oversized batches

Quadrupling the batch size (256→1024) reduced per-epoch parameter updates to 1/4, while the learning rate was not scaled to compensate (the linear scaling rule suggests `lr` should increase proportionally to `sqrt(batch_size)`). Combined with the hard cap of 3 epochs, the total number of gradient steps was far too low for convergence.

### 2. Destructive cold restarts

Setting `reinit_cardinality_threshold=10000` triggered Xavier reinitialization of all embeddings with vocab > 10K after every epoch. In a 3-epoch training run, this meant the model's learned representations for high-frequency items were destroyed twice, preventing the retention of long-tail knowledge.

### 3. Inactive encoder switch (code bug)

The intended change from Transformer to SwiGLU sequence encoding did not take effect due to a `del seq_encoder_type` statement in `model.py` line 480 of the `PCVRInterFormer.__init__` constructor. The parameter was silently discarded before being used. This bug was identified and fixed in v0.5.

### Additional note: Focal Loss alpha under-tuned

With `focal_alpha=0.1`, the positive class received very low weight, which — combined with the already-reduced gradient signal from Focal Loss's modulating factor — further suppressed learning on the minority class. v0.5 corrected this to `alpha=0.25`.

---

## Key Takeaway

**Ablate one variable at a time.** v0.3 modified loss function, batch size, epoch count, and embedding reinitialization simultaneously. The −0.0029 regression could not be attributed to any single change, making it impossible to identify which modifications were harmful versus neutral. This methodological lesson directly informed the more disciplined approach in v0.5.

---

## File Structure

```
v0.3/
├── README.md          # This file
├── dataset.py         # Parquet dataset loader
├── model.py           # PCVRInterFormer (⚠️ seq_encoder_type bug present)
├── train.py           # Training entry point (modified defaults)
├── trainer.py         # Training loop
├── utils.py           # Utilities
├── ns_groups.json     # NS feature grouping config
├── run.sh             # Launch script (batch_size=1024)
├── AUC.png            # Validation AUC curve
├── LogLoss.png        # Validation LogLoss curve
├── Loss.png           # Training loss curve
└── training_log.txt   # Experiment notes
```

---

## Quick Start

```bash
bash run.sh
```

Or with explicit parameters:

```bash
python train.py \
    --batch_size 1024 \
    --loss_type focal \
    --focal_alpha 0.1 \
    --focal_gamma 2.0 \
    --num_epochs 3 \
    --reinit_cardinality_threshold 10000
```

---

## References

- [InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction](https://dl.acm.org/doi/10.1145/3627673.3680088) (CIKM 2025)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [MultiEpoch: Reusing Training Data for CTR Prediction](https://arxiv.org/abs/2305.19531)
