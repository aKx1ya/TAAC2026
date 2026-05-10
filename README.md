> **GitHub Repository:** https://github.com/aKx1ya/TAAC2026

# TAAC 2026 — Unified Feature Interaction and Sequence Modeling for CTR Prediction

> **Tencent Advertising Algorithm Competition 2026 · KDD Cup · Academic Track**
>
> Course project for AMA564 Deep Learning, The Hong Kong Polytechnic University

---

## Overview

This repository contains our work on the [TAAC 2026 Academic Track](https://algo.qq.com), which challenges participants to design a unified model that deeply fuses static feature interactions and sequential user behavior for click-through rate (CTR) prediction. Starting from the official **PCVRHyFormer** baseline, we explored two complementary optimization paths and achieved significant improvements.

**Key result:** An untuned InterFormer architecture outperforms the heavily optimized HyFormer baseline by **+0.0115 AUC** at half the model dimension.

---

## Main Results

| Model | Version | AUC | d_model | Key Changes |
|-------|---------|-----|---------|-------------|
| PCVRHyFormer | v0.2 (optimized) | 0.7952 | 128 | Focal Loss, LongerEncoder, RoPE, ItemFeatureInteraction |
| **PCVRInterFormer** | **v0.1 (no tuning)** | **0.8067** | **64** | **InterFormer architecture (CrossSummary + PersonalizedFFN)** |
| PCVRInterFormer | v0.3 | 0.8038 | 64 | + Focal Loss, batch 1024, cold restart |
| PCVRInterFormer | v0.5 (peak) | 0.8520* | 64 | + DHEN, DCN, GradScaler, warmup |

*Peak validation AUC at step 2396; subsequent training exhibited overfitting.

---

## Deliverables

| Item | Link |
|------|------|
| Final Report (PDF) | [`docs/Report Versions/AMA564_Final_Report_PolyU_Group.pdf`](docs/Report%20Versions/AMA564_Final_Report_PolyU_Group.pdf) |
| Project Video (7 min) | [YouTube](https://youtu.be/-JgE9-YpROI) |
| Presentation Slides (HTML) | [`docs/Report Versions/TAAC2026_presentation (6).html`](docs/Report%20Versions/TAAC2026_presentation%20(6).html) |

---

## Repository Structure

```
TAAC2026/
│
├── Hyformer/                    # HyFormer (PCVRHyFormer) experiments
│   ├── HyFormer-v0.0/          # Official baseline code (unmodified)
│   ├── HyFormer-v0.1/          # First training run
│   ├── HyFormer-v0.2/          # Optimized baseline (AUC 0.7952)
│   ├── HyFormer-v0.3.0-base/   # Base HyFormer for ablation
│   ├── HyFormer-v0.3.1-*/      # Ablation experiment variants
│   └── ...
│
├── Interformer/                 # InterFormer (PCVRInterFormer) experiments
│   ├── v0.1/                   # Baseline InterFormer (AUC 0.8067)
│   ├── v0.3/                   # Hyperparameter tuning (AUC 0.8038)
│   ├── v0.4/                   # Training stability fixes
│   ├── v0.5/                   # DHEN + DCN integration (peak AUC 0.852)
│   ├── Hyformer_baseline/      # HyFormer baseline for comparison
│   └── data/                   # Demo dataset + schema
│
├── datasetAnalysis/             # Dataset exploration and feature engineering
│   ├── data_analysis.py        # EDA scripts
│   ├── schema.json             # Full dataset schema (120 columns)
│   └── demo_1000.parquet       # 1000-row demo dataset
│
├── docs/                        # Documentation and deliverables
│   ├── Report Versions/        # Final report PDF, presentation HTML
│   └── Readings/               # Reference materials
│
└── references/                  # Related papers (HyFormer, InterFormer, OneTrans)
```

Each experiment folder contains its own `README.md` with detailed configuration, training logs, and results.

---

## Approach Summary

### Problem

Industrial CTR prediction requires jointly modeling static user/item attributes (feature interaction) and temporal behavioral sequences (sequential recommendation). These two paradigms have evolved independently for 15+ years, resulting in shallow fusion, inconsistent optimization objectives, and limited scalability.

### What We Did

**Path 1 — Optimize the HyFormer baseline (v0.2):**
We systematically improved the official PCVRHyFormer with 13 modifications including doubled model dimension (64→128), Focal Loss, LongerEncoder with RoPE, and a novel ItemFeatureInteraction module.

**Path 2 — Explore InterFormer as an alternative (v0.1–v0.5):**
We implemented PCVRInterFormer, replacing the query-token bottleneck with direct NS–sequence modulation via PersonalizedFeedForward and cross-domain summary attention. The untuned InterFormer (v0.1) outperformed the heavily optimized HyFormer at half the parameters.

**Key finding:** Architecture choice is the dominant factor (+0.0115 AUC). Training tricks alone have a ceiling — v0.5 peaks at 0.852 but overfits, and per-user GAUC remains near zero, suggesting the bottleneck is feature representation rather than optimization.

---

## Dataset

The TAAC 2026 Academic Track dataset contains ~1M fully anonymized impression records from Tencent advertising logs, organized into 120 flat columns across 6 categories:

| Category | Columns | Type |
|----------|---------|------|
| ID & Label | 5 | int64 / int32 |
| User Integer Features | 46 | int64 / list |
| User Dense Features | 10 | list\<float\> |
| Item Integer Features | 14 | int64 / list |
| Item Dense Features | 0 | — |
| Domain Sequences | 45 | list\<int64\> (4 domains: A/B/C/D) |

See [`datasetAnalysis/`](datasetAnalysis/) for schema details and EDA scripts.

---

## Quick Start

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA-capable GPU (12–16 GB VRAM recommended)

### Training (InterFormer v0.1 — best result)

```bash
cd Interformer/v0.1
bash run.sh \
    --data_dir /path/to/taac2026_data \
    --schema_path ../data/schema.json \
    --ckpt_dir ./checkpoints \
    --log_dir ./logs
```

### Training (HyFormer v0.2)

```bash
cd Hyformer/HyFormer-v0.2
bash run.sh \
    --data_dir /path/to/taac2026_data \
    --schema_path ./data/schema.json \
    --ckpt_dir ./checkpoints \
    --log_dir ./logs
```

> The TAAC 2026 dataset should be downloaded from the [official competition portal](https://algo.qq.com). For local development, a 1000-row demo dataset is available at `Interformer/data/demo_1000.parquet`.

---

## Team

| Name | Student ID |
|------|-----------|
| Zijian Xu | 25059032G |
| Zesong Qiu | 25125622G |
| Qianqian Yang | 25126863G |
| Sauhon Wong | 25123101G |
| Bingchen Zhang | 25123092G |
| Xingzhou Xu | 25100545G |

The Hong Kong Polytechnic University · AMA564 Deep Learning · May 2026

---

## Development Log

<details>
<summary>Click to expand update history</summary>

### 2026-05-05
- Uploaded official baseline (trained, no tuning)
- Implemented InterFormer v0.1 (training) and v0.2 (queued)
- Optimized: hardware adaptation, loss function
- TODO: training speed, learning rate tuning, feature engineering

### 2026-05-07
- Uploaded InterFormer v0.1 and v0.3 with training results
- Uploaded InterFormer v0.4 (training in progress)
- Next focus: feature engineering and hyperparameter tuning

### 2026-05-08
- Uploaded HyFormer v0.2 training and evaluation results (AUC 0.7952)
- Uploaded InterFormer v0.5 (training in progress)
- InterFormer v0.5 results uploaded; inference timeout issue identified
- Uploaded full-data schema.json with analysis
- Ablation experiments launched
- Feature engineering: log item frequency (marginal gain), Bayesian smoothed CTR (significant gain)
- Combined log-freq + Bayesian CTR embedding (queued)

</details>

---

## References

- [HyFormer: Revisiting the Roles of Sequence Modeling and Feature Interaction in CTR Prediction](https://arxiv.org/abs/2601.12681)
- [InterFormer: Effective Heterogeneous Interaction Learning for CTR Prediction](https://dl.acm.org/doi/10.1145/3627673.3680088) (CIKM 2025)
- [OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer](https://arxiv.org/abs/2510.26104)
- [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
