#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ═══════════════════════════════════════════════════════════════════════════════
# HyFormer v0.2 — TAAC 2026 Optimized Configuration
# ═══════════════════════════════════════════════════════════════════════════════
#
# Constraint check (RankMixer full mode):
#   d_model=128  num_queries=1  num_sequences=4
#   user_ns_tokens=7  item_ns_tokens=4
#   num_ns = 7(user_int) + 1(user_dense) + 4(item_int) + 0(item_dense) = 12
#   T = 1*4 + 12 = 16  →  128 % 16 == 0 ✓
#
# Key optimizations for ~1M-scale PCVR data with extreme label sparsity:
#   1. Focal Loss (α=0.25, γ=2.0) — handles severe positive/negative imbalance
#   2. LongerEncoder (top_k=50, causal) — efficient long-sequence modeling
#   3. Higher dropout (0.1) + RoPE — anti-overfitting + position awareness
#   4. Cold-restart for high-cardinality embeddings after epoch 1
#   5. Timestamp-based train/valid split — realistic future-prediction eval
#   6. Increased item_ns_tokens (4) — better item-side feature disentanglement
#   7. Larger batch_size (512) + d_model (128) — stabler gradients, more capacity
#   8. seq_id_threshold=5000 — aggressive id-feature dropout for稀疏场景
# ═══════════════════════════════════════════════════════════════════════════════

python3 -u "${SCRIPT_DIR}/train.py" \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 7 \
    --item_ns_tokens 4 \
    --num_queries 1 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --seq_id_threshold 5000 \
    --batch_size 512 \
    --d_model 128 \
    --emb_dim 64 \
    --num_hyformer_blocks 2 \
    --num_heads 4 \
    --hidden_mult 4 \
    --dropout_rate 0.1 \
    --seq_encoder_type longer \
    --seq_top_k 50 \
    --seq_causal \
    --use_rope \
    --rope_base 10000.0 \
    --rank_mixer_mode full \
    --loss_type focal \
    --focal_alpha 0.25 \
    --focal_gamma 2.0 \
    --lr 1e-4 \
    --sparse_lr 0.05 \
    --sparse_weight_decay 1e-6 \
    --reinit_sparse_after_epoch 1 \
    --reinit_cardinality_threshold 10000 \
    --sort_by_timestamp \
    --valid_time_ratio 0.1 \
    --num_workers 8 \
    --buffer_batches 10 \
    --patience 5 \
    "$@"

# ═══════════════════════════════════════════════════════════════════════════════
# Alternative: lightweight config for quick experiments / debugging
# ═══════════════════════════════════════════════════════════════════════════════
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type rankmixer \
#     --user_ns_tokens 5 \
#     --item_ns_tokens 2 \
#     --num_queries 2 \
#     --ns_groups_json "" \
#     --emb_skip_threshold 1000000 \
#     --batch_size 256 \
#     --d_model 64 \
#     --emb_dim 64 \
#     --dropout_rate 0.01 \
#     --seq_encoder_type transformer \
#     --loss_type bce \
#     --lr 1e-4 \
#     --sparse_lr 0.05 \
#     --sort_by_timestamp \
#     --valid_time_ratio 0.1 \
#     --num_workers 8 \
#     "$@"
#
# ═══════════════════════════════════════════════════════════════════════════════
# Alternative: GroupNSTokenizer driven by ns_groups.json
# ═══════════════════════════════════════════════════════════════════════════════
# Uses feature grouping from ns_groups.json (7 user groups + 4 item groups).
# With d_model=64 and num_ns=12, only num_queries=1 satisfies d_model%T==0.
# To switch, comment out the block above and uncomment the block below.
#
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --ns_tokenizer_type group \
#     --ns_groups_json "${SCRIPT_DIR}/ns_groups.json" \
#     --num_queries 1 \
#     --emb_skip_threshold 1000000 \
#     --num_workers 8 \
#     "$@"
