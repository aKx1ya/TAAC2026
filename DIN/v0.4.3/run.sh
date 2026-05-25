#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- v9.0.3: 方案A —— 在 v9.0.2 基础上把 loss 从 focal 切回 bce_pairwise ----
# 与 v9.0.2 相比的唯一改动：
#   loss_type focal → bce_pairwise，其他参数保持不变
# 目的：验证 v9.0.2 掉分是否源于 Focal Loss 替换了 Pairwise Ranking Loss

python3 -u "${SCRIPT_DIR}/train.py" \
    --batch_size 128 \
    --ns_tokenizer_type rankmixer \
    --user_ns_tokens 3 \
    --item_ns_tokens 4 \
    --num_queries 2 \
    --ns_groups_json "" \
    --emb_skip_threshold 1000000 \
    --hash_bucket_size 100000 \
    --num_workers 8 \
    --num_cross_layers 2 \
    --cross_low_rank 64 \
    --use_ns_output_fusion \
    --use_temporal_bias \
    --use_time_gap \
    --precision bf16 \
    --lr_schedule cosine \
    --warmup_steps 500 \
    --ema_decay 0.999 \
    --weight_decay 0.02 \
    --loss_type bce_pairwise \
    --pairwise_lambda 0.05 \
    "$@"

