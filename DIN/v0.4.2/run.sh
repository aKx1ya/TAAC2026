#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# ---- v9.0.2: d_model=128, num_heads=8, SE-Net, NS self-attn, Focal Loss, 训练技巧 ----
# 与 v9.0 基线相比的改动：
#   train.py 默认值变化:
#     d_model 64→128, num_heads 4→8, use_se_net→True, use_ns_self_attn→True
#     dropout_rate 0.01→0.05, loss_type bce→focal, focal_alpha 0.1→0.25
#     label_smoothing 0.0→0.05
#   run.sh 变化:
#     batch_size 256→128 (d_model翻倍后显存不够)

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
    "$@"

# ---- 备选：如果 focal loss 效果不好，切回 bce_pairwise ----
# python3 -u "${SCRIPT_DIR}/train.py" \
#     --batch_size 128 \
#     --ns_tokenizer_type rankmixer \
#     ... (同上) ...
#     --loss_type bce_pairwise \
#     --pairwise_lambda 0.05 \
#     "$@"

