#!/bin/bash
# 단일 샘플 테스트 - Uncertainty 분해 visualize용
start_time=$(date +%s)
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt \
    --benchmark gsm8k \
    --num_samples 1 \
    --batch_size 16 \
    --steps 64 \
    --gen_length 256 \
    --block_length 256 \
    --remasking uncertainty_aware \
    --logits_eos_inf \
    --mc_samples 6 \
    --alpha 1.0 \
    --beta 1.0 \
    --dropout_p 0.1 \
    --device cuda:2 \
    --output_dir ./result \
    --exp_name gsm8k_single_uncertainty

echo "Single sample experiment for uncertainty visualization completed!"
echo "Check ./result/gsm8k_single_uncertainty/ directory"
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"