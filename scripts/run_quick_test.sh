#!/bin/bash
# Quick test - 빠른 테스트용 (짧은 길이, 적은 스텝)

python3 visualize_uncertainty_by_timestep.py \
    --use_prompt \
    --benchmark custom \
    --custom_prompt "The quick brown fox" \
    --steps 16 \
    --gen_length 32 \
    --block_length 32 \
    --remasking uncertainty_aware \
    --mc_samples 4 \
    --device cuda:2 \
    --output_dir ./result \
    --exp_name quick_test

echo "Quick test completed! Check ./result/quick_test/ directory"
