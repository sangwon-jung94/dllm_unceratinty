#!/bin/bash
# Remasking Strategy 비교 실험

echo "Comparing different remasking strategies..."

# 1. Uncertainty-aware
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt --benchmark gsm8k --batch_size 8 \
    --remasking uncertainty_aware \
    --mc_samples 8 --alpha 1.0 --beta 1.0 \
    --steps 512 --gen_length 512 --block_length 512 \
    --device cuda:2 --exp_name remasking_uncertainty_aware

echo "Uncertainty-aware done!"
echo "================================"

# 2. Low confidence (baseline)
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt --benchmark gsm8k --batch_size 8 \
    --remasking low_confidence \
    --steps 512 --gen_length 512 --block_length 512 \
    --device cuda:2 --exp_name remasking_low_confidence

echo "Low confidence done!"
echo "================================"

# 3. Random
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt --benchmark gsm8k --batch_size 8 \
    --remasking random \
    --steps 512 --gen_length 512 --block_length 512 \
    --device cuda:2 --exp_name remasking_random

echo "Random done!"
echo "================================"

echo "All remasking strategy experiments completed!"
echo "Compare results in:"
echo "  - ./result/remasking_uncertainty_aware/"
echo "  - ./result/remasking_low_confidence/"
echo "  - ./result/remasking_random/"
