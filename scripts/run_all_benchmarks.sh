#!/bin/bash
# 여러 벤치마크에서 실험 실행

# GSM8K
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt --benchmark gsm8k --num_samples 1 \
    --remasking uncertainty_aware \
    --steps 64 --gen_length 128 --mc_samples 8 \
    --device cuda:2 --exp_name gsm8k_test

echo "GSM8K done!"
echo "================================"

# HellaSwag
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt --benchmark hellaswag --num_samples 1 \
    --remasking uncertainty_aware \
    --steps 64 --gen_length 128 --mc_samples 8 \
    --device cuda:2 --exp_name hellaswag_test

echo "HellaSwag done!"
echo "================================"

# Winogrande
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt --benchmark winogrande --num_samples 1 \
    --remasking uncertainty_aware \
    --steps 64 --gen_length 128 --mc_samples 8 \
    --device cuda:2 --exp_name winogrande_test

echo "Winogrande done!"
echo "================================"

# ARC Easy
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt --benchmark arc_easy --num_samples 1 \
    --remasking uncertainty_aware \
    --steps 64 --gen_length 128 --mc_samples 8 \
    --device cuda:2 --exp_name arc_easy_test

echo "ARC Easy done!"
echo "================================"

echo "All benchmarks completed!"
echo "Check results in ./result/ directory"
