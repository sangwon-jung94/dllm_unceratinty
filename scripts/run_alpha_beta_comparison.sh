#!/bin/bash
# Alpha/Beta 파라미터 조정 실험

# Baseline (alpha=1.0, beta=1.0)
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt --benchmark gsm8k \
    --remasking uncertainty_aware \
    --alpha 1.0 --beta 1.0 \
    --steps 64 --gen_length 128 --mc_samples 8 \
    --device cuda:2 --exp_name gsm8k_alpha1_beta1

echo "Baseline (α=1.0, β=1.0) done!"
echo "================================"

# Epistemic-heavy (alpha=2.0, beta=1.0)
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt --benchmark gsm8k \
    --remasking uncertainty_aware \
    --alpha 2.0 --beta 1.0 \
    --steps 64 --gen_length 128 --mc_samples 8 \
    --device cuda:2 --exp_name gsm8k_alpha2_beta1

echo "Epistemic-heavy (α=2.0, β=1.0) done!"
echo "================================"

# Aleatoric-heavy (alpha=1.0, beta=2.0)
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt --benchmark gsm8k \
    --remasking uncertainty_aware \
    --alpha 1.0 --beta 2.0 \
    --steps 64 --gen_length 128 --mc_samples 8 \
    --device cuda:2 --exp_name gsm8k_alpha1_beta2

echo "Aleatoric-heavy (α=1.0, β=2.0) done!"
echo "================================"

# Both heavy (alpha=2.0, beta=2.0)
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt --benchmark gsm8k \
    --remasking uncertainty_aware \
    --alpha 2.0 --beta 2.0 \
    --steps 64 --gen_length 128 --mc_samples 8 \
    --device cuda:2 --exp_name gsm8k_alpha2_beta2

echo "Both heavy (α=2.0, β=2.0) done!"
echo "================================"

echo "All alpha/beta experiments completed!"
echo "Check results in ./result/ directory"
