#!/bin/bash
# MC Dropout 샘플 수 변화에 따른 비교 실험

for mc_samples in 4 8 16; do
    echo "Running with MC samples = ${mc_samples}..."
    
    python3 visualize_uncertainty_by_timestep.py \
        --use_prompt \
        --benchmark gsm8k \
        --num_samples 1 \
        --steps 64 \
        --gen_length 128 \
        --remasking uncertainty_aware \
        --mc_samples ${mc_samples} \
        --device cuda:2 \
        --exp_name gsm8k_mc${mc_samples}
    
    echo "MC samples = ${mc_samples} done!"
    echo "================================"
done

echo "All MC samples experiments completed!"
echo "Check results in ./result/ directory"
