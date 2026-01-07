#!/bin/bash
# MC Dropout 샘플 수 변화에 따른 비교 실험

for mc_samples in 2 4 8 16 32; do
    echo "- gsm8k_mc${mc_samples}/"
    start_time=$(date +%s)
    python3 visualize_uncertainty_by_timestep.py \
        --use_prompt \
        --benchmark gsm8k \
        --batch_size 4 \
        --num_samples 256 \
        --steps 256 \
        --gen_length 256 \
        --block_length 256 \
        --remasking uncertainty_aware \
        --mc_samples ${mc_samples}  --alpha 1.0 --beta 1.0  --dropout_p 0.1 \
        --use_mc_dropout_logit \
        --logits_eos_inf \
        --device cuda:0 cuda:1 \
        --use_ensemble_model \
        --memory_efficient \
        --output_dir ./result \
        --exp_name gsm8k_uncertainty__mc${mc_samples}__only_last_layer__dropout_logit__eos_inf
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"
done

for mc_samples in 2 4 8 16 32; do
    echo "- gsm8k_mc${mc_samples}/"
    start_time=$(date +%s)
    python3 visualize_uncertainty_by_timestep.py \
        --use_prompt \
        --benchmark gsm8k \
        --batch_size 4 \
        --num_samples 256 \
        --steps 256 \
        --gen_length 256 \
        --block_length 256 \
        --remasking uncertainty_aware \
        --mc_samples ${mc_samples}  --alpha 0.0 --beta 1.0  --dropout_p 0.1 \
        --use_mc_dropout_logit \
        --logits_eos_inf \
        --device cuda:0 cuda:1 \
        --use_ensemble_model \
        --memory_efficient \
        --output_dir ./result \
        --exp_name gsm8k_uncertainty_aleatoric__mc${mc_samples}__only_last_layer__dropout_logit__eos_inf
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"
done

for mc_samples in 2 4 8 16 32; do
    echo "- gsm8k_mc${mc_samples}/"
    start_time=$(date +%s)
    python3 visualize_uncertainty_by_timestep.py \
        --use_prompt \
        --benchmark gsm8k \
        --batch_size 4 \
        --num_samples 256 \
        --steps 256 \
        --gen_length 256 \
        --block_length 256 \
        --remasking uncertainty_aware \
        --mc_samples ${mc_samples}  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
        --use_mc_dropout_logit \
        --logits_eos_inf \
        --device cuda:0 cuda:1 \
        --use_ensemble_model \
        --memory_efficient \
        --output_dir ./result \
        --exp_name gsm8k_uncertainty_epistemic__mc${mc_samples}__only_last_layer__dropout_logit__eos_inf
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"
done
