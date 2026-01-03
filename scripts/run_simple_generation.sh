#!/bin/bash
# Simple generation without uncertainty tracking

python generate_simple.py \
    --use_prompt \
    --benchmark gsm8k \
    --num_samples 20000 \
    --remasking entropy \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --batch_size 4 \
    --device cuda:0 cuda:1 cuda:2 \
    --logits_eos_inf \
    --output_dir ./result \
    --exp_name gsm8k_dropout0.2_entropy_all \
    --dropout_p 0.2

python generate_simple.py \
    --use_prompt \
    --benchmark gsm8k \
    --num_samples 20000 \
    --remasking entropy \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --batch_size 4 \
    --device cuda:0 cuda:1 cuda:2 \
    --logits_eos_inf \
    --output_dir ./result \
    --exp_name gsm8k_dropout0.1_entropy_all \
    --dropout_p 0.1

python generate_simple.py \
    --use_prompt \
    --benchmark gsm8k \
    --num_samples 20000 \
    --remasking entropy \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --batch_size 4 \
    --device cuda:0 cuda:1 cuda:2 \
    --logits_eos_inf \
    --output_dir ./result \
    --exp_name gsm8k_entropy_all