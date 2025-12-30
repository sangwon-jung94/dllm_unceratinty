#!/bin/bash
# Simple generation without uncertainty tracking

python generate_simple.py \
    --use_prompt \
    --benchmark gsm8k \
    --num_samples 200000 \
    --remasking low_confidence \
    --steps 512 \
    --gen_length 512 \
    --block_length 512 \
    --batch_size 64 \
    --device cuda:1 \
    --confidence_eos_eot_inf \
    --output_dir ./result \
    --exp_name gsm8k_low_confidence_all 
