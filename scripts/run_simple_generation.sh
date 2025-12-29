#!/bin/bash
# Simple generation without uncertainty tracking

read -p "GSM8K 10 샘플 예시를 실행하시겠습니까? (y/n): " choice

if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
    echo ""
    echo "GSM8K 10 샘플 생성 중 (low_confidence remasking)..."
    python generate_simple.py \
        --use_prompt \
        --benchmark gsm8k \
        --num_samples 320 \
        --remasking low_confidence \
        --steps 512 \
        --gen_length 512 \
        --block_length 512 \
        --batch_size 32 \
        --device cuda:1 \
        --confidence_eos_eot_inf \
        --output_dir ./result \
        --exp_name gsm8k_simple_generation_low_confidence \
        --dropout_p 0.2
fi
