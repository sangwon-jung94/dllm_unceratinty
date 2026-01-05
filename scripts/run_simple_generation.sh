#!/bin/bash
#SBATCH -c 8
#SBATCH -t 600
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:3
#SBATCH --mem=128000
#SBATCH --open-mode=append
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --mail-user=sangwonjung@g.harvard.edu
#SBATCH --mail-type=ALL

module load python/3.10 cudnn cuda/12.4 gcc
source activate RERD

# Simple generation without uncertainty tracking


python generate_simple.py \
    --use_prompt \
    --benchmark gsm8k \
    --num_samples 20000 \
    --remasking entropy \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --batch_size 8 \
    --device cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 \
    --logits_eos_inf \
    --use_ensemble_model \
    --output_dir ./result \
    --exp_name gsm8k_dropout0.1_only_last_layer_entropy_all \
    --dropout_p 0.1
