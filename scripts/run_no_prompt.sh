#!/bin/bash
# 프롬프트 없이 실험 (Unconditional Generation)
# LLaDA 논문 Appendix B.4: EOS 토큰 처리로 짧은 생성 방지

python3 visualize_uncertainty_by_timestep.py \
    --model_path GSAI-ML/LLaDA-8B-Base \
    --batch_size 1 \
    --steps 64 \
    --gen_length 4096 \
    --block_length 4096 \
    --remasking low_confidence \
    --logits_eos_inf \
    --mc_samples 8 \
    --alpha 1.0 \
    --beta 1.0 \
    --dropout_p 0.1 \
    --device cuda:2 \
    --output_dir ./result \
    --exp_name no_prompt_baseline

echo "No prompt experiment completed! Check ./result/no_prompt_baseline/ directory"
echo "Note: Using --logits_eos_inf to prevent early EOS generation (LLaDA paper)"
