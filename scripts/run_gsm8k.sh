#!/bin/bash
# 프롬프트와 함께 실험 - GSM8K 벤치마크 전체 (1319 samples)
# LLaDA 논문 Appendix B.4: GSM8K에서는 EOS 토큰 confidence를 0으로 설정

# Record start and end time
start_time=$(date +%s)
python3 visualize_uncertainty_by_timestep.py \
    --model_path GSAI-ML/LLaDA-8B-Base \
    --use_prompt \
    --benchmark gsm8k \
    --batch_size 32 \
    --num_samples 320 \
    --steps 512 \
    --gen_length 512 \
    --block_length 512 \
    --remasking uncertainty_aware \
    --mc_samples 4  --alpha 1.0 --beta 1.0 \
    --use_mc_dropout_logit \
    --device cuda:0 \
    --output_dir ./result \
    --exp_name gsm8k_uncertainty_aware
end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "Note: Using --confidence_eos_eot_inf for GSM8K (as per LLaDA paper)"
echo "Total samples processed from GSM8K test set"
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"
