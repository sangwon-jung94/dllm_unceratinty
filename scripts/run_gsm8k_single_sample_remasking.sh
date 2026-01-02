#!/bin/bash
# 프롬프트와 함께 실험 - GSM8K 벤치마크 전체 (1319 samples)
# LLaDA 논문 Appendix B.4: GSM8K에서는 EOS 토큰 confidence를 0으로 설정

# Record start and end time
start_time=$(date +%s)
python3 visualize_uncertainty_single_sample_remasking.py \
    --use_prompt \
    --benchmark gsm8k \
    --batch_size 4 \
    --num_samples 256 \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --remasking low_confidence \
    --mc_samples 4  --alpha 1.0 --beta 0.0  --dropout_p 0.2 \
    --use_mc_dropout_logit \
    --logits_eos_inf \
    --device cuda:0 cuda:1 cuda:2 \
    --output_dir ./result \
    --exp_name gsm8k_low_confidence_single__dropout_logit__eos_inf
end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"
