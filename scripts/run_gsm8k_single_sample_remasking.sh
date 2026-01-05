#!/bin/bash
# 프롬프트와 함께 실험 - GSM8K 벤치마크 전체 (1319 samples)
# LLaDA 논문 Appendix B.4: GSM8K에서는 EOS 토큰 confidence를 0으로 설정

# Record start and end time
start_time=$(date +%s)
python3 visualize_uncertainty_single_sample_remasking.py \
    --use_prompt \
    --benchmark gsm8k \
    --batch_size 2 \
    --num_samples 256 \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --remasking entropy \
    --mc_samples 6  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
    --use_mc_dropout_logit \
    --use_ensemble_for_remasking \
    --use_single_sample_for_sampling \
    --use_ensemble_model \
    --logits_eos_inf \
    --device cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --output_dir ./result \
    --exp_name gsm8k_entropy_single_sample__only_last_layer__dropout_logit__eos_inf
end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

start_time=$(date +%s)
python3 visualize_uncertainty_single_sample_remasking.py \
    --use_prompt \
    --benchmark gsm8k \
    --batch_size 2 \
    --num_samples 256 \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --remasking entropy \
    --mc_samples 6  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
    --use_mc_dropout_logit \
    --use_ensemble_for_remasking \
    --use_single_sample_for_sampling \
    --use_ensemble_model \
    --logits_eos_inf \
    --device cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --output_dir ./result \
    --exp_name gsm8k_entropy_single_sample__only_last_layer__dropout_logit__eos_inf
end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

start_time=$(date +%s)
python3 visualize_uncertainty_single_sample_remasking.py \
    --use_prompt \
    --benchmark gsm8k \
    --batch_size 2 \
    --num_samples 256 \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --remasking low_confidence \
    --mc_samples 6  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
    --use_mc_dropout_logit \
    --use_ensemble_for_remasking \
    --use_single_sample_for_sampling \
    --use_ensemble_model \
    --logits_eos_inf \
    --device cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --output_dir ./result \
    --exp_name gsm8k_low_confidence_single_sample__only_last_layer__dropout_logit__eos_inf
end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

start_time=$(date +%s)
python3 visualize_uncertainty_single_sample_remasking.py \
    --use_prompt \
    --benchmark gsm8k \
    --batch_size 2 \
    --num_samples 256 \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --remasking entropy \
    --mc_samples 6  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
    --use_mc_dropout_logit \
    --use_single_sample_for_remasking \
    --use_ensemble_for_sampling \
    --use_ensemble_model \
    --logits_eos_inf \
    --device cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --output_dir ./result \
    --exp_name gsm8k_entropy_single_remask__only_last_layer__dropout_logit__eos_inf
end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

start_time=$(date +%s)
python3 visualize_uncertainty_single_sample_remasking.py \
    --use_prompt \
    --benchmark gsm8k \
    --batch_size 2 \
    --num_samples 256 \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --remasking low_confidence \
    --mc_samples 6  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
    --use_mc_dropout_logit \
    --use_single_sample_for_remasking \
    --use_ensemble_for_sampling \
    --use_ensemble_model \
    --logits_eos_inf \
    --device cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --output_dir ./result \
    --exp_name gsm8k_low_confidence_single_remask__only_last_layer__dropout_logit__eos_inf
end_time=$(date +%s)
elapsed=$((end_time - start_time))

echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"
