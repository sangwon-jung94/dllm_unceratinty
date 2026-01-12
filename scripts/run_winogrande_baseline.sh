#!/usr/bin/env bash
################################################
# 기본 벤치마크 측정
################################################

start_time=$(date +%s)
python generate_simple.py \
    --track_sampling_order \
    --use_prompt \
    --benchmark winogrande \
    --num_samples 20000 \
    --remasking topk_entropy \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --batch_size 4 \
    --device cuda:3 cuda:5 cuda:6 cuda:7 \
    --logits_eos_inf \
    --use_ensemble_model \
    --output_dir ./result \
    --exp_name winogrande/1267_samples/topk_entropy/baseline_logits_eos_inf \
    --dropout_p 0.0
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"


start_time=$(date +%s)
python generate_simple.py \
    --track_sampling_order \
    --use_prompt \
    --benchmark winogrande \
    --num_samples 20000 \
    --remasking low_confidence \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --batch_size 4 \
    --device cuda:3 cuda:5 cuda:6 cuda:7 \
    --logits_eos_inf \
    --use_ensemble_model \
    --output_dir ./result \
    --exp_name winogrande/1267_samples/low_confidence/baseline_logits_eos_inf \
    --dropout_p 0.0
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"


start_time=$(date +%s)
python generate_simple.py \
    --track_sampling_order \
    --use_prompt \
    --benchmark winogrande \
    --num_samples 20000 \
    --remasking weighted_entropy \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --batch_size 4 \
    --device cuda:3 cuda:5 cuda:6 cuda:7 \
    --logits_eos_inf \
    --use_ensemble_model \
    --output_dir ./result \
    --exp_name winogrande/1267_samples/weighted_entropy/baseline_logits_eos_inf \
    --dropout_p 0.0
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"


start_time=$(date +%s)
python generate_simple.py \
    --track_sampling_order \
    --use_prompt \
    --benchmark winogrande \
    --num_samples 20000 \
    --remasking entropy \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --batch_size 4 \
    --device cuda:3 cuda:5 cuda:6 cuda:7 \
    --logits_eos_inf \
    --use_ensemble_model \
    --output_dir ./result \
    --exp_name winogrande/1267_samples/entropy/baseline_logits_eos_inf \
    --dropout_p 0.0
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

################################################
# dropout 0.1 벤치마크 측정
################################################
start_time=$(date +%s)
python generate_simple.py \
    --track_sampling_order \
    --use_prompt \
    --benchmark winogrande \
    --num_samples 20000 \
    --remasking topk_entropy \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --batch_size 4 \
    --device cuda:3 cuda:5 cuda:6 cuda:7 \
    --logits_eos_inf \
    --use_ensemble_model \
    --output_dir ./result \
    --exp_name winogrande/1267_samples/only_last_layer/topk_entropy/dropout0.1_logits_eos_inf \
    --dropout_p 0.1
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"


start_time=$(date +%s)
python generate_simple.py \
    --track_sampling_order \
    --use_prompt \
    --benchmark winogrande \
    --num_samples 20000 \
    --remasking low_confidence \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --batch_size 4 \
    --device cuda:3 cuda:5 cuda:6 cuda:7 \
    --logits_eos_inf \
    --use_ensemble_model \
    --output_dir ./result \
    --exp_name winogrande/1267_samples/only_last_layer/low_confidence/dropout0.1_logits_eos_inf \
    --dropout_p 0.1
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"


start_time=$(date +%s)
python generate_simple.py \
    --track_sampling_order \
    --use_prompt \
    --benchmark winogrande \
    --num_samples 20000 \
    --remasking weighted_entropy \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --batch_size 4 \
    --device cuda:3 cuda:5 cuda:6 cuda:7 \
    --logits_eos_inf \
    --use_ensemble_model \
    --output_dir ./result \
    --exp_name winogrande/1267_samples/only_last_layer/weighted_entropy/dropout0.1_logits_eos_inf \
    --dropout_p 0.1
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"


start_time=$(date +%s)
python generate_simple.py \
    --track_sampling_order \
    --use_prompt \
    --benchmark winogrande \
    --num_samples 20000 \
    --remasking entropy \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --batch_size 4 \
    --device cuda:3 cuda:5 cuda:6 cuda:7 \
    --logits_eos_inf \
    --use_ensemble_model \
    --output_dir ./result \
    --exp_name winogrande/1267_samples/only_last_layer/entropy/dropout0.1_logits_eos_inf \
    --dropout_p 0.1
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"
