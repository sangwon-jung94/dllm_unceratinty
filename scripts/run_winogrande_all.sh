#!/usr/bin/env bash
# ################################################
# # 기본 벤치마크 측정
# ################################################

# start_time=$(date +%s)
# python generate_simple.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --num_samples 20000 \
#     --remasking topk_entropy \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --batch_size 4 \
#     --device cuda:3 cuda:5 cuda:6 cuda:7 \
#     --logits_eos_inf \
#     --use_ensemble_model \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/topk_entropy/baseline_logits_eos_inf \
#     --dropout_p 0.0
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"


# start_time=$(date +%s)
# python generate_simple.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --num_samples 20000 \
#     --remasking low_confidence \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --batch_size 4 \
#     --device cuda:3 cuda:5 cuda:6 cuda:7 \
#     --logits_eos_inf \
#     --use_ensemble_model \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/low_confidence/baseline_logits_eos_inf \
#     --dropout_p 0.0
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"


# start_time=$(date +%s)
# python generate_simple.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --num_samples 20000 \
#     --remasking weighted_entropy \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --batch_size 4 \
#     --device cuda:3 cuda:5 cuda:6 cuda:7 \
#     --logits_eos_inf \
#     --use_ensemble_model \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/weighted_entropy/baseline_logits_eos_inf \
#     --dropout_p 0.0
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"


# start_time=$(date +%s)
# python generate_simple.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --num_samples 20000 \
#     --remasking entropy \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --batch_size 4 \
#     --device cuda:3 cuda:5 cuda:6 cuda:7 \
#     --logits_eos_inf \
#     --use_ensemble_model \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/entropy/baseline_logits_eos_inf \
#     --dropout_p 0.0
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

# ################################################
# # dropout 0.1 벤치마크 측정
# ################################################
# start_time=$(date +%s)
# python generate_simple.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --num_samples 20000 \
#     --remasking topk_entropy \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --batch_size 4 \
#     --device cuda:3 cuda:5 cuda:6 cuda:7 \
#     --logits_eos_inf \
#     --use_ensemble_model \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/only_last_layer/topk_entropy/dropout0.1_logits_eos_inf \
#     --dropout_p 0.1
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"


# start_time=$(date +%s)
# python generate_simple.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --num_samples 20000 \
#     --remasking low_confidence \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --batch_size 4 \
#     --device cuda:3 cuda:5 cuda:6 cuda:7 \
#     --logits_eos_inf \
#     --use_ensemble_model \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/only_last_layer/low_confidence/dropout0.1_logits_eos_inf \
#     --dropout_p 0.1
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"


# start_time=$(date +%s)
# python generate_simple.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --num_samples 20000 \
#     --remasking weighted_entropy \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --batch_size 4 \
#     --device cuda:3 cuda:5 cuda:6 cuda:7 \
#     --logits_eos_inf \
#     --use_ensemble_model \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/only_last_layer/weighted_entropy/dropout0.1_logits_eos_inf \
#     --dropout_p 0.1
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"


# start_time=$(date +%s)
# python generate_simple.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --num_samples 20000 \
#     --remasking entropy \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --batch_size 4 \
#     --device cuda:3 cuda:5 cuda:6 cuda:7 \
#     --logits_eos_inf \
#     --use_ensemble_model \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/only_last_layer/entropy/dropout0.1_logits_eos_inf \
#     --dropout_p 0.1
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

################################################
# mc dropout 벤치마크 측정
################################################
# start_time=$(date +%s)
# python3 visualize_uncertainty_by_timestep.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking uncertainty_aware \
#     --mc_samples 8 --alpha 1.0 --beta 1.0 --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --logits_eos_inf \
#     --device cuda:1 cuda:2 cuda:3 \
#     --use_ensemble_model \
#     --memory_efficient \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/only_last_layer/uncertainty/uncertainty__mc8__use_mc_logit__dropout0.1__logits_eos_inf
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# minutes=$((elapsed / 60))
# echo "Elapsed time: ${elapsed} seconds (${minutes} minutes)"

# start_time=$(date +%s)
# python3 visualize_uncertainty_by_timestep.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking uncertainty_aware \
#     --mc_samples 8 --alpha 1.0 --beta 0.0 --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --logits_eos_inf \
#     --device cuda:1 cuda:2 cuda:3 \
#     --use_ensemble_model \
#     --memory_efficient \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/only_last_layer/uncertainty/uncertainty_epistemic__mc8__use_mc_logit__dropout0.1__logits_eos_inf 
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# minutes=$((elapsed / 60))
# echo "Elapsed time: ${elapsed} seconds (${minutes} minutes)"

# start_time=$(date +%s)
# python3 visualize_uncertainty_by_timestep.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking uncertainty_aware \
#     --mc_samples 8 --alpha 0.0 --beta 1.0 --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --logits_eos_inf \
#     --device cuda:1 cuda:2 cuda:3 \
#     --use_ensemble_model \
#     --memory_efficient \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/only_last_layer/uncertainty/uncertainty_aleatoric__mc8__use_mc_logit__dropout0.1__logits_eos_inf 
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# minutes=$((elapsed / 60))
# echo "Elapsed time: ${elapsed} seconds (${minutes} minutes)"

# ################################################
# # single sample / single remasking 실험
# ################################################
# start_time=$(date +%s)
# python3 visualize_uncertainty_single_sample_remasking.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking uncertainty_aware \
#     --mc_samples 8  --alpha 1.0 --beta 1.0  --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --use_ensemble_for_remasking \
#     --use_single_sample_for_sampling \
#     --use_ensemble_model \
#     --logits_eos_inf \
#     --device cuda:1 cuda:2 cuda:3 \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/only_last_layer/uncertainty/uncertainty__single_sample__mc8__use_mc_logit__dropout0.1__logits_eos_inf 
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# minutes=$((elapsed / 60))
# echo "Elapsed time: ${elapsed} seconds (${minutes} minutes)"

# start_time=$(date +%s)
# python3 visualize_uncertainty_single_sample_remasking.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking uncertainty_aware \
#     --mc_samples 8  --alpha 1.0 --beta 1.0  --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --use_single_sample_for_remasking \
#     --use_ensemble_for_sampling \
#     --use_ensemble_model \
#     --logits_eos_inf \
#     --device cuda:1 cuda:2 cuda:3 \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/only_last_layer/uncertainty/uncertainty__single_unmask__mc8__use_mc_logit__dropout0.1__logits_eos_inf 
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# minutes=$((elapsed / 60))
# echo "Elapsed time: ${elapsed} seconds (${minutes} minutes)"

# start_time=$(date +%s)
# python3 visualize_uncertainty_single_sample_remasking.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking uncertainty_aware \
#     --mc_samples 8  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --use_ensemble_for_remasking \
#     --use_single_sample_for_sampling \
#     --use_ensemble_model \
#     --logits_eos_inf \
#     --device cuda:1 cuda:2 cuda:3 \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/only_last_layer/uncertainty/uncertainty_epistemic__single_sample__mc8__use_mc_logit__dropout0.1__logits_eos_inf 
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# minutes=$((elapsed / 60))
# echo "Elapsed time: ${elapsed} seconds (${minutes} minutes)"

# start_time=$(date +%s)
# python3 visualize_uncertainty_single_sample_remasking.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking uncertainty_aware \
#     --mc_samples 8  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --use_single_sample_for_remasking \
#     --use_ensemble_for_sampling \
#     --use_ensemble_model \
#     --logits_eos_inf \
#     --device cuda:1 cuda:2 cuda:3 \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/only_last_layer/uncertainty/uncertainty_epistemic__single_unmask__mc8__use_mc_logit__dropout0.1__logits_eos_inf 
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# minutes=$((elapsed / 60))
# echo "Elapsed time: ${elapsed} seconds (${minutes} minutes)"

# start_time=$(date +%s)
# python3 visualize_uncertainty_single_sample_remasking.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking uncertainty_aware \
#     --mc_samples 8  --alpha 0.0 --beta 1.0  --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --use_ensemble_for_remasking \
#     --use_single_sample_for_sampling \
#     --use_ensemble_model \
#     --logits_eos_inf \
#     --device cuda:1 cuda:2 cuda:3 \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/only_last_layer/uncertainty/uncertainty_aleatoric__single_sample__mc8__use_mc_logit__dropout0.1__logits_eos_inf 
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# minutes=$((elapsed / 60))
# echo "Elapsed time: ${elapsed} seconds (${minutes} minutes)"

# start_time=$(date +%s)
# python3 visualize_uncertainty_single_sample_remasking.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking uncertainty_aware \
#     --mc_samples 8  --alpha 0.0 --beta 1.0  --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --use_single_sample_for_remasking \
#     --use_ensemble_for_sampling \
#     --use_ensemble_model \
#     --logits_eos_inf \
#     --device cuda:1 cuda:2 cuda:3 \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/only_last_layer/uncertainty/uncertainty_aleatoric__single_unmask__mc8__use_mc_logit__dropout0.1__logits_eos_inf 
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# minutes=$((elapsed / 60))
# echo "Elapsed time: ${elapsed} seconds (${minutes} minutes)"


# start_time=$(date +%s)
# python3 visualize_uncertainty_single_sample_remasking.py \
#     --track_sampling_order \
#     --use_prompt \
#     --benchmark winogrande \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking low_confidence \
#     --mc_samples 8  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --use_ensemble_for_remasking \
#     --use_single_sample_for_sampling \
#     --use_ensemble_model \
#     --logits_eos_inf \
#     --device cuda:0 cuda:1 cuda:2 cuda:3 \
#     --output_dir ./result \
#     --exp_name winogrande/1267_samples/only_last_layer/low_confidence/low_confidence__single_sample__mc8__use_mc_logit__dropout0.1__logits_eos_inf 
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# minutes=$((elapsed / 60))
# echo "Elapsed time: ${elapsed} seconds (${minutes} minutes)"

start_time=$(date +%s)
python3 visualize_uncertainty_single_sample_remasking.py \
    --track_sampling_order \
    --use_prompt \
    --benchmark winogrande \
    --batch_size 16 \
    --num_samples 20000 \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --remasking low_confidence \
    --mc_samples 8  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
    --use_mc_dropout_logit \
    --use_single_sample_for_remasking \
    --use_ensemble_for_sampling \
    --use_ensemble_model \
    --logits_eos_inf \
    --device cuda:0 cuda:1 cuda:2 cuda:3 \
    --output_dir ./result \
    --exp_name winogrande/1267_samples/only_last_layer/low_confidence/low_confidence__single_unmask__mc8__use_mc_logit__dropout0.1__logits_eos_inf 
end_time=$(date +%s)
elapsed=$((end_time - start_time))
minutes=$((elapsed / 60))
echo "Elapsed time: ${elapsed} seconds (${minutes} minutes)"

start_time=$(date +%s)
python3 visualize_uncertainty_single_sample_remasking.py \
    --track_sampling_order \
    --use_prompt \
    --benchmark winogrande \
    --batch_size 16 \
    --num_samples 20000 \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --remasking entropy \
    --mc_samples 8  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
    --use_mc_dropout_logit \
    --use_ensemble_for_remasking \
    --use_single_sample_for_sampling \
    --use_ensemble_model \
    --logits_eos_inf \
    --device cuda:0 cuda:1 cuda:2 cuda:3 \
    --output_dir ./result \
    --exp_name winogrande/1267_samples/only_last_layer/entropy/entropy__single_sample__mc8__use_mc_logit__dropout0.1__logits_eos_inf 
end_time=$(date +%s)
elapsed=$((end_time - start_time))
minutes=$((elapsed / 60))
echo "Elapsed time: ${elapsed} seconds (${minutes} minutes)"

start_time=$(date +%s)
python3 visualize_uncertainty_single_sample_remasking.py \
    --track_sampling_order \
    --use_prompt \
    --benchmark winogrande \
    --batch_size 16 \
    --num_samples 20000 \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --remasking entropy \
    --mc_samples 8  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
    --use_mc_dropout_logit \
    --use_single_sample_for_remasking \
    --use_ensemble_for_sampling \
    --use_ensemble_model \
    --logits_eos_inf \
    --device cuda:0 cuda:1 cuda:2 cuda:3 \
    --output_dir ./result \
    --exp_name winogrande/1267_samples/only_last_layer/entropy/entropy__single_unmask__mc8__use_mc_logit__dropout0.1__logits_eos_inf 
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"