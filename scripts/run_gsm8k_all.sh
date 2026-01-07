start_time=$(date +%s)
python generate_simple.py \
    --use_prompt \
    --benchmark gsm8k \
    --num_samples 20000 \
    --remasking topk_entropy \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --batch_size 4 \
    --device cuda:0 \
    --logits_eos_inf \
    --use_ensemble_model \
    --output_dir ./result \
    --exp_name only_last_layer/gsm8k_dropout0.1_only_last_layer_topk_entropy_all \
    --dropout_p 0.1
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"


start_time=$(date +%s)
python generate_simple.py \
    --use_prompt \
    --benchmark gsm8k \
    --num_samples 20000 \
    --remasking topk_entropy \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --batch_size 4 \
    --device cuda:0 \
    --logits_eos_inf \
    --use_ensemble_model \
    --output_dir ./result \
    --exp_name only_last_layer/gsm8k_only_last_layer_topk_entropy_all \
    --dropout_p 0.1
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"


start_time=$(date +%s)
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt \
    --benchmark gsm8k \
    --batch_size 4 \
    --num_samples 20000 \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --remasking topk_entropy \
    --mc_samples 6  --dropout_p 0.1 \
    --use_mc_dropout_logit \
    --logits_eos_inf \
    --device cuda:0 \
    --use_ensemble_model \
    --memory_efficient \
    --output_dir ./result \
    --exp_name only_last_layer/gsm8k_topk_entropy__mc6__only_last_layer__dropout_logit__eos_inf
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

# start_time=$(date +%s)
# python generate_simple.py \
#     --use_prompt \
#     --benchmark gsm8k \
#     --num_samples 20000 \
#     --remasking weighted_entropy \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --batch_size 16 \
#     --device cuda:2 cuda:3 \
#     --logits_eos_inf \
#     --use_ensemble_model \
#     --output_dir ./result \
#     --exp_name only_last_layer/gsm8k_dropout0.1_only_last_layer_weighted_entropy_all \
#     --dropout_p 0.1
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"


# start_time=$(date +%s)
# python generate_simple.py \
#     --use_prompt \
#     --benchmark gsm8k \
#     --num_samples 20000 \
#     --remasking weighted_entropy \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --batch_size 16 \
#     --device cuda:2 cuda:3 \
#     --logits_eos_inf \
#     --use_ensemble_model \
#     --output_dir ./result \
#     --exp_name only_last_layer/gsm8k_only_last_layer_weighted_entropy_all \
#     --dropout_p 0.1
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"


# start_time=$(date +%s)
# python3 visualize_uncertainty_by_timestep.py \
#     --use_prompt \
#     --benchmark gsm8k \
#     --batch_size 4 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking weighted_entropy \
#     --mc_samples 6  --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --logits_eos_inf \
#     --device cuda:2 cuda:3 \
#     --use_ensemble_model \
#     --memory_efficient \
#     --output_dir ./result \
#     --exp_name only_last_layer/gsm8k_weighted_entropy__mc6__only_last_layer__dropout_logit__eos_inf
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

# # =============================================================================

# start_time=$(date +%s)
# python3 visualize_uncertainty_single_sample_remasking.py \
#     --use_prompt \
#     --benchmark gsm8k \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking entropy \
#     --mc_samples 6  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --use_ensemble_for_remasking \
#     --use_single_sample_for_sampling \
#     --use_ensemble_model \
#     --logits_eos_inf \
#     --device cuda:2 cuda:3 \
#     --output_dir ./result \
#     --exp_name only_last_layer/gsm8k_entropy_single_sample__only_last_layer__dropout_logit__eos_inf
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

# start_time=$(date +%s)
# python3 visualize_uncertainty_single_sample_remasking.py \
#     --use_prompt \
#     --benchmark gsm8k \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking low_confidence \
#     --mc_samples 6  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --use_ensemble_for_remasking \
#     --use_single_sample_for_sampling \
#     --use_ensemble_model \
#     --logits_eos_inf \
#     --device cuda:2 cuda:3 \
#     --output_dir ./result \
#     --exp_name only_last_layer/gsm8k_low_confidence_single_sample__only_last_layer__dropout_logit__eos_inf
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

# start_time=$(date +%s)
# python3 visualize_uncertainty_single_sample_remasking.py \
#     --use_prompt \
#     --benchmark gsm8k \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking entropy \
#     --mc_samples 6  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --use_single_sample_for_remasking \
#     --use_ensemble_for_sampling \
#     --use_ensemble_model \
#     --logits_eos_inf \
#     --device cuda:2 cuda:3 \
#     --output_dir ./result \
#     --exp_name only_last_layer/gsm8k_entropy_single_remask__only_last_layer__dropout_logit__eos_inf
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

# start_time=$(date +%s)
# python3 visualize_uncertainty_single_sample_remasking.py \
#     --use_prompt \
#     --benchmark gsm8k \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking low_confidence \
#     --mc_samples 6  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --use_single_sample_for_remasking \
#     --use_ensemble_for_sampling \
#     --use_ensemble_model \
#     --logits_eos_inf \
#     --device cuda:2 cuda:3 \
#     --output_dir ./result \
#     --exp_name only_last_layer/gsm8k_low_confidence_single_remask__only_last_layer__dropout_logit__eos_inf
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

# #=============================================================================
# # 표준 실험

# start_time=$(date +%s)
# python3 visualize_uncertainty_by_timestep.py \
#     --use_prompt \
#     --benchmark gsm8k \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking entropy \
#     --mc_samples 6  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --logits_eos_inf \
#     --device cuda:2 cuda:3 \
#     --use_ensemble_model \
#     --memory_efficient \
#     --output_dir ./result \
#     --exp_name only_last_layer/gsm8k_entropy__mc6__only_last_layer__dropout_logit__eos_inf
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

# start_time=$(date +%s)
# python3 visualize_uncertainty_by_timestep.py \
#     --use_prompt \
#     --benchmark gsm8k \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking low_confidence \
#     --mc_samples 6  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --logits_eos_inf \
#     --device cuda:2 cuda:3 \
#     --use_ensemble_model \
#     --memory_efficient \
#     --output_dir ./result \
#     --exp_name only_last_layer/gsm8k_low_confidence__mc6__only_last_layer__dropout_logit__eos_inf
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

# start_time=$(date +%s)
# python3 visualize_uncertainty_by_timestep.py \
#     --use_prompt \
#     --benchmark gsm8k \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking uncertainty_aware \
#     --mc_samples 6  --alpha 1.0 --beta 1.0  --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --logits_eos_inf \
#     --device cuda:2 cuda:3 \
#     --use_ensemble_model \
#     --memory_efficient \
#     --output_dir ./result \
#     --exp_name only_last_layer/gsm8k_uncertainty__mc6__only_last_layer__dropout_logit__eos_inf
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

# start_time=$(date +%s)
# python3 visualize_uncertainty_by_timestep.py \
#     --use_prompt \
#     --benchmark gsm8k \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking uncertainty_aware \
#     --mc_samples 6  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --logits_eos_inf \
#     --device cuda:2 cuda:3 \
#     --use_ensemble_model \
#     --memory_efficient \
#     --output_dir ./result \
#     --exp_name only_last_layer/gsm8k_uncertainty_epistemic__mc6__only_last_layer__dropout_logit__eos_inf
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

# start_time=$(date +%s)
# python3 visualize_uncertainty_by_timestep.py \
#     --use_prompt \
#     --benchmark gsm8k \
#     --batch_size 16 \
#     --num_samples 20000 \
#     --steps 256 \
#     --gen_length 256 \
#     --block_length 256 \
#     --remasking uncertainty_aware \
#     --mc_samples 6  --alpha 0.0 --beta 1.0  --dropout_p 0.1 \
#     --use_mc_dropout_logit \
#     --logits_eos_inf \
#     --device cuda:2 cuda:3 \
#     --use_ensemble_model \
#     --memory_efficient \
#     --output_dir ./result \
#     --exp_name only_last_layer/gsm8k_uncertainty_aleatoric__mc6__only_last_layer__dropout_logit__eos_inf
# end_time=$(date +%s)
# elapsed=$((end_time - start_time))
# echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"