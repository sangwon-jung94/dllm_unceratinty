start_time=$(date +%s)
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt \
    --benchmark gsm8k \
    --batch_size 4 \
    --num_samples 2000 \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --remasking uncertainty_aware \
    --mc_samples 8  --alpha 1.0 --beta 1.0  --dropout_p 0.1 \
    --logits_eos_inf \
    --device cuda:3 cuda:5 cuda:6 cuda:7 \
    --use_ensemble_model \
    --memory_efficient \
    --output_dir ./result \
    --exp_name /gsm8k/only_last_layer/1319_samples/uncertainty/uncertainty_eos_inf
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"


start_time=$(date +%s)
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt \
    --benchmark gsm8k \
    --batch_size 4 \
    --num_samples 2000 \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --remasking uncertainty_aware \
    --mc_samples 8  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
    --logits_eos_inf \
    --device cuda:3 cuda:5 cuda:6 cuda:7 \
    --use_ensemble_model \
    --memory_efficient \
    --output_dir ./result \
    --exp_name /gsm8k/only_last_layer/1319_samples/uncertainty/uncertainty_epistemic_eos_inf
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"

start_time=$(date +%s)
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt \
    --benchmark gsm8k \
    --batch_size 4 \
    --num_samples 2000 \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --remasking uncertainty_aware \
    --mc_samples 8  --alpha 0.0 --beta 1.0  --dropout_p 0.1 \
    --logits_eos_inf \
    --device cuda:3 cuda:5 cuda:6 cuda:7 \
    --use_ensemble_model \
    --memory_efficient \
    --output_dir ./result \
    --exp_name /gsm8k/only_last_layer/1319_samples/uncertainty/uncertainty_aleatoric_eos_inf
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"
