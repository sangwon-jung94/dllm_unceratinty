start_time=$(date +%s)
python3 visualize_uncertainty_by_timestep.py \
    --use_prompt \
    --benchmark gsm8k \
    --batch_size 4 \
    --num_samples 256 \
    --steps 256 \
    --gen_length 256 \
    --block_length 256 \
    --remasking entropy \
    --mc_samples 6  --alpha 1.0 --beta 0.0  --dropout_p 0.1 \
    --use_mc_dropout_logit \
    --logits_eos_inf \
    --device cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7 \
    --use_ensemble_model \
    --memory_efficient \
    --output_dir ./result \
    --exp_name gsm8k_entropy__mc6__only_last_layer__dropout_logit__eos_inf
end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "Elapsed time: ${elapsed} seconds ($(($elapsed / 60)) minutes)"
