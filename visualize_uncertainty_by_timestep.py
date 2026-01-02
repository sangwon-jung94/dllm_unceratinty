import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
from generate import (
    add_gumbel_noise,
    get_num_transfer_tokens,
    compute_uncertainty_decomposition,
    compute_uncertainty_score,
    compute_entropy
)
import torch.nn.functional as F
import argparse
import os
import signal
import sys
from datetime import datetime
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from tqdm import tqdm


@torch.no_grad()
def generate_with_uncertainty_tracking(
    model, 
    prompt, 
    attention_mask=None, 
    steps=128, 
    gen_length=128, 
    block_length=128, 
    temperature=0.,
    cfg_scale=0., 
    remasking='uncertainty_aware', 
    mask_id=126336, 
    logits_eos_inf=False, 
    confidence_eos_eot_inf=False,
    mc_samples=8, 
    alpha=1.0, 
    beta=1.0, 
    dropout_p=None,
    use_mc_dropout_logit=False
):
    '''
    Generate text while tracking uncertainty at each timestep.
    
    Returns:
        x: Generated sequences
        uncertainty_history: Dict containing timestep-wise uncertainty metrics
    '''
    # Initialize uncertainty tracking
    uncertainty_history = {
        'timesteps': [],
        'mean_epistemic': [],
        'mean_aleatoric': [],
        'mean_total': [],
        'std_epistemic': [],
        'std_aleatoric': [],
        'std_total': [],
        # For tracking uncertainty of actually unmasked tokens
        'mean_epistemic_unmasked': [],
        'mean_aleatoric_unmasked': [],
        'mean_total_unmasked': [],
        'std_epistemic_unmasked': [],
        'std_aleatoric_unmasked': [],
        'std_total_unmasked': []
    }
    
    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask, 
            torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)
        ], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    global_step = 0
    
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in tqdm(range(steps_per_block), desc=f"Block {num_block+1}/{num_blocks} steps", leave=False):
            mask_index = (x == mask_id)
            
            # Compute uncertainty decomposition with MC Dropout (for uncertainty estimation only)
            mean_logits_mc, H_epistemic, H_aleatoric, p_bar = compute_uncertainty_decomposition(
                model=model,
                x=x,
                attention_mask=attention_mask,
                mc_samples=mc_samples,
                cfg_scale=cfg_scale,
                prompt_index=prompt_index,
                mask_id=mask_id,
                dropout_p=dropout_p
            )
            
            # Calculate total uncertainty
            H_total = H_epistemic + H_aleatoric
            
            # Only consider masked positions for uncertainty statistics
            masked_positions = mask_index
            
            # Compute statistics over all masked positions (across all sequences in batch)
            if masked_positions.any():
                epistemic_masked = H_epistemic[masked_positions]
                aleatoric_masked = H_aleatoric[masked_positions]
                total_masked = H_total[masked_positions]
                
                # Store mean and std across all masked token positions
                uncertainty_history['timesteps'].append(global_step)
                uncertainty_history['mean_epistemic'].append(epistemic_masked.mean().item())
                uncertainty_history['mean_aleatoric'].append(aleatoric_masked.mean().item())
                uncertainty_history['mean_total'].append(total_masked.mean().item())
                uncertainty_history['std_epistemic'].append(epistemic_masked.std().item())
                uncertainty_history['std_aleatoric'].append(aleatoric_masked.std().item())
                uncertainty_history['std_total'].append(total_masked.std().item())
            
            # Get logits for actual token sampling
            # Use MC dropout logits or clean logits based on use_mc_dropout_logit option
            if use_mc_dropout_logit:
                # Use MC dropout averaged logits (already computed above)
                logits = mean_logits_mc
            else:
                # Use clean logits without dropout for high-quality generation
                if cfg_scale > 0:
                    logits_cond = model(
                        input_ids=torch.where(prompt_index, x, mask_id),
                        attention_mask=attention_mask
                    ).logits
                    logits_uncond = model(
                        input_ids=torch.full_like(x, mask_id),
                        attention_mask=attention_mask
                    ).logits
                    logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
                else:
                    logits = model(
                        input_ids=x,
                        attention_mask=attention_mask
                    ).logits
            
            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Apply confidence_eos_eot_inf before computing remasking scores
            # This ensures EOS/EoT tokens are unmasked last
            if confidence_eos_eot_inf:
                logits[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            # Choose remasking strategy
            if remasking == 'uncertainty_aware':
                # Compute unmasking score based on uncertainty
                x0_p = compute_uncertainty_score(H_epistemic, H_aleatoric, alpha, beta)
            elif remasking == 'low_confidence':
                # Use confidence (probability of predicted token)
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                # Random selection
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # # Original for-loop implementation (commented out - use vectorized version below for better performance)
            # transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            # for j in range(confidence.shape[0]):
            #     _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
            #     transfer_index[j, select_index] = True
            # x[transfer_index] = x0[transfer_index]

            # Vectorized topk operation (much faster than for loop)
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            max_k = num_transfer_tokens[:, i].max().item()
            if max_k > 0:
                _, topk_indices = torch.topk(confidence, k=max_k, dim=1)
                valid_mask = torch.arange(max_k, device=x0.device).unsqueeze(0) < num_transfer_tokens[:, i].unsqueeze(1)
                batch_indices = torch.arange(confidence.shape[0], device=x0.device).unsqueeze(1).expand(-1, max_k)
                transfer_index[batch_indices[valid_mask], topk_indices[valid_mask]] = True
            
            # Track uncertainty for actually unmasked tokens (before updating x)
            if transfer_index.any():
                epistemic_unmasked = H_epistemic[transfer_index]
                aleatoric_unmasked = H_aleatoric[transfer_index]
                total_unmasked = H_total[transfer_index]
                
                uncertainty_history['mean_epistemic_unmasked'].append(epistemic_unmasked.mean().item())
                uncertainty_history['mean_aleatoric_unmasked'].append(aleatoric_unmasked.mean().item())
                uncertainty_history['mean_total_unmasked'].append(total_unmasked.mean().item())
                uncertainty_history['std_epistemic_unmasked'].append(epistemic_unmasked.std().item())
                uncertainty_history['std_aleatoric_unmasked'].append(aleatoric_unmasked.std().item())
                uncertainty_history['std_total_unmasked'].append(total_unmasked.std().item())
            else:
                # No tokens unmasked in this step (shouldn't happen but just in case)
                uncertainty_history['mean_epistemic_unmasked'].append(0.0)
                uncertainty_history['mean_aleatoric_unmasked'].append(0.0)
                uncertainty_history['mean_total_unmasked'].append(0.0)
                uncertainty_history['std_epistemic_unmasked'].append(0.0)
                uncertainty_history['std_aleatoric_unmasked'].append(0.0)
                uncertainty_history['std_total_unmasked'].append(0.0)
            
            x[transfer_index] = x0[transfer_index]
            global_step += 1

    return x, uncertainty_history


def plot_uncertainty_over_time(uncertainty_history, save_path='uncertainty_over_timesteps.png', title_suffix=''):
    '''
    Plot uncertainty metrics over diffusion timesteps with mean ± std bands.
    
    Args:
        uncertainty_history: Dict containing timestep-wise uncertainty metrics
        save_path: Path to save the plot
        title_suffix: Additional text to add to the title
    '''
    fig, ax = plt.subplots(figsize=(12, 8))
    
    timesteps = uncertainty_history['timesteps']
    
    # Convert to numpy arrays for easier computation
    timesteps_arr = np.array(timesteps)
    mean_total = np.array(uncertainty_history['mean_total'])
    mean_epistemic = np.array(uncertainty_history['mean_epistemic'])
    mean_aleatoric = np.array(uncertainty_history['mean_aleatoric'])
    std_total = np.array(uncertainty_history['std_total'])
    std_epistemic = np.array(uncertainty_history['std_epistemic'])
    std_aleatoric = np.array(uncertainty_history['std_aleatoric'])
    
    # Plot Total Uncertainty with band
    ax.plot(timesteps, mean_total, 
            label='Total Uncertainty', linewidth=2.5, marker='o', markersize=4, 
            color='#2ca02c', alpha=0.9)
    ax.fill_between(timesteps, mean_total - std_total, mean_total + std_total,
                     alpha=0.2, color='#2ca02c')
    
    # Plot Epistemic Uncertainty with band
    ax.plot(timesteps, mean_epistemic, 
            label='Epistemic Uncertainty', linewidth=2.5, marker='s', markersize=4, 
            color='#1f77b4', alpha=0.9)
    ax.fill_between(timesteps, mean_epistemic - std_epistemic, mean_epistemic + std_epistemic,
                     alpha=0.2, color='#1f77b4')
    
    # Plot Aleatoric Uncertainty with band
    ax.plot(timesteps, mean_aleatoric, 
            label='Aleatoric Uncertainty', linewidth=2.5, marker='^', markersize=4, 
            color='#ff7f0e', alpha=0.9)
    ax.fill_between(timesteps, mean_aleatoric - std_aleatoric, mean_aleatoric + std_aleatoric,
                     alpha=0.2, color='#ff7f0e')
    
    ax.set_xlabel('Diffusion Timestep (t)', fontsize=12)
    ax.set_ylabel('Uncertainty (Entropy)', fontsize=12)
    title = 'Uncertainty Over Diffusion Timesteps (Mean ± Std)\n(Averaged over all masked token positions)'
    if title_suffix:
        title += f'\n{title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


def plot_uncertainty_components(uncertainty_history, save_path='uncertainty_components.png', title_suffix=''):
    '''
    Plot uncertainty decomposition over time with filled areas.
    '''
    fig, ax = plt.subplots(figsize=(12, 6))
    
    timesteps = uncertainty_history['timesteps']
    
    # Create stacked area plot
    ax.fill_between(timesteps, 0, uncertainty_history['mean_aleatoric'], 
                    label='Aleatoric (Ambiguity)', alpha=0.6, color='#ff7f0e')
    ax.fill_between(timesteps, uncertainty_history['mean_aleatoric'], 
                    uncertainty_history['mean_total'],
                    label='Epistemic (Model Uncertainty)', alpha=0.6, color='#1f77b4')
    
    # Also plot the total line
    ax.plot(timesteps, uncertainty_history['mean_total'], 
           label='Total Uncertainty', linewidth=2.5, color='black', linestyle='--')
    
    ax.set_xlabel('Diffusion Timestep (t)', fontsize=12)
    ax.set_ylabel('Mean Uncertainty (Entropy)', fontsize=12)
    title = 'Uncertainty Decomposition Over Diffusion Process\n(Stacked: Aleatoric + Epistemic)'
    if title_suffix:
        title += f'\n{title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


def get_benchmark_prompts(benchmark_name, num_samples=None):
    '''
    Load prompts from various benchmarks used in eval_llada.py
    
    Args:
        benchmark_name: Name of the benchmark ('gsm8k', 'hellaswag', 'winogrande', etc.)
        num_samples: Number of samples to use (None = all samples)
    
    Returns:
        List of prompts
    '''
    prompts = []
    
    if benchmark_name == 'gsm8k':
        dataset = load_dataset('gsm8k', 'main', split='test')
        num_to_use = len(dataset) if num_samples is None else min(num_samples, len(dataset))
        prompts = [item['question'] for item in dataset.select(range(num_to_use))]
    
    elif benchmark_name == 'hellaswag':
        dataset = load_dataset('Rowan/hellaswag', split='validation')
        num_to_use = len(dataset) if num_samples is None else min(num_samples, len(dataset))
        for item in dataset.select(range(num_to_use)):
            ctx = item['ctx']
            prompts.append(ctx)
    
    elif benchmark_name == 'winogrande':
        dataset = load_dataset('winogrande', 'winogrande_xl', split='validation')
        num_to_use = len(dataset) if num_samples is None else min(num_samples, len(dataset))
        for item in dataset.select(range(num_to_use)):
            sentence = item['sentence']
            prompts.append(sentence)
    
    elif benchmark_name == 'arc_easy':
        dataset = load_dataset('ai2_arc', 'ARC-Easy', split='test')
        num_to_use = len(dataset) if num_samples is None else min(num_samples, len(dataset))
        for item in dataset.select(range(num_to_use)):
            question = item['question']
            prompts.append(question)
    
    elif benchmark_name == 'arc_challenge':
        dataset = load_dataset('ai2_arc', 'ARC-Challenge', split='test')
        num_to_use = len(dataset) if num_samples is None else min(num_samples, len(dataset))
        for item in dataset.select(range(num_to_use)):
            question = item['question']
            prompts.append(question)
    
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")
    
    return prompts


def worker_init():
    '''Initialize worker process to handle interrupts properly'''
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_batch_on_device(args_dict):
    '''
    Worker function to process a batch on a specific GPU device.
    This function is called in parallel for each GPU.
    
    Args:
        args_dict: Dictionary containing all necessary arguments
    
    Returns:
        Tuple of (batch_outputs, uncertainty_history)
    '''
    # Extract arguments
    device = args_dict['device']
    batch_prompts = args_dict['batch_prompts']
    model_path = args_dict['model_path']
    steps = args_dict['steps']
    gen_length = args_dict['gen_length']
    block_length = args_dict['block_length']
    temperature = args_dict['temperature']
    cfg_scale = args_dict['cfg_scale']
    remasking = args_dict['remasking']
    logits_eos_inf = args_dict['logits_eos_inf']
    confidence_eos_eot_inf = args_dict['confidence_eos_eot_inf']
    mc_samples = args_dict['mc_samples']
    alpha = args_dict['alpha']
    beta = args_dict['beta']
    dropout_p = args_dict['dropout_p']
    use_mc_dropout_logit = args_dict['use_mc_dropout_logit']
    use_prompt = args_dict['use_prompt']
    batch_idx = args_dict['batch_idx']
    
    # Set device
    torch.cuda.set_device(device)
    
    # Load model and tokenizer on this GPU
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        trust_remote_code=True
    )
    
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    
    # Process this batch
    if use_prompt:
        # Apply chat template if using Instruct model
        messages = [{"role": "user", "content": prompt} for prompt in batch_prompts]
        formatted_prompts = [
            tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) 
            for message in messages
        ]

        encoded_outputs = tokenizer(
            formatted_prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt"
        )
        input_ids = encoded_outputs['input_ids'].to(device)
        attention_mask = encoded_outputs['attention_mask'].to(device)
    else:
        # For Instruct models, use minimal chat template
        if 'Instruct' in model_path:
            messages = [{'role': 'user', 'content': ''}]
            template = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompts = [template] * len(batch_prompts)
            
            encoded_outputs = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt"
            )
            input_ids = encoded_outputs['input_ids'].to(device)
            attention_mask = encoded_outputs['attention_mask'].to(device)
        else:
            # For Base models, just use start token
            start_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
            input_ids = torch.tensor([[start_token]] * len(batch_prompts), device=device)
            attention_mask = torch.ones_like(input_ids)
    
    # Generate with uncertainty tracking
    out, uncertainty_history = generate_with_uncertainty_tracking(
        model=model,
        prompt=input_ids,
        attention_mask=attention_mask,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=temperature,
        cfg_scale=cfg_scale,
        remasking=remasking,
        logits_eos_inf=logits_eos_inf,
        confidence_eos_eot_inf=confidence_eos_eot_inf,
        mc_samples=mc_samples,
        alpha=alpha,
        beta=beta,
        dropout_p=dropout_p,
        use_mc_dropout_logit=use_mc_dropout_logit
    )

    # Decode output for this batch
    batch_output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    
    # Clean up to free GPU memory
    del model
    torch.cuda.empty_cache()
    
    return batch_output, uncertainty_history, batch_idx


def main():
    parser = argparse.ArgumentParser(description='Visualize uncertainty over diffusion timesteps')
    
    # Model and device settings
    parser.add_argument('--model_path', type=str, default='GSAI-ML/LLaDA-8B-Instruct',
                        help='Path or name of the model')
    parser.add_argument('--device', type=str, nargs='+', default=['cuda:0'],
                        help='Device(s) to run on (e.g., cuda:0 cuda:1 cuda:2). Multiple devices enable parallel processing.')
    
    # Generation parameters
    parser.add_argument('--steps', type=int, default=64,
                        help='Number of diffusion steps')
    parser.add_argument('--gen_length', type=int, default=128,
                        help='Number of tokens to generate')
    parser.add_argument('--block_length', type=int, default=128,
                        help='Block length for generation')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature')
    parser.add_argument('--cfg_scale', type=float, default=0.0,
                        help='Classifier-free guidance scale')
    
    # Uncertainty estimation parameters
    parser.add_argument('--remasking', type=str, default='uncertainty_aware',
                        choices=['uncertainty_aware', 'low_confidence', 'random'],
                        help='Remasking strategy')
    parser.add_argument('--mc_samples', type=int, default=4,
                        help='Number of MC Dropout samples (only for uncertainty_aware)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for epistemic uncertainty (only for uncertainty_aware)')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for aleatoric uncertainty (only for uncertainty_aware)')
    parser.add_argument('--dropout_p', type=float, default=0.1,
                        help='Dropout probability for MC Dropout (only for uncertainty_aware)')
    parser.add_argument('--use_mc_dropout_logit', action='store_true',
                        help='Use MC dropout averaged logits for sampling instead of clean logits (may reduce generation quality)')
    
    # EOS token handling (from LLaDA paper Appendix B.4)
    parser.add_argument('--logits_eos_inf', action='store_true',
                        help='Set EOS token logits to -inf to prevent early termination')
    parser.add_argument('--confidence_eos_eot_inf', action='store_true',
                        help='Set confidence of EOS and EoT tokens to -inf')
    
    # Prompt settings
    parser.add_argument('--use_prompt', action='store_true',
                        help='Use prompt for generation (if False, generates unconditionally)')
    parser.add_argument('--benchmark', type=str, default='gsm8k',
                        choices=['gsm8k', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'custom'],
                        help='Benchmark to use for prompts')
    parser.add_argument('--custom_prompt', type=str, default=None,
                        help='Custom prompt text (used when benchmark=custom)')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples from benchmark (None = all samples)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing multiple samples at once (careful with memory)')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./result',
                        help='Directory to save outputs')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (used for output filenames)')
    
    args = parser.parse_args()
    
    # Generate experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        prompt_type = 'with_prompt' if args.use_prompt else 'no_prompt'
        benchmark_name = args.benchmark if args.use_prompt else 'unconditional'
        args.exp_name = f'{prompt_type}_{benchmark_name}_{args.remasking}_{timestamp}'
    
    # Create experiment-specific output directory
    exp_output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # Set devices - handle both single and multiple GPUs
    devices = args.device if isinstance(args.device, list) else [args.device]
    # Validate devices
    valid_devices = []
    for dev in devices:
        if dev == 'cpu' or (torch.cuda.is_available() and 'cuda' in dev):
            valid_devices.append(dev)
        else:
            print(f"Warning: Device {dev} not available, skipping...")
    
    if not valid_devices:
        valid_devices = ['cpu']
        print("Warning: No valid devices found, falling back to CPU")
    
    devices = valid_devices
    num_devices = len(devices)
    
    print("=" * 80)
    print(f"Experiment: {args.exp_name}")
    print("=" * 80)
    print(f"Output Directory: {exp_output_dir}")
    print(f"Device(s): {devices} ({num_devices} device{'s' if num_devices > 1 else ''})")
    print(f"Model: {args.model_path}")
    print(f"Steps: {args.steps}, Gen Length: {args.gen_length}, Block Length: {args.block_length}")
    print(f"Remasking Strategy: {args.remasking}")
    if args.remasking == 'uncertainty_aware':
        print(f"MC Samples: {args.mc_samples}, Alpha: {args.alpha}, Beta: {args.beta}, Dropout: {args.dropout_p}")
        print(f"Use MC Dropout Logit for Sampling: {args.use_mc_dropout_logit}")
    print(f"Use Prompt: {args.use_prompt}")
    if args.use_prompt:
        print(f"Benchmark: {args.benchmark}")
    print("=" * 80)
    
    # For single device, use the original sequential approach
    # For multiple devices, we'll use parallel processing
    use_parallel = num_devices > 1
    
    if not use_parallel:
        # Original single-device code path
        device = devices[0]
        print("\nLoading model...")
        model = AutoModel.from_pretrained(
            args.model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to(device).eval()
        
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, 
            trust_remote_code=True
        )
        
        if tokenizer.padding_side != 'left':
            tokenizer.padding_side = 'left'
        
        assert tokenizer.pad_token_id != 126336
    
    # Prepare prompts
    if args.use_prompt:
        if args.benchmark == 'custom' and args.custom_prompt:
            all_prompts = [args.custom_prompt]
        elif args.benchmark == 'custom':
            all_prompts = ["Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?"]
        else:
            all_prompts = get_benchmark_prompts(args.benchmark, args.num_samples)
        
        print(f"\nTotal prompts to process: {len(all_prompts)}")
        print(f"Batch size: {args.batch_size}")
        num_batches = (len(all_prompts) + args.batch_size - 1) // args.batch_size
        print(f"Number of batches: {num_batches}")
        
        # Process in batches
        all_outputs = []
        all_uncertainty_histories = []
        
        if use_parallel:
            # Parallel processing across multiple GPUs
            print(f"\nUsing parallel processing across {num_devices} GPUs...")
            
            # Prepare batch arguments for all batches
            batch_args_list = []
            for batch_idx in range(0, len(all_prompts), args.batch_size):
                batch_prompts = all_prompts[batch_idx:batch_idx + args.batch_size]
                device_idx = (batch_idx // args.batch_size) % num_devices
                
                batch_args = {
                    'device': devices[device_idx],
                    'batch_prompts': batch_prompts,
                    'model_path': args.model_path,
                    'steps': args.steps,
                    'gen_length': args.gen_length,
                    'block_length': args.block_length,
                    'temperature': args.temperature,
                    'cfg_scale': args.cfg_scale,
                    'remasking': args.remasking,
                    'logits_eos_inf': args.logits_eos_inf,
                    'confidence_eos_eot_inf': args.confidence_eos_eot_inf,
                    'mc_samples': args.mc_samples,
                    'alpha': args.alpha,
                    'beta': args.beta,
                    'dropout_p': args.dropout_p,
                    'use_mc_dropout_logit': args.use_mc_dropout_logit,
                    'use_prompt': args.use_prompt,
                    'batch_idx': batch_idx // args.batch_size
                }
                batch_args_list.append(batch_args)
            
            # Process batches in parallel
            # Use spawn method to avoid CUDA initialization issues
            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # Already set
            
            try:
                with ProcessPoolExecutor(max_workers=num_devices, initializer=worker_init) as executor:
                    # Submit all batch jobs
                    futures = {executor.submit(process_batch_on_device, batch_args): batch_args['batch_idx'] 
                              for batch_args in batch_args_list}
                    
                    # Collect results as they complete
                    results = {}
                    with tqdm(total=len(futures), desc="Processing batches") as pbar:
                        for future in as_completed(futures):
                            batch_idx = futures[future]
                            try:
                                batch_output, uncertainty_history, returned_batch_idx = future.result()
                                results[returned_batch_idx] = (batch_output, uncertainty_history)
                                pbar.update(1)
                            except Exception as e:
                                print(f"\nError processing batch {batch_idx}: {e}")
                                raise
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Shutting down workers...")
                executor.shutdown(wait=False, cancel_futures=True)
                print("Cleanup complete. Exiting.")
                sys.exit(1)
            
            # Sort results by batch index and extract outputs
            for batch_idx in sorted(results.keys()):
                batch_output, uncertainty_history = results[batch_idx]
                all_outputs.extend(batch_output)
                all_uncertainty_histories.append(uncertainty_history)
        
        else:
            # Sequential processing on single device (original code)
            for batch_idx in tqdm(range(0, len(all_prompts), args.batch_size), desc="Batch", total=num_batches):
                batch_prompts = all_prompts[batch_idx:batch_idx + args.batch_size]

                # Apply chat template if using Instruct model
                messages = [{"role": "user", "content": prompt} for prompt in batch_prompts]
                formatted_prompts = [
                    tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) 
                    for message in messages
                ]

                encoded_outputs = tokenizer(
                    formatted_prompts,
                    add_special_tokens=False,
                    padding=True,
                    return_tensors="pt"
                )
                input_ids = encoded_outputs['input_ids'].to(device)
                attention_mask = encoded_outputs['attention_mask'].to(device)

                # Generate with uncertainty tracking
                out, uncertainty_history = generate_with_uncertainty_tracking(
                    model=model,
                    prompt=input_ids,
                    attention_mask=attention_mask,
                    steps=args.steps,
                    gen_length=args.gen_length,
                    block_length=args.block_length,
                    temperature=args.temperature,
                    cfg_scale=args.cfg_scale,
                    remasking=args.remasking,
                    logits_eos_inf=args.logits_eos_inf,
                    confidence_eos_eot_inf=args.confidence_eos_eot_inf,
                    mc_samples=args.mc_samples,
                    alpha=args.alpha,
                    beta=args.beta,
                    dropout_p=args.dropout_p,
                    use_mc_dropout_logit=args.use_mc_dropout_logit
                )

                # Decode output for this batch
                batch_output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
                all_outputs.extend(batch_output)
                all_uncertainty_histories.append(uncertainty_history)
        
        # Aggregate uncertainty histories (average across batches)
        aggregated_history = {
            'timesteps': all_uncertainty_histories[0]['timesteps'],
            'mean_epistemic': [],
            'mean_aleatoric': [],
            'mean_total': [],
            'std_epistemic': [],
            'std_aleatoric': [],
            'std_total': []
        }
        
        # Aggregate all metrics (including unmasked token metrics)
        all_keys = ['mean_epistemic', 'mean_aleatoric', 'mean_total', 
                    'std_epistemic', 'std_aleatoric', 'std_total',
                    'mean_epistemic_unmasked', 'mean_aleatoric_unmasked', 'mean_total_unmasked',
                    'std_epistemic_unmasked', 'std_aleatoric_unmasked', 'std_total_unmasked']
        
        for key in all_keys:
            values_per_timestep = []
            for hist in all_uncertainty_histories:
                values_per_timestep.append(hist[key])
            # Average across batches
            aggregated_history[key] = np.mean(values_per_timestep, axis=0).tolist()
        
        output = all_outputs
        uncertainty_history = aggregated_history
        original_prompts = all_prompts  # Keep original prompts for saving
        
        print(f"\nFirst prompt example: {all_prompts[0][:100]}...")
    else:
        # No prompt - unconditional generation
        # For unconditional generation, we currently only support single device
        # (parallel processing is mainly useful when processing many prompts)
        if use_parallel:
            print("\nWarning: Parallel processing not needed for unconditional generation, using single device")
            use_parallel = False
            device = devices[0]
            # Load model if not already loaded
            if 'model' not in locals():
                print("\nLoading model...")
                model = AutoModel.from_pretrained(
                    args.model_path, 
                    trust_remote_code=True, 
                    torch_dtype=torch.bfloat16
                ).to(device).eval()
                
                tokenizer = AutoTokenizer.from_pretrained(
                    args.model_path, 
                    trust_remote_code=True
                )
                
                if tokenizer.padding_side != 'left':
                    tokenizer.padding_side = 'left'
                
                assert tokenizer.pad_token_id != 126336
        
        batch_size = args.batch_size
        original_prompts = None  # No prompts for unconditional generation
        
        # For Instruct models, we need to use the chat template even for "unconditional" generation
        # Otherwise the model immediately outputs EOS tokens
        if 'Instruct' in args.model_path:
            # Create minimal chat template with empty user message
            messages = [{'role': 'user', 'content': ''}]
            template = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            prompts = [template] * batch_size
            
            encoded_outputs = tokenizer(
                prompts,
                add_special_tokens=False,
                padding=True,
                return_tensors="pt"
            )
            input_ids = encoded_outputs['input_ids'].to(device)
            attention_mask = encoded_outputs['attention_mask'].to(device)
            print(f"\nGenerating with minimal Instruct template (empty prompt) - Batch size: {batch_size}")
        else:
            # For Base models, just use start token
            start_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
            input_ids = torch.tensor([[start_token]] * batch_size, device=device)
            attention_mask = torch.ones_like(input_ids)
            print(f"\nGenerating unconditionally (no prompt, Base model) - Batch size: {batch_size}")
        
        print("\nGenerating with uncertainty tracking...")
        
        # Generate with uncertainty tracking
        out, uncertainty_history = generate_with_uncertainty_tracking(
            model=model,
            prompt=input_ids,
            attention_mask=attention_mask,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
            remasking=args.remasking,
            logits_eos_inf=args.logits_eos_inf,
            confidence_eos_eot_inf=args.confidence_eos_eot_inf,
            mc_samples=args.mc_samples,
            alpha=args.alpha,
            beta=args.beta,
            dropout_p=args.dropout_p,
            use_mc_dropout_logit=args.use_mc_dropout_logit
        )
        
        # Decode output
        output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    
    # Print and save results
    print("\n" + "=" * 80)
    print(f"Generated output: (showing first 5 of {len(output)} samples)")
    print("=" * 80)
    
    # Print only first 5 samples to terminal
    num_to_print = min(5, len(output))
    for i in range(num_to_print):
        if original_prompts:
            print(f"\n[Sample {i+1}]")
            print(f"Question: {original_prompts[i][:200]}{'...' if len(original_prompts[i]) > 200 else ''}")
            print(f"Answer: {output[i][:500]}{'...' if len(output[i]) > 500 else ''}")
        else:
            print(f"\n[Sample {i+1}]")
            print(output[i][:500] + ('...' if len(output[i]) > 500 else ''))
        print('-' * 80)
    
    if len(output) > num_to_print:
        print(f"\n... and {len(output) - num_to_print} more samples (saved to file)")
    
    # Save ALL generated text to file
    text_output_path = os.path.join(exp_output_dir, 'output.txt')
    with open(text_output_path, 'w') as f:
        f.write(f"Experiment: {args.exp_name}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Steps: {args.steps}, Gen Length: {args.gen_length}\n")
        f.write(f"Remasking Strategy: {args.remasking}\n")
        if args.remasking == 'uncertainty_aware':
            f.write(f"MC Samples: {args.mc_samples}, Alpha: {args.alpha}, Beta: {args.beta}\n")
            f.write(f"Use MC Dropout Logit for Sampling: {args.use_mc_dropout_logit}\n")
        f.write(f"Use Prompt: {args.use_prompt}\n")
        if args.use_prompt:
            f.write(f"Benchmark: {args.benchmark}\n")
            f.write(f"Total Samples: {len(output)}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, o in enumerate(output):
            f.write(f"[Sample {i+1}]\n")
            if original_prompts:
                f.write(f"Question:\n{original_prompts[i]}\n\n")
                f.write(f"Answer:\n{o}\n\n")
            else:
                f.write(f"{o}\n\n")
            f.write("-" * 80 + "\n\n")
    print(f"\nAll {len(output)} generated texts saved to: {text_output_path}")
    
    # Print uncertainty statistics
    print("\nUncertainty Statistics:")
    print(f"Number of timesteps tracked: {len(uncertainty_history['timesteps'])}")
    print(f"Initial mean total uncertainty: {uncertainty_history['mean_total'][0]:.4f}")
    print(f"Final mean total uncertainty: {uncertainty_history['mean_total'][-1]:.4f}")
    print(f"Initial epistemic/aleatoric ratio: {uncertainty_history['mean_epistemic'][0] / (uncertainty_history['mean_aleatoric'][0] + 1e-6):.4f}")
    print(f"Final epistemic/aleatoric ratio: {uncertainty_history['mean_epistemic'][-1] / (uncertainty_history['mean_aleatoric'][-1] + 1e-6):.4f}")
    
    # Save uncertainty data
    data_output_path = os.path.join(exp_output_dir, 'uncertainty_data.npz')
    np.savez(data_output_path, **uncertainty_history)
    print(f"\nUncertainty data saved to: {data_output_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    title_suffix = f"Remasking: {args.remasking}"
    
    # Plot 1: All masked tokens - over timesteps
    plot1_path = os.path.join(exp_output_dir, 'uncertainty_over_timesteps.png')
    plot_uncertainty_over_time(uncertainty_history, plot1_path, title_suffix)
    
    # Plot 2: All masked tokens - components
    plot2_path = os.path.join(exp_output_dir, 'uncertainty_components.png')
    plot_uncertainty_components(uncertainty_history, plot2_path, title_suffix)
    
    # Additional plots for uncertainty_aware mode: same plots but for unmasked tokens only
    if args.remasking == 'uncertainty_aware':
        print("\n** Creating additional plots for unmasked tokens (uncertainty_aware mode)...")
        
        # Create a modified history dict with unmasked token data
        unmasked_history = {
            'timesteps': uncertainty_history['timesteps'],
            'mean_epistemic': uncertainty_history['mean_epistemic_unmasked'],
            'mean_aleatoric': uncertainty_history['mean_aleatoric_unmasked'],
            'mean_total': uncertainty_history['mean_total_unmasked'],
            'std_epistemic': uncertainty_history['std_epistemic_unmasked'],
            'std_aleatoric': uncertainty_history['std_aleatoric_unmasked'],
            'std_total': uncertainty_history['std_total_unmasked']
        }
        
        # Plot 3: Unmasked tokens - over timesteps
        plot3_path = os.path.join(exp_output_dir, 'uncertainty_over_timesteps_unmasked.png')
        unmasked_title_suffix = f"{title_suffix} (Unmasked Tokens Only)"
        plot_uncertainty_over_time(unmasked_history, plot3_path, unmasked_title_suffix)
        
        # Plot 4: Unmasked tokens - components
        plot4_path = os.path.join(exp_output_dir, 'uncertainty_components_unmasked.png')
        plot_uncertainty_components(unmasked_history, plot4_path, unmasked_title_suffix)
    
    print("\n" + "=" * 80)
    print("All outputs saved to:", exp_output_dir)
    print("=" * 80)


if __name__ == '__main__':
    main()
