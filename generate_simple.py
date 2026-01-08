import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel
from generate import (
    add_gumbel_noise,
    get_num_transfer_tokens,
    enable_mc_dropout,
    disable_mc_dropout,
)
from models.EnsembleLLaDA import get_ensemble_model
import torch.nn.functional as F
import argparse
import os
import json
import sys
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm
import multiprocessing as mp


@torch.no_grad()
def generate_simple(
    model, 
    prompt, 
    attention_mask=None, 
    steps=128, 
    gen_length=128, 
    block_length=128, 
    temperature=0.,
    cfg_scale=0., 
    remasking='low_confidence', 
    mask_id=126336, 
    logits_eos_inf=False, 
    confidence_eos_eot_inf=False,
    dropout_p=None,
    topk_entropy_k_ratio=10,
    weighted_entropy_lambda=2.0,
):
    '''
    Generate text without uncertainty tracking.
    Supports 'low_confidence', 'random', 'entropy', 'topk_entropy', and 'weighted_entropy' remasking.
    
    Returns:
        x: Generated sequences
    '''
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
    if dropout_p is not None:
        enable_mc_dropout(model, p=dropout_p)
    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in tqdm(range(steps_per_block), desc=f"Block {num_block+1}/{num_blocks} steps", leave=False):
            mask_index = (x == mask_id)
            
            if cfg_scale > 0:
                logits_cond = model(
                    input_ids=torch.where(prompt_index, x, mask_id),
                    attention_mask=attention_mask,
                ).logits
                logits_uncond = model(
                    input_ids=torch.full_like(x, mask_id),
                    attention_mask=attention_mask,
                ).logits
                logits = logits_uncond + cfg_scale * (logits_cond - logits_uncond)
            else:
                logits = model(
                    input_ids=x,
                    attention_mask=attention_mask,
                ).logits
            
            if logits_eos_inf:
                logits[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Apply confidence_eos_eot_inf before computing remasking scores
            if confidence_eos_eot_inf:
                logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

            # Choose remasking strategy
            if remasking == 'low_confidence':
                # Use confidence (probability of predicted token)
                p = F.softmax(logits, dim=-1)
                x0_p = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                # Random selection
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            elif remasking == 'entropy':
                # Use entropy (lower entropy = more confident = unmask first)
                p = F.softmax(logits, dim=-1)
                log_p = torch.log(p + 1e-10)  # Add small epsilon to avoid log(0)
                entropy = -torch.sum(p * log_p, dim=-1)
                # Negate entropy so lower entropy (more confident) gets higher priority
                x0_p = -entropy
            elif remasking == 'topk_entropy':
                # Step 1: Get top-k positions by probability, but restrict to masked tokens in the current block
                p = F.softmax(logits, dim=-1)
                top1_probs = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                current_block_start = prompt.shape[1] + num_block * block_length
                current_block_end = prompt.shape[1] + (num_block + 1) * block_length
                candidate_mask = mask_index.clone()
                candidate_mask[:, :current_block_start] = False
                candidate_mask[:, current_block_end:] = False
                masked_top1_probs = top1_probs.masked_fill(~candidate_mask, -torch.inf)
                # Calculate k based on num_transfer_tokens for this step
                current_transfer = num_transfer_tokens[:, i].max().item()
                k = max(1, int(current_transfer * topk_entropy_k_ratio))
                available = candidate_mask.sum(dim=1).max().item()

                if available == 0:
                    x0_p = torch.full_like(top1_probs, -np.inf)
                else:
                    k = min(k, available)
                    # Get top-k positions by probability (higher prob = more confident)
                    _, topk_prob_indices = torch.topk(masked_top1_probs, k=min(k, masked_top1_probs.shape[-1]), dim=-1)
                    # Step 2: Among top-k, use entropy to decide remasking order
                    log_p = torch.log(p + 1e-10)
                    entropy = -torch.sum(p * log_p, dim=-1)
                    # Initialize with -inf so only top-k positions are considered
                    x0_p = torch.full_like(top1_probs, -np.inf)
                    # Set entropy-based scores for top-k positions (negative entropy = lower uncertainty = unmask first)
                    batch_indices = torch.arange(x0_p.shape[0], device=x0_p.device).unsqueeze(-1).expand_as(topk_prob_indices)
                    x0_p[batch_indices, topk_prob_indices] = -entropy[batch_indices, topk_prob_indices]
            elif remasking == 'weighted_entropy':
                # H = lambda * H(p1, 1-p1) + (1-p1) * H(p2, p3, ...)
                p = F.softmax(logits, dim=-1)
                # Get top-1 probability
                p1 = torch.squeeze(
                    torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                # Binary entropy of top-1: H(p1, 1-p1)
                eps = 1e-10
                H_binary = -p1 * torch.log(p1 + eps) - (1 - p1) * torch.log(1 - p1 + eps)
                # Residual entropy
                log_p = torch.log(p + eps)
                H_full = -torch.sum(p * log_p, dim=-1)
                H_residual = H_full + p1 * torch.log(p1 + eps)
                # Weighted entropy: emphasize top-1 uncertainty
                weighted_H = weighted_entropy_lambda * H_binary + (1 - p1) * (H_residual / (1 - p1 + eps))
                x0_p = -weighted_H  # Lower weighted entropy = unmask first
            else:
                raise ValueError(f"Unsupported remasking strategy: {remasking}. Use 'low_confidence', 'random', 'entropy', 'topk_entropy', or 'weighted_entropy'.")
            
            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            # Vectorized topk operation
            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
            max_k = num_transfer_tokens[:, i].max().item()
            if max_k > 0:
                _, topk_indices = torch.topk(confidence, k=max_k, dim=1)
                valid_mask = torch.arange(max_k, device=x0.device).unsqueeze(0) < num_transfer_tokens[:, i].unsqueeze(1)
                batch_indices = torch.arange(confidence.shape[0], device=x0.device).unsqueeze(1).expand(-1, max_k)
                transfer_index[batch_indices[valid_mask], topk_indices[valid_mask]] = True
            
            x[transfer_index] = x0[transfer_index]
            global_step += 1
    if dropout_p is not None:
        disable_mc_dropout(model)

    return x


def get_benchmark_prompts(benchmark_name, num_samples=None):
    '''
    Load prompts from various benchmarks
    
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


def gpu_worker(device, task_queue, result_queue, common_args):
    """Worker process that loads model once and processes batches from queue dynamically."""
    if device != 'cpu':
        torch.cuda.set_device(device)

    # Load model ONCE at worker startup
    if common_args.get('use_ensemble_model', False):
        model = get_ensemble_model(
            common_args['model_path'],
            mlp_dropout_p=common_args.get('dropout_p'),
            torch_dtype=torch.bfloat16
        ).to(device).eval()
    else:
        model = AutoModel.from_pretrained(
            common_args['model_path'], 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16
        ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        common_args['model_path'], 
        trust_remote_code=True
    )

    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    assert tokenizer.pad_token_id != 126336

    # Process batches from queue until we get None (poison pill)
    while True:
        task = task_queue.get()
        if task is None:
            break
        
        batch_idx, batch_prompts = task
        
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

        out = generate_simple(
            model=model,
            prompt=input_ids,
            attention_mask=attention_mask,
            steps=common_args['steps'],
            gen_length=common_args['gen_length'],
            block_length=common_args['block_length'],
            temperature=common_args['temperature'],
            cfg_scale=common_args['cfg_scale'],
            remasking=common_args['remasking'],
            logits_eos_inf=common_args['logits_eos_inf'],
            confidence_eos_eot_inf=common_args['confidence_eos_eot_inf'],
            dropout_p=common_args['dropout_p'],
            topk_entropy_k_ratio=common_args['topk_entropy_k_ratio'],
            weighted_entropy_lambda=common_args['weighted_entropy_lambda'],
        )

        batch_output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
        result_queue.put((batch_idx, batch_output))

    # Clean up
    del model
    if device != 'cpu':
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description='Simple generation without uncertainty tracking')
    
    # Model and device settings
    parser.add_argument('--model_path', type=str, default='GSAI-ML/LLaDA-8B-Instruct',
                        help='Path or name of the model')
    parser.add_argument('--device', type=str, nargs='+', default=['cuda:0'],
                        help='Device(s) to run on (e.g., cuda:0 cuda:1 cpu)')
    
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
    
    # Remasking strategy (only low_confidence and random)
    parser.add_argument('--remasking', type=str, default='low_confidence',
                        choices=['low_confidence', 'random', 'entropy', 'topk_entropy', 'weighted_entropy'],
                        help='Remasking strategy')
    parser.add_argument('--topk_entropy_k_ratio', type=int, default=10,
                        help='Ratio for top-k filtering in topk_entropy remasking (k = num_transfer_tokens * ratio)')
    parser.add_argument('--weighted_entropy_lambda', type=float, default=2.0,
                        help='Lambda for weighting top-1 entropy in weighted_entropy remasking')

    parser.add_argument('--dropout_p', type=float, default=None,
                        help='Dropout probability to set (used for MC dropout and EnsembleLLaDA MLP; None uses model default)')

    parser.add_argument('--use_ensemble_model', action='store_true',
                        help='Use EnsembleLLaDA model (last-layer MLP dropout controlled by --dropout_p)')
    
    # EOS token handling
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
                        help='Batch size for processing multiple samples at once')
    
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
    
    # Validate devices
    devices = args.device if isinstance(args.device, list) else [args.device]
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
    use_parallel = num_devices > 1
    
    print("=" * 80)
    print(f"Experiment: {args.exp_name}")
    print("=" * 80)
    print(f"Output Directory: {exp_output_dir}")
    print(f"Device(s): {devices} ({num_devices} device{'s' if num_devices > 1 else ''})")
    print(f"Model: {args.model_path}")
    print(f"Steps: {args.steps}, Gen Length: {args.gen_length}, Block Length: {args.block_length}")
    print(f"Remasking Strategy: {args.remasking}")
    if args.use_ensemble_model:
        dropout_str = f"{args.dropout_p}" if args.dropout_p is not None else "config.residual_dropout"
        print(f"Using EnsembleLLaDA: True (last layer MLP dropout: {dropout_str})")
    print(f"Use Prompt: {args.use_prompt}")
    if args.use_prompt:
        print(f"Benchmark: {args.benchmark}")
    print("=" * 80)
    
    if not use_parallel:
        device = devices[0]
        print("\nLoading model...")
        if args.use_ensemble_model:
            print("Using EnsembleLLaDA model...")
            model = get_ensemble_model(
                args.model_path,
                mlp_dropout_p=args.dropout_p,
                torch_dtype=torch.bfloat16
            ).to(device).eval()
        else:
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
        print(f"Number of batches: {(len(all_prompts) + args.batch_size - 1) // args.batch_size}")
        
        # Process in batches to avoid OOM
        all_outputs = []
        num_batches = (len(all_prompts) + args.batch_size - 1) // args.batch_size

        if use_parallel:
            print(f"\nUsing parallel processing across {num_devices} GPUs with dynamic load balancing...")
            print(f"  → Model loaded ONCE per GPU, fast GPUs process more batches")

            try:
                mp.set_start_method('spawn', force=True)
            except RuntimeError:
                pass

            common_args = {
                'model_path': args.model_path,
                'steps': args.steps,
                'gen_length': args.gen_length,
                'block_length': args.block_length,
                'temperature': args.temperature,
                'cfg_scale': args.cfg_scale,
                'remasking': args.remasking,
                'logits_eos_inf': args.logits_eos_inf,
                'confidence_eos_eot_inf': args.confidence_eos_eot_inf,
                'dropout_p': args.dropout_p,
                'use_ensemble_model': args.use_ensemble_model,
                'topk_entropy_k_ratio': args.topk_entropy_k_ratio,
                'weighted_entropy_lambda': args.weighted_entropy_lambda
            }

            # Create shared queues for dynamic task distribution
            task_queue = mp.Queue()
            result_queue = mp.Queue()

            # Add all batches to the task queue
            total_batches = 0
            for batch_idx in range(0, len(all_prompts), args.batch_size):
                batch_prompts = all_prompts[batch_idx:batch_idx + args.batch_size]
                batch_num = batch_idx // args.batch_size
                task_queue.put((batch_num, batch_prompts))
                total_batches += 1

            # Add poison pills (None) to signal workers to exit
            for _ in range(num_devices):
                task_queue.put(None)

            # Start worker processes - one per GPU
            workers = []
            for device in devices:
                p = mp.Process(
                    target=gpu_worker,
                    args=(device, task_queue, result_queue, common_args)
                )
                p.start()
                workers.append(p)

            try:
                # Collect results as they complete
                results = {}
                with tqdm(total=total_batches, desc="Processing batches") as pbar:
                    for _ in range(total_batches):
                        batch_idx, batch_output = result_queue.get()
                        results[batch_idx] = batch_output
                        pbar.update(1)

                # Wait for all workers to finish
                for p in workers:
                    p.join()

            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Shutting down workers...")
                for p in workers:
                    p.terminate()
                for p in workers:
                    p.join()
                print("Cleanup complete. Exiting.")
                sys.exit(1)

            for batch_idx in sorted(results.keys()):
                all_outputs.extend(results[batch_idx])

        else:
            with tqdm(total=num_batches, desc="Batch", position=0, leave=True) as batch_pbar:
                for batch_idx in range(0, len(all_prompts), args.batch_size):
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

                    # Generate
                    out = generate_simple(
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
                        dropout_p=args.dropout_p,
                        topk_entropy_k_ratio=args.topk_entropy_k_ratio,
                        weighted_entropy_lambda=args.weighted_entropy_lambda,
                    )

                    # Decode output for this batch
                    batch_output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
                    all_outputs.extend(batch_output)
                    batch_pbar.update(1)
        
        output = all_outputs
        original_prompts = all_prompts
        
        print(f"\nFirst prompt example: {all_prompts[0][:100]}...")
    else:
        # No prompt - unconditional generation
        if use_parallel:
            print("\nWarning: Parallel processing not needed for unconditional generation, using single device")
            use_parallel = False
            device = devices[0]
            if 'model' not in locals():
                print("\nLoading model...")
                if args.use_ensemble_model:
                    print("Using EnsembleLLaDA model...")
                    model = get_ensemble_model(
                        args.model_path,
                        mlp_dropout_p=args.dropout_p,
                        torch_dtype=torch.bfloat16
                    ).to(device).eval()
                else:
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
        original_prompts = None
        
        # For Instruct models, use minimal chat template
        if 'Instruct' in args.model_path:
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
        
        print("\nGenerating...")
        
        # Generate
        out = generate_simple(
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
            dropout_p=args.dropout_p,
            topk_entropy_k_ratio=args.topk_entropy_k_ratio,
            weighted_entropy_lambda=args.weighted_entropy_lambda,
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
    
    # Create evaluation.json
    evaluation_data = {
        'experiment_name': args.exp_name,
        'model_path': args.model_path,
        'steps': args.steps,
        'gen_length': args.gen_length,
        'block_length': args.block_length,
        'temperature': args.temperature,
        'cfg_scale': args.cfg_scale,
        'remasking': args.remasking,
        'use_prompt': args.use_prompt,
        'benchmark': args.benchmark if args.use_prompt else None,
        'num_samples': len(output),
        'samples': []
    }
    
    for i, o in enumerate(output):
        sample_data = {
            'sample_id': i + 1,
            'output': o
        }
        if original_prompts:
            sample_data['prompt'] = original_prompts[i]
        evaluation_data['samples'].append(sample_data)
    
    json_output_path = os.path.join(exp_output_dir, 'evaluation.json')
    with open(json_output_path, 'w') as f:
        json.dump(evaluation_data, f, indent=2)
    print(f"Evaluation data saved to: {json_output_path}")
    
    print("\n" + "=" * 80)
    print("All outputs saved to:", exp_output_dir)
    print("=" * 80)


if __name__ == '__main__':
    main()
