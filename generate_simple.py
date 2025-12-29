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
import torch.nn.functional as F
import argparse
import os
import json
from datetime import datetime
from datasets import load_dataset
from tqdm import tqdm


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
):
    '''
    Generate text without uncertainty tracking.
    Only supports 'low_confidence' and 'random' remasking.
    
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
            else:
                raise ValueError(f"Unsupported remasking strategy: {remasking}. Use 'low_confidence' or 'random'.")
            
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


def main():
    parser = argparse.ArgumentParser(description='Simple generation without uncertainty tracking')
    
    # Model and device settings
    parser.add_argument('--model_path', type=str, default='GSAI-ML/LLaDA-8B-Instruct',
                        help='Path or name of the model')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on (e.g., cuda:0, cpu)')
    
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
                        choices=['low_confidence', 'random'],
                        help='Remasking strategy (uncertainty_aware not supported)')

    parser.add_argument('--dropout_p', type=float, default=None,
                        help='Dropout probability to set (if None, uses model default)')
    
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
    
    # Set device
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    
    print("=" * 80)
    print(f"Experiment: {args.exp_name}")
    print("=" * 80)
    print(f"Output Directory: {exp_output_dir}")
    print(f"Device: {device}")
    print(f"Model: {args.model_path}")
    print(f"Steps: {args.steps}, Gen Length: {args.gen_length}, Block Length: {args.block_length}")
    print(f"Remasking Strategy: {args.remasking}")
    print(f"Use Prompt: {args.use_prompt}")
    if args.use_prompt:
        print(f"Benchmark: {args.benchmark}")
    print("=" * 80)
    
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
        print(f"Number of batches: {(len(all_prompts) + args.batch_size - 1) // args.batch_size}")
        
        # Process in batches to avoid OOM
        all_outputs = []
        
        num_batches = (len(all_prompts) + args.batch_size - 1) // args.batch_size
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
            )

            # Decode output for this batch
            batch_output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
            all_outputs.extend(batch_output)
        
        output = all_outputs
        original_prompts = all_prompts
        
        print(f"\nFirst prompt example: {all_prompts[0][:100]}...")
    else:
        # No prompt - unconditional generation
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
            confㅌidence_eos_eot_inf=args.confidence_eos_eot_inf,
            dropout_p=args.dropout_p,
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
