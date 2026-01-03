"""
일회용 실험 스크립트: MC Dropout 앙상블 로짓으로 샘플링하되, 
remasking strategy는 단일 샘플 로짓 사용

차이점:
- 샘플링: 앙상블된 mean_logits 사용 (기존과 동일)
- Remasking: 앙상블 되기 전 마지막 단일 샘플의 로짓/확률 사용
"""
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from datetime import datetime
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import signal
import sys
from tqdm import tqdm
import psutil


def compute_entropy(p, dim=-1):
    """
    Compute entropy of probability distribution.
    H = -sum(p * log(p))
    """
    p_log_p = p * torch.log(p + 1e-10)
    entropy = -torch.sum(p_log_p, dim=dim)
    return entropy


def enable_mc_dropout(model, p=None):
    """MC Dropout을 위해 dropout 레이어 활성화"""
    dropout_count = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Dropout) and not name.endswith("emb_drop"):
            if p is not None:
                m.p = p
            m.train()
            dropout_count += 1
        # if m.__class__.__name__ == "LLaDALlamaBlock":
        #     m.config.attention_dropout = p if p is not None else m.config.attention_dropout
        #     m.train()
        #     dropout_count += 1
    return dropout_count


def disable_mc_dropout(model):
    """Dropout 레이어 비활성화"""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
        if m.__class__.__name__ == "LLaDALlamaBlock":
            m.eval()
            m.config.attention_dropout = 0.


def compute_uncertainty_decomposition_with_single_sample(
    model, x, attention_mask, mc_samples, cfg_scale, prompt_index, mask_id, dropout_p=None
):
    """
    MC Dropout으로 epistemic/aleatoric uncertainty 계산하되,
    마지막 단일 샘플의 로짓과 확률도 함께 리턴
    
    Returns:
        mean_logits: MC 샘플들의 평균 로짓 (앙상블)
        H_epistemic: Epistemic uncertainty
        H_aleatoric: Aleatoric uncertainty
        p_bar: 평균 확률 분포 (앙상블)
        single_logits: 마지막 MC 샘플의 로짓 (단일)
        single_probs: 마지막 MC 샘플의 확률 분포 (단일)
    """
    probs_sum = None
    entropy_sum = None
    logits_sum = None
    single_logits = None
    single_probs = None

    dropout_count = enable_mc_dropout(model, p=dropout_p)

    try:
        for k in range(mc_samples):
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                    logits = model(x_, attention_mask=attention_mask_).logits
                else:
                    logits = model(x_).logits
                logits, un_logits = torch.chunk(logits, 2, dim=0)
                logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
            else:
                if attention_mask is not None:
                    logits = model(x, attention_mask=attention_mask).logits
                else:
                    logits = model(x).logits

            p = F.softmax(logits, dim=-1)
            h_k = compute_entropy(p, dim=-1)

            # 마지막 샘플 저장 (단일 샘플용)
            single_logits = logits.clone()
            single_probs = p.clone()

            if probs_sum is None:
                probs_sum = p.clone()
                entropy_sum = h_k.clone()
                logits_sum = logits.clone()
            else:
                probs_sum = probs_sum + p
                entropy_sum = entropy_sum + h_k
                logits_sum = logits_sum + logits
    except Exception as e:
        raise e
    finally:
        disable_mc_dropout(model)

    p_bar = probs_sum / mc_samples
    mean_logits = logits_sum / mc_samples
    H_total = compute_entropy(p_bar, dim=-1)
    H_aleatoric = entropy_sum / mc_samples
    H_epistemic = (H_total - H_aleatoric).clamp_min(0.0)

    return mean_logits, H_epistemic, H_aleatoric, p_bar, single_logits, single_probs


def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(block_mask_index, steps_per_block):
    """Calculate number of tokens to unmask at each step"""
    num_mask_tokens_list = torch.sum(block_mask_index, dim=-1, keepdim=True)
    frac_mask = torch.arange(1, steps_per_block + 1, device=block_mask_index.device).unsqueeze(0) / steps_per_block
    num_transfer_tokens = torch.round(frac_mask * num_mask_tokens_list).long()
    num_transfer_tokens = torch.cat([
        num_transfer_tokens[:, 0:1],
        num_transfer_tokens[:, 1:] - num_transfer_tokens[:, :-1]
    ], dim=-1)
    return num_transfer_tokens


def compute_uncertainty_score(epistemic, aleatoric, alpha, beta):
    """
    Compute unmasking priority score: score_i = -alpha * U_epi_i - beta * U_ale_i
    Higher score = lower uncertainty = unmask first.
    """
    return -alpha * epistemic - beta * aleatoric


@torch.no_grad()
def generate_with_single_sample_remasking(
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
    use_single_sample_for_remasking=True
):
    '''
    실험용 생성 함수:
    - 샘플링은 앙상블된 로짓 사용
    - Remasking은 단일 샘플 로짓 사용 (use_single_sample_for_remasking=True인 경우)
    
    Returns:
        x: Generated sequences
        uncertainty_history: Dict containing timestep-wise uncertainty metrics
    '''
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
        'std_total_unmasked': [],
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

        for i in tqdm(range(steps_per_block), desc=f"Block {num_block+1}/{num_blocks} steps", leave=False, position=1):
            mask_index = (x == mask_id)
            
            # Compute uncertainty decomposition with single sample
            mean_logits_mc, H_epistemic, H_aleatoric, p_bar, single_logits, single_probs = \
                compute_uncertainty_decomposition_with_single_sample(
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
            
            # Track uncertainty statistics
            masked_positions = mask_index
            if masked_positions.any():
                epistemic_masked = H_epistemic[masked_positions]
                aleatoric_masked = H_aleatoric[masked_positions]
                total_masked = H_total[masked_positions]
                
                uncertainty_history['timesteps'].append(global_step)
                uncertainty_history['mean_epistemic'].append(epistemic_masked.mean().item())
                uncertainty_history['mean_aleatoric'].append(aleatoric_masked.mean().item())
                uncertainty_history['mean_total'].append(total_masked.mean().item())
                uncertainty_history['std_epistemic'].append(epistemic_masked.std().item())
                uncertainty_history['std_aleatoric'].append(aleatoric_masked.std().item())
                uncertainty_history['std_total'].append(total_masked.std().item())
            
            # **핵심 차이점**: 샘플링은 앙상블 로짓 사용
            logits_for_sampling = mean_logits_mc
            
            if logits_eos_inf:
                logits_for_sampling[:, :, 126081] = -torch.inf

            logits_with_noise = add_gumbel_noise(logits_for_sampling, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # **핵심 차이점**: Remasking은 단일 샘플 또는 앙상블 선택
            if use_single_sample_for_remasking:
                # 단일 샘플 로짓으로 remasking 결정
                logits_for_remasking = single_logits
                probs_for_remasking = single_probs
            else:
                # 앙상블 로짓으로 remasking 결정 (기존 방식)
                logits_for_remasking = mean_logits_mc
                probs_for_remasking = p_bar

            # Apply confidence_eos_eot_inf before computing remasking scores
            if confidence_eos_eot_inf:
                logits_for_remasking[:, :, 126081] = logits_for_remasking[:, :, 126348] = -torch.inf

            # Choose remasking strategy
            if remasking == 'uncertainty_aware':
                # 단일 샘플로 계산된 uncertainty 사용하려면 여기도 수정 필요
                # 하지만 uncertainty는 여러 샘플이 필요하므로 앙상블 사용
                x0_p = compute_uncertainty_score(H_epistemic, H_aleatoric, alpha, beta)
            elif remasking == 'low_confidence':
                # 단일 샘플의 확률 사용
                x0_p = torch.squeeze(
                    torch.gather(probs_for_remasking, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
            elif remasking == 'random':
                x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            
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
    '''Plot uncertainty metrics over diffusion timesteps with mean ± std bands.'''
    fig, ax = plt.subplots(figsize=(12, 8))
    
    timesteps = uncertainty_history['timesteps']
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
    title = 'Uncertainty Over Diffusion Timesteps (Mean ± Std)\n(Single Sample Remasking Experiment)'
    if title_suffix:
        title += f'\n{title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    plt.close()


def worker_init():
    """Ignore SIGINT in worker processes so the main process can handle interrupts."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def process_device_batches(args_dict):
    """Run all assigned batches sequentially on one device to avoid overlapping loads."""
    device = args_dict['device']
    device_batches = args_dict['device_batches']  # list of (batch_idx, batch_prompts)
    common_args = args_dict['common_args']

    if device != 'cpu':
        torch.cuda.set_device(device)

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

    results = []

    for batch_idx, batch_prompts in device_batches:
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

        out, uncertainty_history = generate_with_single_sample_remasking(
            model=model,
            prompt=input_ids,
            attention_mask=attention_mask,
            steps=common_args['steps'],
            gen_length=common_args['gen_length'],
            block_length=common_args['block_length'],
            temperature=common_args['temperature'],
            cfg_scale=common_args['cfg_scale'],
            remasking=common_args['remasking'],
            mask_id=common_args['mask_id'],
            logits_eos_inf=common_args['logits_eos_inf'],
            confidence_eos_eot_inf=common_args['confidence_eos_eot_inf'],
            mc_samples=common_args['mc_samples'],
            alpha=common_args['alpha'],
            beta=common_args['beta'],
            dropout_p=common_args['dropout_p'],
            use_single_sample_for_remasking=common_args['use_single_sample_for_remasking']
        )

        batch_output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
        results.append((batch_idx, batch_output, uncertainty_history))

    del model
    if device != 'cpu':
        torch.cuda.empty_cache()

    return results


def get_benchmark_prompts(benchmark_name, num_samples=None):
    """Load prompts from selected benchmark for prompt-based generation."""
    prompts = []

    if benchmark_name == 'gsm8k':
        dataset = load_dataset('gsm8k', 'main', split='test')
        num_to_use = len(dataset) if num_samples is None else min(num_samples, len(dataset))
        prompts = [item['question'] for item in dataset.select(range(num_to_use))]
    else:
        raise ValueError(f"Unknown benchmark: {benchmark_name}")

    return prompts


def main():
    parser = argparse.ArgumentParser(description='실험: 앙상블 샘플링 + 단일 샘플 remasking')
    
    # Model and device settings
    parser.add_argument('--model_path', type=str, default='GSAI-ML/LLaDA-8B-Instruct',
                        help='Path or name of the model')
    parser.add_argument('--device', type=str, nargs='+', default=['cuda:0'],
                        help='Device(s) to run on (e.g., cuda:0 cuda:1 cpu)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Number of prompts to process per batch (per device). Default = all prompts.')
    
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
    parser.add_argument('--remasking', type=str, default='low_confidence',
                        choices=['uncertainty_aware', 'low_confidence', 'random'],
                        help='Remasking strategy')
    parser.add_argument('--mc_samples', type=int, default=4,
                        help='Number of MC Dropout samples')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for epistemic uncertainty')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Weight for aleatoric uncertainty')
    parser.add_argument('--dropout_p', type=float, default=0.1,
                        help='Dropout probability for MC Dropout')
    parser.add_argument('--use_mc_dropout_logit', action='store_true',
                        help='(Compatibility) Accept flag; logits already use MC dropout by design')
    
    # Experiment toggle
    parser.add_argument('--use_single_sample_for_remasking', action='store_true',
                        help='Use single sample logits for remasking (default: True for this experiment)')
    parser.add_argument('--use_ensemble_for_remasking', dest='use_single_sample_for_remasking', 
                        action='store_false',
                        help='Use ensemble logits for remasking (baseline)')
    parser.set_defaults(use_single_sample_for_remasking=True)
    
    # EOS token handling
    parser.add_argument('--logits_eos_inf', action='store_true',
                        help='Set EOS token logits to -inf')
    parser.add_argument('--confidence_eos_eot_inf', action='store_true',
                        help='Set confidence of EOS and EoT tokens to -inf')
    
    # Prompt settings
    parser.add_argument('--prompt', type=str, 
                        default="Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
                        help='Prompt text')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to generate')
    parser.add_argument('--use_prompt', action='store_true',
                        help='Use benchmark/custom prompts instead of repeating --prompt')
    parser.add_argument('--benchmark', type=str, default='gsm8k',
                        choices=['gsm8k', 'custom'],
                        help='Benchmark name when --use_prompt is set')
    parser.add_argument('--custom_prompt', type=str, default=None,
                        help='Custom prompt text when benchmark=custom')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./result',
                        help='Directory to save outputs')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    
    args = parser.parse_args()
    
    # Generate experiment name if not provided
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        remasking_type = 'single_sample' if args.use_single_sample_for_remasking else 'ensemble'
        args.exp_name = f'experiment_{args.remasking}_{remasking_type}_{timestamp}'
    
    # Create experiment-specific output directory
    exp_output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_output_dir, exist_ok=True)
    
    # Device setup
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

    # Prepare prompts
    if args.use_prompt:
        if args.benchmark == 'custom':
            if args.custom_prompt is not None:
                all_prompts = [args.custom_prompt]
            else:
                all_prompts = [args.prompt]
        else:
            all_prompts = get_benchmark_prompts(args.benchmark, args.num_samples)
    else:
        all_prompts = [args.prompt] * args.num_samples
    batch_size = args.batch_size if args.batch_size is not None else len(all_prompts)
    num_batches = (len(all_prompts) + batch_size - 1) // batch_size

    print("=" * 80)
    print(f"실험: 앙상블 샘플링 + {'단일 샘플' if args.use_single_sample_for_remasking else '앙상블'} Remasking")
    print("=" * 80)
    print(f"Output Directory: {exp_output_dir}")
    print(f"Device(s): {devices} ({num_devices} device{'s' if num_devices > 1 else ''})")
    print(f"Batch size: {batch_size}")
    print(f"Model: {args.model_path}")
    print(f"Remasking Strategy: {args.remasking}")
    print(f"MC Samples: {args.mc_samples}, Dropout: {args.dropout_p}")
    print(f"Use Single Sample for Remasking: {args.use_single_sample_for_remasking}")
    print("=" * 80)
    if args.use_prompt:
        print(f"\nBenchmark: {args.benchmark}")
        if args.benchmark == 'custom' and args.custom_prompt:
            print(f"Custom prompt: {args.custom_prompt[:80]}{'...' if len(args.custom_prompt) > 80 else ''}")
    else:
        print(f"\nPrompt: {args.prompt}")
    print(f"Total samples: {len(all_prompts)} (batches: {num_batches})\n")

    all_outputs = []
    all_uncertainty_histories = []

    if use_parallel:
        print(f"Using parallel processing across {num_devices} devices...\n")

        # Round-robin assign batches to devices but process sequentially per device to avoid overlapping models on the same GPU.
        per_device_batches = {dev: [] for dev in devices}
        for batch_idx in range(0, len(all_prompts), batch_size):
            batch_prompts = all_prompts[batch_idx:batch_idx + batch_size]
            device_idx = (batch_idx // batch_size) % num_devices
            per_device_batches[devices[device_idx]].append((batch_idx // batch_size, batch_prompts))

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
            'mc_samples': args.mc_samples,
            'alpha': args.alpha,
            'beta': args.beta,
            'dropout_p': args.dropout_p,
            'use_single_sample_for_remasking': args.use_single_sample_for_remasking,
            'mask_id': 126336
        }

        executor = None
        try:
            with ProcessPoolExecutor(max_workers=num_devices, initializer=worker_init) as executor:
                futures = {
                    executor.submit(
                        process_device_batches,
                        {
                            'device': device,
                            'device_batches': per_device_batches[device],
                            'common_args': common_args
                        }
                    ): device for device in devices if per_device_batches[device]
                }

                results = {}
                with tqdm(total=len(futures), desc="Processing batches", position=0, leave=True) as pbar:
                    for future in as_completed(futures):
                        try:
                            batch_results = future.result()
                            for batch_idx, batch_output, uncertainty_history in batch_results:
                                results[batch_idx] = (batch_output, uncertainty_history)
                            pbar.update(1)
                        except Exception as e:
                            print(f"\n\n[!] Error processing device {futures[future]}: {e}")
                            print("[!] Shutting down due to error...")
                            executor.shutdown(wait=False, cancel_futures=True)
                            current_process = psutil.Process()
                            children = current_process.children(recursive=True)
                            for child in children:
                                try:
                                    child.terminate()
                                except:
                                    pass
                            raise
        except KeyboardInterrupt:
            print("\n\n[!] Interrupted by user (Ctrl+C). Shutting down...")
            if executor:
                print("Cancelling running tasks...")
                executor.shutdown(wait=False, cancel_futures=True)
            print("Terminating worker processes...")
            current_process = psutil.Process()
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except:
                    pass
            print("Cleanup complete. Exiting.")
            sys.exit(0)

        for batch_idx in sorted(results.keys()):
            batch_output, uncertainty_history = results[batch_idx]
            all_outputs.extend(batch_output)
            all_uncertainty_histories.append(uncertainty_history)

    else:
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

        for batch_idx in tqdm(range(0, len(all_prompts), batch_size), desc="Processing batches", total=num_batches, position=0, leave=True):
            batch_prompts = all_prompts[batch_idx:batch_idx + batch_size]

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

            out, uncertainty_history = generate_with_single_sample_remasking(
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
                use_single_sample_for_remasking=args.use_single_sample_for_remasking
            )

            batch_output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
            all_outputs.extend(batch_output)
            all_uncertainty_histories.append(uncertainty_history)

    # Aggregate histories across batches/devices
    history_keys = ['mean_epistemic', 'mean_aleatoric', 'mean_total', 
                    'std_epistemic', 'std_aleatoric', 'std_total',
                    'mean_epistemic_unmasked', 'mean_aleatoric_unmasked', 'mean_total_unmasked',
                    'std_epistemic_unmasked', 'std_aleatoric_unmasked', 'std_total_unmasked']
    if not all_uncertainty_histories:
        raise RuntimeError("No uncertainty histories collected.")

    if len(all_uncertainty_histories) == 1:
        uncertainty_history = all_uncertainty_histories[0]
    else:
        aggregated_history = {'timesteps': all_uncertainty_histories[0]['timesteps']}
        for key in history_keys:
            aggregated_history[key] = np.mean([hist[key] for hist in all_uncertainty_histories], axis=0).tolist()
        uncertainty_history = aggregated_history

    output = all_outputs

    # Print results
    print("\n" + "=" * 80)
    print("Generated outputs:")
    print("=" * 80)
    for i, o in enumerate(output):
        print(f"\n[Sample {i+1}]")
        print(o)
        print('-' * 80)
    
    # Save results
    text_output_path = os.path.join(exp_output_dir, 'output.txt')
    with open(text_output_path, 'w') as f:
        f.write(f"Experiment: {args.exp_name}\n")
        f.write(f"Use Single Sample for Remasking: {args.use_single_sample_for_remasking}\n")
        f.write(f"Remasking Strategy: {args.remasking}\n")
        f.write(f"MC Samples: {args.mc_samples}, Dropout: {args.dropout_p}\n")
        f.write(f"Prompt: {args.prompt}\n")
        f.write("=" * 80 + "\n\n")
        
        for i, o in enumerate(output):
            f.write(f"[Sample {i+1}]\n")
            f.write(o + "\n")
            f.write("-" * 80 + "\n\n")
    print(f"\nResults saved to: {text_output_path}")
    
    # Save uncertainty data
    data_output_path = os.path.join(exp_output_dir, 'uncertainty_data.npz')
    np.savez(data_output_path, **uncertainty_history)
    print(f"Uncertainty data saved to: {data_output_path}")
    
    # Create visualization
    print("\nCreating visualization...")
    plot_path = os.path.join(exp_output_dir, 'uncertainty_over_timesteps.png')
    title_suffix = f"Remasking: {args.remasking} ({'Single Sample' if args.use_single_sample_for_remasking else 'Ensemble'})"
    plot_uncertainty_over_time(uncertainty_history, plot_path, title_suffix)
    
    print("\n" + "=" * 80)
    print(f"All outputs saved to: {exp_output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
