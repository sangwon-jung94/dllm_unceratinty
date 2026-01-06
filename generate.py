import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import warnings

from transformers import AutoTokenizer, AutoModel


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


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


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


def compute_entropy(probs, dim=-1, eps=1e-9):
    """
    Compute entropy H(p) = -sum(p * log(p)) along specified dimension.
    """
    return (-probs * probs.clamp_min(eps).log()).sum(dim=dim)


def is_ensemble_model(model):
    """
    Check if the model is using EnsembleLLaDABlock for the last layer.
    """
    try:
        from models.EnsembleLLaDA import EnsembleLLaDABlock
        last_block = model.model.transformer.blocks[-1]
        return isinstance(last_block, EnsembleLLaDABlock)
    except (ImportError, AttributeError):
        return False


def compute_uncertainty_decomposition(model, x, attention_mask, mc_samples, cfg_scale, prompt_index, mask_id, dropout_p=None, memory_efficient=False):
    """
    Compute epistemic and aleatoric uncertainty via MC Dropout.
    
    If using EnsembleLLaDA, performs a single forward pass with num_ensembles=mc_samples.
    Otherwise, performs mc_samples forward passes with dropout enabled.

    Args:
        dropout_p: Dropout probability for MC Dropout. If None, uses existing layer probabilities.
        memory_efficient: If True and using EnsembleLLaDA, uses generator mode to save memory.
                         If False (default), uses stacked tensor mode for better compatibility.

    Returns:
        Tuple of (mean_logits, epistemic_uncertainty, aleatoric_uncertainty, mean_probs)
    """
    use_ensemble = is_ensemble_model(model)
    
    if use_ensemble:
        # EnsembleLLaDA: Single forward pass with num_ensembles=mc_samples
        last_block = model.model.transformer.blocks[-1]
        prev_training = last_block.mlp_dropout.training
        last_block.mlp_dropout.train()

        try:
            if cfg_scale > 0.:
                un_x = x.clone()
                un_x[prompt_index] = mask_id
                x_ = torch.cat([x, un_x], dim=0)
                if attention_mask is not None:
                    attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                    logits_output = model.model(x_, attention_mask=attention_mask_, num_ensembles=mc_samples, memory_efficient=memory_efficient).logits
                else:
                    logits_output = model.model(x_, num_ensembles=mc_samples, memory_efficient=memory_efficient).logits
            else:
                if attention_mask is not None:
                    logits_output = model.model(x, attention_mask=attention_mask, num_ensembles=mc_samples, memory_efficient=memory_efficient).logits
                else:
                    logits_output = model.model(x, num_ensembles=mc_samples, memory_efficient=memory_efficient).logits
            
            # Online calculation of uncertainty statistics
            # We accumulate sum of probs and sum of entropies to compute exact MI
            # without storing all logits in memory.
            
            sum_probs = None
            sum_entropy = None
            mean_logits = None  # We also track mean logits for final prediction if needed
            
            # Handle both generator (memory_efficient=True) and tensor (memory_efficient=False)
            import types
            if isinstance(logits_output, types.GeneratorType):
                iterator = logits_output
            else:
                # Stacked tensor [num_ensembles, Batch, Seq, Vocab]
                iterator = (logits_output[i] for i in range(logits_output.shape[0]))

            for i, logits_i in enumerate(iterator):
                # logits_i: [Batch, Seq, Vocab] or [2*Batch, Seq, Vocab] (if CFG)
                
                if cfg_scale > 0.:
                    # Apply CFG per sample
                    logits_cond, logits_uncond = torch.chunk(logits_i, 2, dim=0)
                    logits_i = logits_uncond + (cfg_scale + 1) * (logits_cond - logits_uncond)
                
                probs_i = F.softmax(logits_i, dim=-1)
                entropy_i = compute_entropy(probs_i, dim=-1)
                
                if i == 0:
                    sum_probs = probs_i
                    sum_entropy = entropy_i
                    mean_logits = logits_i
                else:
                    sum_probs += probs_i
                    sum_entropy += entropy_i
                    # Online mean for logits (optional, but good for consistency)
                    mean_logits += (logits_i - mean_logits) / (i + 1)
                
                # Explicitly delete to help GC
                del logits_i, probs_i, entropy_i
            
            # Compute final statistics
            p_bar = sum_probs / mc_samples
            H_aleatoric = sum_entropy / mc_samples
            H_total = compute_entropy(p_bar, dim=-1)
            H_epistemic = (H_total - H_aleatoric).clamp_min(0.0)
            
        finally:
            # Restore original training mode to avoid side-effects
            if not prev_training:
                last_block.mlp_dropout.eval()
        
    else:
        # Standard MC Dropout: Multiple forward passes
        probs_sum = None
        entropy_sum = None
        logits_sum = None

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

    return mean_logits, H_epistemic, H_aleatoric, p_bar


def compute_uncertainty_score(epistemic, aleatoric, alpha, beta):
    """
    Compute unmasking priority score: score_i = -alpha * U_epi_i - beta * U_ale_i
    Higher score = lower uncertainty = unmask first.
    """
    return -alpha * epistemic - beta * aleatoric


@ torch.no_grad()
def generate(model, prompt, attention_mask=None, steps=128, gen_length=128, block_length=128, temperature=0.,
             cfg_scale=0., remasking='low_confidence', mask_id=126336, logits_eos_inf=False, confidence_eos_eot_inf=False,
             mc_samples=8, alpha=1.0, beta=1.0, dropout_p=None, topk_entropy_k_ratio=10, weighted_entropy_lambda=2.0):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence', 'random', 'uncertainty_aware', 'topk_entropy', or 'weighted_entropy'.
        mask_id: The token id of [MASK] is 126336.
        logits_eos_inf: Whether to set the logits of EOS token to -inf. See Appendix B.4 of LLaDA for details.
        confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf. See Appendix B.4 of LLaDA for details.
        mc_samples: Number of MC Dropout forward passes for uncertainty estimation (default: 8).
                    Only used when remasking='uncertainty_aware'.
        alpha: Weight for epistemic uncertainty in unmasking score (default: 1.0).
               Higher alpha = more penalty for high epistemic uncertainty = delay uncertain positions.
        beta: Weight for aleatoric uncertainty in unmasking score (default: 1.0).
              Higher beta = more penalty for high aleatoric uncertainty = late-commit for ambiguous positions.
        dropout_p: Dropout probability for MC Dropout (default: None, uses existing layer probabilities).
                   Only used when remasking='uncertainty_aware'. Typical values: 0.1 to 0.3.
        topk_entropy_k_ratio: Ratio for top-k filtering in 'topk_entropy' remasking (default: 10).
                              k = num_transfer_tokens * topk_entropy_k_ratio.
        weighted_entropy_lambda: Lambda for weighting top-1 entropy in 'weighted_entropy' remasking (default: 2.0).
                                 H = lambda * H(p1, 1-p1) + (1-p1) * H(p2, p3, ...).
    '''
    # Check for dropout layers if using uncertainty-aware remasking
    if remasking == 'uncertainty_aware':
        dropout_count = sum(1 for m in model.modules() if isinstance(m, nn.Dropout))
        if dropout_count == 0:
            warnings.warn(
                "No nn.Dropout layers found in model. Falling back to 'low_confidence' remasking. "
                "MC Dropout requires dropout layers to be present in the model architecture.",
                UserWarning
            )
            remasking = 'low_confidence'

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([attention_mask, torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)], dim=-1)

    prompt_index = (x != mask_id)

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    for num_block in range(num_blocks):
        block_mask_index = (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length:] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        for i in range(steps):
            mask_index = (x == mask_id)
            if remasking == 'uncertainty_aware':
                # Use MC Dropout for uncertainty decomposition
                mean_logits, H_epistemic, H_aleatoric, p_bar = compute_uncertainty_decomposition(
                    model=model,
                    x=x,
                    attention_mask=attention_mask,
                    mc_samples=mc_samples,
                    cfg_scale=cfg_scale,
                    prompt_index=prompt_index,
                    mask_id=mask_id,
                    dropout_p=dropout_p
                )
                # Use mean probs (p_bar) instead of mean_logits for sampling
                # p_bar is the average of softmax outputs, which is more principled
                if logits_eos_inf:
                    p_bar[:, :, 126081] = 0.0  # Zero out EOS probability

                # Sample from mean probability distribution
                if temperature == 0:
                    x0 = torch.argmax(p_bar, dim=-1)
                else:
                    # Apply Gumbel noise to log probabilities
                    log_p_bar = torch.log(p_bar + 1e-10)
                    log_p_bar_with_noise = add_gumbel_noise(log_p_bar, temperature=temperature)
                    x0 = torch.argmax(log_p_bar_with_noise, dim=-1)

                # Compute unmasking score based on uncertainty
                x0_p = compute_uncertainty_score(H_epistemic, H_aleatoric, alpha, beta)

            else:
                # Original single-pass logic for 'low_confidence' and 'random'
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    if attention_mask is not None:
                        attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
                    logits = model(x_, attention_mask=attention_mask_).logits
                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                else:
                    logits = model(x, attention_mask=attention_mask).logits

                if logits_eos_inf:
                    logits[:, :, 126081] = -torch.inf

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if confidence_eos_eot_inf:
                    logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                elif remasking == 'entropy':
                    p = F.softmax(logits, dim=-1)
                    x0_p = -compute_entropy(p, dim=-1)
                elif remasking == 'topk_entropy':
                    # Step 1: Get top-k positions by probability
                    p = F.softmax(logits, dim=-1)
                    top1_probs = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                    # Calculate k based on num_transfer_tokens for this step
                    current_transfer = num_transfer_tokens[:, i].max().item()
                    k = max(1, int(current_transfer * topk_entropy_k_ratio))
                    # Get top-k positions by probability (higher prob = more confident)
                    _, topk_prob_indices = torch.topk(top1_probs, k=min(k, top1_probs.shape[-1]), dim=-1)
                    # Step 2: Among top-k, use entropy to decide remasking order
                    entropy = compute_entropy(p, dim=-1)
                    # Initialize with -inf so only top-k positions are considered
                    x0_p = torch.full_like(top1_probs, -np.inf)
                    # Set entropy-based scores for top-k positions (negative entropy = lower uncertainty = unmask first)
                    batch_indices = torch.arange(x0_p.shape[0], device=x0_p.device).unsqueeze(-1).expand_as(topk_prob_indices)
                    x0_p[batch_indices, topk_prob_indices] = -entropy[batch_indices, topk_prob_indices]
                elif remasking == 'weighted_entropy':
                    # H = lambda * H(p1, 1-p1) + (1-p1) * H(p2, p3, ...)
                    # where H(p1, 1-p1) = -p1*log(p1) - (1-p1)*log(1-p1) is binary entropy
                    p = F.softmax(logits, dim=-1)
                    # Get top-1 probability
                    p1 = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                    # Binary entropy of top-1: H(p1, 1-p1)
                    eps = 1e-10
                    H_binary = -p1 * torch.log(p1 + eps) - (1 - p1) * torch.log(1 - p1 + eps)
                    # Residual entropy: H(p2, p3, ...) normalized by (1-p1)
                    # Full entropy minus contribution from p1
                    H_full = compute_entropy(p, dim=-1)
                    H_residual = H_full + p1 * torch.log(p1 + eps)  # H - (-p1*log(p1)) = H + p1*log(p1)
                    # Weighted entropy: emphasize top-1 uncertainty
                    weighted_H = weighted_entropy_lambda * H_binary + (1 - p1) * (H_residual / (1 - p1 + eps))
                    x0_p = -weighted_H  # Lower weighted entropy = unmask first

            x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device) # Indicates which tokens to unmask
            for j in range(confidence.shape[0]):
                _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                transfer_index[j, select_index] = True
            x[transfer_index] = x0[transfer_index]

    return x


def main():
    device = 'cuda:2'

    model = AutoModel.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)

    # The LLaDA architecture theoretically supports both left-padding and right-padding. 
    # However, the sampling code implementation is simpler with left-padding.
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # If the padding ID equals the mask ID, you need to modify our generate function to achieve correct inference.
    assert tokenizer.pad_token_id != 126336

    prompts = [ "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 kilometers per hour. How many kilometers can she run in 8 hours?",
             "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?",
             "Randy has 60 mango trees on his farm. He also has 5 less than half as many coconut trees as mango trees. How many trees does Randy have in all on his farm?"]

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    messages = [{"role": "user", "content": prompt} for prompt in prompts]
    prompts = [tokenizer.apply_chat_template([message], add_generation_prompt=True, tokenize=False) for message in messages]

    encoded_outputs = tokenizer(
        prompts,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt"
    )
    input_ids = encoded_outputs['input_ids'].to(device)
    attention_mask = encoded_outputs['attention_mask'].to(device)

    out = generate(model, input_ids, attention_mask, steps=128, gen_length=128, block_length=32, temperature=0., cfg_scale=0., remasking='low_confidence')
    output = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)
    for o in output:
        print(o)
        print('-' * 50)

if __name__ == '__main__':
    main()
