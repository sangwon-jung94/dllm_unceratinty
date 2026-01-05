"""
Test uncertainty decomposition on simple masked examples
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
import argparse


def compute_entropy(p, dim=-1):
    """Compute entropy: H = -sum(p * log(p))"""
    p_log_p = p * torch.log(p + 1e-10)
    entropy = -torch.sum(p_log_p, dim=dim)
    return entropy


def enable_mc_dropout(model, p=None):
    """Enable dropout layers for MC Dropout"""
    dropout_count = 0
    for name, m in model.named_modules():
        if isinstance(m, nn.Dropout) and not name.endswith("emb_drop"):
            if p is not None:
                m.p = p
            m.train()
            dropout_count += 1
    return dropout_count


def disable_mc_dropout(model):
    """Disable dropout layers"""
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()


def compute_uncertainty_for_masked_tokens(model, input_ids, mask_positions, mc_samples=10, dropout_p=0.1):
    """
    Compute epistemic and aleatoric uncertainty for masked tokens using MC Dropout
    
    Args:
        model: The model
        input_ids: Input token IDs with masked tokens
        mask_positions: Boolean tensor indicating masked positions
        mc_samples: Number of MC dropout samples
        dropout_p: Dropout probability
        
    Returns:
        Dictionary with uncertainty metrics and predictions
    """
    model.eval()  # Keep model in eval mode (only dropout will be in train)
    
    probs_sum = None
    entropy_sum = None
    logits_list = []
    
    dropout_count = enable_mc_dropout(model, p=dropout_p)
    print(f"  Enabled {dropout_count} dropout layers with p={dropout_p}")
    
    if dropout_count == 0:
        print("  Warning: No dropout layers found! Uncertainty decomposition may not work properly.")
    
    # MC Dropout sampling
    with torch.no_grad():
        for k in range(mc_samples):
            logits = model(input_ids).logits
            logits_list.append(logits)
            
            probs = F.softmax(logits, dim=-1)
            entropy = compute_entropy(probs, dim=-1)
            
            if probs_sum is None:
                probs_sum = probs
                entropy_sum = entropy
            else:
                probs_sum += probs
                entropy_sum += entropy
    
    disable_mc_dropout(model)
    
    # Compute mean
    p_bar = probs_sum / mc_samples
    H_bar = entropy_sum / mc_samples
    
    # Total uncertainty
    H_total = compute_entropy(p_bar, dim=-1)
    
    # Aleatoric uncertainty (average entropy of individual predictions)
    H_aleatoric = H_bar
    
    # Epistemic uncertainty (difference between total and aleatoric)
    H_epistemic = (H_total - H_aleatoric).clamp_min(0.0)
    
    # Get predictions for masked positions
    mean_logits = torch.stack(logits_list).mean(dim=0)
    masked_logits = mean_logits[0, mask_positions]
    masked_probs = F.softmax(masked_logits, dim=-1)
    
    # Get top predictions for each masked position
    predictions = []
    for pos_idx in range(masked_logits.shape[0]):
        top_probs, top_indices = torch.topk(masked_probs[pos_idx], k=10)
        predictions.append({
            'position': pos_idx,
            'top_tokens': top_indices.tolist(),
            'top_probs': top_probs.tolist()
        })
    
    # Average uncertainties over masked positions
    avg_epistemic = H_epistemic[0, mask_positions].mean().item()
    avg_aleatoric = H_aleatoric[0, mask_positions].mean().item()
    avg_total = H_total[0, mask_positions].mean().item()
    
    return {
        'epistemic': avg_epistemic,
        'aleatoric': avg_aleatoric,
        'total': avg_total,
        'predictions': predictions,
        'mean_logits': mean_logits,
        'dropout_count': dropout_count
    }


def create_masked_prompt(tokenizer, text_with_mask):
    """
    Convert text with [MASK] placeholder to token IDs with mask tokens.
    """
    mask_token_id = 126336  # LLaDA mask token
    
    # Replace [MASK] with actual mask token
    text = text_with_mask.replace("[MASK]", tokenizer.decode([mask_token_id]))
    
    # Tokenize
    tokens = tokenizer(text, return_tensors="pt", add_special_tokens=True)
    
    return tokens.input_ids


def test_examples(model_path="GSAI-ML/LLaDA-8B-Instruct", device="cuda", mc_samples=10):
    """
    Test uncertainty decomposition on the provided examples
    """
    print("=" * 80)
    print("Loading model...")
    print("=" * 80)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )
    model = model.to(device).eval()
    
    # Test examples
    examples = [
        {
            "text": "I went to the bank to deposit the [MASK].",
            "candidates": ["Money", "cash", "check"],
            "description": "Banking deposit example"
        },
        {
            "text": "He answered the question [MASK].",
            "candidates": ["Quickly", "confidently", "honestly"],
            "description": "Manner of answering"
        },
        {
            "text": "He [MASK] answered the question.",
            "candidates": ["Quickly", "confidently", "honestly"],
            "description": "Manner of answering"
        },
        {
            "text": "The protein p53 regulates [MASK].",
            "candidates": ["Apoptosis", "cell cycle", "transcription"],
            "description": "Biological function"
        },
        {
            "text": "[MASK] wrote the novel [MASK].",
            "candidates": ["Orwell – 1984", "Huxley – Brave New World", "Tolkien – The Hobbit"],
            "description": "Author and book (multiple masks)"
        },
        {
            "text": "In a binary search tree, the in-order traversal always outputs the keys in [MASK] order.",
            "candidates": ["ascending", "descending", "random"],
            "description": "Binary search tree traversal order"
        },
        {
            "text": "The professor of MIIL is [MASK].",
            "candidates": ["Paul", "John", "Alice"],
            "description": "Professor of MIIL"
        }
    ]
    
    results = []
    
    for idx, example in enumerate(examples, 1):
        print(f"\n{'=' * 80}")
        print(f"Example {idx}: {example['description']}")
        print(f"Text: {example['text']}")
        print(f"Candidates: {', '.join(example['candidates'])}")
        print("=" * 80)
        
        # Create masked input
        input_ids = create_masked_prompt(tokenizer, example['text'])
        input_ids = input_ids.to(device)
        
        # Find mask positions
        mask_token_id = 126336
        mask_positions = (input_ids[0] == mask_token_id)
        num_masks = mask_positions.sum().item()
        
        print(f"Number of masks: {num_masks}")
        print(f"Input tokens: {tokenizer.decode(input_ids[0])}")
        
        # Compute uncertainty
        print(f"\nComputing uncertainty with MC Dropout (mc_samples={mc_samples})...")
        
        try:
            unc_result = compute_uncertainty_for_masked_tokens(
                model=model,
                input_ids=input_ids,
                mask_positions=mask_positions,
                mc_samples=mc_samples,
                dropout_p=0.1
            )
            
            print(f"\n✓ Uncertainty Analysis:")
            print(f"  Total uncertainty:     {unc_result['total']:.4f}")
            print(f"  Epistemic uncertainty: {unc_result['epistemic']:.4f}")
            print(f"  Aleatoric uncertainty: {unc_result['aleatoric']:.4f}")
            print(f"  Epistemic/Aleatoric ratio: {unc_result['epistemic']/max(unc_result['aleatoric'], 1e-6):.2f}")
            
            # Show top predictions for each masked position
            print(f"\nTop predictions for masked positions:")
            for pred in unc_result['predictions']:
                pos = pred['position']
                print(f"  Mask position {pos}:")
                for i, (token_id, prob) in enumerate(zip(pred['top_tokens'][:10], pred['top_probs'][:10])):
                    token_text = tokenizer.decode([token_id])
                    print(f"    {i+1}. '{token_text}' (p={prob:.4f})")
            
            # Store results
            results.append({
                'example': example,
                'uncertainty': {
                    'total': unc_result['total'],
                    'epistemic': unc_result['epistemic'],
                    'aleatoric': unc_result['aleatoric']
                },
                'predictions': unc_result['predictions']
            })
        except Exception as e:
            print(f"\n✗ Error during uncertainty computation: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'example': example,
                'error': str(e)
            })
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for idx, result in enumerate(results, 1):
        print(f"\nExample {idx}: {result['example']['description']}")
        print(f"  Input:  {result['example']['text']}")
        
        if 'error' in result:
            print(f"  ✗ Error: {result['error']}")
        else:
            if result.get('uncertainty'):
                unc = result['uncertainty']
                print(f"  Uncertainty:")
                print(f"    Total:     {unc['total']:.4f}")
                print(f"    Epistemic: {unc['epistemic']:.4f}")
                print(f"    Aleatoric: {unc['aleatoric']:.4f}")
                print(f"    Ratio (E/A): {unc['epistemic']/max(unc['aleatoric'], 1e-6):.2f}")
                
                # Show top prediction
                if result.get('predictions'):
                    for pred in result['predictions']:
                        print(f"  Top prediction for position {pred['position']}: {result['example']['candidates'][0] if pred['position'] < len(result['example']['candidates']) else 'N/A'}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mc_samples", type=int, default=10,
                        help="Number of MC dropout samples for uncertainty estimation")
    
    args = parser.parse_args()
    
    results = test_examples(
        model_path=args.model_path,
        device=args.device,
        mc_samples=args.mc_samples
    )
