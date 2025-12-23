"""
Test script to visualize uncertainty-aware remasking behavior.

This script verifies that tokens with LOW uncertainty are decoded FIRST,
as intended by the uncertainty-aware remasking strategy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel
from generate import (
    add_gumbel_noise,
    get_num_transfer_tokens,
    enable_mc_dropout,
    disable_mc_dropout,
    compute_entropy,
    compute_uncertainty_decomposition,
    compute_uncertainty_score,
)


class UncertaintyVisualizer:
    """Captures and visualizes uncertainty-aware remasking behavior."""

    def __init__(self):
        self.step_data = []

    def log_step(self, step, mask_positions, scores, selected_indices,
                 epistemic, aleatoric, tokens_before, tokens_after, tokenizer=None):
        """Log data for a single step."""
        self.step_data.append({
            'step': step,
            'mask_positions': mask_positions.cpu().numpy() if torch.is_tensor(mask_positions) else mask_positions,
            'scores': scores.cpu().numpy() if torch.is_tensor(scores) else scores,
            'selected_indices': selected_indices.cpu().numpy() if torch.is_tensor(selected_indices) else selected_indices,
            'epistemic': epistemic.cpu().numpy() if torch.is_tensor(epistemic) else epistemic,
            'aleatoric': aleatoric.cpu().numpy() if torch.is_tensor(aleatoric) else aleatoric,
            'tokens_before': tokens_before.cpu().numpy() if torch.is_tensor(tokens_before) else tokens_before,
            'tokens_after': tokens_after.cpu().numpy() if torch.is_tensor(tokens_after) else tokens_after,
            'tokenizer': tokenizer,
        })

    def print_summary(self):
        """Print a summary of the uncertainty-aware remasking process."""
        print("\n" + "="*80)
        print("UNCERTAINTY-AWARE REMASKING VISUALIZATION")
        print("="*80)

        for data in self.step_data:
            step = data['step']
            scores = data['scores']
            selected = data['selected_indices']
            epistemic = data['epistemic']
            aleatoric = data['aleatoric']
            tokenizer = data['tokenizer']
            tokens_after = data['tokens_after']

            print(f"\n--- Step {step} ---")
            print(f"Number of tokens unmasked this step: {len(selected)}")

            if len(selected) > 0:
                # Show selected tokens with their uncertainties
                print("\nTokens unmasked (sorted by selection order):")
                print(f"{'Pos':>5} | {'Score':>10} | {'Epistemic':>10} | {'Aleatoric':>10} | {'Token':>15}")
                print("-" * 60)

                for idx in selected[:10]:  # Show first 10
                    score = scores[idx] if idx < len(scores) else float('nan')
                    epi = epistemic[idx] if idx < len(epistemic) else float('nan')
                    ale = aleatoric[idx] if idx < len(aleatoric) else float('nan')
                    token_id = tokens_after[idx]
                    token_str = tokenizer.decode([token_id]) if tokenizer else str(token_id)
                    token_str = repr(token_str)[:15]
                    print(f"{idx:>5} | {score:>10.4f} | {epi:>10.4f} | {ale:>10.4f} | {token_str:>15}")

                if len(selected) > 10:
                    print(f"  ... and {len(selected) - 10} more tokens")

                # Verify ordering
                selected_scores = [scores[idx] for idx in selected if idx < len(scores)]
                if len(selected_scores) > 1:
                    is_sorted = all(selected_scores[i] >= selected_scores[i+1]
                                   for i in range(len(selected_scores)-1))
                    print(f"\n✓ Tokens selected in descending score order: {is_sorted}")
                    print(f"  (Higher score = lower uncertainty = unmasked first)")

    def verify_correctness(self):
        """Verify that low uncertainty tokens are indeed unmasked first."""
        print("\n" + "="*80)
        print("VERIFICATION: Low Uncertainty Tokens Unmasked First")
        print("="*80)

        all_correct = True
        for data in self.step_data:
            step = data['step']
            scores = data['scores']
            selected = data['selected_indices']

            if len(selected) == 0:
                continue

            # Get scores of selected vs non-selected masked positions
            mask_positions = np.where(scores > -np.inf)[0]
            if len(mask_positions) == 0:
                continue

            selected_set = set(selected)
            selected_scores = [scores[i] for i in selected if i < len(scores)]
            non_selected_scores = [scores[i] for i in mask_positions
                                   if i not in selected_set and scores[i] > -np.inf]

            if len(selected_scores) > 0 and len(non_selected_scores) > 0:
                min_selected = min(selected_scores)
                max_non_selected = max(non_selected_scores)

                correct = min_selected >= max_non_selected
                all_correct = all_correct and correct

                status = "✓ PASS" if correct else "✗ FAIL"
                print(f"Step {step}: {status}")
                print(f"  Min selected score: {min_selected:.4f}")
                print(f"  Max non-selected score: {max_non_selected:.4f}")

        print("\n" + "="*80)
        if all_correct:
            print("✓ ALL STEPS PASSED: Low uncertainty tokens are correctly unmasked first!")
        else:
            print("✗ SOME STEPS FAILED: Check the implementation!")
        print("="*80)

        return all_correct


@torch.no_grad()
def generate_with_visualization(model, prompt, attention_mask=None, steps=128, gen_length=128,
                                 block_length=128, temperature=0., cfg_scale=0., mask_id=126336,
                                 mc_samples=8, alpha=1.0, beta=1.0, dropout_p=None, tokenizer=None):
    """
    Generate with uncertainty-aware remasking and capture visualization data.
    """
    visualizer = UncertaintyVisualizer()

    # Check for dropout layers
    dropout_count = sum(1 for m in model.modules() if isinstance(m, nn.Dropout))
    if dropout_count == 0:
        print(f"WARNING: No dropout layers found. MC Dropout will not provide meaningful uncertainty estimates.")
        print(f"         For testing, we'll proceed but results may not show variance.")
    else:
        print(f"Found {dropout_count} dropout layers for MC Dropout.")

    x = torch.full((prompt.shape[0], prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        attention_mask = torch.cat([
            attention_mask,
            torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=model.device)
        ], dim=-1)

    prompt_index = (x != mask_id)
    prompt_length = prompt.shape[1]

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    global_step = 0

    for num_block in range(num_blocks):
        block_start = prompt_length + num_block * block_length
        block_end = prompt_length + (num_block + 1) * block_length

        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = (x == mask_id)
            tokens_before = x.clone()

            # Compute uncertainty decomposition via MC Dropout
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

            logits = mean_logits
            logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
            x0 = torch.argmax(logits_with_noise, dim=-1)

            # Compute uncertainty-based unmasking score
            x0_p = compute_uncertainty_score(H_epistemic, H_aleatoric, alpha, beta)

            # Mask out positions beyond current block
            x0_p[:, block_end:] = -np.inf

            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p, -np.inf)

            transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

            for j in range(confidence.shape[0]):
                k = num_transfer_tokens[j, i].item()
                if k > 0:
                    _, select_index = torch.topk(confidence[j], k=k)
                    transfer_index[j, select_index] = True

                    # Log visualization data (only for first batch item)
                    if j == 0:
                        # Get mask positions for this batch
                        mask_pos = torch.where(mask_index[j])[0]
                        visualizer.log_step(
                            step=global_step,
                            mask_positions=mask_pos,
                            scores=confidence[j],
                            selected_indices=select_index,
                            epistemic=H_epistemic[j],
                            aleatoric=H_aleatoric[j],
                            tokens_before=tokens_before[j],
                            tokens_after=x0[j],
                            tokenizer=tokenizer,
                        )

            x[transfer_index] = x0[transfer_index]
            global_step += 1

    return x, visualizer


def run_synthetic_test():
    """
    Run a synthetic test to verify the score computation logic.
    No model needed - just tests the math.
    """
    print("\n" + "="*80)
    print("SYNTHETIC TEST: Verify Score Computation")
    print("="*80)

    # Create synthetic uncertainty values
    epistemic = torch.tensor([0.1, 0.5, 0.3, 0.8, 0.2])
    aleatoric = torch.tensor([0.2, 0.4, 0.6, 0.1, 0.9])

    alpha, beta = 1.0, 1.0
    scores = compute_uncertainty_score(epistemic, aleatoric, alpha, beta)

    print("\nInput uncertainties and computed scores:")
    print(f"{'Position':>8} | {'Epistemic':>10} | {'Aleatoric':>10} | {'Total U':>10} | {'Score':>10}")
    print("-" * 55)

    for i in range(len(epistemic)):
        total_u = epistemic[i] + aleatoric[i]
        print(f"{i:>8} | {epistemic[i]:>10.4f} | {aleatoric[i]:>10.4f} | {total_u:>10.4f} | {scores[i]:>10.4f}")

    # Verify that higher scores correspond to lower uncertainty
    sorted_indices = torch.argsort(scores, descending=True)
    total_uncertainty = epistemic + aleatoric

    print(f"\nSelection order (by descending score): {sorted_indices.tolist()}")
    print(f"Total uncertainty at each position: {total_uncertainty.tolist()}")

    # Check if selection order matches uncertainty order
    selected_uncertainties = total_uncertainty[sorted_indices]
    is_correct = all(selected_uncertainties[i] <= selected_uncertainties[i+1]
                     for i in range(len(selected_uncertainties)-1))

    print(f"\n✓ Tokens selected in ascending uncertainty order: {is_correct}")

    if is_correct:
        print("✓ SYNTHETIC TEST PASSED: Score computation is correct!")
    else:
        print("✗ SYNTHETIC TEST FAILED: Check score computation!")

    return is_correct


def run_model_test(model_path='GSAI-ML/LLaDA-8B-Instruct', device='cuda',
                   gen_length=32, steps=32, block_length=32,
                   mc_samples=4, alpha=1.0, beta=1.0):
    """
    Run a test with the actual model.
    """
    print("\n" + "="*80)
    print("MODEL TEST: Uncertainty-Aware Remasking with LLaDA")
    print("="*80)

    print(f"\nLoading model: {model_path}")
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'

    # Simple test prompt
    prompt = "What is 2 + 2?"
    messages = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    encoded = tokenizer(prompt_text, add_special_tokens=False, return_tensors="pt")
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    print(f"\nPrompt: {prompt}")
    print(f"Parameters: gen_length={gen_length}, steps={steps}, mc_samples={mc_samples}")
    print(f"           alpha={alpha}, beta={beta}")

    print("\nGenerating with uncertainty-aware remasking...")
    output, visualizer = generate_with_visualization(
        model=model,
        prompt=input_ids,
        attention_mask=attention_mask,
        steps=steps,
        gen_length=gen_length,
        block_length=block_length,
        temperature=0.,
        cfg_scale=0.,
        mc_samples=mc_samples,
        alpha=alpha,
        beta=beta,
        tokenizer=tokenizer,
    )

    # Decode and print output
    generated_text = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nGenerated text: {generated_text}")

    # Print visualization
    visualizer.print_summary()

    # Verify correctness
    return visualizer.verify_correctness()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test uncertainty-aware remasking visualization')
    parser.add_argument('--synthetic-only', action='store_true',
                        help='Run only synthetic test (no model loading)')
    parser.add_argument('--model', type=str, default='GSAI-ML/LLaDA-8B-Instruct',
                        help='Model path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--gen-length', type=int, default=32,
                        help='Generation length')
    parser.add_argument('--steps', type=int, default=32,
                        help='Number of diffusion steps')
    parser.add_argument('--mc-samples', type=int, default=4,
                        help='Number of MC Dropout samples')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Epistemic uncertainty weight')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='Aleatoric uncertainty weight')
    args = parser.parse_args()

    # Always run synthetic test first
    synthetic_passed = run_synthetic_test()

    if not args.synthetic_only:
        # Run model test
        model_passed = run_model_test(
            model_path=args.model,
            device=args.device,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.gen_length,  # Use same as gen_length for simplicity
            mc_samples=args.mc_samples,
            alpha=args.alpha,
            beta=args.beta,
        )

        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"Synthetic test: {'✓ PASSED' if synthetic_passed else '✗ FAILED'}")
        print(f"Model test:     {'✓ PASSED' if model_passed else '✗ FAILED'}")
    else:
        print("\n(Skipping model test - use without --synthetic-only to run full test)")


if __name__ == '__main__':
    main()
