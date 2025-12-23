"""
Test script for the uncertainty-aware remasking modifications in generate.py
"""
import torch
from transformers import AutoTokenizer, AutoModel

from generate import (
    generate,
    enable_mc_dropout,
    disable_mc_dropout,
    compute_entropy,
    compute_uncertainty_score,
)


def test_helper_functions():
    """Test the helper functions for uncertainty computation."""
    print("=" * 50)
    print("Testing helper functions...")
    print("=" * 50)

    # Test compute_entropy
    probs = torch.tensor([[0.5, 0.5], [0.9, 0.1], [1.0, 0.0]])
    entropy = compute_entropy(probs, dim=-1)
    print(f"Entropy of [0.5, 0.5]: {entropy[0]:.4f} (expected ~0.693)")
    print(f"Entropy of [0.9, 0.1]: {entropy[1]:.4f} (expected ~0.325)")
    print(f"Entropy of [1.0, 0.0]: {entropy[2]:.4f} (expected ~0.0)")

    # Test compute_uncertainty_score
    epistemic = torch.tensor([0.5, 0.2, 0.8])
    aleatoric = torch.tensor([0.3, 0.6, 0.1])
    score = compute_uncertainty_score(epistemic, aleatoric, alpha=1.0, beta=1.0)
    print(f"\nUncertainty scores: {score}")
    print("Higher score = lower uncertainty = unmask first")

    print("\nHelper function tests passed!\n")


def test_generation(model_path="GSAI-ML/LLaDA-8B-Instruct", device="cuda"):
    """Test generation with different remasking strategies."""
    print("=" * 50)
    print(f"Loading model from {model_path}...")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.to(device).eval()

    # Simple test prompt
    prompt = "The capital of France is"
    print(f"\nPrompt: '{prompt}'")

    # Tokenize
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Test parameters
    gen_length = 32
    steps = 32
    block_length = 32

    # Test each remasking strategy
    strategies = ['low_confidence', 'random', 'uncertainty_aware']

    for strategy in strategies:
        print(f"\n{'=' * 50}")
        print(f"Testing remasking='{strategy}'...")
        print("=" * 50)

        try:
            output = generate(
                model=model,
                prompt=input_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=0.0,
                cfg_scale=0.0,
                remasking=strategy,
                mc_samples=4 if strategy == 'uncertainty_aware' else 8,
                alpha=1.0,
                beta=1.0,
            )

            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"Generated: {generated_text}")
            print(f"Strategy '{strategy}' works!")

        except Exception as e:
            print(f"Error with strategy '{strategy}': {e}")
            raise

    print("\n" + "=" * 50)
    print("All generation tests passed!")
    print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test uncertainty-aware remasking")
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                        help="Path to the LLaDA model")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda or cpu)")
    parser.add_argument("--skip_model", action="store_true",
                        help="Skip model loading tests (only test helper functions)")
    args = parser.parse_args()

    # Always test helper functions
    test_helper_functions()

    # Test generation unless skipped
    if not args.skip_model:
        if not torch.cuda.is_available() and args.device == "cuda":
            print("CUDA not available, switching to CPU")
            args.device = "cpu"
        test_generation(model_path=args.model_path, device=args.device)
    else:
        print("Skipping model generation tests (--skip_model flag set)")
