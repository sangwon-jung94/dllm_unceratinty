"""
Debug script for uncertainty_aware remasking issue
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from generate import generate, enable_mc_dropout, disable_mc_dropout


def check_dropout_layers(model):
    """Check all dropout layers in the model."""
    print("=" * 60)
    print("1. Checking Dropout Layers in Model")
    print("=" * 60)

    dropout_info = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Dropout):
            dropout_info.append((name, m.p))
            print(f"  {name}: p={m.p}")

    print(f"\nTotal nn.Dropout layers: {len(dropout_info)}")

    if len(dropout_info) == 0:
        print("WARNING: No nn.Dropout layers found!")
        print("MC Dropout will NOT work - model has no dropout layers.")
    elif all(p == 0 for _, p in dropout_info):
        print("WARNING: All dropout layers have p=0!")
        print("MC Dropout needs non-zero dropout probability.")

    return dropout_info


def test_mc_dropout_effect(model, tokenizer, device):
    """Test how MC Dropout affects model output."""
    print("\n" + "=" * 60)
    print("2. Testing MC Dropout Effect on Logits")
    print("=" * 60)

    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Append mask tokens after prompt to simulate generation scenario
    mask_id = 126336
    num_masks = 16
    x = torch.cat([input_ids, torch.full((1, num_masks), mask_id, device=device)], dim=1)
    prompt_len = input_ids.shape[1]

    print(f"Prompt length: {prompt_len}, Appended masks: {num_masks}, Total length: {x.shape[1]}")

    # Get logits without dropout
    model.eval()
    with torch.no_grad():
        logits_eval = model(x).logits

    # Get logits with MC dropout (p=0.1)
    dropout_count = enable_mc_dropout(model, p=0.1)
    print(f"Enabled dropout on {dropout_count} layers with p=0.1")

    with torch.no_grad():
        logits_mc1 = model(x).logits
        logits_mc2 = model(x).logits

    disable_mc_dropout(model)

    # Compare logits at masked positions only
    mask_positions = (x == mask_id)[0]

    diff_eval_mc1 = (logits_eval[0, mask_positions] - logits_mc1[0, mask_positions]).abs().mean().item()
    diff_mc1_mc2 = (logits_mc1[0, mask_positions] - logits_mc2[0, mask_positions]).abs().mean().item()

    print(f"\nLogits difference at masked positions (eval vs MC1): {diff_eval_mc1:.6f}")
    print(f"Logits difference at masked positions (MC1 vs MC2): {diff_mc1_mc2:.6f}")

    if diff_eval_mc1 > 1.0:
        print("WARNING: Large difference! dropout_p=0.1 may be too high.")

    # Check predicted tokens at first masked position
    first_mask_idx = prompt_len
    pred_eval = logits_eval[0, first_mask_idx].argmax().item()
    pred_mc1 = logits_mc1[0, first_mask_idx].argmax().item()
    pred_mc2 = logits_mc2[0, first_mask_idx].argmax().item()

    print(f"\nPredicted token at first mask (eval): {tokenizer.decode([pred_eval])} (id={pred_eval})")
    print(f"Predicted token at first mask (MC1):  {tokenizer.decode([pred_mc1])} (id={pred_mc1})")
    print(f"Predicted token at first mask (MC2):  {tokenizer.decode([pred_mc2])} (id={pred_mc2})")


def test_generation_comparison(model, tokenizer, device):
    """Compare generation output across strategies and dropout_p values."""
    print("\n" + "=" * 60)
    print("3. Comparing Generation Outputs")
    print("=" * 60)

    prompt = "Question: What is 2 + 2?\nAnswer:"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    gen_length = 4
    steps = 4
    block_length = 4

    configs = [
        ("low_confidence", None),
        ("uncertainty_aware", None),  # Use original dropout p
        ("uncertainty_aware", 0.01),  # Very small dropout
        ("uncertainty_aware", 0.05),  # Small dropout
        ("uncertainty_aware", 0.1),   # Original setting (likely problem)
    ]

    print(f"Prompt: {prompt}\n")

    for remasking, dropout_p in configs:
        label = f"{remasking} (dropout_p={dropout_p})"
        print(f"\n--- {label} ---")

        try:
            kwargs = {
                "model": model,
                "prompt": input_ids,
                "steps": steps,
                "gen_length": gen_length,
                "block_length": block_length,
                "temperature": 0.0,
                "cfg_scale": 0.0,
                "remasking": remasking,
                "mc_samples": 4,
                "alpha": 1.0,
                "beta": 1.0,
            }
            if dropout_p is not None:
                kwargs["dropout_p"] = dropout_p

            output = generate(**kwargs)
            generated = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
            print(f"Output: {generated[:200]}")

        except Exception as e:
            print(f"Error: {e}")


def test_uncertainty_values(model, tokenizer, device):
    """Check actual uncertainty values being computed."""
    print("\n" + "=" * 60)
    print("4. Checking Uncertainty Values")
    print("=" * 60)

    from generate import compute_uncertainty_decomposition

    prompt = "The capital of France is"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Create masked input (simulate generation)
    mask_id = 126336
    x = torch.cat([input_ids, torch.full((1, 16), mask_id, device=device)], dim=1)
    prompt_index = (x != mask_id)

    for dropout_p in [None, 0.01, 0.1]:
        print(f"\n--- dropout_p={dropout_p} ---")

        mean_logits, H_epi, H_ale, p_bar = compute_uncertainty_decomposition(
            model=model,
            x=x,
            attention_mask=None,
            mc_samples=4,
            cfg_scale=0.0,
            prompt_index=prompt_index,
            mask_id=mask_id,
            dropout_p=dropout_p
        )

        # Only look at masked positions
        mask_positions = (x == mask_id)[0]

        print(f"Epistemic uncertainty (masked pos): min={H_epi[0, mask_positions].min():.4f}, "
              f"max={H_epi[0, mask_positions].max():.4f}, mean={H_epi[0, mask_positions].mean():.4f}")
        print(f"Aleatoric uncertainty (masked pos): min={H_ale[0, mask_positions].min():.4f}, "
              f"max={H_ale[0, mask_positions].max():.4f}, mean={H_ale[0, mask_positions].mean():.4f}")

        # Check if epistemic is always 0 (no variation across MC samples)
        if H_epi[0, mask_positions].max() < 0.001:
            print("WARNING: Epistemic uncertainty is ~0! MC Dropout is not creating variation.")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "GSAI-ML/LLaDA-8B-Base"

    print(f"Loading model: {model_path}")
    print(f"Device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()

    # Run all checks
    check_dropout_layers(model)
    test_mc_dropout_effect(model, tokenizer, device)
    test_uncertainty_values(model, tokenizer, device)
    test_generation_comparison(model, tokenizer, device)

    print("\n" + "=" * 60)
    print("Debug complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
