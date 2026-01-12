"""Quick test for simplified blank tokenization."""

import torch
from transformers import AutoTokenizer

# Simulate the simplified logic
def test_simplified(sentence):
    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True,
    )
    mask_id = 126336
    
    print(f"\nOriginal: {sentence}")
    
    # Split at underscore
    parts = sentence.split('_', 1)
    prefix, suffix = parts
    
    print(f"Prefix: '{prefix}'")
    print(f"Suffix: '{suffix}'")
    
    # Tokenize
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False) if suffix else []
    
    # Construct
    input_ids_list = prefix_ids + [mask_id] + suffix_ids
    blank_position = len(prefix_ids)
    
    print(f"\nTokens: {input_ids_list}")
    print(f"Blank position: {blank_position}")
    
    # Decode to verify
    decoded_before = tokenizer.decode(prefix_ids)
    decoded_after = tokenizer.decode(suffix_ids)
    print(f"\nReconstruction: '{decoded_before}' <MASK> '{decoded_after}'")
    print("✅ Correct!" if decoded_before.strip() + decoded_after.strip() == sentence.replace('_', '').strip() else "❌ Mismatch")

# Test cases
test_simplified("I _ you")
test_simplified("The _ is small.")
test_simplified("The trophy doesn't fit because the _ is too large.")
