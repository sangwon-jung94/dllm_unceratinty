"""
Test script to verify blank tokenization logic.
Tests both with and without chat template.
"""

import torch
from transformers import AutoTokenizer


def test_blank_tokenization(tokenizer, sentence, option1, option2, use_chat_template=True):
    """Test the blank tokenization approach."""
    mask_id = 126336
    
    print("\n" + "="*80)
    print(f"TEST: use_chat_template={use_chat_template}")
    print("="*80)
    print(f"Original sentence: {sentence}")
    print(f"Option1: {option1}, Option2: {option2}")
    
    # Check if sentence contains underscore
    if '_' not in sentence:
        print("ERROR: Sentence does not contain '_'")
        return
    
    # Split sentence at the underscore
    parts = sentence.split('_', 1)
    if len(parts) != 2:
        print(f"ERROR: Expected exactly one '_' in sentence")
        return
    
    prefix, suffix = parts
    print(f"\nPrefix: '{prefix}'")
    print(f"Suffix: '{suffix}'")
    
    # Format with chat template if needed
    if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
        # Create prompt with placeholder
        prompt_template = f"Complete the sentence by filling the blank with one of the options:\n\nSentence: {{sentence}}\n\nOptions:\n1) {option1}\n2) {option2}"
        
        # First, apply chat template with placeholder
        messages = [{"role": "user", "content": prompt_template.format(sentence="PREFIX_PLACEHOLDER_SUFFIX")}]
        formatted_template = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        
        print(f"\n--- Chat Template Applied ---")
        print(f"Template with placeholder:\n{formatted_template[:200]}...")
        
        # Now replace placeholder with actual prefix and suffix
        formatted_with_blank = formatted_template.replace("PREFIX_PLACEHOLDER_SUFFIX", prefix + "_" + suffix)
        
        print(f"\nFormatted with blank:\n{formatted_with_blank[:300]}...")
        
        # Split the formatted text at underscore
        formatted_parts = formatted_with_blank.split('_', 1)
        if len(formatted_parts) == 2:
            prefix, suffix = formatted_parts
            print(f"\n--- After template, split at '_' ---")
            print(f"New prefix: '{prefix[:100]}...'")
            print(f"New suffix: '{suffix[:100]}...'")
    
    # Tokenize prefix and suffix separately
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False) if suffix else []
    
    print(f"\n--- Tokenization ---")
    print(f"Prefix tokens ({len(prefix_ids)}): {prefix_ids[:10]}...")
    print(f"Suffix tokens ({len(suffix_ids)}): {suffix_ids[:10]}...")
    
    # Construct input_ids: prefix + mask + suffix
    input_ids_list = prefix_ids + [mask_id] + suffix_ids
    blank_position = len(prefix_ids)
    
    print(f"\n--- Final Construction ---")
    print(f"Total length: {len(input_ids_list)}")
    print(f"Blank (mask) position: {blank_position}")
    print(f"Context around mask:")
    print(f"  Tokens before mask: {input_ids_list[max(0, blank_position-3):blank_position]}")
    print(f"  Mask token: {input_ids_list[blank_position]}")
    print(f"  Tokens after mask: {input_ids_list[blank_position+1:min(len(input_ids_list), blank_position+4)]}")
    
    # Decode to verify
    print(f"\n--- Verification (decode) ---")
    if blank_position > 0:
        before_text = tokenizer.decode(input_ids_list[max(0, blank_position-5):blank_position])
        print(f"Text before mask: '...{before_text}'")
    
    # Note: mask_id might not decode properly, that's expected
    try:
        mask_text = tokenizer.decode([mask_id])
        print(f"Mask token decodes to: '{mask_text}'")
    except:
        print(f"Mask token ({mask_id}) cannot be decoded (expected)")
    
    if blank_position < len(input_ids_list) - 1:
        after_text = tokenizer.decode(input_ids_list[blank_position+1:min(len(input_ids_list), blank_position+6)])
        print(f"Text after mask: '{after_text}...'")
    
    # Show what option tokens look like
    print(f"\n--- Option Tokenization ---")
    opt1_ids = tokenizer.encode(option1, add_special_tokens=False)
    opt2_ids = tokenizer.encode(option2, add_special_tokens=False)
    opt1_space_ids = tokenizer.encode(' ' + option1, add_special_tokens=False)
    opt2_space_ids = tokenizer.encode(' ' + option2, add_special_tokens=False)
    
    print(f"Option1 '{option1}': {opt1_ids}")
    print(f"Option1 with space ' {option1}': {opt1_space_ids}")
    print(f"Option2 '{option2}': {opt2_ids}")
    print(f"Option2 with space ' {option2}': {opt2_space_ids}")
    
    return {
        'input_ids': input_ids_list,
        'blank_position': blank_position,
        'prefix_ids': prefix_ids,
        'suffix_ids': suffix_ids,
        'option1_ids': opt1_ids,
        'option2_ids': opt2_ids,
    }


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        'GSAI-ML/LLaDA-8B-Instruct',
        trust_remote_code=True,
    )
    
    # Test case from Winogrande
    test_cases = [
        {
            'sentence': 'John moved the couch from the garage to the backyard to create space. The _ is small.',
            'option1': 'garage',
            'option2': 'backyard',
        },
        {
            'sentence': 'I _ you',
            'option1': 'love',
            'option2': 'hate',
        },
        {
            'sentence': 'The trophy doesn\'t fit into the brown suitcase because the _ is too large.',
            'option1': 'trophy',
            'option2': 'suitcase',
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n\n{'#'*80}")
        print(f"TEST CASE {i}")
        print(f"{'#'*80}")
        
        # Test WITH chat template
        result_with = test_blank_tokenization(
            tokenizer,
            test_case['sentence'],
            test_case['option1'],
            test_case['option2'],
            use_chat_template=True
        )
        
        # Test WITHOUT chat template
        result_without = test_blank_tokenization(
            tokenizer,
            test_case['sentence'],
            test_case['option1'],
            test_case['option2'],
            use_chat_template=False
        )
        
        # Compare
        if result_with and result_without:
            print(f"\n{'='*80}")
            print("COMPARISON")
            print(f"{'='*80}")
            print(f"With chat template:")
            print(f"  Total length: {len(result_with['input_ids'])}")
            print(f"  Blank position: {result_with['blank_position']}")
            print(f"\nWithout chat template:")
            print(f"  Total length: {len(result_without['input_ids'])}")
            print(f"  Blank position: {result_without['blank_position']}")


if __name__ == '__main__':
    main()
