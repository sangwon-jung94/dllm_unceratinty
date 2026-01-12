"""
Winogrande result evaluation script.

Parses model outputs formatted with [Sample N] sections and compares
predictions against the Winogrande validation set (defaults to the XL config).
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Dict

from datasets import load_dataset


def extract_predicted_choice(answer_text: str, options: Optional[Dict[str, str]] = None, question_text: str = "") -> Optional[str]:
    """Extract the model's chosen option ("1" or "2") from an answer block."""
    if not answer_text:
        return None

    # Try explicit answer patterns first (highest priority)
    explicit_patterns = [
        r"(?:the\s+)?correct\s+answer\s+is\s*[:\-]?\s*\(?\s*([12])\s*\)?",
        r"answer\s+is\s*[:\-]?\s*\(?\s*([12])\s*\)?",
        r"the\s+(?:correct\s+)?(?:choice|option|answer)\s+is\s*[:\-]?\s*\(?\s*([12])\s*\)?",
        r"(?:correct|right)\s+(?:choice|option)\s*[:\-]?\s*\(?\s*([12])\s*\)?",
    ]

    text = answer_text.lower()
    for pattern in explicit_patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            return matches[-1]

    # Parse question sentence with blank filled in (contextual matching)
    # Example: "Joe went to _ because..." -> look for "Joe went to bakery because..."
    if options and question_text and "_" in question_text:
        opt1 = options.get("1", "").strip()
        opt2 = options.get("2", "").strip()
        
        parts = question_text.split("_")
        if len(parts) == 2:
            before = parts[0].strip().split()[-5:] if parts[0].strip() else []
            after = parts[1].strip().split()[:5] if parts[1].strip() else []
            
            before_pattern = r'\s+'.join(re.escape(w.lower()) for w in before[-3:]) if before else ""
            after_pattern = r'\s+'.join(re.escape(w.lower()) for w in after[:3]) if after else ""
            
            answer_lower = answer_text.lower()
            count1 = 0
            count2 = 0
            
            if before_pattern and after_pattern:
                pattern1 = f"{before_pattern}\\s+{re.escape(opt1.lower())}\\s+{after_pattern}"
                pattern2 = f"{before_pattern}\\s+{re.escape(opt2.lower())}\\s+{after_pattern}"
                count1 = len(re.findall(pattern1, answer_lower))
                count2 = len(re.findall(pattern2, answer_lower))
            elif before_pattern:
                pattern1 = f"{before_pattern}\\s+{re.escape(opt1.lower())}"
                pattern2 = f"{before_pattern}\\s+{re.escape(opt2.lower())}"
                count1 = len(re.findall(pattern1, answer_lower))
                count2 = len(re.findall(pattern2, answer_lower))
            elif after_pattern:
                pattern1 = f"{re.escape(opt1.lower())}\\s+{after_pattern}"
                pattern2 = f"{re.escape(opt2.lower())}\\s+{after_pattern}"
                count1 = len(re.findall(pattern1, answer_lower))
                count2 = len(re.findall(pattern2, answer_lower))
            
            if count1 > count2:
                return "1"
            elif count2 > count1:
                return "2"

    # Very strict fallback: only match explicit parenthetical or numbered answer formats
    # e.g., "(1)", "(2)", "1)", "2)" at end of sentences or after keywords
    strict_patterns = [
        r"(?:answer|choice|option|response)\s*[:\-]?\s*\(([12])\)",
        r"\(([12])\)\s*(?:is\s+)?(?:correct|right)",
        r"^([12])\)",  # Line starts with "1)" or "2)"
    ]
    
    for pattern in strict_patterns:
        matches = re.findall(pattern, answer_text, flags=re.MULTILINE | re.IGNORECASE)
        if matches:
            return matches[-1]

    return None


def parse_output_file(output_file: Path, max_samples: Optional[int] = None) -> List[dict]:
    """Parse output.txt into a list of samples with optional question/answer labels."""
    content = output_file.read_text(encoding="utf-8")

    samples = []
    # Split on [Sample N] boundaries; handle both labeled (Question/Answer) and unlabeled blocks.
    block_pattern = r"\[Sample (\d+)\]\s*(.*?)(?=\n\s*\[Sample \d+\]|$)"
    matches = re.findall(block_pattern, content, flags=re.DOTALL)

    for sample_num, block in matches:
        block_text = block.strip()

        # Prefer explicit Question/Answer labels when present.
        qa_match = re.search(r"Question:\s*(.*?)\s*Answer:\s*(.*)", block_text, flags=re.DOTALL)
        if qa_match:
            question_text = qa_match.group(1).strip()
            answer_text = qa_match.group(2).strip()
        else:
            question_match = re.search(r"Question:\s*(.*)", block_text, flags=re.DOTALL)
            question_text = question_match.group(1).strip() if question_match else ""
            answer_text = block_text[question_match.end():].strip() if question_match else block_text

        cleaned_answer = re.sub(r"-{10,}", "", answer_text).strip()

        options: Dict[str, str] = {}
        opt_matches = re.findall(r"^\s*([12])\)\s*(.+?)\s*$", question_text, flags=re.MULTILINE)
        for idx, text in opt_matches:
            options[idx] = text.strip()

        samples.append(
            {
                "sample_num": int(sample_num),
                "question": question_text,
                "answer": cleaned_answer,
                "options": options,
            }
        )

        if max_samples is not None and len(samples) >= max_samples:
            break

    return samples


def evaluate_winogrande(
    output_file: Path,
    dataset_config: str = "winogrande_xl",
    split: str = "validation",
    verbose: bool = False,
    max_samples: Optional[int] = None,
) -> dict:
    """Evaluate Winogrande predictions stored in an output.txt file."""
    print(f"Parsing output file: {output_file}")
    samples = parse_output_file(output_file, max_samples=max_samples)
    print(
        f"Found {len(samples)} samples"
        + (f" (limited to {max_samples})" if max_samples else "")
    )

    print(f"Loading Winogrande dataset ({dataset_config}, {split})...")
    dataset = load_dataset("winogrande", dataset_config, split=split)

    correct = 0
    total = len(samples)
    results = []
    printed = 0

    for sample in samples:
        sample_idx = sample["sample_num"] - 1
        if sample_idx >= len(dataset):
            print(f"Warning: Sample {sample['sample_num']} exceeds dataset size")
            continue

        ground_truth_choice = str(dataset[sample_idx]["answer"]).strip()
        
        # Get question and options from dataset if not in parsed sample
        question_text = sample.get("question", "") or dataset[sample_idx].get("sentence", "")
        options_dict = sample.get("options", {})
        if not options_dict and dataset[sample_idx]:
            # Extract options from dataset
            opt1 = dataset[sample_idx].get("option1", "")
            opt2 = dataset[sample_idx].get("option2", "")
            if opt1 and opt2:
                options_dict = {"1": opt1, "2": opt2}
        
        predicted_choice = extract_predicted_choice(
            sample["answer"], 
            options=options_dict,
            question_text=question_text
        )

        is_correct = predicted_choice is not None and predicted_choice == ground_truth_choice
        if is_correct:
            correct += 1

        result = {
            "sample_num": sample["sample_num"],
            "question": question_text[:120] + ("..." if len(question_text) > 120 else ""),
            "ground_truth": ground_truth_choice,
            "predicted": predicted_choice,
            "correct": is_correct,
        }
        results.append(result)

        if verbose or (not is_correct and printed < 10):
            status = "\u2713" if is_correct else "\u2717"
            print(f"\n{status} Sample {sample['sample_num']}:")
            print(f"  Question: {result['question']}")
            print(f"  Ground Truth: {ground_truth_choice}")
            print(f"  Predicted: {predicted_choice}")
            printed += 1

    accuracy = (correct / total * 100) if total > 0 else 0

    metrics = {
        "total_samples": total,
        "correct": correct,
        "incorrect": total - correct,
        "accuracy": accuracy,
        "results": results,
    }

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Winogrande results from output.txt")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output.txt file")
    parser.add_argument("--dataset_config", type=str, default="winogrande_xl", help="Winogrande config (default: winogrande_xl)")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split to use (default: validation)")
    parser.add_argument("--verbose", action="store_true", help="Print detailed per-sample results")
    parser.add_argument("--save_results", type=str, default=None, help="Path to save detailed results (JSON format)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to evaluate")

    args = parser.parse_args()

    output_path = Path(args.output_file)
    if not output_path.exists():
        print(f"Error: File not found: {output_path}")
        return

    metrics = evaluate_winogrande(
        output_file=output_path,
        dataset_config=args.dataset_config,
        split=args.split,
        verbose=args.verbose,
        max_samples=args.max_samples,
    )

    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Total Samples: {metrics['total_samples']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Incorrect: {metrics['incorrect']}")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print("=" * 80)

    if args.save_results:
        save_path = Path(args.save_results)
        with save_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {save_path}")


if __name__ == "__main__":
    main()
