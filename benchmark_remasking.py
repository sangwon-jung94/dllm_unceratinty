#!/usr/bin/env python
"""
Benchmark script for comparing remasking strategies on GSM8K.

Usage:
    python benchmark_remasking.py --model_path GSAI-ML/LLaDA-8B-Base

    # Run specific strategies only:
    python benchmark_remasking.py --strategies low_confidence random

    # Adjust generation parameters:
    python benchmark_remasking.py --gen_length 1024 --steps 1024 --block_length 1024
"""

import argparse
import subprocess
import json
import os
from datetime import datetime
from pathlib import Path


def run_evaluation(model_path, strategy, gen_length, steps, block_length,
                   mc_samples=8, alpha=1.0, beta=1.0, dropout_p=None, num_fewshot=5,
                   limit=None, output_dir="benchmark_results"):
    """Run lm-eval for a single remasking strategy."""

    # Build model_args string
    model_args = f"model_path={model_path}"
    model_args += f",gen_length={gen_length}"
    model_args += f",steps={steps}"
    model_args += f",block_length={block_length}"
    model_args += f",remasking='{strategy}'"

    if strategy == 'uncertainty_aware':
        model_args += f",mc_samples={mc_samples}"
        model_args += f",alpha={alpha}"
        model_args += f",beta={beta}"
        if dropout_p is not None:
            model_args += f",dropout_p={dropout_p}"

    # Build command
    cmd = [
        "python", "eval_llada.py",
        "--model", "llada_dist",
        "--tasks", "gsm8k",
        "--num_fewshot", str(num_fewshot),
        "--batch_size", "1",
        "--model_args", model_args,
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # lm-eval's --output_path creates a directory, not a file
    output_subdir = Path(output_dir) / f"gsm8k_{strategy}_{timestamp}"
    cmd.extend(["--output_path", str(output_subdir)])

    print(f"\n{'='*60}")
    print(f"Running GSM8K evaluation with remasking='{strategy}'")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    # Run evaluation
    result = subprocess.run(cmd, capture_output=False, text=True)

    return output_subdir, result.returncode


def parse_results(output_dir):
    """Parse results from lm-eval output directory.

    lm-eval creates a directory structure like:
        output_dir/model_name/results_*.json
    """
    if not output_dir.exists():
        return None

    # Find the results JSON file inside the directory structure
    results_files = list(output_dir.glob("**/results*.json"))
    if not results_files:
        return None

    # Use the most recent results file
    results_file = max(results_files, key=lambda p: p.stat().st_mtime)

    with open(results_file) as f:
        data = json.load(f)

    results = data.get("results", {})

    # Extract GSM8K scores
    gsm8k_scores = {}
    for task, metrics in results.items():
        # GSM8K uses exact_match or flexible_extract for accuracy
        if "exact_match,strict-match" in metrics:
            gsm8k_scores[task] = metrics["exact_match,strict-match"]
        elif "exact_match,flexible-extract" in metrics:
            gsm8k_scores[task] = metrics["exact_match,flexible-extract"]
        elif "acc" in metrics:
            gsm8k_scores[task] = metrics["acc"]
        elif "acc,none" in metrics:
            gsm8k_scores[task] = metrics["acc,none"]

    return gsm8k_scores


def print_comparison(all_results):
    """Print a comparison table of all results."""
    print("\n" + "="*80)
    print("BENCHMARK COMPARISON: Remasking Strategies on GSM8K")
    print("="*80)

    # Get all tasks
    all_tasks = set()
    for results in all_results.values():
        if results:
            all_tasks.update(results.keys())

    # Sort tasks
    all_tasks = sorted(all_tasks)

    # Print header
    strategies = list(all_results.keys())
    header = f"{'Task':<40} " + " ".join(f"{s:>15}" for s in strategies)
    print(header)
    print("-" * len(header))

    # Print each task
    for task in all_tasks:
        row = f"{task:<40} "
        for strategy in strategies:
            results = all_results.get(strategy, {})
            if results and task in results:
                row += f"{results[task]:>15.4f}"
            else:
                row += f"{'N/A':>15}"
        print(row)

    # Print averages
    print("-" * len(header))
    row = f"{'AVERAGE':<40} "
    for strategy in strategies:
        results = all_results.get(strategy, {})
        if results:
            avg = sum(results.values()) / len(results)
            row += f"{avg:>15.4f}"
        else:
            row += f"{'N/A':>15}"
    print(row)
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Benchmark remasking strategies on GSM8K")
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Base",
                        help="Path to the model")
    parser.add_argument("--strategies", nargs="+",
                        default=["low_confidence", "random", "uncertainty_aware"],
                        choices=["low_confidence", "random", "uncertainty_aware"],
                        help="Remasking strategies to benchmark")
    parser.add_argument("--gen_length", type=int, default=256,
                        help="Generation length (longer for GSM8K math reasoning)")
    parser.add_argument("--steps", type=int, default=128,
                        help="Number of diffusion steps")
    parser.add_argument("--block_length", type=int, default=256,
                        help="Block length for semi-autoregressive generation")
    parser.add_argument("--mc_samples", type=int, default=8,
                        help="MC Dropout samples for uncertainty_aware")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Epistemic uncertainty weight")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="Aleatoric uncertainty weight")
    parser.add_argument("--dropout_p", type=float, default=None,
                        help="Dropout probability for MC Dropout")
    parser.add_argument("--num_fewshot", type=int, default=5,
                        help="Number of few-shot examples")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of examples (for quick testing)")
    parser.add_argument("--output_dir", type=str, default="benchmark_results",
                        help="Directory to save results")

    args = parser.parse_args()

    print("="*60)
    print("REMASKING STRATEGY BENCHMARK")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Strategies: {args.strategies}")
    print(f"Generation params: gen_length={args.gen_length}, steps={args.steps}, block_length={args.block_length}")
    if "uncertainty_aware" in args.strategies:
        print(f"Uncertainty params: mc_samples={args.mc_samples}, alpha={args.alpha}, beta={args.beta}, dropout_p={args.dropout_p}")
    print()

    all_results = {}
    output_files = {}

    for strategy in args.strategies:
        output_file, returncode = run_evaluation(
            model_path=args.model_path,
            strategy=strategy,
            gen_length=args.gen_length,
            steps=args.steps,
            block_length=args.block_length,
            mc_samples=args.mc_samples,
            alpha=args.alpha,
            beta=args.beta,
            dropout_p=args.dropout_p,
            num_fewshot=args.num_fewshot,
            limit=args.limit,
            output_dir=args.output_dir
        )

        output_files[strategy] = output_file

        if returncode == 0:
            results = parse_results(output_file)
            all_results[strategy] = results
        else:
            print(f"WARNING: Evaluation failed for strategy '{strategy}'")
            all_results[strategy] = None

    # Print comparison
    print_comparison(all_results)

    # Save summary
    summary_file = Path(args.output_dir) / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "model_path": args.model_path,
            "strategies": args.strategies,
            "params": {
                "gen_length": args.gen_length,
                "steps": args.steps,
                "block_length": args.block_length,
                "mc_samples": args.mc_samples,
                "alpha": args.alpha,
                "beta": args.beta,
                "dropout_p": args.dropout_p,
            },
            "results": all_results,
            "output_files": {k: str(v) for k, v in output_files.items()}
        }, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    main()
