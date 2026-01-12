"""
Winogrande Blank Uncertainty Analysis Script.

This script analyzes uncertainty at the blank position (_) in Winogrande sentences.
It computes:
1. Epistemic and aleatoric uncertainty at the blank position using MC Dropout
2. Confidence and entropy metrics
3. Correlation with ground truth correctness
4. Integration with existing evaluation results from result/winogrande/

Usage:
    python analyze_winogrande_blank_uncertainty.py \
        --model_path GSAI-ML/LLaDA-8B-Instruct \
        --num_samples 100 \
        --mc_samples 8 \
        --output_dir ./result/winogrande_blank_uncertainty/analysis
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from datasets import load_dataset
from scipy import stats
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from generate import (
    compute_entropy,
    compute_uncertainty_decomposition,
    enable_mc_dropout,
    disable_mc_dropout,
)
from models.EnsembleLLaDA import replace_last_block_with_ensemble, EnsembleLLaDAModel
from evaluate_winogrande import evaluate_winogrande, parse_output_file


# Set style for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def compute_blank_uncertainty(
    model,
    tokenizer,
    sentence: str,
    option1: str,
    option2: str,
    device: str,
    mc_samples: int = 8,
    dropout_p: float = 0.1,
    use_ensemble: bool = False,
) -> Dict:
    """
    Compute uncertainty metrics for the blank position in a Winogrande sentence.
    
    This approach:
    1. Splits sentence at "_" into prefix and suffix
    2. Tokenizes prefix and suffix separately
    3. Inserts mask_id between them
    4. Computes uncertainty at the masked position
    
    Example: "I _ you" -> tokenize("I ") + [<mask>] + tokenize(" you")
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        sentence: Winogrande sentence with _
        option1: First option
        option2: Second option
        device: Device to use
        mc_samples: Number of MC Dropout samples
        dropout_p: Dropout probability
        use_ensemble: Whether to use ensemble model
    
    Returns:
        Dict with uncertainty metrics
    """
    mask_id = 126336
    
    # Check if sentence contains underscore
    if '_' not in sentence:
        raise ValueError(f"Sentence does not contain '_': {sentence}")
    
    # Split sentence at the underscore
    parts = sentence.split('_', 1)  # Split only at first occurrence
    if len(parts) != 2:
        raise ValueError(f"Expected exactly one '_' in sentence: {sentence}")
    
    prefix, suffix = parts
    
    # Tokenize prefix and suffix separately (no chat template)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False) if prefix else []
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False) if suffix else []
    
    # Construct input_ids: prefix + mask + suffix
    input_ids_list = prefix_ids + [mask_id] + suffix_ids
    blank_position = len(prefix_ids)  # Position of the mask token
    
    # Convert to tensor
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    
    # Create prompt_index (all positions except mask are prompt)
    prompt_index = torch.ones_like(input_ids, dtype=torch.bool)
    prompt_index[0, blank_position] = False  # Only mask position is not prompt
    
    x = input_ids.clone()
    blank_start = blank_position
    blank_end = blank_position + 1
    
    # Compute uncertainty decomposition
    with torch.no_grad():
        mean_logits, H_epistemic, H_aleatoric, p_bar = compute_uncertainty_decomposition(
            model=model,
            x=x,
            attention_mask=attention_mask,
            mc_samples=mc_samples,
            cfg_scale=0.0,
            prompt_index=prompt_index,
            mask_id=mask_id,
            dropout_p=dropout_p,
        )
    
    # Extract metrics for the blank position
    blank_pos = blank_start
    
    epistemic = H_epistemic[0, blank_pos].item()
    aleatoric = H_aleatoric[0, blank_pos].item()
    total_uncertainty = epistemic + aleatoric
    
    # Get probability distribution at blank position
    probs_at_blank = p_bar[0, blank_pos]
    
    # Compute entropy
    entropy = compute_entropy(probs_at_blank.unsqueeze(0), dim=-1).item()
    
    # Get predicted token and confidence
    predicted_token_id = probs_at_blank.argmax().item()
    predicted_token = tokenizer.decode([predicted_token_id])
    confidence = probs_at_blank.max().item()
    
    # Get probabilities for option tokens - try multiple variations
    def get_option_prob(option_text, probs):
        """Get the probability of an option, trying different tokenizations."""
        max_prob = 0.0
        
        # Try different variations
        variations = [
            option_text,
            ' ' + option_text,
            option_text.lower(),
            ' ' + option_text.lower(),
        ]
        
        for var in variations:
            ids = tokenizer.encode(var, add_special_tokens=False)
            if ids:
                prob = probs[ids[0]].item()
                max_prob = max(max_prob, prob)
        
        return max_prob
    
    option1_prob = get_option_prob(option1, probs_at_blank)
    option2_prob = get_option_prob(option2, probs_at_blank)
    
    # Also try with space prefix
    option1_with_space_ids = tokenizer.encode(' ' + option1, add_special_tokens=False)
    option2_with_space_ids = tokenizer.encode(' ' + option2, add_special_tokens=False)
    
    option1_prob_space = probs_at_blank[option1_with_space_ids[0]].item() if option1_with_space_ids else 0.0
    option2_prob_space = probs_at_blank[option2_with_space_ids[0]].item() if option2_with_space_ids else 0.0
    
    # Use max of with/without space
    option1_prob = max(option1_prob, option1_prob_space)
    option2_prob = max(option2_prob, option2_prob_space)
    
    # Determine predicted choice based on option probabilities
    if option1_prob > option2_prob:
        predicted_choice = "1"
    elif option2_prob > option1_prob:
        predicted_choice = "2"
    else:
        predicted_choice = None
    
    return {
        "blank_start": blank_start,
        "blank_end": blank_end,
        "epistemic": epistemic,
        "aleatoric": aleatoric,
        "total": total_uncertainty,
        "predicted_token": predicted_token,
        "option1_prob": option1_prob,
        "option2_prob": option2_prob,
        "confidence": confidence,
        "entropy": entropy,
        "predicted_choice": predicted_choice,
    }


def load_evaluation_results(result_dir: Path) -> Optional[Dict]:
    """
    Load evaluation results from an existing result directory.
    
    Args:
        result_dir: Path to result directory containing output.txt
    
    Returns:
        Dict with evaluation metrics or None
    """
    output_file = result_dir / "output.txt"
    eval_file = result_dir / "evaluation.json"
    
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            return json.load(f)
    
    if output_file.exists():
        try:
            metrics = evaluate_winogrande(output_file)
            return metrics
        except Exception as e:
            print(f"Warning: Could not evaluate {output_file}: {e}")
            return None
    
    return None


def analyze_uncertainty_correctness_correlation(
    uncertainty_data: List[Dict],
    save_dir: Path,
) -> Dict:
    """
    Analyze correlation between uncertainty metrics and correctness.
    
    Args:
        uncertainty_data: List of dicts with uncertainty and correctness info
        save_dir: Directory to save plots
    
    Returns:
        Dict with correlation statistics
    """
    correct_samples = [d for d in uncertainty_data if d.get('correct', False)]
    incorrect_samples = [d for d in uncertainty_data if not d.get('correct', False)]
    
    stats_dict = {
        "correct": {
            "count": len(correct_samples),
            "epistemic": {
                "mean": np.mean([d['epistemic'] for d in correct_samples]) if correct_samples else 0,
                "std": np.std([d['epistemic'] for d in correct_samples]) if correct_samples else 0,
            },
            "aleatoric": {
                "mean": np.mean([d['aleatoric'] for d in correct_samples]) if correct_samples else 0,
                "std": np.std([d['aleatoric'] for d in correct_samples]) if correct_samples else 0,
            },
            "total": {
                "mean": np.mean([d['total'] for d in correct_samples]) if correct_samples else 0,
                "std": np.std([d['total'] for d in correct_samples]) if correct_samples else 0,
            },
            "confidence": {
                "mean": np.mean([d['confidence'] for d in correct_samples]) if correct_samples else 0,
                "std": np.std([d['confidence'] for d in correct_samples]) if correct_samples else 0,
            },
            "entropy": {
                "mean": np.mean([d['entropy'] for d in correct_samples]) if correct_samples else 0,
                "std": np.std([d['entropy'] for d in correct_samples]) if correct_samples else 0,
            },
        },
        "incorrect": {
            "count": len(incorrect_samples),
            "epistemic": {
                "mean": np.mean([d['epistemic'] for d in incorrect_samples]) if incorrect_samples else 0,
                "std": np.std([d['epistemic'] for d in incorrect_samples]) if incorrect_samples else 0,
            },
            "aleatoric": {
                "mean": np.mean([d['aleatoric'] for d in incorrect_samples]) if incorrect_samples else 0,
                "std": np.std([d['aleatoric'] for d in incorrect_samples]) if incorrect_samples else 0,
            },
            "total": {
                "mean": np.mean([d['total'] for d in incorrect_samples]) if incorrect_samples else 0,
                "std": np.std([d['total'] for d in incorrect_samples]) if incorrect_samples else 0,
            },
            "confidence": {
                "mean": np.mean([d['confidence'] for d in incorrect_samples]) if incorrect_samples else 0,
                "std": np.std([d['confidence'] for d in incorrect_samples]) if incorrect_samples else 0,
            },
            "entropy": {
                "mean": np.mean([d['entropy'] for d in incorrect_samples]) if incorrect_samples else 0,
                "std": np.std([d['entropy'] for d in incorrect_samples]) if incorrect_samples else 0,
            },
        },
    }
    
    # Statistical tests
    if len(correct_samples) > 1 and len(incorrect_samples) > 1:
        metrics = ['epistemic', 'aleatoric', 'total', 'confidence', 'entropy']
        for metric in metrics:
            correct_vals = [d[metric] for d in correct_samples]
            incorrect_vals = [d[metric] for d in incorrect_samples]
            
            # Mann-Whitney U test
            try:
                u_stat, p_value = stats.mannwhitneyu(correct_vals, incorrect_vals, alternative='two-sided')
                stats_dict[f'{metric}_mannwhitney'] = {
                    'u_statistic': u_stat,
                    'p_value': p_value,
                }
            except Exception:
                pass
            
            # Point-biserial correlation
            try:
                all_vals = correct_vals + incorrect_vals
                labels = [1] * len(correct_vals) + [0] * len(incorrect_vals)
                corr, p_val = stats.pointbiserialr(labels, all_vals)
                stats_dict[f'{metric}_pointbiserial'] = {
                    'correlation': corr,
                    'p_value': p_val,
                }
            except Exception:
                pass
    
    return stats_dict


def plot_uncertainty_distributions(
    uncertainty_data: List[Dict],
    save_path: Path,
    title_suffix: str = "",
):
    """
    Plot distributions of uncertainty metrics by correctness.
    """
    correct = [d for d in uncertainty_data if d.get('correct', False)]
    incorrect = [d for d in uncertainty_data if not d.get('correct', False)]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    metrics = [
        ('epistemic', 'Epistemic Uncertainty'),
        ('aleatoric', 'Aleatoric Uncertainty'),
        ('total', 'Total Uncertainty'),
        ('confidence', 'Confidence'),
        ('entropy', 'Entropy'),
        ('option1_prob', 'Option 1 Probability'),
    ]
    
    for ax, (metric, label) in zip(axes.flatten(), metrics):
        correct_vals = [d[metric] for d in correct if metric in d]
        incorrect_vals = [d[metric] for d in incorrect if metric in d]
        
        if correct_vals:
            ax.hist(correct_vals, bins=20, alpha=0.6, label=f'Correct (n={len(correct_vals)})', color='green')
        if incorrect_vals:
            ax.hist(incorrect_vals, bins=20, alpha=0.6, label=f'Incorrect (n={len(incorrect_vals)})', color='red')
        
        ax.set_xlabel(label)
        ax.set_ylabel('Count')
        ax.legend()
        ax.set_title(f'{label} Distribution')
    
    plt.suptitle(f'Uncertainty Distributions by Correctness{title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved distribution plot to: {save_path}")


def plot_uncertainty_vs_confidence(
    uncertainty_data: List[Dict],
    save_path: Path,
    title_suffix: str = "",
):
    """
    Plot uncertainty vs confidence scatter plots.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    correct = [d for d in uncertainty_data if d.get('correct', False)]
    incorrect = [d for d in uncertainty_data if not d.get('correct', False)]
    
    # Epistemic vs Confidence
    ax = axes[0]
    if correct:
        ax.scatter([d['confidence'] for d in correct], [d['epistemic'] for d in correct],
                   alpha=0.6, label='Correct', c='green', s=50)
    if incorrect:
        ax.scatter([d['confidence'] for d in incorrect], [d['epistemic'] for d in incorrect],
                   alpha=0.6, label='Incorrect', c='red', s=50)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Epistemic Uncertainty')
    ax.set_title('Epistemic vs Confidence')
    ax.legend()
    
    # Aleatoric vs Confidence
    ax = axes[1]
    if correct:
        ax.scatter([d['confidence'] for d in correct], [d['aleatoric'] for d in correct],
                   alpha=0.6, label='Correct', c='green', s=50)
    if incorrect:
        ax.scatter([d['confidence'] for d in incorrect], [d['aleatoric'] for d in incorrect],
                   alpha=0.6, label='Incorrect', c='red', s=50)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Aleatoric Uncertainty')
    ax.set_title('Aleatoric vs Confidence')
    ax.legend()
    
    # Entropy vs Confidence
    ax = axes[2]
    if correct:
        ax.scatter([d['confidence'] for d in correct], [d['entropy'] for d in correct],
                   alpha=0.6, label='Correct', c='green', s=50)
    if incorrect:
        ax.scatter([d['confidence'] for d in incorrect], [d['entropy'] for d in incorrect],
                   alpha=0.6, label='Incorrect', c='red', s=50)
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Entropy')
    ax.set_title('Entropy vs Confidence')
    ax.legend()
    
    plt.suptitle(f'Uncertainty vs Confidence{title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved uncertainty vs confidence plot to: {save_path}")


def plot_comparison_boxplots(
    uncertainty_data: List[Dict],
    save_path: Path,
    title_suffix: str = "",
):
    """
    Create box plots comparing uncertainty metrics between correct and incorrect predictions.
    """
    correct = [d for d in uncertainty_data if d.get('correct', False)]
    incorrect = [d for d in uncertainty_data if not d.get('correct', False)]
    
    # Skip if no data or only one category
    if not correct and not incorrect:
        print(f"Warning: No data for boxplot, skipping: {save_path}")
        return
    
    metrics = ['epistemic', 'aleatoric', 'total', 'entropy']
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 6))
    
    for ax, metric in zip(axes, metrics):
        data = []
        labels = []
        
        if correct:
            data.append([d[metric] for d in correct])
            labels.append(f'Correct\n(n={len(correct)})')
        if incorrect:
            data.append([d[metric] for d in incorrect])
            labels.append(f'Incorrect\n(n={len(incorrect)})')
        
        if data:
            bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
            colors = ['lightgreen', 'lightcoral'][:len(data)]
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
        
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} by Correctness')
    
    plt.suptitle(f'Uncertainty Comparison{title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved boxplot comparison to: {save_path}")


def plot_correlation_heatmap(
    uncertainty_data: List[Dict],
    save_path: Path,
    title_suffix: str = "",
):
    """
    Plot correlation heatmap between different metrics.
    """
    import pandas as pd
    
    metrics = ['epistemic', 'aleatoric', 'total', 'confidence', 'entropy', 'option1_prob', 'option2_prob']
    
    # Filter data with all required metrics
    filtered_data = [d for d in uncertainty_data if all(m in d for m in metrics)]
    
    if not filtered_data:
        print("Warning: No data with all metrics for correlation heatmap")
        return
    
    # Add 'correct' as a numeric column
    df_data = {m: [d[m] for d in filtered_data] for m in metrics}
    df_data['correct'] = [1 if d.get('correct', False) else 0 for d in filtered_data]
    
    df = pd.DataFrame(df_data)
    
    # Compute correlations
    corr_matrix = df.corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
    
    # Add labels
    columns = list(corr_matrix.columns)
    ax.set_xticks(range(len(columns)))
    ax.set_yticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=45, ha='right')
    ax.set_yticklabels(columns)
    
    # Add correlation values as text
    for i in range(len(columns)):
        for j in range(len(columns)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    ax.set_title(f'Correlation Heatmap{title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved correlation heatmap to: {save_path}")


def integrate_with_existing_results(
    uncertainty_data: List[Dict],
    result_dirs: List[Path],
    save_dir: Path,
) -> Dict:
    """
    Integrate blank uncertainty analysis with existing evaluation results.
    Computes separate statistics for each result directory.
    
    Args:
        uncertainty_data: List of uncertainty measurements
        result_dirs: List of paths to existing result directories
        save_dir: Directory to save analysis results
    
    Returns:
        Dict with integrated analysis (per-directory statistics)
    """
    integrated_results = {}
    
    for result_dir in result_dirs:
        eval_results = load_evaluation_results(result_dir)
        if eval_results is None:
            continue
        
        dir_name = result_dir.name
        integrated_results[dir_name] = {
            "path": str(result_dir),
            "evaluation": {
                "total_samples": eval_results.get('total_samples', 0),
                "accuracy": eval_results.get('accuracy', 0),
                "correct": eval_results.get('correct', 0),
            },
        }
        
        # Match samples by index
        if 'results' in eval_results:
            matched_samples = []
            for eval_sample in eval_results['results']:
                sample_num = eval_sample.get('sample_num', 0)
                # Find corresponding uncertainty data
                for unc_sample in uncertainty_data:
                    if unc_sample.get('sample_num') == sample_num:
                        matched_samples.append({
                            **unc_sample,
                            'eval_predicted': eval_sample.get('predicted'),
                            'eval_correct': eval_sample.get('correct'),
                        })
                        break
            
            if matched_samples:
                # Separate by eval correctness
                eval_correct = [d for d in matched_samples if d.get('eval_correct', False)]
                eval_incorrect = [d for d in matched_samples if not d.get('eval_correct', False)]
                
                integrated_results[dir_name]["matched_samples"] = len(matched_samples)
                
                # Compute per-directory statistics with std and p-values
                dir_stats = {
                    "correct": {
                        "count": len(eval_correct),
                    },
                    "incorrect": {
                        "count": len(eval_incorrect),
                    },
                    "statistical_tests": {},
                }
                
                # Calculate mean, std for each metric
                metrics = ['epistemic', 'aleatoric', 'total', 'entropy', 'confidence']
                for metric in metrics:
                    if eval_correct:
                        correct_vals = [d[metric] for d in eval_correct]
                        dir_stats["correct"][f"mean_{metric}"] = float(np.mean(correct_vals))
                        dir_stats["correct"][f"std_{metric}"] = float(np.std(correct_vals))
                    else:
                        dir_stats["correct"][f"mean_{metric}"] = 0.0
                        dir_stats["correct"][f"std_{metric}"] = 0.0
                    
                    if eval_incorrect:
                        incorrect_vals = [d[metric] for d in eval_incorrect]
                        dir_stats["incorrect"][f"mean_{metric}"] = float(np.mean(incorrect_vals))
                        dir_stats["incorrect"][f"std_{metric}"] = float(np.std(incorrect_vals))
                    else:
                        dir_stats["incorrect"][f"mean_{metric}"] = 0.0
                        dir_stats["incorrect"][f"std_{metric}"] = 0.0
                    
                    # Statistical test (Mann-Whitney U) for this directory
                    if len(eval_correct) > 1 and len(eval_incorrect) > 1:
                        try:
                            u_stat, p_value = stats.mannwhitneyu(
                                [d[metric] for d in eval_correct],
                                [d[metric] for d in eval_incorrect],
                                alternative='two-sided'
                            )
                            dir_stats["statistical_tests"][metric] = {
                                "u_statistic": float(u_stat),
                                "p_value": float(p_value),
                                "significant": bool(p_value < 0.05),
                            }
                        except Exception:
                            pass
                
                integrated_results[dir_name]["uncertainty_by_eval_correctness"] = dir_stats
                
                # Generate only plots that differ per directory (boxplots depend on correctness)
                dir_plot_path = save_dir / "per_directory_plots" / dir_name
                dir_plot_path.mkdir(parents=True, exist_ok=True)
                
                # Create modified data with eval_correct as the 'correct' field for plotting
                plot_data = []
                for d in matched_samples:
                    plot_d = d.copy()
                    plot_d['correct'] = d.get('eval_correct', False)
                    plot_data.append(plot_d)
                
                # Only generate boxplots (distributions/correlation are same for all dirs)
                title_suffix = f" ({dir_name})"
                
                plot_comparison_boxplots(
                    plot_data,
                    dir_plot_path / 'uncertainty_boxplots.png',
                    title_suffix=title_suffix,
                )
                
                print(f"  Saved boxplot for {dir_name}")
    
    return integrated_results


def main():
    parser = argparse.ArgumentParser(description='Analyze Winogrande blank uncertainty')
    
    # Model settings
    parser.add_argument('--model_path', type=str, default='GSAI-ML/LLaDA-8B-Instruct',
                        help='Path or name of the model')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to run on')
    
    # Uncertainty settings
    parser.add_argument('--mc_samples', type=int, default=8,
                        help='Number of MC Dropout samples')
    parser.add_argument('--dropout_p', type=float, default=0.1,
                        help='Dropout probability for MC Dropout')
    parser.add_argument('--use_ensemble', action='store_true',
                        help='Use EnsembleLLaDA instead of MC Dropout')
    
    # Data settings
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to analyze (None = all)')
    parser.add_argument('--dataset_config', type=str, default='winogrande_xl',
                        help='Winogrande config')
    parser.add_argument('--split', type=str, default='validation',
                        help='Dataset split')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./result/winogrande_blank_uncertainty',
                        help='Directory to save outputs')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')
    
    # Integration with existing results
    parser.add_argument('--integrate_results', action='store_true',
                        help='Integrate with existing results in result/winogrande/')
    parser.add_argument('--existing_results_dir', type=str, default='./result/winogrande',
                        help='Directory containing existing evaluation results')
    
    args = parser.parse_args()
    
    # Generate experiment name
    if args.exp_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.exp_name = f'blank_uncertainty_mc{args.mc_samples}_{args.num_samples or "all"}samples_{timestamp}'
    
    # Create output directory
    exp_output_dir = Path(args.output_dir) / args.exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print(f"Winogrande Blank Uncertainty Analysis")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Device: {args.device}")
    print(f"MC Samples: {args.mc_samples}")
    print(f"Dropout: {args.dropout_p}")
    print(f"Use Ensemble: {args.use_ensemble}")
    print(f"Output Directory: {exp_output_dir}")
    print("=" * 80)
    
    # Load model and tokenizer
    print("\nLoading model...")
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to(args.device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    
    if tokenizer.padding_side != 'left':
        tokenizer.padding_side = 'left'
    
    # Convert to EnsembleLLaDA if requested
    if args.use_ensemble:
        import types
        print("Converting to EnsembleLLaDA...")
        replace_last_block_with_ensemble(model, mlp_dropout_p=args.dropout_p)
        # Patch forward method to support num_ensembles
        model.model.forward = types.MethodType(EnsembleLLaDAModel.forward, model.model)
        print("Model converted to EnsembleLLaDA")
    
    # Load Winogrande dataset
    print(f"\nLoading Winogrande dataset ({args.dataset_config}, {args.split})...")
    dataset = load_dataset('winogrande', args.dataset_config, split=args.split)
    
    num_samples = args.num_samples if args.num_samples else len(dataset)
    num_samples = min(num_samples, len(dataset))
    
    print(f"Analyzing {num_samples} samples...")
    
    # Analyze each sample
    uncertainty_data = []
    
    for idx in tqdm(range(num_samples), desc="Analyzing blanks"):
        sample = dataset[idx]
        sentence = sample['sentence']
        option1 = sample['option1']
        option2 = sample['option2']
        ground_truth = str(sample['answer']).strip()
        
        try:
            result = compute_blank_uncertainty(
                model=model,
                tokenizer=tokenizer,
                sentence=sentence,
                option1=option1,
                option2=option2,
                device=args.device,
                mc_samples=args.mc_samples,
                dropout_p=args.dropout_p,
                use_ensemble=args.use_ensemble,
            )
            
            result['sample_num'] = idx + 1
            result['sentence'] = sentence
            result['option1'] = option1
            result['option2'] = option2
            result['ground_truth'] = ground_truth
            result['correct'] = result['predicted_choice'] == ground_truth
            
            uncertainty_data.append(result)
            
        except Exception as e:
            print(f"\nError processing sample {idx + 1}: {e}")
            continue
    
    print(f"\nSuccessfully analyzed {len(uncertainty_data)} samples")
    
    # Save raw data
    data_path = exp_output_dir / 'blank_uncertainty_data.json'
    with open(data_path, 'w') as f:
        json.dump(uncertainty_data, f, indent=2)
    print(f"Saved uncertainty data to: {data_path}")
    
    # Compute statistics
    stats = analyze_uncertainty_correctness_correlation(uncertainty_data, exp_output_dir)
    
    stats_path = exp_output_dir / 'uncertainty_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to: {stats_path}")
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_uncertainty_distributions(
        uncertainty_data,
        exp_output_dir / 'uncertainty_distributions.png',
    )
    
    plot_uncertainty_vs_confidence(
        uncertainty_data,
        exp_output_dir / 'uncertainty_vs_confidence.png',
    )
    
    plot_comparison_boxplots(
        uncertainty_data,
        exp_output_dir / 'uncertainty_boxplots.png',
    )
    
    plot_correlation_heatmap(
        uncertainty_data,
        exp_output_dir / 'correlation_heatmap.png',
    )
    
    # Integrate with existing results if requested
    if args.integrate_results:
        print("\nIntegrating with existing results...")
        existing_dir = Path(args.existing_results_dir)
        
        if existing_dir.exists():
            # Find all result directories
            result_dirs = []
            for subdir in existing_dir.rglob('*'):
                if subdir.is_dir() and (subdir / 'output.txt').exists():
                    result_dirs.append(subdir)
            
            if result_dirs:
                print(f"Found {len(result_dirs)} result directories")
                integrated = integrate_with_existing_results(
                    uncertainty_data,
                    result_dirs,
                    exp_output_dir,
                )
                
                integrated_path = exp_output_dir / 'integrated_analysis.json'
                with open(integrated_path, 'w') as f:
                    json.dump(integrated, f, indent=2)
                print(f"Saved integrated analysis to: {integrated_path}")
            else:
                print("No result directories found with output.txt")
        else:
            print(f"Warning: Existing results directory not found: {existing_dir}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    correct_count = sum(1 for d in uncertainty_data if d.get('correct', False))
    total_count = len(uncertainty_data)
    accuracy = correct_count / total_count * 100 if total_count > 0 else 0
    
    print(f"Blank-based Prediction Accuracy: {accuracy:.2f}% ({correct_count}/{total_count})")
    print(f"\nCorrect predictions (n={stats['correct']['count']}):")
    print(f"  Epistemic: {stats['correct']['epistemic']['mean']:.4f} ± {stats['correct']['epistemic']['std']:.4f}")
    print(f"  Aleatoric: {stats['correct']['aleatoric']['mean']:.4f} ± {stats['correct']['aleatoric']['std']:.4f}")
    print(f"  Entropy:   {stats['correct']['entropy']['mean']:.4f} ± {stats['correct']['entropy']['std']:.4f}")
    
    print(f"\nIncorrect predictions (n={stats['incorrect']['count']}):")
    print(f"  Epistemic: {stats['incorrect']['epistemic']['mean']:.4f} ± {stats['incorrect']['epistemic']['std']:.4f}")
    print(f"  Aleatoric: {stats['incorrect']['aleatoric']['mean']:.4f} ± {stats['incorrect']['aleatoric']['std']:.4f}")
    print(f"  Entropy:   {stats['incorrect']['entropy']['mean']:.4f} ± {stats['incorrect']['entropy']['std']:.4f}")
    
    print("\n" + "=" * 80)
    print(f"All outputs saved to: {exp_output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
