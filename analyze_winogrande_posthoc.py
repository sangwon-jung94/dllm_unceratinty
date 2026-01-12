"""
Post-hoc Analysis Script for Winogrande Blank Uncertainty Results.

This script takes the output from analyze_winogrande_blank_uncertainty.py
and creates detailed visualizations and statistical analyses.

Usage:
    python analyze_winogrande_posthoc.py \
        --uncertainty_data ./result/winogrande_blank_uncertainty/full_analysis_100samples/blank_uncertainty_data.json \
        --integrated_results ./result/winogrande_blank_uncertainty/full_analysis_100samples/integrated_analysis.json \
        --output_dir ./result/winogrande_blank_uncertainty/full_analysis_100samples/posthoc
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


plt.style.use('seaborn-v0_8-whitegrid')


def load_data(uncertainty_path: Path, integrated_path: Path = None):
    """Load uncertainty and integrated analysis data."""
    with open(uncertainty_path, 'r') as f:
        uncertainty_data = json.load(f)
    
    integrated_data = None
    if integrated_path and integrated_path.exists():
        with open(integrated_path, 'r') as f:
            integrated_data = json.load(f)
    
    return uncertainty_data, integrated_data


def plot_epistemic_vs_aleatoric(data: List[Dict], save_path: Path):
    """Create scatter plot of epistemic vs aleatoric uncertainty."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    correct = [d for d in data if d.get('correct', False)]
    incorrect = [d for d in data if not d.get('correct', False)]
    
    # Plot with different markers and colors
    if correct:
        ax.scatter([d['epistemic'] for d in correct], [d['aleatoric'] for d in correct],
                   c='green', alpha=0.6, label=f'Correct (n={len(correct)})', s=80, marker='o')
    if incorrect:
        ax.scatter([d['epistemic'] for d in incorrect], [d['aleatoric'] for d in incorrect],
                   c='red', alpha=0.6, label=f'Incorrect (n={len(incorrect)})', s=80, marker='x')
    
    ax.set_xlabel('Epistemic Uncertainty', fontsize=12)
    ax.set_ylabel('Aleatoric Uncertainty', fontsize=12)
    ax.set_title('Epistemic vs Aleatoric Uncertainty\n(Blank Position Prediction)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_option_probability_analysis(data: List[Dict], save_path: Path):
    """Analyze option probabilities vs uncertainty."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Calculate probability ratio (higher is more confident in chosen option)
    for d in data:
        if d.get('predicted_choice') == '1':
            d['prob_ratio'] = d['option1_prob'] / (d['option2_prob'] + 1e-10)
        else:
            d['prob_ratio'] = d['option2_prob'] / (d['option1_prob'] + 1e-10)
        d['prob_diff'] = abs(d['option1_prob'] - d['option2_prob'])
    
    correct = [d for d in data if d.get('correct', False)]
    incorrect = [d for d in data if not d.get('correct', False)]
    
    # 1. Probability difference vs Epistemic
    ax = axes[0, 0]
    if correct:
        ax.scatter([d['prob_diff'] for d in correct], [d['epistemic'] for d in correct],
                   c='green', alpha=0.6, label='Correct', s=50)
    if incorrect:
        ax.scatter([d['prob_diff'] for d in incorrect], [d['epistemic'] for d in incorrect],
                   c='red', alpha=0.6, label='Incorrect', s=50)
    ax.set_xlabel('|P(option1) - P(option2)|')
    ax.set_ylabel('Epistemic Uncertainty')
    ax.set_title('Option Probability Difference vs Epistemic')
    ax.legend()
    
    # 2. Probability difference vs Aleatoric
    ax = axes[0, 1]
    if correct:
        ax.scatter([d['prob_diff'] for d in correct], [d['aleatoric'] for d in correct],
                   c='green', alpha=0.6, label='Correct', s=50)
    if incorrect:
        ax.scatter([d['prob_diff'] for d in incorrect], [d['aleatoric'] for d in incorrect],
                   c='red', alpha=0.6, label='Incorrect', s=50)
    ax.set_xlabel('|P(option1) - P(option2)|')
    ax.set_ylabel('Aleatoric Uncertainty')
    ax.set_title('Option Probability Difference vs Aleatoric')
    ax.legend()
    
    # 3. Option1 vs Option2 probability
    ax = axes[1, 0]
    if correct:
        ax.scatter([d['option1_prob'] for d in correct], [d['option2_prob'] for d in correct],
                   c='green', alpha=0.6, label='Correct', s=50)
    if incorrect:
        ax.scatter([d['option1_prob'] for d in incorrect], [d['option2_prob'] for d in incorrect],
                   c='red', alpha=0.6, label='Incorrect', s=50)
    ax.plot([0, max(max(d['option1_prob'] for d in data), max(d['option2_prob'] for d in data))],
            [0, max(max(d['option1_prob'] for d in data), max(d['option2_prob'] for d in data))],
            'k--', alpha=0.3, label='y=x')
    ax.set_xlabel('P(option1)')
    ax.set_ylabel('P(option2)')
    ax.set_title('Option 1 vs Option 2 Probability')
    ax.legend()
    
    # 4. Confidence distribution
    ax = axes[1, 1]
    if correct:
        ax.hist([d['confidence'] for d in correct], bins=20, alpha=0.6, label='Correct', color='green')
    if incorrect:
        ax.hist([d['confidence'] for d in incorrect], bins=20, alpha=0.6, label='Incorrect', color='red')
    ax.set_xlabel('Confidence (max probability at blank)')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution')
    ax.legend()
    
    plt.suptitle('Option Probability Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_strategy_comparison(integrated_data: Dict, save_path: Path):
    """Compare different remasking strategies with uncertainty metrics."""
    if not integrated_data:
        print("No integrated data available")
        return
    
    # Extract strategy metrics
    strategies = []
    accuracies = []
    correct_epistemic = []
    incorrect_epistemic = []
    correct_aleatoric = []
    incorrect_aleatoric = []
    
    for strategy_name, data in integrated_data.items():
        if 'uncertainty_by_eval_correctness' not in data:
            continue
        
        strategies.append(strategy_name[:40])  # Truncate long names
        accuracies.append(data['evaluation']['accuracy'])
        
        unc_data = data['uncertainty_by_eval_correctness']
        correct_epistemic.append(unc_data['correct'].get('mean_epistemic', 0))
        incorrect_epistemic.append(unc_data['incorrect'].get('mean_epistemic', 0))
        correct_aleatoric.append(unc_data['correct'].get('mean_aleatoric', 0))
        incorrect_aleatoric.append(unc_data['incorrect'].get('mean_aleatoric', 0))
    
    if not strategies:
        print("No valid strategy data found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # 1. Accuracy bar chart
    ax = axes[0, 0]
    y_pos = np.arange(len(strategies))
    bars = ax.barh(y_pos, accuracies, color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(strategies, fontsize=8)
    ax.set_xlabel('Accuracy (%)')
    ax.set_title('Accuracy by Remasking Strategy')
    ax.bar_label(bars, fmt='%.1f', fontsize=8)
    
    # 2. Epistemic comparison
    ax = axes[0, 1]
    width = 0.35
    x = np.arange(len(strategies))
    bars1 = ax.bar(x - width/2, correct_epistemic, width, label='Correct', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, incorrect_epistemic, width, label='Incorrect', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Mean Epistemic Uncertainty')
    ax.set_title('Epistemic Uncertainty: Correct vs Incorrect')
    ax.legend()
    
    # 3. Aleatoric comparison
    ax = axes[1, 0]
    bars1 = ax.bar(x - width/2, correct_aleatoric, width, label='Correct', color='green', alpha=0.7)
    bars2 = ax.bar(x + width/2, incorrect_aleatoric, width, label='Incorrect', color='red', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Mean Aleatoric Uncertainty')
    ax.set_title('Aleatoric Uncertainty: Correct vs Incorrect')
    ax.legend()
    
    # 4. Scatter: Accuracy vs Epistemic difference
    ax = axes[1, 1]
    epistemic_diff = np.array(correct_epistemic) - np.array(incorrect_epistemic)
    ax.scatter(epistemic_diff, accuracies, s=100, c='purple', alpha=0.7)
    for i, strat in enumerate(strategies):
        ax.annotate(strat[-15:], (epistemic_diff[i], accuracies[i]), fontsize=7, 
                   rotation=20, ha='left')
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Epistemic Difference (Correct - Incorrect)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Accuracy vs Epistemic Difference')
    
    plt.suptitle('Strategy Comparison with Blank Uncertainty', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_uncertainty_calibration(data: List[Dict], save_path: Path):
    """Analyze calibration: is uncertainty predictive of correctness?"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Sort by uncertainty and bin
    n_bins = 10
    
    for idx, (metric, title) in enumerate([
        ('epistemic', 'Epistemic'),
        ('aleatoric', 'Aleatoric'),
        ('total', 'Total')
    ]):
        ax = axes[idx]
        
        # Sort by metric
        sorted_data = sorted(data, key=lambda x: x[metric])
        
        # Bin
        bin_size = len(sorted_data) // n_bins
        accuracies = []
        uncertainties = []
        
        for i in range(n_bins):
            start = i * bin_size
            end = start + bin_size if i < n_bins - 1 else len(sorted_data)
            bin_data = sorted_data[start:end]
            
            acc = sum(1 for d in bin_data if d.get('correct', False)) / len(bin_data) * 100
            unc = np.mean([d[metric] for d in bin_data])
            
            accuracies.append(acc)
            uncertainties.append(unc)
        
        ax.bar(range(n_bins), accuracies, color='steelblue', alpha=0.8)
        ax.set_xlabel(f'{title} Uncertainty Bin (Low → High)')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Accuracy by {title} Uncertainty Level')
        ax.set_xticks(range(n_bins))
        ax.set_xticklabels([f'{i+1}' for i in range(n_bins)])
        
        # Add trend line
        z = np.polyfit(range(n_bins), accuracies, 1)
        p = np.poly1d(z)
        ax.plot(range(n_bins), p(range(n_bins)), 'r--', alpha=0.8, label=f'Trend')
        ax.legend()
    
    plt.suptitle('Uncertainty Calibration: Does Higher Uncertainty Mean Lower Accuracy?', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_detailed_distributions(data: List[Dict], save_path: Path):
    """Create violin plots for detailed distribution comparison."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    
    correct = [d for d in data if d.get('correct', False)]
    incorrect = [d for d in data if not d.get('correct', False)]
    
    metrics = ['epistemic', 'aleatoric', 'total', 'entropy']
    titles = ['Epistemic', 'Aleatoric', 'Total', 'Entropy']
    
    for ax, metric, title in zip(axes, metrics, titles):
        plot_data = []
        labels = []
        
        if correct:
            plot_data.append([d[metric] for d in correct])
            labels.append('Correct')
        if incorrect:
            plot_data.append([d[metric] for d in incorrect])
            labels.append('Incorrect')
        
        parts = ax.violinplot(plot_data, showmeans=True, showmedians=True)
        
        # Color the violins
        colors = ['green', 'red']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i] if i < len(colors) else 'blue')
            pc.set_alpha(0.6)
        
        ax.set_xticks([1, 2])
        ax.set_xticklabels(labels)
        ax.set_ylabel(title)
        ax.set_title(f'{title} Distribution')
        
        # Add statistical test result
        if len(correct) > 1 and len(incorrect) > 1:
            u_stat, p_val = stats.mannwhitneyu(
                [d[metric] for d in correct],
                [d[metric] for d in incorrect],
                alternative='two-sided'
            )
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
            ax.text(1.5, ax.get_ylim()[1] * 0.95, f'p={p_val:.4f} {sig}', 
                   ha='center', fontsize=10, fontweight='bold')
    
    plt.suptitle('Uncertainty Distributions by Correctness\n(Violin Plots with Statistical Tests)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def create_summary_report(data: List[Dict], integrated_data: Dict, save_path: Path):
    """Create a text summary report of the analysis."""
    correct = [d for d in data if d.get('correct', False)]
    incorrect = [d for d in data if not d.get('correct', False)]
    
    report = []
    report.append("=" * 80)
    report.append("WINOGRANDE BLANK UNCERTAINTY ANALYSIS - SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Basic statistics
    report.append("## BASIC STATISTICS")
    report.append(f"Total samples analyzed: {len(data)}")
    report.append(f"Correct predictions: {len(correct)} ({len(correct)/len(data)*100:.1f}%)")
    report.append(f"Incorrect predictions: {len(incorrect)} ({len(incorrect)/len(data)*100:.1f}%)")
    report.append("")
    
    # Uncertainty comparison
    report.append("## UNCERTAINTY BY CORRECTNESS")
    report.append("")
    report.append("| Metric     | Correct (mean±std)    | Incorrect (mean±std)  | Difference |")
    report.append("|------------|----------------------|----------------------|------------|")
    
    for metric in ['epistemic', 'aleatoric', 'total', 'entropy', 'confidence']:
        c_vals = [d[metric] for d in correct]
        i_vals = [d[metric] for d in incorrect]
        c_mean, c_std = np.mean(c_vals), np.std(c_vals)
        i_mean, i_std = np.mean(i_vals), np.std(i_vals)
        diff = c_mean - i_mean
        report.append(f"| {metric:10} | {c_mean:7.4f} ± {c_std:7.4f} | {i_mean:7.4f} ± {i_std:7.4f} | {diff:+.4f}   |")
    
    report.append("")
    
    # Statistical tests
    report.append("## STATISTICAL TESTS (Mann-Whitney U)")
    report.append("")
    
    for metric in ['epistemic', 'aleatoric', 'total', 'entropy']:
        c_vals = [d[metric] for d in correct]
        i_vals = [d[metric] for d in incorrect]
        u_stat, p_val = stats.mannwhitneyu(c_vals, i_vals, alternative='two-sided')
        sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'
        report.append(f"- {metric}: U={u_stat:.1f}, p={p_val:.6f} {sig}")
    
    report.append("")
    
    # Key findings
    report.append("## KEY FINDINGS")
    report.append("")
    
    # Check epistemic significance
    c_epi = [d['epistemic'] for d in correct]
    i_epi = [d['epistemic'] for d in incorrect]
    _, p_epi = stats.mannwhitneyu(c_epi, i_epi, alternative='two-sided')
    
    if p_epi < 0.05:
        epi_direction = "higher" if np.mean(c_epi) > np.mean(i_epi) else "lower"
        report.append(f"1. Epistemic uncertainty is significantly {epi_direction} for correct predictions (p={p_epi:.4f})")
    else:
        report.append("1. No significant difference in epistemic uncertainty between correct/incorrect predictions")
    
    # Check aleatoric
    c_ale = [d['aleatoric'] for d in correct]
    i_ale = [d['aleatoric'] for d in incorrect]
    _, p_ale = stats.mannwhitneyu(c_ale, i_ale, alternative='two-sided')
    
    if p_ale < 0.05:
        ale_direction = "higher" if np.mean(c_ale) > np.mean(i_ale) else "lower"
        report.append(f"2. Aleatoric uncertainty is significantly {ale_direction} for correct predictions (p={p_ale:.4f})")
    else:
        report.append("2. No significant difference in aleatoric uncertainty between correct/incorrect predictions")
    
    report.append("")
    
    # Strategy comparison if available
    if integrated_data:
        report.append("## REMASKING STRATEGY PERFORMANCE")
        report.append("")
        
        strategy_accs = [(name, data['evaluation']['accuracy']) 
                        for name, data in integrated_data.items() 
                        if 'evaluation' in data]
        strategy_accs.sort(key=lambda x: x[1], reverse=True)
        
        report.append("Top 5 strategies by accuracy:")
        for name, acc in strategy_accs[:5]:
            report.append(f"  - {name[:50]}: {acc:.2f}%")
    
    report.append("")
    report.append("=" * 80)
    
    # Write report
    with open(save_path, 'w') as f:
        f.write('\n'.join(report))
    
    print(f"Saved summary report: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Post-hoc analysis of Winogrande blank uncertainty')
    
    parser.add_argument('--uncertainty_data', type=str, required=True,
                        help='Path to blank_uncertainty_data.json')
    parser.add_argument('--integrated_results', type=str, default=None,
                        help='Path to integrated_analysis.json (optional)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for plots (default: same as input)')
    
    args = parser.parse_args()
    
    # Setup paths
    uncertainty_path = Path(args.uncertainty_data)
    integrated_path = Path(args.integrated_results) if args.integrated_results else None
    output_dir = Path(args.output_dir) if args.output_dir else uncertainty_path.parent / 'posthoc'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Winogrande Blank Uncertainty - Post-hoc Analysis")
    print("=" * 60)
    print(f"Input: {uncertainty_path}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    
    # Load data
    data, integrated_data = load_data(uncertainty_path, integrated_path)
    print(f"Loaded {len(data)} samples")
    
    # Generate plots
    print("\nGenerating plots...")
    
    plot_epistemic_vs_aleatoric(data, output_dir / 'epistemic_vs_aleatoric.png')
    plot_option_probability_analysis(data, output_dir / 'option_probability_analysis.png')
    plot_uncertainty_calibration(data, output_dir / 'uncertainty_calibration.png')
    plot_detailed_distributions(data, output_dir / 'detailed_distributions.png')
    
    if integrated_data:
        plot_strategy_comparison(integrated_data, output_dir / 'strategy_comparison.png')
    
    # Create summary report
    create_summary_report(data, integrated_data, output_dir / 'summary_report.txt')
    
    print("\n" + "=" * 60)
    print(f"All outputs saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
