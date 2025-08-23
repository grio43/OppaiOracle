 #!/usr/bin/env python3
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns

def load_results(results_file):
    """Load evaluation results from JSON file"""
    with open(results_file, 'r') as f:
        data = json.load(f)
    return data

def plot_metrics_distribution(results_data, output_dir):
    """Plot distribution of precision, recall, and F1 scores"""
    detailed = results_data.get('detailed_results', [])
    
    if not detailed:
        print("No detailed results to plot")
        return
    
    # Extract metrics
    precisions = [r['metrics']['precision'] for r in detailed]
    recalls = [r['metrics']['recall'] for r in detailed]
    f1_scores = [r['metrics']['f1'] for r in detailed]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Histogram of metrics
    axes[0, 0].hist(precisions, bins=20, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Precision Distribution')
    axes[0, 0].set_xlabel('Precision')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].axvline(np.mean(precisions), color='red', linestyle='--', label=f'Mean: {np.mean(precisions):.3f}')
    axes[0, 0].legend()
    
    axes[0, 1].hist(recalls, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_title('Recall Distribution')
    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].axvline(np.mean(recalls), color='red', linestyle='--', label=f'Mean: {np.mean(recalls):.3f}')
    axes[0, 1].legend()
    
    axes[1, 0].hist(f1_scores, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('F1 Score Distribution')
    axes[1, 0].set_xlabel('F1 Score')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].axvline(np.mean(f1_scores), color='red', linestyle='--', label=f'Mean: {np.mean(f1_scores):.3f}')
    axes[1, 0].legend()
    
    # Scatter plot of precision vs recall
    axes[1, 1].scatter(precisions, recalls, alpha=0.5)
    axes[1, 1].set_title('Precision vs Recall')
    axes[1, 1].set_xlabel('Precision')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add diagonal line for F1 contours
    x = np.linspace(0, 1, 100)
    for f1 in [0.3, 0.5, 0.7, 0.9]:
        y = (f1 * x) / (2 * x - f1)
        axes[1, 1].plot(x, y, '--', alpha=0.3, label=f'F1={f1}')
    axes[1, 1].legend(loc='best', fontsize='small')
    
    plt.tight_layout()
    output_file = output_dir / 'metrics_distribution.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved metrics distribution plot to {output_file}")
    plt.close()

def plot_tag_performance(results_data, output_dir, top_n=20):
    """Plot tag performance (best and worst)"""
    summary = results_data.get('summary', {})
    tag_perf = summary.get('tag_performance', {})
    
    best_tags = tag_perf.get('best_tags', [])[:top_n]
    worst_tags = tag_perf.get('worst_tags', [])[:top_n]
    
    if not best_tags and not worst_tags:
        print("No tag performance data to plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Best performing tags
    if best_tags:
        tags = [t['tag'][:20] for t in best_tags]  # Truncate long tag names
        f1_scores = [t['f1'] for t in best_tags]
        supports = [t['support'] for t in best_tags]
        
        bars1 = ax1.barh(tags, f1_scores, color='green', alpha=0.7)
        ax1.set_xlabel('F1 Score')
        ax1.set_title(f'Top {len(best_tags)} Best Performing Tags')
        ax1.set_xlim([0, 1])
        
        # Add support numbers
        for i, (bar, support) in enumerate(zip(bars1, supports)):
            ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'n={support}', va='center', fontsize=8)
    
    # Worst performing tags
    if worst_tags:
        tags = [t['tag'][:20] for t in worst_tags]
        f1_scores = [t['f1'] for t in worst_tags]
        supports = [t['support'] for t in worst_tags]
        
        bars2 = ax2.barh(tags, f1_scores, color='red', alpha=0.7)
        ax2.set_xlabel('F1 Score')
        ax2.set_title(f'Top {len(worst_tags)} Worst Performing Tags')
        ax2.set_xlim([0, 1])
        
        # Add support numbers
        for i, (bar, support) in enumerate(zip(bars2, supports)):
            ax2.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'n={support}', va='center', fontsize=8)
    
    plt.tight_layout()
    output_file = output_dir / 'tag_performance.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved tag performance plot to {output_file}")
    plt.close()

def plot_tag_counts(results_data, output_dir):
    """Plot distribution of number of tags per image"""
    detailed = results_data.get('detailed_results', [])
    
    if not detailed:
        print("No detailed results for tag count analysis")
        return
    
    gt_counts = [r['num_gt_tags'] for r in detailed]
    pred_counts = [r['num_pred_tags'] for r in detailed]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ground truth tag counts
    ax1.hist(gt_counts, bins=range(0, max(gt_counts)+2), alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title('Ground Truth Tags per Image')
    ax1.set_xlabel('Number of Tags')
    ax1.set_ylabel('Count')
    ax1.axvline(np.mean(gt_counts), color='red', linestyle='--', label=f'Mean: {np.mean(gt_counts):.1f}')
    ax1.legend()
    
    # Predicted tag counts
    ax2.hist(pred_counts, bins=range(0, max(pred_counts)+2), alpha=0.7, color='green', edgecolor='black')
    ax2.set_title('Predicted Tags per Image')
    ax2.set_xlabel('Number of Tags')
    ax2.set_ylabel('Count')
    ax2.axvline(np.mean(pred_counts), color='red', linestyle='--', label=f'Mean: {np.mean(pred_counts):.1f}')
    ax2.legend()
    
    plt.tight_layout()
    output_file = output_dir / 'tag_counts.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved tag count distribution to {output_file}")
    plt.close()

def generate_performance_report(results_data, output_dir):
    """Generate a detailed text report"""
    summary = results_data.get('summary', {})
    detailed = results_data.get('detailed_results', [])
    
    report_lines = []
    report_lines.append("="*60)
    report_lines.append("MODEL EVALUATION REPORT")
    report_lines.append("="*60)
    report_lines.append("")
    
    # Overall statistics
    report_lines.append("OVERALL STATISTICS:")
    report_lines.append(f"  Total files: {summary.get('total_files', 0)}")
    report_lines.append(f"  Processed files: {summary.get('processed_files', 0)}")
    report_lines.append(f"  Failed files: {summary.get('failed_files', 0)}")
    report_lines.append(f"  Threshold: {summary.get('threshold', 0.5)}")
    report_lines.append("")
    
    # Average metrics
    avg_metrics = summary.get('average_metrics', {})
    report_lines.append("AVERAGE METRICS:")
    report_lines.append(f"  Precision: {avg_metrics.get('precision', 0):.4f} ± {avg_metrics.get('precision_std', 0):.4f}")
    report_lines.append(f"  Recall: {avg_metrics.get('recall', 0):.4f} ± {avg_metrics.get('recall_std', 0):.4f}")
    report_lines.append(f"  F1 Score: {avg_metrics.get('f1', 0):.4f} ± {avg_metrics.get('f1_std', 0):.4f}")
    report_lines.append("")
    
    # Performance distribution
    if detailed:
        f1_scores = [r['metrics']['f1'] for r in detailed]
        report_lines.append("F1 SCORE DISTRIBUTION:")
        report_lines.append(f"  Excellent (F1 > 0.8): {sum(1 for s in f1_scores if s > 0.8)} images")
        report_lines.append(f"  Good (0.6 < F1 <= 0.8): {sum(1 for s in f1_scores if 0.6 < s <= 0.8)} images")
        report_lines.append(f"  Fair (0.4 < F1 <= 0.6): {sum(1 for s in f1_scores if 0.4 < s <= 0.6)} images")
        report_lines.append(f"  Poor (F1 <= 0.4): {sum(1 for s in f1_scores if s <= 0.4)} images")
        report_lines.append("")
    
    # Best performing files
    if detailed:
        sorted_by_f1 = sorted(detailed, key=lambda x: x['metrics']['f1'], reverse=True)
        report_lines.append("TOP 10 BEST PERFORMING IMAGES:")
        for i, result in enumerate(sorted_by_f1[:10], 1):
            report_lines.append(f"  {i}. {result['filename']}: F1={result['metrics']['f1']:.3f}")
        report_lines.append("")
        
        report_lines.append("TOP 10 WORST PERFORMING IMAGES:")
        for i, result in enumerate(sorted_by_f1[-10:], 1):
            report_lines.append(f"  {i}. {result['filename']}: F1={result['metrics']['f1']:.3f}")
        report_lines.append("")
    
    # Tag performance
    tag_perf = summary.get('tag_performance', {})
    best_tags = tag_perf.get('best_tags', [])[:10]
    worst_tags = tag_perf.get('worst_tags', [])[:10]
    
    if best_tags:
        report_lines.append("TOP 10 BEST PERFORMING TAGS:")
        for i, tag in enumerate(best_tags, 1):
            report_lines.append(f"  {i}. {tag['tag']}: F1={tag['f1']:.3f}, Support={tag['support']}")
        report_lines.append("")
    
    if worst_tags:
        report_lines.append("TOP 10 WORST PERFORMING TAGS:")
        for i, tag in enumerate(worst_tags, 1):
            report_lines.append(f"  {i}. {tag['tag']}: F1={tag['f1']:.3f}, Support={tag['support']}")
        report_lines.append("")
    
    # Save report
    report_file = output_dir / 'evaluation_report.txt'
    with open(report_file, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Saved detailed report to {report_file}")
    
    # Also print to console
    print('\n'.join(report_lines))

def main():
    parser = argparse.ArgumentParser(description="Visualize model evaluation results")
    parser.add_argument("--results", default="evaluation_results.json", 
                       help="Path to evaluation results JSON file")
    parser.add_argument("--output_dir", default="./evaluation_plots",
                       help="Directory to save visualization outputs")
    parser.add_argument("--top_n", type=int, default=20,
                       help="Number of top/bottom tags to show")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load results
    print(f"Loading results from {args.results}")
    results_data = load_results(args.results)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_metrics_distribution(results_data, output_dir)
    plot_tag_performance(results_data, output_dir, args.top_n)
    plot_tag_counts(results_data, output_dir)
    
    # Generate text report
    print("\nGenerating performance report...")
    generate_performance_report(results_data, output_dir)
    
    print(f"\nAll outputs saved to {output_dir}")

if __name__ == "__main__":
    main()