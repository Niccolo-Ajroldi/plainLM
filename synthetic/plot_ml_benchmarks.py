"""
Script to generate plots from ML benchmark results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_results(results_file="ml_benchmark_results/all_results.json"):
    """
    Load all results from JSON file.
    
    Returns:
        Dictionary containing all results
    """
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_best_loss_over_samples(results, output_dir="ml_plots"):
    """
    Plot cumulative best loss over optimizer samples for all methods and baselines.
    
    Args:
        results: Dictionary containing all results
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    optimizers = results['optimizers']
    benchmarks = results.get('benchmarks_list', [])
    
    if not benchmarks and optimizers:
        benchmarks = list(optimizers[0]['benchmarks'].keys())
    
    # Identify all unique methods and baselines
    method_names = set()

    for e in optimizers:
        method_names.add(e.get('opt'))
    
    method_names = sorted(list(method_names))
    
    # Determine number of iterations (use max optimizer_idx + 1)
    max_idx = max(e.get('optimizer_idx', 0) for e in optimizers)
    n_samples = max_idx + 1
    
    for benchmark_name in benchmarks:
        print(f"Plotting best-so-far for {benchmark_name}...")
        
        # Build cumulative best for each method
        all_best_sequences = {}
        
        for method_name in method_names:
            method_entries = [e for e in optimizers if e.get('opt') == method_name]
            method_entries.sort(key=lambda x: x.get('optimizer_idx', 0))
            
            final_losses = []
            for e in method_entries:
                bench = e.get('benchmarks', {}).get(benchmark_name, {})
                final_loss = bench.get('final_loss', float('inf'))
                final_losses.append(float(final_loss) if final_loss is not None else float('inf'))
            
            best_seq = []
            cur = float('inf')
            for v in final_losses:
                if v < cur:
                    cur = v
                best_seq.append(cur)
            
            all_best_sequences[method_name] = best_seq
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        iterations = list(range(1, n_samples + 1))
        
        # Define colors and styles
        method_colors = {'RS': "#21C9D8", 'RE': "#080CE7", 'Adam': '#9b1d1d', 'AdamW': '#ff7f0e', 'SGD': '#2ca02c'}
        method_markers = {'RS': 'o', 'RE': 's', 'Adam': 'D', 'AdamW': '^', 'SGD': 'v'}
        
        # Plot methods with solid lines
        for method_name in method_names:
            if method_name in all_best_sequences:
                ax.plot(iterations, all_best_sequences[method_name], 
                       marker=method_markers.get(method_name, 'o'), linewidth=2, markersize=6, 
                       label=f'{method_name} Best-So-Far', 
                       color=method_colors.get(method_name, '#000000'))
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Best Loss Found So Far', fontsize=12, fontweight='bold')
        ax.set_title(f'{benchmark_name.replace("_", " ").title()} - All Methods Comparison', 
                     fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        plot_file = output_dir / f"{benchmark_name}_all_methods_comparison.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved {plot_file}")


if __name__ == "__main__":
    print("Loading ML benchmark results...")
    results = load_results("ml_benchmark_results/all_results.json")
    print(f"Loaded {len(results['optimizers'])} optimizer entries")
    
    print("\nGenerating best-so-far comparison plots...")
    plot_best_loss_over_samples(results, output_dir="ml_plots")
    
    print("\nDone! All plots saved to ml_plots/")
