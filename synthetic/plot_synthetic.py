"""
Script to generate plots from benchmark results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict


def load_results(results_file="benchmark_results/all_results.json"):
    """
    Load all results from JSON file.
    
    Returns:
        Dictionary containing all results
    """
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_benchmark_results(results, output_dir="plots"):
    """
    Generate plots for each benchmark showing final loss vs number of dimensions.
    
    Args:
        results: Dictionary containing all results
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    optimizers = results['optimizers']
    
    if not optimizers:
        print("No results to plot!")
        return
    
    benchmarks = list(optimizers[0]['benchmarks'].keys())
    
    # Aggregate results by benchmark and dimension
    for benchmark_name in benchmarks:
        print(f"Plotting {benchmark_name}...")
        
        # Collect data: dimension -> list of final losses across optimizers
        dim_to_losses = defaultdict(list)
        
        for optimizer_result in optimizers:
            benchmark_data = optimizer_result['benchmarks'].get(benchmark_name, {})
            for n_dims_str, data in benchmark_data.items():
                n_dims = int(n_dims_str)
                final_loss = data.get('final_loss')
                
                # Skip invalid results
                if final_loss is None or np.isinf(final_loss) or np.isnan(final_loss):
                    continue
                    
                dim_to_losses[n_dims].append(final_loss)
        
        if not dim_to_losses:
            print(f"  No valid data for {benchmark_name}, skipping...")
            continue
        
        dimensions = sorted(dim_to_losses.keys())
        
        mean_losses = []
        std_losses = []
        min_losses = []
        max_losses = []
        
        for dim in dimensions:
            losses = dim_to_losses[dim]
            mean_losses.append(np.mean(losses))
            std_losses.append(np.std(losses))
            min_losses.append(np.min(losses))
            max_losses.append(np.max(losses))
        
        mean_losses = np.array(mean_losses)
        std_losses = np.array(std_losses)
        min_losses = np.array(min_losses)
        max_losses = np.array(max_losses)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.errorbar(dimensions, mean_losses, yerr=std_losses, 
                    marker='o', linewidth=2, markersize=8, 
                    capsize=5, capthick=2, label='Mean ± Std')
        
        ax.fill_between(dimensions, min_losses, max_losses, 
                        alpha=0.2, label='Min-Max Range')
        
        ax.set_xlabel('Number of Dimensions', fontsize=12, fontweight='bold')
        ax.set_ylabel('Final Loss', fontsize=12, fontweight='bold')
        ax.set_title(f'{benchmark_name.replace("_", " ").title()} Benchmark', 
                     fontsize=14, fontweight='bold')
        #ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        plot_file = output_dir / f"{benchmark_name}_vs_dims.png"
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved to {plot_file}")
    
    print(f"\nAll plots saved to {output_dir}/")


def plot_best_loss_over_optimizer_samples(results, n_dims, output_dir="plots"):
    """
    Plot the best loss value found so far at each optimizer sample (random search iteration).
    X-axis: optimizer sample index (1 to n_optimizer_samples)
    Y-axis: best loss found so far
    
    Args:
        results: Dictionary containing all results
        n_dims: Number of dimensions to plot
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    optimizers = results['optimizers']
    benchmarks = list(optimizers[0]['benchmarks'].keys())
    
    n_benchmarks = len(benchmarks)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, benchmark_name in enumerate(benchmarks):
        ax = axes[idx]
        
        best_losses = []
        current_best = float('inf')
        
        for optimizer_result in optimizers:
            benchmark_data = optimizer_result['benchmarks'].get(benchmark_name, {})
            dim_data = benchmark_data.get(str(n_dims), {})
            final_loss = dim_data.get('final_loss', float('inf'))
            
            if final_loss is not None and not np.isnan(final_loss):
                current_best = min(current_best, final_loss)
            
            best_losses.append(current_best)
        
        optimizer_indices = list(range(1, len(optimizers) + 1))
        ax.plot(optimizer_indices, best_losses, marker='o', linewidth=2, markersize=6)
        
        ax.set_xlabel('Iteration', fontsize=10, fontweight='bold')
        ax.set_ylabel('Best Loss Found', fontsize=10, fontweight='bold')
        ax.set_title(f'{benchmark_name.replace("_", " ").title()} ({n_dims}D)', 
                     fontsize=11, fontweight='bold')
        #ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_file = output_dir / f"best_loss_over_samples_{n_dims}d.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Best loss trajectory plot saved to {plot_file}")


def plot_best_loss_over_optimizer_samples_individual(results, n_dims, output_dir="plots"):
    """
    Create individual plots for each benchmark showing best loss over optimizer samples.
    
    Args:
        results: Dictionary containing all results
        n_dims: Number of dimensions to plot
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    optimizers = results['optimizers']
    benchmarks = list(optimizers[0]['benchmarks'].keys())
    
    for benchmark_name in benchmarks:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Collect best loss at each optimizer sample
        best_losses = []
        current_best = float('inf')
        
        for optimizer_result in optimizers:
            benchmark_data = optimizer_result['benchmarks'].get(benchmark_name, {})
            dim_data = benchmark_data.get(str(n_dims), {})
            final_loss = dim_data.get('final_loss', float('inf'))
            
            # Skip invalid results but keep tracking
            if final_loss is not None and not np.isnan(final_loss):
                current_best = min(current_best, final_loss)
            
            best_losses.append(current_best)
        
        optimizer_indices = list(range(1, len(optimizers) + 1))
        ax.plot(optimizer_indices, best_losses, marker='o', linewidth=2, markersize=8,
                color='#2E86AB', markerfacecolor='#A23B72')
        
        ax.set_xlabel('Iteration', 
                      fontsize=12, fontweight='bold')
        ax.set_ylabel('Best Loss Found So Far', fontsize=12, fontweight='bold')
        ax.set_title(f'{benchmark_name.replace("_", " ").title()} - Best Loss Trajectory ({n_dims}D)', 
                     fontsize=14, fontweight='bold')
        #ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = output_dir / f"{benchmark_name}_best_loss_trajectory_{n_dims}d.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  {benchmark_name} trajectory saved to {plot_file}")


def plot_baselines_vs_method(results, n_dims, output_dir="plots"):
    """
    Plot cumulative best loss over optimizer samples for the method (neps-derived optimizers)
    and baseline optimizers (Adam, AdamW, SGD) evaluated using the same sampled learning rates.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    optimizers = results['optimizers']
    benchmarks = list(optimizers[0]['benchmarks'].keys())

    baseline_names = [
        'Adam', 
        'AdamW', 
        'SGD'
        ]

    # Build method-only sequence (exclude baseline entries)
    method_entries = [e for e in optimizers if e.get('optimizer_type') != 'baseline']

    for benchmark_name in benchmarks:
        print(f"Plotting baselines vs method for {benchmark_name} ({n_dims}D)...")

        # Method final losses in sampling order
        method_final_losses = []
        for e in method_entries:
            bench = e.get('benchmarks', {}).get(benchmark_name, {})
            ent = bench.get(n_dims) if n_dims in bench else bench.get(str(n_dims))
            if ent and 'final_loss' in ent:
                method_final_losses.append(float(ent['final_loss']))
            else:
                method_final_losses.append(float('inf'))

        # cumulative best for method
        method_best = []
        cur = float('inf')
        for v in method_final_losses:
            if v < cur:
                cur = v
            method_best.append(cur)

        n_method_samples = len(method_final_losses)

        # Build baseline cumulative-best sequences aligned by optimizer_idx
        baseline_best_map = {}
        for bname in baseline_names:
            seq = []
            cur_b = float('inf')
            for idx in range(n_method_samples):
                found_val = None
                # search for baseline entry with matching optimizer_idx and baseline_name
                for be in optimizers:
                    if be.get('optimizer_type') == 'baseline' and be.get('baseline_name') == bname and be.get('optimizer_idx') == idx:
                        bench = be.get('benchmarks', {}).get(benchmark_name, {})
                        ent = bench.get(n_dims) if n_dims in bench else bench.get(str(n_dims))
                        if ent and 'final_loss' in ent:
                            found_val = float(ent['final_loss'])
                        break

                if found_val is None:
                    # missing baseline entry for this idx: treat as inf
                    val = float('inf')
                else:
                    val = found_val

                if val < cur_b:
                    cur_b = val
                seq.append(cur_b)

            baseline_best_map[bname] = seq

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        iterations = list(range(1, n_method_samples + 1))

        ax.plot(iterations, method_best, marker='o', linewidth=2, markersize=6, label='NePS Best-So-Far', color='#2E86AB')

        colors = {'Adam': "#b1171c", 'AdamW': '#ff7f0e', 'SGD': '#2ca02c'}
        markers = {'Adam': 'x', 'AdamW': 's', 'SGD': '^'}
        for bname in baseline_names:
            ax.plot(iterations, baseline_best_map[bname], marker=markers.get(bname), linewidth=1.5, markersize=5, label=f'{bname} Best-So-Far', color=colors.get(bname))

        ax.set_xlabel('Iteration', fontsize=12, fontweight='bold')
        ax.set_ylabel('Best Loss Found So Far', fontsize=12, fontweight='bold')
        ax.set_title(f'{benchmark_name.replace("_", " ").title()} - Method vs Baselines ({n_dims}D)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_xscale('log')
        #ax.set_yscale('log')

        plt.tight_layout()
        plot_file = output_dir / f"{benchmark_name}_method_vs_baselines_{n_dims}d.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  Saved {plot_file}")


if __name__ == "__main__":
    print("Loading results...")
    results = load_results("benchmark_results/all_results.json")
    print(f"Loaded {len(results['optimizers'])} optimizer samples")
    
    print("\nGenerating benchmark plots (final loss vs dimensions)...")
    plot_benchmark_results(results, output_dir="plots")
    
    print("\nGenerating best loss trajectory plots...")
    dimension_list = results['dimension_list']
    
    for n_dims in dimension_list:
        print(f"\n  Dimension {n_dims}:")
        plot_best_loss_over_optimizer_samples_individual(results, n_dims=n_dims, output_dir="plots")
        # Also plot baselines vs method for this dimensionality
        plot_baselines_vs_method(results, n_dims=n_dims, output_dir="plots")
    
    print(f"\nGenerating combined trajectory plot for 10D...")
    plot_best_loss_over_optimizer_samples(results, n_dims=10, output_dir="plots")
    
    print("\nDone!")