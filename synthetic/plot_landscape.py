"""
Plot 2D function landscape with optimization trajectories for the best found method and AdamW baseline.

This script expects `benchmark` to be one of the synthetic benchmarks used in `nos_synthetic.py` and
`n_dims` to be 2.

Usage (example):
    python plot_landscape.py --results_file benchmark_results/all_results.json --benchmark rosenbrock --n_dims 2 --output_dir plots

"""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from nos_synthetic import get_benchmark_function


def load_results(results_file: str):
    with open(results_file, 'r') as f:
        return json.load(f)


def find_best_method_and_baseline(results, benchmark_name, n_dims, baseline_name='AdamW'):
    optimizers = results['optimizers']

    best_entry = None
    best_loss = float('inf')

    # find best among method entries (non-baseline)
    for e in optimizers:
        if e.get('optimizer_type') == 'baseline':
            continue
        bench = e.get('benchmarks', {}).get(benchmark_name, {})
        ent = bench.get(n_dims) if n_dims in bench else bench.get(str(n_dims))
        if ent and 'final_loss' in ent:
            try:
                val = float(ent['final_loss'])
            except Exception:
                continue
            if val < best_loss:
                best_loss = val
                best_entry = e

    if best_entry is None:
        raise RuntimeError('No method (neps) entries found with valid final_loss for this benchmark/dim')

    # find AdamW baseline entry with same optimizer_idx
    opt_idx = best_entry.get('optimizer_idx')
    baseline_entry = None
    for e in optimizers:
        if e.get('optimizer_type') == 'baseline' and e.get('baseline_name') == baseline_name and e.get('optimizer_idx') == opt_idx:
            baseline_entry = e
            break

    return best_entry, baseline_entry


def plot_2d_landscape_and_trajectories(fn, method_traj, base_traj, out_file, title=None):
    # Combine trajectories to determine plotting range
    all_points = np.array(method_traj + base_traj)
    xmin, ymin = all_points.min(axis=0) - 0.5
    xmax, ymax = all_points.max(axis=0) + 0.5

    # grid
    nx, ny = 200, 200
    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)
    X, Y = np.meshgrid(xs, ys)

    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = np.array([fn(p) for p in pts]).reshape(X.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    levels = 60
    cf = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.9)
    fig.colorbar(cf, ax=ax, label='Function value')

    # plot trajectories
    method_traj = np.array(method_traj)
    base_traj = np.array(base_traj)

    ax.plot(method_traj[:, 0], method_traj[:, 1], '-o', color='#FF3366', label='Best Method', linewidth=2, markersize=4)
    ax.plot(base_traj[:, 0], base_traj[:, 1], '-s', color='#1f77b4', label='AdamW (baseline)', linewidth=2, markersize=4)

    # mark start and end
    ax.scatter(method_traj[0, 0], method_traj[0, 1], color='#FF3366', marker='D', s=60, label='Method start')
    ax.scatter(method_traj[-1, 0], method_traj[-1, 1], color='#FF3366', marker='X', s=60, label='Method end')
    ax.scatter(base_traj[0, 0], base_traj[0, 1], color='#1f77b4', marker='d', s=60, label='AdamW start')
    ax.scatter(base_traj[-1, 0], base_traj[-1, 1], color='#1f77b4', marker='x', s=60, label='AdamW end')

    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    if title:
        ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_file', type=str, default='benchmark_results/all_results.json')
    parser.add_argument('--benchmark', type=str, required=True)
    parser.add_argument('--n_dims', type=int, default=2)
    parser.add_argument('--output_dir', type=str, default='plots')
    args = parser.parse_args()

    results = load_results(args.results_file)

    if args.n_dims != 2:
        raise ValueError('This script currently supports only 2D (n_dims==2)')

    best_entry, baseline_entry = find_best_method_and_baseline(results, args.benchmark, args.n_dims, baseline_name='AdamW')

    # extract x_history
    bench_best = best_entry.get('benchmarks', {}).get(args.benchmark, {})
    ent_best = bench_best.get(args.n_dims) if args.n_dims in bench_best else bench_best.get(str(args.n_dims))
    xhist_best = ent_best.get('x_history') if ent_best else None

    xhist_base = None
    if baseline_entry:
        bench_base = baseline_entry.get('benchmarks', {}).get(args.benchmark, {})
        ent_base = bench_base.get(args.n_dims) if args.n_dims in bench_base else bench_base.get(str(args.n_dims))
        xhist_base = ent_base.get('x_history') if ent_base else None

    if xhist_best is None:
        raise RuntimeError('Best method entry does not contain x_history; ensure nos_synthetic recorded positions for 2D runs')
    if xhist_base is None:
        raise RuntimeError('AdamW baseline entry does not contain x_history; ensure nos_synthetic recorded positions for 2D runs')

    # get function for plotting
    fn, grad, start_pos = get_benchmark_function(args.benchmark, args.n_dims)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{args.benchmark}_best_vs_AdamW_2d.png"

    plot_2d_landscape_and_trajectories(fn, xhist_best, xhist_base, out_file, title=f"{args.benchmark} - Best Method vs AdamW")

    print(f"Saved landscape trajectory plot to {out_file}")
