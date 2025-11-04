"""
A demo script to test the training pipeline with a small model and dataset.
"""

import torch
import numpy as np
import json
import os
from pathlib import Path
import neps
from neps.space.neps_spaces import sampling, neps_space
from nos_space import NOSSpace2
import argparse
import torch.optim as optim


def get_benchmark_function(name: str, n_dims: int):
    """
    Factory function to get the objective function, its gradient,
    and a recommended starting position.
    """
    if name == 'sphere':
        def fn(x):
            return np.sum(x**2)
        def grad(x):
            return 2 * x
        start_pos = np.full(n_dims, 5.0)

    elif name == 'ellipsoid':
        c = 10**np.linspace(0, 6, n_dims)
        def fn(x):
            return np.sum(c * (x**2))
        def grad(x):
            return 2 * c * x
        start_pos = np.full(n_dims, 5.0)

    elif name == 'rotated_quadratic':
        rng = np.random.RandomState(42)
        M = rng.rand(n_dims, n_dims)
        Q, _ = np.linalg.qr(M)
        D = np.diag(10**np.linspace(0, 3, n_dims))
        A = Q.T @ D @ Q
        b = rng.rand(n_dims)

        def fn(x):
            return 0.5 * x.T @ A @ x - b.T @ x
        def grad(x):
            return A @ x - b
        start_pos = np.zeros(n_dims)

    elif name == 'rosenbrock':
        def fn(x):
            return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        def grad(x):
            g = np.zeros_like(x)
            g[:-1] = -400 * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (1 - x[:-1])
            g[1:] += 200 * (x[1:] - x[:-1]**2)
            return g
        start_pos = np.zeros(n_dims)

    elif name == 'rastrigin':
        def fn(x):
            return 10 * n_dims + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
        def grad(x):
            return 2 * x + 20 * np.pi * np.sin(2 * np.pi * x)
        start_pos = np.random.RandomState(42).uniform(-5.12, 5.12, n_dims)

    elif name == 'ackley':
        def fn(x):
            term1 = -20 * np.exp(-0.2 * np.sqrt(np.mean(x**2)))
            term2 = -np.exp(np.mean(np.cos(2 * np.pi * x)))
            return term1 + term2 + 20 + np.e
        def grad(x):
            s = np.sqrt(np.mean(x**2))
            if s == 0:
                return np.zeros_like(x)
            g1 = -20 * np.exp(-0.2*s) * (-0.2 * (0.5/s) * (2*x/n_dims))
            g2 = -np.exp(np.mean(np.cos(2*np.pi*x))) * (-2*np.pi/n_dims) * np.sin(2*np.pi*x)
            return g1 + g2
        start_pos = np.random.RandomState(42).uniform(-32.7, 32.7, n_dims)

    else:
        raise ValueError(f"Unknown function: {name}")

    return fn, grad, start_pos


def evaluate_pipeline(optimizer_cls, learning_rate, benchmark_name='sphere', n_dims=10, n_steps=100):
    """
    Evaluate an optimizer on a synthetic benchmark function.
    
    Args:
        optimizer_cls: Optimizer class or creator
        learning_rate: Learning rate for the optimizer
        benchmark_name: Name of the benchmark function
        n_dims: Number of dimensions
        n_steps: Number of optimization steps
    
    Returns:
        Tuple of (final_loss, loss_history)
    """
    # Get benchmark function and starting position
    fn, grad, start_pos = get_benchmark_function(benchmark_name, n_dims)
    
    x = torch.nn.Parameter(torch.from_numpy(start_pos).float(), requires_grad=True)
    opt = optimizer_cls([x], lr=learning_rate)
    loss_history = []
    x_history = []

    for step in range(n_steps):
        opt.zero_grad()

        x_np = x.detach().numpy()
        loss_value = fn(x_np)
        loss_history.append(float(loss_value))
        # store position for possible 2D plotting
        if n_dims == 2:
            x_history.append(x_np.copy().tolist())

        grad_np = grad(x_np)
        x.grad = torch.from_numpy(grad_np).float()
        opt.step()

    final_x_np = x.detach().numpy()
    final_loss = fn(final_x_np)

    return final_loss, loss_history, (x_history if n_dims == 2 else None)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run optimizer benchmarks on synthetic functions.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=['sphere', 'ellipsoid', 'rotated_quadratic', 'rosenbrock', 'rastrigin', 'ackley'],
        help="List of benchmark names (space separated)."
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[10000],
        help="List of problem dimensions (space separated ints)."
    )
    parser.add_argument(
        "--n_optimizer_samples",
        type=int,
        default=1000,
        help="Number of different optimizers to sample (random search iterations)."
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=500,
        help="Number of optimization steps per optimizer."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="benchmark_results",
        help="Directory to write benchmark results."
    )

    args = parser.parse_args()

    benchmarks = args.benchmarks
    dimension_list = args.dimensions
    n_optimizer_samples = args.n_optimizer_samples
    n_steps = args.n_steps

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    # Store all results across all sampled optimizers
    all_results = {
        'n_optimizer_samples': n_optimizer_samples,
        'n_steps': n_steps,
        'dimension_list': dimension_list,
        'optimizers': []
    }

    space = NOSSpace2(max_lines=10)
    random_sampler = sampling.RandomSampler({})
    prior_sampler = sampling.PriorOrFallbackSampler(random_sampler)

    for optimizer_idx in range(n_optimizer_samples):
        print(f"\n{'='*60}")
        print(f"Optimizer Sample {optimizer_idx + 1}/{n_optimizer_samples}")
        print(f"{'='*60}")

        # Sample optimizer and learning rate
        resolved_pipeline, resolution_context = neps_space.resolve(space, domain_sampler=prior_sampler)
        optimizer_creator_object = resolved_pipeline.optimizer_cls
        learning_rate = resolved_pipeline.learning_rate
        optimizer_creator = neps_space.convert_operation_to_callable(optimizer_creator_object)

        try:
            lr_float = float(learning_rate)
        except Exception:
            lr_float = learning_rate

        print(f"\nLearning rate: {lr_float}")

        # Store results for this optimizer
        optimizer_results = {
            'optimizer_idx': optimizer_idx,
            'learning_rate': float(lr_float) if isinstance(lr_float, (int, float, np.floating)) else str(lr_float),
            'optimizer_info': str(optimizer_creator_object),
            'optimizer_type': 'neps',
            'benchmarks': {}
        }

        # Test on all benchmark functions and dimensions
        for benchmark_name in benchmarks:
            print(f"\n  Benchmark: {benchmark_name}")
            optimizer_results['benchmarks'][benchmark_name] = {}

            for n_dims in dimension_list:
                try:
                    def opt_wrapper(params, lr):
                        return optimizer_creator(params, lr=lr, variables=(1, 1))

                    final_loss, loss_history, x_history = evaluate_pipeline(
                        optimizer_cls=opt_wrapper,
                        learning_rate=learning_rate,
                        benchmark_name=benchmark_name,
                        n_dims=n_dims,
                        n_steps=n_steps
                    )

                    entry = {
                        'final_loss': float(final_loss),
                        'loss_history': [float(l) for l in loss_history]
                    }

                    if x_history is not None:
                        entry['x_history'] = x_history

                    optimizer_results['benchmarks'][benchmark_name][n_dims] = entry

                    # Compute best loss so far for this benchmark and dimensionality across previous optimizers + current
                    best_loss = float(final_loss)
                    for prev in all_results['optimizers']:
                        prev_benchmarks = prev.get('benchmarks', {})
                        prev_bench = prev_benchmarks.get(benchmark_name, {})
                        prev_entry = prev_bench.get(n_dims)
                        if prev_entry and 'final_loss' in prev_entry:
                            try:
                                prev_loss = float(prev_entry['final_loss'])
                            except Exception:
                                continue
                            if prev_loss < best_loss:
                                best_loss = prev_loss

                    print(f"    n_dims={n_dims:3d}: final_loss={final_loss:.6e}  best_so_far={best_loss:.6e}")

                except Exception as e:
                    print(f"    n_dims={n_dims:3d}: ERROR - {str(e)}")
                    optimizer_results['benchmarks'][benchmark_name][n_dims] = {
                        'final_loss': float('inf'),
                        'loss_history': [],
                        'error': str(e)
                    }

        all_results['optimizers'].append(optimizer_results)

        # --- Baseline optimizers (PyTorch) evaluated with the same sampled learning rate ---
        # We'll run Adam, AdamW, and SGD as baseline optimizers and store their results
        baseline_optimizers = {
            'Adam': lambda params, lr: optim.Adam(params, lr=lr),
            'AdamW': lambda params, lr: optim.AdamW(params, lr=lr),
            'SGD': lambda params, lr: optim.SGD(params, lr=lr)
        }

        for base_name, base_creator in baseline_optimizers.items():
            baseline_results = {
                'optimizer_idx': optimizer_idx,
                'learning_rate': float(lr_float) if isinstance(lr_float, (int, float, np.floating)) else str(lr_float),
                'optimizer_info': f'{base_name} (baseline)',
                'optimizer_type': 'baseline',
                'baseline_name': base_name,
                'benchmarks': {}
            }

            for benchmark_name in benchmarks:
                baseline_results['benchmarks'][benchmark_name] = {}

                for n_dims in dimension_list:
                    try:
                        # Pass a callable that matches evaluate_pipeline expectation: (params, lr) -> optimizer
                        final_loss, loss_history, x_history = evaluate_pipeline(
                            optimizer_cls=lambda params, lr: base_creator(params, lr=lr),
                            learning_rate=learning_rate,
                            benchmark_name=benchmark_name,
                            n_dims=n_dims,
                            n_steps=n_steps
                        )

                        entry = {
                            'final_loss': float(final_loss),
                            'loss_history': [float(l) for l in loss_history]
                        }
                        if x_history is not None:
                            entry['x_history'] = x_history

                        baseline_results['benchmarks'][benchmark_name][n_dims] = entry

                        print(f"    [baseline:{base_name}] n_dims={n_dims:3d}: final_loss={final_loss:.6e}")

                    except Exception as e:
                        print(f"    [baseline:{base_name}] n_dims={n_dims:3d}: ERROR - {str(e)}")
                        baseline_results['benchmarks'][benchmark_name][n_dims] = {
                            'final_loss': float('inf'),
                            'loss_history': [],
                            'error': str(e)
                        }

            all_results['optimizers'].append(baseline_results)

    # Save all results to a single JSON file
    output_file = results_dir / "all_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*60}")
    print(f"All results saved to {output_file}")


    # Compute and print the overall best optimizer entry (smallest final_loss)
    best = {'loss': float('inf'), 'optimizer': None, 'benchmark': None, 'n_dims': None}
    for opt in all_results.get('optimizers', []):
        bench_map = opt.get('benchmarks', {})
        for bench_name, bench in bench_map.items():
            for dim_k, entry in bench.items():
                try:
                    val = float(entry.get('final_loss', float('inf')))
                except Exception:
                    continue
                if val < best['loss']:
                    best['loss'] = val
                    best['optimizer'] = opt
                    best['benchmark'] = bench_name
                    best['n_dims'] = dim_k

    if best['optimizer'] is not None:
        opt = best['optimizer']
        info = opt.get('optimizer_info')
        otype = opt.get('optimizer_type', 'unknown')
        idx = opt.get('optimizer_idx')
        bname = opt.get('baseline_name') if otype == 'baseline' else None
        print(f"Best optimizer found -> final_loss={best['loss']:.6e}")
        print(f"  optimizer_idx: {idx}")
        print(f"  optimizer_type: {otype}")
        if bname:
            print(f"  baseline_name: {bname}")
        print(f"  optimizer_info: {info}")
        print(f"  benchmark: {best['benchmark']}")
        print(f"  n_dims: {best['n_dims']}")
    else:
        print('No valid optimizer results found to determine best optimizer')
    print(f"{'='*60}")