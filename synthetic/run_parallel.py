#!/usr/bin/env python3
"""
Run synthetic benchmarks in parallel across dimensions and benchmark functions on a local (CPU) machine.

- Samples optimizers from NOSSpace2 (via neps)
- Evaluates all (benchmark, dimension) pairs in parallel with bounded workers
- Aggregates results into a single JSON file compatible with `nos_synthetic.py` outputs

Example:
    python synthetic/run_parallel.py \
        --benchmarks all \
        --dims 2,5,10,20,50 \
        --n-optimizers 2 \
        --n-steps 200 \
        --max-workers 4

Notes:
- To avoid CPU over-subscription, we set OMP_NUM_THREADS=1 for workers by default.
- Very large dimensions (e.g., 100k) can be slow on laptops; use --dims to select practical sizes.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import neps
from neps.space.neps_spaces import sampling, neps_space
from nos_space import NOSSpace2, resolve_expression


# ------------------------------
# Benchmarks and evaluation core
# ------------------------------

def get_benchmark_function(name: str, n_dims: int):
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


def evaluate_pipeline(
    optimizer_cls: Callable,
    learning_rate: float,
    benchmark_name: str,
    n_dims: int,
    n_steps: int,
) -> Tuple[float, List[float]]:
    fn, grad, start_pos = get_benchmark_function(benchmark_name, n_dims)
    x = torch.nn.Parameter(torch.from_numpy(start_pos).float(), requires_grad=True)
    opt = optimizer_cls([x], lr=learning_rate)
    loss_history: List[float] = []

    for _ in range(n_steps):
        opt.zero_grad()
        x_np = x.detach().numpy()
        loss_value = float(fn(x_np))
        loss_history.append(loss_value)
        grad_np = grad(x_np)
        x.grad = torch.from_numpy(grad_np).float()
        opt.step()

    final_x_np = x.detach().numpy()
    final_loss = float(fn(final_x_np))
    return final_loss, loss_history


@dataclass
class Task:
    benchmark: str
    n_dims: int


def _make_optimizer_cls_from_lines(lines):
    """Build an Optimizer class that applies the provided lines logic.

    This mirrors NOSSpace2.create_optimizer but avoids capturing a local class
    from another process, so it's safe to construct inside workers.
    """
    class CustomOptimizer(torch.optim.Optimizer):
        def __init__(self, params, lr=1e-3, variables=(0.1, 0.1)):
            defaults = dict(lr=lr, vars=variables)
            super().__init__(params, defaults)
            self.lr = lr
            for group in self.param_groups:
                for p in group.get("params", []):
                    state = self.state.setdefault(p, {})
                    if "v1" not in state:
                        state["v1"] = torch.ones_like(p.data) * variables[0]
                    if "v2" not in state:
                        state["v2"] = torch.ones_like(p.data) * variables[1]
                    if "u" not in state:
                        state["u"] = torch.zeros_like(p.data)

        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()

            for group in self.param_groups:
                for p in group.get("params", []):
                    d_p = torch.zeros_like(p.data) if p.grad is None else p.grad
                    state = self.state.setdefault(p, {})

                    def as_tensor(x):
                        if isinstance(x, torch.Tensor):
                            try:
                                return x.to(device=p.data.device, dtype=p.data.dtype)
                            except Exception:
                                return x
                        else:
                            return torch.tensor(x, dtype=p.data.dtype, device=p.data.device)

                    var_dict = {
                        "g": d_p,
                        "w": p.data,
                        "u": state.get("u", torch.zeros_like(p.data)),
                        "v1": state.get("v1", torch.zeros_like(p.data)),
                        "v2": state.get("v2", torch.zeros_like(p.data)),
                    }

                    for line in lines:
                        target_var, expr = line
                        result = resolve_expression(expr, var_dict)
                        result = as_tensor(result)
                        state[target_var] = result
                        var_dict[target_var] = result

                    p.data = p.data - self.lr * state["u"]

            return loss

    return CustomOptimizer


def _worker_task(task: Task, lr: float, lines, n_steps: int) -> Tuple[str, int, Dict]:
    # Each worker should avoid BLAS oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    OptimizerCls = _make_optimizer_cls_from_lines(lines)

    def opt_wrapper(params, lr):
        return OptimizerCls(params, lr=lr, variables=(1, 1))

    try:
        final_loss, loss_history = evaluate_pipeline(
            optimizer_cls=opt_wrapper,
            learning_rate=lr,
            benchmark_name=task.benchmark,
            n_dims=task.n_dims,
            n_steps=n_steps,
        )
        payload = {
            'final_loss': float(final_loss),
            'loss_history': [float(l) for l in loss_history],
        }
    except Exception as e:
        payload = {
            'final_loss': float('inf'),
            'loss_history': [],
            'error': str(e),
        }
    return task.benchmark, task.n_dims, payload


# ------------------------------
# Orchestrator
# ------------------------------

def run_parallel(
    benchmarks: List[str],
    dims: List[int],
    n_optimizer_samples: int,
    n_steps: int,
    max_workers: int,
    output_dir: Path,
    output_name: str | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bound workers to at least 1
    max_workers = max(1, max_workers)

    # neps optimizer sampling space
    space = NOSSpace2(max_lines=10)
    random_sampler = sampling.RandomSampler({})
    prior_sampler = sampling.PriorOrFallbackSampler(random_sampler)

    all_results = {
        'n_optimizer_samples': n_optimizer_samples,
        'n_steps': n_steps,
        'dimension_list': dims,
        'benchmarks': benchmarks,
        'optimizers': [],
    }

    # Single timestamp for all outputs
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    if not output_name:
        output_name = f"all_results_parallel_{ts}.json"
    output_file = output_dir / output_name

    for optimizer_idx in range(n_optimizer_samples):
        print("=" * 60)
        print(f"Optimizer Sample {optimizer_idx + 1}/{n_optimizer_samples}")
        print("=" * 60)

        # Sample optimizer and learning rate via neps
        resolved_pipeline, resolution_context = neps_space.resolve(space, domain_sampler=prior_sampler)
        optimizer_creator_object = resolved_pipeline.optimizer_cls
        learning_rate = float(resolved_pipeline.learning_rate)
        optimizer_creator = neps_space.convert_operation_to_callable(optimizer_creator_object)
        # Extract a serializable representation of the optimizer lines to avoid pickling the class
        try:
            # Instantiate a temporary optimizer to read lines; empty param list is fine
            temp_opt = optimizer_creator(params=[], lr=learning_rate, variables=(1, 1))
            lines = temp_opt.get_lines()
        except Exception:
            # Fallback: no lines accessor; treat as no-op single-line update (unlikely with NOSSpace2)
            lines = (('u', ('mul', 'g', 0.0)),)
        print(f"Learning rate: {learning_rate}")

        # Prepare tasks
        tasks = [Task(b, d) for b in benchmarks for d in dims]
        optimizer_results = {
            'optimizer_idx': optimizer_idx,
            'learning_rate': learning_rate,
            'optimizer_info': str(optimizer_creator_object),
            'benchmarks': {b: {} for b in benchmarks},
        }

        # Run tasks in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [
                ex.submit(_worker_task, t, learning_rate, lines, n_steps)
                for t in tasks
            ]
            for fut in as_completed(futures):
                bench, dim, payload = fut.result()
                optimizer_results['benchmarks'][bench][dim] = payload
                print(f"  {bench:18s} n_dims={dim:6d}: final_loss={payload.get('final_loss', float('inf')):.6e}")

        all_results['optimizers'].append(optimizer_results)

        # Flush to disk incrementally for safety
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Partial results saved to {output_file}")

    print("=" * 60)
    print(f"All results saved to {output_file}")
    print("=" * 60)


# ------------------------------
# CLI
# ------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--benchmarks', type=str, default='all',
                        help='Comma-separated list or "all". Available: sphere,ellipsoid,rotated_quadratic,rosenbrock,rastrigin,ackley')
    parser.add_argument('--dims', type=str, default='2,5,10,20,50',
                        help='Comma-separated list of dimensions (e.g., 2,5,10). Large dims can be slow.')
    parser.add_argument('--n-optimizers', type=int, default=1, help='Number of optimizer samples to evaluate')
    parser.add_argument('--n-steps', type=int, default=200, help='Optimization steps per run')
    parser.add_argument('--max-workers', type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help='Max parallel workers (processes). Default: cpu_count - 1')
    parser.add_argument('--output-dir', type=str, default='benchmark_results', help='Directory to write JSON output')
    parser.add_argument('--output-name', type=str, default=None, help='Optional JSON filename; default includes timestamp')
    return parser.parse_args()


def main():
    # Reduce thread contention globally
    os.environ.setdefault('OMP_NUM_THREADS', '1')

    args = parse_args()

    all_benchmarks = ['sphere', 'ellipsoid', 'rotated_quadratic', 'rosenbrock', 'rastrigin', 'ackley']
    if args.benchmarks.strip().lower() == 'all':
        benchmarks = all_benchmarks
    else:
        benchmarks = [b.strip() for b in args.benchmarks.split(',') if b.strip()]
        unknown = set(benchmarks) - set(all_benchmarks)
        if unknown:
            raise SystemExit(f"Unknown benchmark(s): {sorted(unknown)}")

    dims: List[int] = []
    for tok in args.dims.split(','):
        tok = tok.strip()
        if not tok:
            continue
        try:
            dims.append(int(tok))
        except ValueError:
            raise SystemExit(f"Invalid dimension value: {tok}")
    if not dims:
        raise SystemExit("No dimensions provided")

    run_parallel(
        benchmarks=benchmarks,
        dims=dims,
        n_optimizer_samples=args.n_optimizers,
        n_steps=args.n_steps,
        max_workers=args.max_workers,
        output_dir=Path(args.output_dir),
        output_name=args.output_name,
    )


if __name__ == '__main__':
    main()
