"""
ML-based optimizer benchmarking script.
Tests optimizers on realistic ML tasks: linear regression, logistic regression,
noisy regression, XOR MLP, and tiny autoencoder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
from pathlib import Path
import neps
from neps import Trial
from neps.state import trial
from neps.space.neps_spaces import sampling, neps_space
from nos_space import NOSSpaceMaxLines
import torch.optim as optim
from functools import partial


# ----------------------------
# ML Benchmark Definitions
# ----------------------------

def get_ml_benchmark_function(name: str, seed: int = 0, n_samples: int = 100, n_dims: int = 5):
    """
    Factory function to get ML benchmark setup.
    
    Args:
        name: Name of the benchmark
        seed: Random seed for reproducibility
        n_samples: Number of data samples (for regression/classification tasks)
        n_dims: Number of input dimensions/features
    
    Returns:
        Tuple of (loss_fn, params_list, description)
        where loss_fn is a callable that returns a loss tensor,
        and params_list is a list of parameters to optimize.
    """
    torch.manual_seed(seed)
    
    if name == 'linear_regression':
        n, d = n_samples, n_dims
        X = torch.randn(n, d)
        true_w = torch.randn(d)
        y = X @ true_w + 0.1 * torch.randn(n)
        w = torch.randn(d, requires_grad=True)
        
        def loss_fn():
            return torch.mean((X @ w - y)**2)
        
        return loss_fn, [w], f"Linear Regression (Convex, n={n}, d={d})"
    
    elif name == 'logistic_regression':
        n, d = n_samples, n_dims
        X = torch.randn(n, d)
        true_w = torch.randn(d)
        logits = X @ true_w
        y = torch.bernoulli(torch.sigmoid(logits))
        w = torch.randn(d, requires_grad=True)
        
        def loss_fn():
            return F.binary_cross_entropy_with_logits(X @ w, y)
        
        return loss_fn, [w], f"Logistic Regression (Mildly Non-Convex, n={n}, d={d})"
    
    elif name == 'noisy_regression':
        n, d = n_samples, n_dims
        X = torch.randn(n, d)
        true_w = torch.randn(d)
        y = X @ true_w
        w = torch.randn(d, requires_grad=True)
        
        def loss_fn():
            noise = 0.1 * torch.randn(())
            return torch.mean((X @ w - y)**2) + noise
        
        return loss_fn, [w], f"Noisy Regression (Convex + Noise, n={n}, d={d})"
    
    elif name == 'xor_mlp':
        # XOR problem is fixed at 2D input, but we can scale hidden layer with n_dims
        X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
        y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)
        
        # Scale hidden layer size with n_dims parameter
        hidden_size = max(4, n_dims)
        model = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        def loss_fn():
            logits = model(X)
            return F.binary_cross_entropy_with_logits(logits, y)
        
        return loss_fn, list(model.parameters()), f"XOR MLP (Nonlinear, hidden={hidden_size})"
    
    elif name == 'autoencoder':
        n, d = n_samples, n_dims
        X = torch.randn(n, d)
        
        # Scale latent dimension with problem size
        dim_latent = max(2, d // 3)
        hidden_size = max(4, (d + dim_latent) // 2)
        
        model = nn.Sequential(
            nn.Linear(d, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, dim_latent),
            nn.ReLU(),
            nn.Linear(dim_latent, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, d)
        )
        
        def loss_fn():
            recon = model(X)
            return torch.mean((recon - X)**2)
        
        return loss_fn, list(model.parameters()), f"Tiny Autoencoder (Deep & Nonlinear, n={n}, d={d}, latent={dim_latent})"
    
    else:
        raise ValueError(f"Unknown benchmark: {name}")


def evaluate_ml_pipeline(optimizer_cls, learning_rate, benchmark_name='linear_regression', 
                        n_steps=500, seed=0, n_samples=100, n_dims=5, weight_decay=0.0):
    """
    Evaluate an optimizer on an ML benchmark.
    
    Args:
        optimizer_cls: Optimizer class or creator
        learning_rate: Learning rate for the optimizer
        benchmark_name: Name of the ML benchmark
        n_steps: Number of optimization steps
        seed: Random seed for reproducibility
        n_samples: Number of data samples
        n_dims: Number of input dimensions
        weight_decay: Weight decay (L2 regularization) coefficient
    
    Returns:
        Tuple of (final_loss, loss_history)
    """
    loss_fn, params, description = get_ml_benchmark_function(benchmark_name, seed=seed, 
                                                             n_samples=n_samples, n_dims=n_dims)
    
    # Create optimizer with weight_decay
    opt = optimizer_cls(params, lr=learning_rate, weight_decay=weight_decay)
    
    loss_history = []
    
    for step in range(n_steps):
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        opt.step()
        loss_history.append(float(loss.item()))
    
    # Final loss
    with torch.no_grad():
        final_loss = float(loss_fn().item())
    
    return final_loss, loss_history

def run_optimizer_on_benchmarks(opt_info):
    """
    Helper function to run an optimizer on all benchmarks.
    
    Args:
    opt_info: Dict with keys 'name', 'creator', 'type', 'opt', 'idx', 'lr'
    
    Returns:
    Dict with optimizer results
    """
    lr_float = opt_info['lr']
    lr_float = float(lr_float) if isinstance(lr_float, (int, float, np.floating)) else str(lr_float)
    results = {
        'optimizer_idx': opt_info['idx'],
        'learning_rate': lr_float,
        'optimizer_info': opt_info['name'],
        'optimizer_type': opt_info['type'],
        'opt': opt_info['opt'],
        'benchmarks': {}
    }
    
    for benchmark_name in benchmarks:
        prefix = f"[{opt_info['type']}:{opt_info.get('name', '')}]"
        print(f"\n  {prefix} Benchmark: {benchmark_name}".strip())
    
    try:
        final_loss, loss_history = evaluate_ml_pipeline(
            optimizer_cls=opt_info['creator'],
            learning_rate=lr_float,
            benchmark_name=benchmark_name,
            n_steps=n_steps,
            seed=seed,
            n_samples=n_samples,
            n_dims=n_dims,
            weight_decay=weight_decay
        )
        
        results['benchmarks'][benchmark_name] = {
            'final_loss': float(final_loss),
            'loss_history': [float(l) for l in loss_history]
        }
        
        # Compute best loss for this specific optimizer type and name
        best_loss = float(final_loss)
        for prev in all_results['optimizers']:
            # Only compare with same optimizer type and name
            if (prev.get('optimizer_type') == opt_info['type'] and prev.get('opt') == opt_info['opt']):
                prev_benchmarks = prev.get('benchmarks', {})
                prev_entry = prev_benchmarks.get(benchmark_name)
                if prev_entry and 'final_loss' in prev_entry:
                    try:
                        prev_loss = float(prev_entry['final_loss'])
                        if prev_loss < best_loss:
                            best_loss = prev_loss
                    except Exception:
                        continue
        print(f"    final_loss={final_loss:.6e}  best_so_far={best_loss:.6e}")
    
    except Exception as e:
        print(f"    ERROR - {str(e)}")
        results['benchmarks'][benchmark_name] = {
            'final_loss': float('inf'),
            'loss_history': [],
            'error': str(e)
        }
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run optimizer benchmarks on ML tasks.")
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=['linear_regression', 'logistic_regression', 'noisy_regression', 'xor_mlp', 'autoencoder'],
        help="List of ML benchmark names (space separated)."
    )
    parser.add_argument(
        "--n_optimizer_samples",
        type=int,
        default=100,
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
        default="ml_benchmark_results",
        help="Directory to write benchmark results."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for benchmark data generation."
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of data samples (for regression/classification tasks)."
    )
    parser.add_argument(
        "--n_dims",
        type=int,
        default=5,
        help="Number of input dimensions/features."
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay (L2 regularization) coefficient for optimizers. Default: 0.01"
    )

    args = parser.parse_args()

    benchmarks = args.benchmarks
    n_optimizer_samples = args.n_optimizer_samples
    n_steps = args.n_steps
    seed = args.seed
    n_samples = args.n_samples
    n_dims = args.n_dims
    weight_decay = args.weight_decay

    results_dir = Path(args.results_dir)
    results_dir.mkdir(exist_ok=True)

    # Store all results across all sampled optimizers
    all_results = {
        'n_optimizer_samples': n_optimizer_samples,
        'n_steps': n_steps,
        'seed': seed,
        'n_samples': n_samples,
        'n_dims': n_dims,
        'weight_decay': weight_decay,
        'benchmarks_list': benchmarks,
        'optimizers': []
    }

    space = NOSSpaceMaxLines(max_lines=10)
    # RS
    random_sampler = sampling.RandomSampler({})
    prior_sampler = sampling.PriorOrFallbackSampler(random_sampler)
    # RE
    trials = {}

    for optimizer_idx in range(n_optimizer_samples):
        print(f"\n{'='*60}")
        print(f"Optimizer Sample {optimizer_idx + 1}/{n_optimizer_samples}")
        print(f"{'='*60}")

        # Sample optimizer and learning rate
        resolved_pipeline, resolution_context = neps_space.resolve(space, domain_sampler=prior_sampler)
        optimizer_creator_object_rs = resolved_pipeline.optimizer_cls
        learning_rate_rs = resolved_pipeline.learning_rate
        optimizer_creator_rs = neps_space.convert_operation_to_callable(optimizer_creator_object_rs)

        # Sample with RE too #############
        config = neps.algorithms.neps_regularized_evolution(NOSSpaceMaxLines(), population_size=20, tournament_size=5)(trials, None)
        assert not isinstance(config, list)
        samplings = neps_space.NepsCompatConverter().from_neps_config(config.config).predefined_samplings

        resolved_pipeline, resolution_context = neps_space.resolve(
            space, domain_sampler=sampling.OnlyPredefinedValuesSampler(predefined_samplings=samplings)
        )
        optimizer_creator_object_re = (resolved_pipeline.optimizer_cls) 
        learning_rate_re = resolved_pipeline.learning_rate  # Extract the learning rate
        optimizer_creator_re = neps_space.convert_operation_to_callable(optimizer_creator_object_re) 

        ###################################

        print(f"\nLearning rate: {learning_rate_rs} (RS), {learning_rate_re} (RE)")

        # Run NEPS optimizer
        def neps_opt_wrapper(params, lr, weight_decay=0.0, type='RS'):
            if type == 'RE':
                return optimizer_creator_re(params, lr=lr, variables=(1, 1))
            elif type == 'RS':
                return optimizer_creator_rs(params, lr=lr, variables=(1, 1))
        
        for neps_name in ['RS', 'RE']:
            neps_creator = partial(neps_opt_wrapper, type=neps_name)
            lr_float = learning_rate_rs if neps_name == 'RS' else learning_rate_re
            optimizer_creator_object = optimizer_creator_object_rs if neps_name == 'RS' else optimizer_creator_object_re
            neps_info = {
                'name': str(optimizer_creator_object),
                'creator': neps_creator,
                'type': 'neps',
                'opt': neps_name,
                'idx': optimizer_idx,
                'lr': lr_float
            }
        
            optimizer_results = run_optimizer_on_benchmarks(neps_info)
            all_results['optimizers'].append(optimizer_results)

            if neps_name == 'RE':
                score = optimizer_results['benchmarks'][benchmarks[0]]['final_loss']
                new_trial = Trial(
                    config=config.config, 
                    metadata=trial.MetaData(
                        id=str(optimizer_idx), 
                        location="", 
                        state=None, 
                        previous_trial_id=None, 
                        previous_trial_location=None, 
                        sampling_worker_id="", 
                        time_sampled=optimizer_idx, 
                        time_end=optimizer_idx
                        ), 
                        report=trial.Report(
                            objective_to_minimize=score, 
                            cost=None, learning_curve=None, 
                            extra={}, err=None, tb=None, 
                            reported_as=trial.State.SUCCESS, 
                            evaluation_duration=0
                        )
                )

                trials[str(optimizer_idx)] = new_trial
        
        # Run baseline optimizers
        baseline_optimizers = {
            'Adam': lambda params, lr, weight_decay: optim.Adam(params, lr=lr, weight_decay=weight_decay),
            'AdamW': lambda params, lr, weight_decay: optim.AdamW(params, lr=lr, weight_decay=weight_decay),
            'SGD': lambda params, lr, weight_decay: optim.SGD(params, lr=lr, weight_decay=weight_decay)
        }
        
        for base_name, base_creator in baseline_optimizers.items():
            baseline_info = {
                'name': f'{base_name}',
                'creator': lambda params, lr, weight_decay=weight_decay: base_creator(params, lr, weight_decay),
                'type': 'baseline',
                'opt': base_name,
                'idx': optimizer_idx,
                'lr': lr_float
            }
            
            baseline_results = run_optimizer_on_benchmarks(baseline_info)
            all_results['optimizers'].append(baseline_results)


    # Save all results to a single JSON file
    output_file = results_dir / "all_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"All results saved to {output_file}")

    # Compute and print the overall best optimizer entry (smallest final_loss)
    best = {'loss': float('inf'), 'optimizer': None, 'benchmark': None}
    for opt in all_results.get('optimizers', []):
        bench_map = opt.get('benchmarks', {})
        for bench_name, entry in bench_map.items():
            try:
                val = float(entry.get('final_loss', float('inf')))
            except Exception:
                continue
            if val < best['loss']:
                best['loss'] = val
                best['optimizer'] = opt
                best['benchmark'] = bench_name

    if best['optimizer'] is not None:
        opt = best['optimizer']
        info = opt.get('optimizer_info')
        otype = opt.get('optimizer_type', 'unknown')
        idx = opt.get('optimizer_idx')
        bname = opt.get('opt')
        print(f"Best optimizer found -> final_loss={best['loss']:.6e}")
        print(f"  optimizer_idx: {idx}")
        print(f"  optimizer_type: {otype}")
        print(f"  optimizer_name: {bname}")
        print(f"  optimizer_info: {info}")
        print(f"  benchmark: {best['benchmark']}")
    else:
        print('No valid optimizer results found to determine best optimizer')

    print(f"{'='*60}")
