# ML Benchmarks for Optimizer Search

This directory contains scripts to evaluate optimizers on realistic ML tasks using the same optimizer search framework from `nos_synthetic.py`.

## Files

- **`nos_ml_benchmarks.py`**: Main benchmarking script that samples optimizers and evaluates them on ML tasks
- **`plot_ml_benchmarks.py`**: Visualization script for ML benchmark results
- **`nos_synthetic.py`**: Original synthetic function benchmarks (Rosenbrock, Rastrigin, etc.)
- **`plot_synthetic.py`**: Visualization for synthetic benchmarks
- **`plot_landscape.py`**: 2D landscape trajectory visualization

## ML Benchmarks

The following ML tasks are implemented:

1. **Linear Regression** (Convex) - Simple least squares regression
2. **Logistic Regression** (Mildly Non-Convex) - Binary classification
3. **Noisy Regression** (Convex + Noise) - Regression with stochastic noise
4. **XOR MLP** (Nonlinear) - Small neural network on XOR problem
5. **Tiny Autoencoder** (Deep & Nonlinear) - Small autoencoder reconstruction task

## Usage

### Run ML Benchmarks

Basic usage:
```bash
cd synthetic
python nos_ml_benchmarks.py
```

With custom settings:
```bash
python nos_ml_benchmarks.py \
  --benchmarks linear_regression logistic_regression xor_mlp \
  --n_optimizer_samples 50 \
  --n_steps 300 \
  --n_samples 200 \
  --n_dims 10 \
  --weight_decay 0.01 \
  --results_dir ml_benchmark_results \
  --seed 42
```

Arguments:
- `--benchmarks`: Space-separated list of benchmark names (default: all 5 benchmarks)
- `--n_optimizer_samples`: Number of optimizer configurations to sample (default: 100)
- `--n_steps`: Number of training steps per optimizer (default: 500)
- `--n_samples`: Number of data samples for regression/classification tasks (default: 100)
- `--n_dims`: Number of input dimensions/features (default: 5)
- `--weight_decay`: Weight decay (L2 regularization) coefficient (default: 0.01)
- `--results_dir`: Output directory for JSON results (default: `ml_benchmark_results`)
- `--seed`: Random seed for reproducibility (default: 0)

### How n_samples and n_dims affect benchmarks:

- **Linear/Logistic/Noisy Regression**: Uses n_samples × n_dims data matrices
- **XOR MLP**: Fixed 2D input, but hidden layer size scales with n_dims (hidden = max(4, n_dims))
- **Autoencoder**: Uses n_samples × n_dims data, latent dimension = d/3, hidden = (d + latent)/2

### Weight Decay:

Weight decay (L2 regularization) is applied to all optimizers (both method and baselines). The default value of 0.01 is a standard choice commonly used with Adam/AdamW optimizers. You can:
- Set `--weight_decay 0.0` to disable regularization
- Increase to `0.1` for stronger regularization on small datasets
- Decrease to `0.001` for lighter regularization on larger datasets

### Generate Plots

After running benchmarks:
```bash
python plot_ml_benchmarks.py
```

This creates three types of plots in `ml_plots/`:
1. **Loss trajectories**: Best method vs average baseline trajectories per benchmark
2. **Best-so-far curves**: Cumulative best loss over optimizer sampling iterations
3. **Final loss comparison**: Bar chart comparing method vs baselines across all benchmarks

### Run Synthetic Benchmarks

For mathematical function benchmarks (with 2D landscape support):
```bash
python nos_synthetic.py \
  --benchmarks rosenbrock rastrigin \
  --dimensions 2 10 50 \
  --n_optimizer_samples 50 \
  --n_steps 200
```

Generate synthetic plots:
```bash
python plot_synthetic.py
```

Generate 2D landscape trajectory plot:
```bash
python plot_landscape.py \
  --benchmark rosenbrock \
  --n_dims 2 \
  --results_file benchmark_results/all_results.json
```

## Results Structure

Both scripts output JSON files with the following structure:
```json
{
  "n_optimizer_samples": 100,
  "n_steps": 500,
  "optimizers": [
    {
      "optimizer_idx": 0,
      "optimizer_type": "neps",
      "learning_rate": 0.001,
      "optimizer_info": "...",
      "benchmarks": {
        "linear_regression": {
          "final_loss": 0.0123,
          "loss_history": [...]
        }
      }
    },
    {
      "optimizer_idx": 0,
      "optimizer_type": "baseline",
      "baseline_name": "Adam",
      "learning_rate": 0.001,
      "benchmarks": {...}
    }
  ]
}
```

## Notes

- Each optimizer sample runs the method optimizer plus 3 baselines (Adam, AdamW, SGD) using the same sampled learning rate
- Results include full loss histories for trajectory visualization
- For 2D synthetic benchmarks, parameter trajectories (`x_history`) are recorded for landscape plotting
- ML benchmarks use PyTorch and are deterministic given the same seed
- The script prints the overall best optimizer found at the end

## Quick Test

Run a quick smoke test (fast):
```bash
# ML benchmarks (fast) - small problem size
python nos_ml_benchmarks.py \
  --benchmarks linear_regression xor_mlp \
  --n_optimizer_samples 10 \
  --n_steps 100 \
  --n_samples 50 \
  --n_dims 3

# ML benchmarks - larger problem
python nos_ml_benchmarks.py \
  --benchmarks autoencoder \
  --n_optimizer_samples 10 \
  --n_steps 200 \
  --n_samples 500 \
  --n_dims 20

# Synthetic benchmarks (2D for landscape plotting)
python nos_synthetic.py \
  --benchmarks rosenbrock \
  --dimensions 2 \
  --n_optimizer_samples 10 \
  --n_steps 100
```
