# Pretrain a Transformer on Causal Language Modeling.
Minimal training script for language modeling in PyTorch. \
It includes custom implementation of a Transformer model, with RoPE, GLU, RMSNorm, compatible with `torch.compile`. \
It supports distributed training via Distributed Data Parallel (DDP).

### Usage

##### Single GPU/CPU:
```bash
  python train.py --config=config/config.yaml
```
##### Multiple GPUs:
```bash
  torchrun --nnodes=1 --nproc_per_node=4 train.py --config=code/config/sweep.yaml
```

##### Run a sweep:

1. **Define Hyperparameters**:
  Create a single YAML file with lists of hyperparameter values. Each value in the list will represent a different configuration, e.g.:
   ```yaml
   lr: [0.1, 0.01]
   wd: [0.1, 0.2, 0.5]
   ...
   ```
2. **Submit the Sweep**: 
  Use `job_idx` to specify which configuration file to use. `job_idx` should range from `0` to `n-1`, where `n` is the number of configurations in the YAML. This is done automatically by `condor.sub`. Python takes care of assigning the corresponding configuration to each job based on the `job_idx`.


### TODO:
- data loading
  - improve readibility
  - add seed to `DistributedSampler`
- test macOS metal support
- add `LinearCooldown` compatible with `WarmupConstant`
- add dummy data
- send eval results when log_every is not a multiple of eval every (also better logger)
- FSDP2 support

