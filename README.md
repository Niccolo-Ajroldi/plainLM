# Pretrain a Transformer on language modeling.
Minimal implementation of a Transformer model and a training script for language modeling in PyTorch. 
Supports multi-GPU training via Distributed Data Parallel (DDP).

### Usage

##### Single GPU/CPU:
```
  python train.py --config=config/config.yaml
```
##### Multiple GPUs:
```
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
  Use `job_idx` to specify which configuration file to use. The `job_idx` should range from `0` to `n-1`, where `n` is the number of configurations in the YAML. This is done automatically by `condor.sub`. Python takes care of assigning the corresponding configuration to each job based on the `job_idx`.


### TODO:
- data loading
  - improve readibility
  - add seed to `DistributedSampler`
- test macOS metal support
- add `LinearCooldown` compatible with `WarmupConstant`

