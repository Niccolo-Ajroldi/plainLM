# Pretrain a Transformer on Language Modeling
A minimal yet efficient implementation of causal language modeling in PyTorch.

It features a custom torch-compilable Transformer model implementation supporting RoPE, GLU, and RMSNorm.
It supports distributed training via Distributed Data Parallel (DDP).

A dedicated script is included for downloading, tokenizing, and chunking data, making data preparation seamless.

## 🛠 Installation
We recommend running `plainLM` in a dedicated Python environment. To install dependencies in an Anaconda environment, execute:
```bash
conda create --name plainLM python=3.12 -y && conda activate plainLM
pip install -r requirements.txt
```

## 📚 Data
We provide a script for downloading, tokenizing, chunking and saving Hugging Face datasets: `data/datasets/slim_pajama/prepare_train.py`.
You can specify any HF dataset and tokenizer. To avoid downloading the entire corpus, we stream, tokenize, and chunk data on-the-fly.

## ⚡️ Usage

Specify hyperparameters in `config.yaml` and launch training as follows:

#### Single GPU/CPU:
```bash
  python train.py --config=config/config.yaml
```
#### Multiple GPUs:
```bash
  torchrun --nnodes=1 --nproc_per_node=4 train.py --config=code/config/sweep.yaml
```

#### Run a sweep in parallel on a SLURM or Condor HPC cluster:

1. **Define hyperparameter sweep**:
  create a single YAML file with lists of hyperparameter values. Each value in the list will represent a different configuration, e.g.:
   ```yaml
   lr: [0.1, 0.01]
   wd: [0.1, 0.2, 0.5]
   beta1: 0.9
   ...
   ```
2. **Submit the sweep**: 
   Submit a job-array, where each job executes the same python script and reads the same configuration, but with a different `job_idx`. We use `job_idx` to map a job to its hyperparameters; `job_idx` should range from `0` to `n-1`, where `n` is the number of configurations in the YAML. This is done automatically by `cluster/slurm.sh` and `cluster/condor.sub`. Python takes care of assigning the corresponding configuration to each job based on the value of `job_idx`.


## 📂 Structure
```
plainLM/
├── cluster/             # HPC scripts (SLURM & Condor)
├── config/              # Configuration files for training and model setup
├── data/                # Everything regarding data preparation and data stream
│   └── datasets/        # Data preprocessing files to download, tokenize, chunk and save data
│   └── dataloaders.py   # Dataloader utilities
│   └── datasamplers.py  # Custom stateful distributed samplers
├── engine/              # Core implementation of the model engine: a torch.nn.Module implementing training steps and evaluations
├── models/              # Model architectures
├── optim/               # Optimization utilities
├── checkpoint_utils.py  # Checkpoint utilities
├── torch_utils.py       # PyTorch utilities (DDP, seed, TF32...)
├── train.py             # Main training script ⭐️
└── utils.py             # Miscellaneous helper functions
```

## ☑️ TODO
- [FSDP2](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) support, ZeRO-2 and tensor parallel compatibility
- dummy data
- unit tests
- add seed to `DistributedSampler`

## Citation
```bibtex
@misc{ajroldi2024plainlm,
  author = {Niccolò Ajroldi},
  title = {plainLM: Language Model Pretraining in PyTorch},
  year = {2024},
  howpublished = {\url{https://github.com/Niccolo-Ajroldi/plainLM}}
}
```

## Credits
This project was inspired by:  
- [Cramming](https://github.com/JonasGeiping/cramming) by Jonas Geiping
- [GPT-NeoX](https://github.com/EleutherAI/gpt-neox) by EleutherAI
- [NanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy

Huge thanks to these projects for their contributions to open-source language model pretraining!

## Some projects using with `plainLM`
- Orvieto, A., & Gower, R. (2025). In search of Adam’s secret sauce https://arxiv.org/abs/2505.21829
- Ajroldi, N., Orvieto, A., & Geiping, J. (2025). When, where and why to average weights? https://arxiv.org/abs/2502.06761
- Srećković, T., Geiping, J., & Orvieto, A. (2025). Is your batch size the problem? Revisiting the Adam-SGD gap in language modeling https://arxiv.org/abs/2506.12543
