"""
A demo script to test the training pipeline with a small model and dataset.
"""

import torch
from nos_train import train

if __name__ == "__main__":
  valid_loss = train(
    optimizer_cls=torch.optim.AdamW,
    lr=3e-4,
    pipeline_directory="runs/adamw_run",
  )
  print(f"Validation loss: {valid_loss}")
