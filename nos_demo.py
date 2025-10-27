"""
A demo script to test the training pipeline with a small model and dataset.
"""

import torch
from nos_train import train
import neps
from nos_space import NOSSpace2

if __name__ == "__main__":
    valid_loss = train(
        optimizer_cls=torch.optim.AdamW,
        lr=3e-4,
        pipeline_directory="runs/adamw_run",
    )
    print(f"Validation loss: {valid_loss}")



    def evaluate_pipeline(optimizer_cls, learning_rate):
        loss = train(
            optimizer_cls=optimizer_cls,
            lr=learning_rate,
            pipeline_directory="runs/neps_run",
        )
        return loss

    space = NOSSpace2(max_lines=10)

    neps.run(
        evaluate_pipeline,
        space,
        root_directory="runs/neps_run",
        evaluations_to_spend=2,
        optimizer=neps.algorithms.random_search
    )