"""
A demo script to test the training pipeline with a small model and dataset.
"""

import torch
from nos_train import train
import neps
from neps.space.neps_spaces import sampling, neps_space
from nos_space import NOSSpace2

if __name__ == "__main__":
    # valid_loss = train(
    #     optimizer_cls=torch.optim.AdamW,
    #     lr=3e-4,
    #     pipeline_directory="runs/adamw_run",
    # )
    # print(f"Validation loss: {valid_loss}")


    def evaluate_pipeline(optimizer_cls, learning_rate):
        # loss = train(
        #     optimizer_cls=optimizer_cls,
        #     lr=learning_rate,
        #     pipeline_directory="runs/neps_run",
        # )

        x = torch.tensor([0.0], requires_grad=True)
        opt = optimizer_cls([x], lr=learning_rate)

        opt.zero_grad()
        loss = (x - 3.0).pow(2).sum()
        loss.backward()
        opt.step()
        loss = (x - 3.0).pow(2).sum()

        return loss.item()

    space = NOSSpace2(max_lines=5)
    random_sampler = sampling.RandomSampler({})
    prior_sampler = sampling.PriorOrFallbackSampler(random_sampler)

    for _ in range(5):
        resolved_pipeline, resolution_context = neps_space.resolve(space, domain_sampler=prior_sampler) # random_sampler would ignore priors 
        optimizer_creator_object = resolved_pipeline.optimizer_cls # Extract the optimizer creator object
        learning_rate = resolved_pipeline.learning_rate # Extract the learning rate
        optimizer_creator = neps_space.convert_operation_to_callable(optimizer_creator_object) # Convert to callable

        x = torch.nn.Parameter(torch.tensor([0.0], dtype=torch.float32), requires_grad=True)
        opt = optimizer_creator([x], lr=learning_rate, variables=(1,1)) # Create optimizer, using the sampled learning rate and start values for internal variables
        print("---Sampled optimizer and learning rate---")
        print(opt)