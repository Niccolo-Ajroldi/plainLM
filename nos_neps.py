"""
A demo script to test the training pipeline with a small model and dataset.
"""

import logging
import torch
import neps
from nos_space import NOSSpace3Lines
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in cast')

if __name__ == "__main__":

    def evaluate_pipeline(optimizer_cls, learning_rate):
        x = torch.tensor([0.0], requires_grad=True)
        opt = optimizer_cls([x], lr=learning_rate)

        opt.zero_grad()
        loss = (x - 3.0).pow(2).sum()
        loss.backward()
        opt.step()
        loss = (x - 3.0).pow(2).sum()

        return loss.item().__float__()

    space = NOSSpace3Lines()  # Samples exactly 3 lines, v1=... v2=... update=...

    # Manually sample a configuration
    config_dict, pipeline_dict = neps.create_config(space)

    # Manually sample optimizer:
    print(
        "Sampled:\n",
        pipeline_dict["optimizer_cls"](
            [torch.tensor([0.0])], lr=pipeline_dict["learning_rate"]
        ),
    )

    # Evaluate the manually sampled pipeline
    loss = evaluate_pipeline(
        optimizer_cls=pipeline_dict["optimizer_cls"],
        learning_rate=pipeline_dict["learning_rate"],
    )
    print(f"Manually sampled pipeline loss: {loss}")

    logging.basicConfig(level=logging.INFO)
    # Import trial into Neps
    neps.import_trials(
        space,
        [(config_dict, neps.UserResultDict(objective_to_minimize=loss))],
        "results/demo_warmstarting",
        overwrite_root_directory=True,
        optimizer=neps.algorithms.neps_regularized_evolution,
    )

    # Continue running RE, using the imported trial as warmstart
    neps.run(
        evaluate_pipeline=evaluate_pipeline,
        pipeline_space=space,
        root_directory="results/demo_warmstarting",
        evaluations_to_spend=30,
        optimizer=neps.algorithms.neps_regularized_evolution,
    )
