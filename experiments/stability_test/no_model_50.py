"""
No model 50
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())

from qrec.Stage_run import run_experiment
from qrec.utils import Hyperparameters, bayes_decision_error_probability

AMOUNT_VALS = 10

EXPERIMENT_INDEX = 2
EXPERIMENT_PATH = "Generic_call/"


def __main__():
    """main"""
    for i in range(AMOUNT_VALS):
        with open("experiments/1/details.pickle", "rb") as f:
            details = pickle.load(f)

        qlearning = details["tables"]
        betas_grid = qlearning.betas_grid

        seed = 0
        np.random.seed(seed=None)  # Use a fixed seed.

        # Set initial parameters
        training_size = int(5e4)
        alpha = 0.25
        details["alpha"] = [1.5, 0.25]  # No estoy seguro para que es esto.

        np.random.seed(seed=None)
        hyperparam = Hyperparameters(0.05, 2.0, 5, 0.0, 1)

        # Run the full program and get the new dictionary with the changes.
        details = run_experiment(details, training_size, alpha, hyperparam)
        plt.plot(betas_grid, qlearning.q0)

    plt.plot(
        betas_grid,
        [1 - bayes_decision_error_probability(b, alpha=alpha) for b in betas_grid],
        label=r"$P_s(\beta)$",
        color="black",
    )
    plt.legend()
    plt.show()


if __name__ == "__main__":
    __main__()
