"""
Model 50
"""

import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())
from qrec.Stage_run import run_experiment
from qrec.utils import (
    Hyperparameters,
    bayes_decision_error_probability,
    model_aware_optimal,
)

EXPERIMENT_INDEX = 2
EXPERIMENT_PATH = "experiments/stability_test/"


def __main__():
    """main"""
    ax = plt.subplot(111)
    amount_vals = 1

    for i in range(amount_vals):
        with open("experiments/1/details.pickle", "rb") as f:
            details = pickle.load(f)

        betas_grid = details["betas"]
        qlearning = details["tables"]

        seed = 0
        np.random.seed(seed=None)  # Random seed

        # Set initial parameters
        training_size = int(5e4)
        alpha = 0.25
        lambd = 0.05
        noise_type = 2
        model_aware_optimal(alpha=alpha, lambd=lambd)

        details["alpha"] = [1.5, 0.25]  # No estoy seguro para que es esto.

        np.random.seed(seed=None)

        hyperparam = Hyperparameters(1, 2 / int(5e4), 25, 250)
        # Run the full program and get the new dictionary with the changes.
        details = run_experiment(
            details,
            training_size,
            alpha,
            hyperparam,
            lambd=lambd,
            model=True,
            noise_type=noise_type,
        )

        # Plots
        stacked_history = np.stack(details["experience"])

        ax.plot(betas_grid, qlearning.q0)
        # ax.plot(details["Ps_greedy"][int(5e5):], label=r"$P_t$")
        # ax.plot(
        #    np.cumsum(stacked_history[int(5e5) :, -1])
        #    / np.arange(1, len(stacked_history[int(5e5) :, -1]) + 1)
        # )

    ax.plot(
        betas_grid,
        [
            1
            - bayes_decision_error_probability(
                b, alpha=alpha, lambd=lambd, noise_type=noise_type
            )
            for b in betas_grid
        ],
        label=r"$P_s(\beta)$",
        color="black",
    )
    ax.plot(
        betas_grid,
        [
            1 - bayes_decision_error_probability(b, alpha=alpha, lambd=0.0)
            for b in betas_grid
        ],
        label=r"$P_s(\beta) model$",
        color="red",
    )
    # ax.set_xscale("log")
    # ax.axhline(1, color="black")
    ax.legend()
    plt.savefig(EXPERIMENT_PATH + "model_images/q0.png")

    # ax=plt.subplot(111)
    # ax.axhline(1.-pstar,color="black",label=r'$P_s^*$')
    # ax.set_xscale("log")
    # ax.legend()
    # plt.savefig(path+"model_images/betas.png")


if __name__ == "__main__":
    __main__()
