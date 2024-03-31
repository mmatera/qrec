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
    amount_vals = 10
    betas = []
    Rts  = []
    q0s = []

    for i in range(amount_vals):
        with open("experiments/1/details.pickle", "rb") as f:
            details = pickle.load(f)

        betas_grid = details["betas"]
        qlearning = details["tables"]

        seed = 0
        np.random.seed(seed=None)  # Random seed

        # Set initial parameters
        training_size = int(5e3)
        alpha = 0.25
        lambd = 0.75
        noise_type = 1
        mmin, p_star, beta_star = model_aware_optimal(betas_grid, alpha=alpha, lambd=lambd, noise_type=noise_type)

        details["alpha"] = [1.5, 0.25]  # No estoy seguro para que es esto.

        np.random.seed(seed=None)

        hyperparam = Hyperparameters(0.01, 1 - 1/100, 20, 50)
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

        stacked_history = np.stack(details["experience"])

        q0s.append(details["tables"][0])
        betas.append(details["greed_beta"])
        Rts.append(np.cumsum(stacked_history[int(5e4):,-1])/np.arange(1,len(stacked_history[int(5e4):,-1])+1))

    means_Rts = [0 for i in range(len(Rts[0]))]
    means_q0s = [0 for i in range(len(q0s[0]))]
    means_betas = [0 for i in range(len(betas[0]))]

    for i in range(len(q0s)):
        for j in range(len(q0s[i])):
            means_q0s[j] += q0s[i][j] / len(q0s)

    for i in range(len(Rts)):
        for j in range(len(Rts[i])):
            means_Rts[j] += Rts[i][j] / len(Rts)

    for i in range(len(betas)):
        for j in range(len(betas[i])):
            means_betas[j] += betas[i][j] / len(betas)

    f, ax = plt.subplots(1,3)

    disp_q0s = [[], []]
    for i in range(len(q0s[0])):
        vals = np.array([q0s[j][i] for j in range(len(q0s))])
        disp_q0s[0].append(vals.min())
        disp_q0s[1].append(vals.max())

    ax[0].fill_between(betas_grid, disp_q0s[0], disp_q0s[1], alpha=0.5, color="blue")
    ax[0].plot(betas_grid, means_q0s, label="mean agent", color="blue")
    ax[0].plot(betas_grid,[1-bayes_decision_error_probability(b, alpha=alpha, noise_val=lambd, noise_type=noise_type) for b in betas_grid],label=r'$Score Function$', color="black")
    ax[0].plot(betas_grid,[1-bayes_decision_error_probability(b, alpha=alpha, noise_val=0.0) for b in betas_grid],label=r'$Surmised Value$', color="red")
    ax[0].set_xlabel(r"$\beta$")
    ax[0].set_ylabel(r"$P_{succ}$")
    ax[0].legend()

    disp_betas = [[], []]
    for i in range(len(betas[0])):
        vals = np.array([betas[j][i] for j in range(len(betas))])
        disp_betas[0].append(vals.min())
        disp_betas[1].append(vals.max())

    ax[1].fill_between([i for i in range(len(disp_betas[0]))], disp_betas[0], disp_betas[1], alpha=0.5, color="blue")
    ax[1].plot(means_betas, color="blue", label=r"mean $\beta$")
    ax[1].set_xscale("log")
    ax[1].axhline(beta_star, color="black", label=r"$\beta^*$")
    ax[1].set_xlabel(r"$t$")
    ax[1].set_ylabel(r"$\beta$")
    ax[1].legend()

    ax[2].plot(means_Rts, label=r"mean $R_t/t$", color="red")
    disp_Rts = [[], []]
    for i in range(len(Rts[0])):
        vals = np.array([Rts[j][i] for j in range(len(Rts))])
        disp_Rts[0].append(vals.min())
        disp_Rts[1].append(vals.max())

    ax[2].fill_between([i for  i in range(len(means_Rts))], disp_Rts[0], disp_Rts[1], alpha=0.5, color="orange")

    ax[2].set_xscale("log")
    ax[2].axhline(1-p_star, color="black", label="optimal reward")
    ax[2].set_xlabel(r"$t$")
    ax[2].set_ylabel(r"$\frac{R_t}{t}$")
    ax[2].legend()
    plt.show()




if __name__ == "__main__":
    __main__()

