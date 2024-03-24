"""
Several sets of hyperparameters
"""

import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())


from qrec.Stage_run import run_experiment
from qrec.utils import Hyperparameters

EXPERIMENT_INDEX = 2
EXPERIMENT_PATH = f"experiments/{EXPERIMENT_INDEX}/"


TRAINING_SIZE = int(5e5)

HYPERPARAMETERS_LIST = [
    Hyperparameters(0.01, 1 + 750 / TRAINING_SIZE, 5.0, 2.5, 20.0),
    Hyperparameters(0.01, 1 + 1000 / TRAINING_SIZE, 5.0, 2.5, 20.0),
    Hyperparameters(0.01, 1 + 1500 / TRAINING_SIZE, 5.0, 5.0, 20.0),
    Hyperparameters(0.01, 1 + 2000 / TRAINING_SIZE, 5.0, 5.0, 20.0),
    Hyperparameters(0.01, 1 + 2000 / TRAINING_SIZE, 5.0, 1.0, 50.0),
    Hyperparameters(0.01, 1 + 2500 / TRAINING_SIZE, 5.0, 2.5, 20.0),
    Hyperparameters(0.01, 1 + 2500 / TRAINING_SIZE, 2.0, 2.5, 20.0),
    Hyperparameters(0.01, 1 + 3000 / TRAINING_SIZE, 5.0, 2.0, 20.0),
    Hyperparameters(0.01, 1 + 3000 / TRAINING_SIZE, 10.0, 5.0, 50.0),
    Hyperparameters(0.01, 1 + 3000 / TRAINING_SIZE, 5.0, 0.2, 20.0),
    Hyperparameters(0.05, 1 + 1500 / TRAINING_SIZE, 5.0, 2.5, 5.0),
    Hyperparameters(0.05, 1 + 1500 / TRAINING_SIZE, 5.0, 2.5, 0.0),
    Hyperparameters(0.05, 1 + 1500 / TRAINING_SIZE, 1.0, 5.0, 20.0),
    Hyperparameters(0.05, 1 + 5000 / TRAINING_SIZE, 0.0, 5.0, 25.0),
    Hyperparameters(0.05, 1 + 3000 / TRAINING_SIZE, 5.0, 1.0, 50.0),
    Hyperparameters(0.05, 1 + 1500 / TRAINING_SIZE, 5.0, 2.5, 20.0),
    Hyperparameters(0.05, 1 + 1500 / TRAINING_SIZE, 5.0, 25.0, 20.0),
    Hyperparameters(0.05, 1 + 1500 / TRAINING_SIZE, 20.0, 5.0, 20.0),
    Hyperparameters(0.05, 1 + 1500 / TRAINING_SIZE, 0.1, 5.0, 50.0),
    Hyperparameters(0.05, 1 + 3000 / TRAINING_SIZE, 1.0, 2.0, 50.0),
]


def main():
    """main"""

    for hyperparameters in HYPERPARAMETERS_LIST:
        delta_eps_0 = np.round(
            (hyperparameters.delta_epsilon - 1) * TRAINING_SIZE, 2
        )
        label_value = (
            rf"$\epsilon = ${hyperparameters.eps_0}, "
            rf"$\epsilon_\delta = ${delta_eps_0}, "
            rf"$\delta_L = ${hyperparameters.delta}, "
            rf"$\sigma = ${hyperparameters.delta_learning_rate}, "
            rf"$\delta_T = ${hyperparameters.temperature}"
        )
        # ## run q-learning
        seed = 0
        np.random.seed(seed)
        with open("experiments/1/details.pickle", "rb") as src_file:
            details = pickle.load(src_file)

        qlearning = details["tables"]
        print(qlearning.n0)

        seed = 0
        ##### CHANGE #####
        alpha = 0.25
        details["alpha"] = [1.5, 0.25]
        # ### CHANGE ####
        np.random.seed(seed)

        details = run_experiment(
            details,
            TRAINING_SIZE,
            alpha,
            hyperparameters,
            N0=int(5e5),
        )

        print(qlearning.n0)
        os.makedirs(
            f"../data_rec/experiments/{EXPERIMENT_INDEX}/",
            exist_ok=True,
        )
        with open(
            f"../data_rec/experiments/{EXPERIMENT_INDEX}/details.pickle",
            "wb",
        ) as dst_file:
            pickle.dump(details, dst_file, protocol=pickle.HIGHEST_PROTOCOL)

        stacked_history = np.stack(details["experience"])
        # plt.plot(np.cumsum(stacked_history[:,-1])/
        # (np.arange(1,
        # len(stacked_history[:,-1])+1) * max(details["Ps_greedy"])),
        # label=label_value)
        plt.plot(details["mean_rewards"])

    plt.xscale("log")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
