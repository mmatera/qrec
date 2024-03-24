"""
Template for experiments

"""

import os
import pathlib
import pickle
import sys

import numpy as np

# Needed if we do not install qrec
sys.path.insert(0, str(pathlib.Path(os.getcwd())))


from qrec.Stage_run import run_experiment
from qrec.utils import Hyperparameters, define_q

# Global Parameters
EXPERIMENT_INDEX = 2
DETAILS_FILE_NAME = f"experiments/{EXPERIMENT_INDEX}/details.pickle"
PATH = "Generic_call/"
SEED = 0


HYPERPARMS = Hyperparameters(0.05, 2.0, 1.0, 0.1, 10)
TRAINING_SIZE = int(5e5)


# --------------------------First define details-------------------------------


# -----------------------------------------------------------------------------


def __main__():
    try:
        with open(DETAILS_FILE_NAME, "rb") as f:
            details = pickle.load(f)
    except FileNotFoundError:
        details = {
            "alpha": [1.5, 0.25],
            "means": [],
            "tables": define_q(),
            "mean_rewards": [0.0, 0.0],
            "experience": [],
            "Ps_greedy": [],
            "ep": 0.5,
        }

    alpha = details["alpha"][1]

    # Run the full program and get the new dictionary with the changes.
    details = run_experiment(
        details,
        TRAINING_SIZE,
        alpha,
        HYPERPARMS,
    )


if __name__ == "__main__":
    __main__()
