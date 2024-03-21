"""
Template for experiments

"""

import pathlib
import os
import sys
import time
import pickle


# Needed if we do not install qrec
sys.path.insert(0, str(pathlib.Path(os.getcwd()).parent))


from qrec.Stage_run import Experiment_run, Hyperparameters
from qrec.utils import *


# Global Parameters
EXPERIMENT_INDEX = 2
DETAILS_FILE_NAME = f"experiments/{EXPERIMENT_INDEX}/details.pickle"
PATH = "Generic_call/"
SEED = 0

details = {}
# TODO: Use namedtuples?
# Hyperparameters: 0-Epsilon0,
#                  1-delta_epsilon,
#                  2-delta_learning_rate,
#                  3-Dispersion_Random,
#                  4-Temperature


HYPERPARMS = Hyperparameters(0.05, 2.0, 1.0, 0.1, 10)
N = int(5e5)
alpha = 0.25
details["alpha"] = [1.5, 0.25]

# --------------------------First define details-------------------------------
try:
    with open(DETAILS_FILE_NAME, "rb") as f:
        details = pickle.load(f)
except FileNotFoundError:
    pass


# -----------------------------------------------------------------------------

# No estoy seguro para que es esto.

# Load parameters from details

alpha = details["alpha"][1]
betas_grid = details["betas"]
q0, q1, n0, n1 = details["tables"]


# Run the full program and get the new dictionary with the changes.
details = Experiment_run(
    details, N, q0, q1, n0, n1, betas_grid, alpha, hiperparam, N0=0
)
