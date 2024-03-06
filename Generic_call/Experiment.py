experiment_index = 2
import os
import sys

path = "Generic_call/"
sys.path.insert(0, os.getcwd())
import time

from qrec.Stage_run import Experiment_run
from qrec.utils import *

# --------------------------First define details-----------------------------------
with open("experiments/1/details.pickle", "rb") as f:
    details = pickle.load(f)

betas_grid = details["betas"]
q0, q1, n0, n1 = details["tables"]

# -----------------------------------------------------------------------------------

seed = 0
np.random.seed(seed)  # Use a fixed seed.

# Set initial parameters
N = int(5e5)
alpha = 0.25
details["alpha"] = [1.5, 0.25]  # No estoy seguro para que es esto.

np.random.seed(seed)

# Hiperparameters: 0-Epsilon0, 1-delta_epsilon, 2-delta_learning_rate, 3-Dispersion_Random, 4-Temperature
hiperparam = [0.05, 2.0, 1.0, 0.1, 10]

# Run the full program and get the new dictionary with the changes.
details = Experiment_run(
    details, N, q0, q1, n0, n1, betas_grid, alpha, hiperparam, N0=0
)
