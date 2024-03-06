experiment_index = 2
import os
import sys

path = "Generic_call/"
sys.path.insert(0, os.getcwd())
import matplotlib.pyplot as plt

from qrec.Stage_run import Experiment_run
from qrec.utils import *

amount_vals = 10
for i in range(amount_vals):
    with open("experiments/1/details.pickle", "rb") as f:
        details = pickle.load(f)

    betas_grid = details["betas"]
    q0, q1, n0, n1 = details["tables"]

    seed = 0
    np.random.seed(seed=None)  # Use a fixed seed.

    # Set initial parameters
    N = int(5e4)
    alpha = 0.25
    details["alpha"] = [1.5, 0.25]  # No estoy seguro para que es esto.

    np.random.seed(seed=None)

    # Hiperparameters: 0-Epsilon0, 1-delta_epsilon, 2-Dispersion_Random, 3-Temperature, 4-Learning reset
    hiperparam = [0.05, 2.0, 5, 0.0, 1]

    # Run the full program and get the new dictionary with the changes.
    details = Experiment_run(details, N, q0, q1, n0, n1, betas_grid, alpha, hiperparam)
    plt.plot(betas_grid, details["tables"][0])

plt.plot(
    betas_grid,
    [1 - Perr(b, alpha=alpha) for b in betas_grid],
    label=r"$P_s(\beta)$",
    color="black",
)
plt.legend()
plt.show()
