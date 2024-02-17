experiment_index=2
import os
import sys
path = "Model_Semi_aware/"
sys.path.insert(0, os.getcwd())
from qrec.utils import *
from qrec.Stage_run import Model_experiment
import matplotlib.pyplot as plt

def probability(alpha, beta_grid):
    results = []
    for i in range(0, len(beta_grid)):
        vals = 0.5 * np.exp(-np.abs(-alpha - beta_grid[i])**2) +  0.5 * (1 - np.exp(-np.abs(alpha - beta_grid[i])**2))
        results.append(vals)
    return results

# --------------------------First define details-----------------------------------
with open("experiments/1/details.pickle","rb") as f:
    details = pickle.load(f)

betas_grid = details["betas"]
q0,q1,n0,n1 = details["tables"]

#-----------------------------------------------------------------------------------

seed = 0
np.random.seed(seed)  # Use a fixed seed.

# Set initial parameters
N=int(5e4)
alpha = 0.25
details["alpha"] = [1.5,0.25]  # No estoy seguro para que es esto.

np.random.seed(seed)

#Hiperparameters: 0-Epsilon0, 1-delta_epsilon, 2-Dispersion_Random, 3-Temperature, 4-Initial learning rate
hiperparam = [0.1, 5, 25, 0.0, 50]
details["ep"] = f"{0.01}"
# Run the full program and get the new dictionary with the changes.
for i in range(0, 1):
    details["seed"] = np.random.uniform(0, 1)
    details = Model_experiment(details, N, q0, q1, n0, n1, betas_grid, alpha, hiperparam, N0=0)

    #plt.plot(details["mean_rewards"])
    #plt.show()

    values = probability(0.25, betas_grid)
    plt.plot(betas_grid, q0)

plt.plot(betas_grid, values, label='theory', color="black")
plt.legend()
plt.show()

