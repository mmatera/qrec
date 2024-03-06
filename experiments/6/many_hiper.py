experiment_index = 2
import os
import sys

path = "experiments/{}/".format(experiment_index)
sys.path.insert(0, os.getcwd())
import time

from qrec.Stage_run import Experiment_run
from qrec.utils import *

N = int(5e5)
# Hiperparameters: 0-Epsilon0, 1-delta_epsilon, 2-delta_learning_rate, 3-Dispersion_Random, 4-Temperature
hiperparameters = [
    [0.01, 1 + 750 / N, 5.0, 2.5, 20.0],
    [0.01, 1 + 1000 / N, 5.0, 2.5, 20.0],
    [0.01, 1 + 1500 / N, 5.0, 5.0, 20.0],
    [0.01, 1 + 2000 / N, 5.0, 5.0, 20.0],
    [0.01, 1 + 2000 / N, 5.0, 1.0, 50.0],
    [0.01, 1 + 2500 / N, 5.0, 2.5, 20.0],
    [0.01, 1 + 2500 / N, 2.0, 2.5, 20.0],
    [0.01, 1 + 3000 / N, 5.0, 2.0, 20.0],
    [0.01, 1 + 3000 / N, 10.0, 5.0, 50.0],
    [0.01, 1 + 3000 / N, 5.0, 0.2, 20.0],
    [0.05, 1 + 1500 / N, 5.0, 2.5, 5.0],
    [0.05, 1 + 1500 / N, 5.0, 2.5, 0.0],
    [0.05, 1 + 1500 / N, 1.0, 5.0, 20.0],
    [0.05, 1 + 5000 / N, 0.0, 5.0, 25.0],
    [0.05, 1 + 3000 / N, 5.0, 1.0, 50.0],
    [0.05, 1 + 1500 / N, 5.0, 2.5, 20.0],
    [0.05, 1 + 1500 / N, 5.0, 25.0, 20.0],
    [0.05, 1 + 1500 / N, 20.0, 5.0, 20.0],
    [0.05, 1 + 1500 / N, 0.1, 5.0, 50.0],
    [0.05, 1 + 3000 / N, 1.0, 2.0, 50.0],
]

for i in range(len(hiperparameters)):
    label_value = rf"$\epsilon = ${hiperparameters[i][0]}, $\epsilon_\delta = ${np.round((hiperparameters[i][1] - 1) * N,2)}, $\delta_L = ${hiperparameters[i][2]}, $\sigma = ${hiperparameters[i][3]}, $\delta_T = ${hiperparameters[i][4]}"
    ### run q-learning
    N = int(5e5)
    seed = 0
    np.random.seed(seed)
    with open("experiments/1/details.pickle", "rb") as f:
        details = pickle.load(f)

    betas_grid = details["betas"]
    q0, q1, n0, n1 = details["tables"]
    print(n0)

    seed = 0
    experiment_index = 1
    ##### CHANGE #####
    alpha = 0.25
    details["alpha"] = [1.5, 0.25]
    #### CHANGE ####
    np.random.seed(seed)

    details = Experiment_run(
        details, N, q0, q1, n0, n1, betas_grid, alpha, hiperparameters[i], N0=int(5e5)
    )

    print(n0)
    os.makedirs("../data_rec/experiments/{}/".format(experiment_index), exist_ok=True)
    with open(
        "../data_rec/experiments/{}/details.pickle".format(experiment_index), "wb"
    ) as f:
        pickle.dump(details, f, protocol=pickle.HIGHEST_PROTOCOL)

    details.keys()
    betas_grid = details["betas"]
    stacked_history = np.stack(details["experience"])
    # plt.plot(np.cumsum(stacked_history[:,-1])/(np.arange(1,len(stacked_history[:,-1])+1) * max(details["Ps_greedy"])), label=label_value)
    plt.plot(details["mean_rewards"])

plt.xscale("log")
plt.legend()
plt.show()
