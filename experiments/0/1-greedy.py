import os
import pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())

from qrec.utils import (
    bayes_decision_error_probability,
    comm_success_prob,
    define_q,
    ep_greedy,
    give_outcome,
    give_reward,
    model_aware_optimal,
)

experiment_index = 0
path = "experiments/{}/".format(experiment_index)


def updates(qlearning, indb, n, g, r, lr=0.01):
    q0 = qlearning.q0
    q1 = qlearning.q1
    n0 = qlearning.n0
    n1 = qlearning.n1
    q1[indb, n, g] += (1 / n1[indb, n, g]) * (r - q1[indb, n, g])
    q0[indb] += (1 / n0[indb]) * np.max(
        [q1[indb, n, g] for g in [0, 1]] - q0[indb]
    )
    n0[indb] += 1
    n1[indb, n, g] += 1
    return


def main():
    """main"""
    seed = 0
    experiment_index = 0
    alpha = 0.4
    epsilon = 1.0
    tables = define_q(beta_steps=10)
    details = {
        "index": experiment_index,
        "alpha": alpha,
        "ep": epsilon,
        "experience": [],
        "Ps_greedy": [],
        "seed": seed,
        "tables": tables,
    }
    np.random.seed(seed)

    ### run q-learning
    N = int(1e5)
    np.random.seed(seed)
    start = time.time()
    qlearning = define_q()
    betas_grid = qlearning.betas_grid
    details = {
        "index": experiment_index,
        "alpha": alpha,
        "ep": "1.",
        "experience": [],
        "Ps_greedy": [],
        "seed": seed,
        "tables": qlearning,
    }
    for experiment in range(N):
        if experiment % int(N / 10) == 0:
            print(experiment)
        hidden_phase = np.random.choice([0, 1])
        indb, b = ep_greedy(tables.q0, qlearning.betas_grid, 1, nr_prob=1.0)
        n = give_outcome(hidden_phase, b)
        indg, g = ep_greedy(tables.q1[indb, n, :], [0, 1], 1, nr_prob=1.0)
        r = give_reward(g, hidden_phase)
        updates(qlearning, indb, n, g, r)
        details["experience"].append([b, n, g, r])
        details["Ps_greedy"].append(comm_success_prob(qlearning, 1.0))
    end = time.time() - start

    details["total_time"] = end

    os.makedirs(
        "../data_rec/experiments/{}/".format(experiment_index), exist_ok=True
    )
    with open(
        "../data_rec/experiments/{}/details.pickle".format(experiment_index),
        "wb",
    ) as f:
        pickle.dump(details, f, protocol=pickle.HIGHEST_PROTOCOL)

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(betas_grid, qlearning.q0, label=r"$Q(\beta)$")
    ax.set_xlabel(r"$\beta$")
    ax.plot(
        betas_grid,
        [1 - bayes_decision_error_probability(b) for b in betas_grid],
        label=r"$P_s(\beta)$",
    )
    ax.legend(prop={"size": 20})
    plt.savefig(path + "q0.png")

    stacked_history = np.stack(details["experience"])
    _, pstar, bstar = model_aware_optimal(betas_grid)

    plt.figure(figsize=(20, 20))
    ax = plt.subplot(311)
    ax.plot(stacked_history[:, 0], linewidth=0.5, label=r"$\beta_t$")
    ax.axhline(bstar, color="red", label=r"$\beta^*$")
    ax.legend(prop={"size": 20})
    ax = plt.subplot(312)
    ax.plot(
        np.cumsum(stacked_history[:, -1])
        / np.arange(1, len(stacked_history[:, -1]) + 1),
        label=r"$R_t/t$",
    )
    ax.legend(prop={"size": 20})
    ax.axhline(1.0 - pstar, color="black", label=r"$P_s^*$")
    ax = plt.subplot(313)
    ax.plot(details["Ps_greedy"], label=r"$P_t$")
    ax.axhline(1.0 - pstar, color="black", label=r"$P_s^*$")
    ax.set_xlabel(r"$experiment$")
    ax.set_xscale("log")
    ax.legend(prop={"size": 20})
    plt.savefig(path + "learning_rates.png")

    #####


if __name__ == "__main__":
    main()
