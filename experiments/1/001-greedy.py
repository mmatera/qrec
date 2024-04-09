import os
import pickle
import sys
import time
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.getcwd())

from qrec.stage_run import run_experiment
from qrec.device_simulation import give_outcome
from qrec.qlearning import Hyperparameters, define_q, give_reward
from qrec.policies import comm_success_prob, ep_greedy
from qrec.utils import (bayes_decision_error_probability,
                        model_aware_optimal)

experiment_index = 1
path = "experiments/{}/".format(experiment_index)


def read_cmd_args():
    """read the command line arguments"""
    parser = ArgumentParser(description="Experiment 1", add_help=False)
    parser.add_argument(
        "--help", "-h", help="show this help message and exit", action="help"
    )
    parser.add_argument(
        "--training-size",
        "-N",
        help="Size of the training set",
        type=int,
        dest="training_size",
        default=int(5e4),
        metavar="TRAINING_SIZE",
    )
    parser.add_argument(
        "--alpha",
        "-a",
        help="Nominal amplitude of the coherent state",
        type=float,
        dest="alpha",
        default=1.5,
        metavar="ALPHA",
    )
    parser.add_argument(
        "--beta-steps",
        "-g",
        help=("Granularity of the beta parameter space."),
        type=int,
        default=10,
        dest="beta_steps",
        metavar="BETA_STEPS",
    )
    parser.add_argument(
        "--dispersion",
        "-d",
        help=("Granularity of the beta parameter space."),
        type=float,
        default=1.0,
        dest="dispersion",
        metavar="DISPERSION",
    )
    parser.add_argument(
        "--epsilon",
        "-e",
        help=("Value of the epsilon parameter."),
        type=float,
        dest="epsilon",
        default=1.0,
        metavar="EPSILON",
    )

    parser.add_argument(
        "--lambda",
        "-l",
        help=("Amout of noise in the source."),
        type=float,
        default=0.0,
        dest="lambd",
        metavar="LAMBDA",
    )
    parser.add_argument(
        "--noise-type",
        "-t",
        help=("Type of noise introduced."),
        type=int,
        default=1,
        dest="noise_type",
        metavar="NOISE_TYPE",
    )
    parser.add_argument(
        "--random-seed",
        "-s",
        help=("Seed of the RNG."),
        type=int,
        dest="random_seed",
        default=0,
        metavar="RANDOM_SEED",
    )
    return parser.parse_args()


def make_plots(details: dict, alpha: float):
    """
    Build plots from details
    """
    details.keys()
    betas_grid = details["tables"].betas_grid
    stacked_history = np.stack(details["experience"])

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(betas_grid, details["tables"].q0, label=r"$Q(\beta)$")
    ax.set_xlabel(r"$\beta$")
    ax.plot(
        betas_grid,
        [1 - bayes_decision_error_probability(b, alpha=alpha) for b in betas_grid],
        label=r"$P_s(\beta)$",
    )
    ax.legend(prop={"size": 20})
    plt.savefig(path + "q0.png")

    _, pstar, bstar = model_aware_optimal(betas_grid, alpha=alpha)
    stacked_history = np.stack(details["experience"])
    counts, bins = np.histogram(stacked_history[:, 0], bins=len(betas_grid))
    x_bins = np.linspace(np.min(bins), np.max(bins), len(bins) - 1)

    plt.figure(figsize=(20, 20))
    ax = plt.subplot(311)
    ax.bar(x_bins, counts, label=r"$\beta_t$", width=np.std(bins[:2]))
    ax.axvline(bstar, color="black")
    ax.set_yscale("log")
    ax.legend(prop={"size": 20})
    ax = plt.subplot(312)
    ax.plot(
        np.cumsum(stacked_history[:, -1])
        / np.arange(1, len(stacked_history[:, -1]) + 1),
        label=r"$R_t/t$",
    )
    ax.legend(prop={"size": 20})
    ax.set_xscale("log")
    ax.axhline(1.0 - pstar, color="black", label=r"$P_s^*$")
    ax = plt.subplot(313)
    ax.plot(details["Ps_greedy"], label=r"$P_t$")
    ax.axhline(1.0 - pstar, color="black", label=r"$P_s^*$")
    ax.set_xlabel(r"$experiment$")
    ax.set_xscale("log")
    ax.legend(prop={"size": 20})
    plt.savefig(path + "learning_curve.png")


def save_data(details, experiment_index):
    """store the results of the experiment"""
    os.makedirs("data_rec/experiments/{}/".format(experiment_index), exist_ok=True)
    with open(
        "data_rec/experiments/{}/details.pickle".format(experiment_index),
        "wb",
    ) as f:
        pickle.dump(details, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    """main"""

    cmd_args = read_cmd_args()

    experiment_index = 1
    alpha = cmd_args.alpha
    beta_steps = cmd_args.beta_steps
    dispersion = cmd_args.dispersion
    epsilon = cmd_args.epsilon
    training_size = cmd_args.training_size

    lambd = cmd_args.lambd
    noise_type = cmd_args.noise_type
    seed = cmd_args.random_seed

    # Initializing
    np.random.seed(seed)
    tables = define_q()
    # ## run q-learning
    start = time.time()
    qlearning = define_q(beta_steps=beta_steps)

    details = {
        "index": experiment_index,
        "alpha": alpha,
        "ep": epsilon,
        "experience": [],
        "Ps_greedy": [],
        "seed": seed,
        "tables": tables,
        "betas": qlearning.betas_grid,
        "witness": [0, 0],
        "means": [],
        "greed_beta": [],
    }
    hyperparam = Hyperparameters(1, 2 / training_size, 0, 1, 3000)
    details = run_experiment(
        details,
        training_size,
        alpha,
        hyperparam,
        lambd=lambd,
        use_model=False,
        noise_type=noise_type,
    )
    print("The simulation took", time.time() - start)
    save_data(details, experiment_index)
    make_plots(details, alpha)

    #####


if __name__ == "__main__":
    main()
