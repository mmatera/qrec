"""
find unobservables


"""


import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import choice, uniform
from scipy.optimize import dual_annealing

PATH = "Model_Semi_aware/"
sys.path.insert(0, os.getcwd())


from .utils import p


def probability(alpha, signs, betas, observations):
    """
    Compute one minus the geometric mean of the probabilities
    associated to a realization
    of signs, betas an observations.
    """
    prob = np.float64(1.0)
    exponent = 1.0 / len(signs)
    for sign, beta, obs in zip(signs, betas, observations):
        prob *= p(alpha * sign + beta, obs) ** exponent
    return 1 - prob


def find_optimal_intensity(states, betas, observations):
    """
    Determine the optimal value of alpha to maximize
    the geometric mean of the success probabilities.
    """
    bounds = [[0, 2]]
    prediction = dual_annealing(probability, bounds, args=(states, betas, observations))

    return prediction.x[0]


def experiments(alpha, duration):
    """
    Simulate the results of a set of experiments
    assuming alpha.

    alpha: float
        the parameter of the distribution.
    duration: int
        the number of experiments.

    return values
    -------------

    observations:


    states: list[int]
        a sequence of +/-1 uniformly distributed.

    betas: list[float]
        a sequence of float uniform random values



    """

    def prob_dist(x):
        prob_0 = p(x, 0)
        return (prob_0, 1 - prob_0)

    betas = uniform(0, 1, duration)
    states = choice([-1, 1], duration)
    observations = [
        choice([0, 1], p=prob_dist(state * alpha + beta))
        for state, beta in zip(states, betas)
    ]

    return observations, states, betas


if __name__ == "__main__":
    alpha = 0.25  # This is not accessible in the experiment

    results = []
    max_experiments = 500
    for i in range(0, max_experiments):
        print(i)
        duration = i + 1
        observations, states, betas = experiments(alpha, duration)
        bounds = [[0, 2]]
        prediction = dual_annealing(
            probability, bounds, args=(states, betas, observations)
        )
        results.append(prediction.x[0])
    x = np.linspace(0, max_experiments, max_experiments)

    plt.scatter(x, results, label="predicted intensities", s=3)
    plt.axhline(alpha, color="black")
    plt.ylabel("prediction")
    plt.xlabel("experiments used")
    plt.legend()
    plt.show()
