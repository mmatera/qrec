"""
Find unobservables


"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import dual_annealing

sys.path.insert(0, os.getcwd())


from qrec.utils import detection_state_probability

PATH = "Model_Semi_aware/"


def probability_geometric_mean_complement(alpha, signs, betas, observations):
    """
    Compute the difference between 1 and the geometric mean of a sequence of
    observations


    results given states |sign*alpha> are detected by
    a detector tunned with an offset beta

    Parameters
    ----------
    alpha : float
        the displacement of the state
    signs : list
        a list of signs for alpha in the source
    betas : list
        a list of displacements in the detector.
    observations : list
        the resulting observations

    Returns
    -------
    float
        the joint probability.

    """
    prob = np.float64(1.0)
    for i in range(len(signs)):
        prob *= detection_state_probability(
            alpha * signs[i] + betas[i], observations[i]
        ) ** (1 / len(signs))
    return 1 - prob


def find_optimal_intensity(states, betas, observations):
    bounds = [[0, 2]]
    prediction = dual_annealing(
        probability_geometric_mean_complement,
        bounds,
        args=(states, betas, observations),
    )

    return prediction.x[0]


def experiments(alpha, duration):
    observations = []
    betas = []
    states = []
    for i in range(duration):
        beta = np.random.uniform(0, 1)
        state = (-1) ** np.random.randint(0, 2)

        if np.random.uniform(0, 1) < detection_state_probability(
            state * alpha + beta, 0
        ):
            observations.append(0)
        else:
            observations.append(1)
        betas.append(beta)
        states.append(state)
    return observations, states, betas


def __main__():
    """main"""
    alpha = 0.25  # This is not accessible in the experiment

    results = []
    max_experiments = 500
    for i in range(0, max_experiments):
        print(i)
        duration = i + 1
        observations, states, betas = experiments(alpha, duration)
        bounds = [[0, 2]]
        prediction = dual_annealing(
            probability_geometric_mean_complement,
            bounds,
            args=(states, betas, observations),
        )
        results.append(prediction.x[0])
    x = np.linspace(0, max_experiments, max_experiments)

    plt.scatter(x, results, label="predicted intensities", s=3)
    plt.axhline(alpha, color="black")
    plt.ylabel("prediction")
    plt.xlabel("experiments used")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    __main__()
