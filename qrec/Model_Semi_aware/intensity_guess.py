import os
import sys

import numpy as np

path = "Model_Semi_aware/"
sys.path.insert(0, os.getcwd())
import matplotlib.pyplot as plt

from qrec.utils import p


def guess_intensity(alpha, delta1, lambd):
    """Guess the intensity from the paramters"""
    # TODO: Implement me
    raise NonImplementedError


def experiments(alpha, duration):
    observations = []
    for i in range(duration):
        if np.random.uniform(0, 1) < p(alpha, 0):
            observations.append(0)
        else:
            observations.append(1)
    return observations


if __name__ == "__main__":
    alpha = 0.5  # This is not accessible in the experiment
    quantity = 10
    for j in range(quantity):
        print(j)
        results = []
        tot_experiments = 500
        observations = experiments(alpha, tot_experiments)
        int_med = 0
        for i in range(0, len(observations)):
            int_med *= i
            int_med += observations[i]
            int_med /= i + 1
            if int_med != 1:
                results.append(np.sqrt(-np.log(1 - int_med)))

        x = np.linspace(
            tot_experiments - len(results), tot_experiments, len(results)
        )

        plt.plot(x, results)
    plt.axhline(alpha, color="black")
    plt.ylabel("prediction")
    plt.xlabel("experiments used")
    plt.legend()
    plt.show()
