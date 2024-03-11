import os
import sys

import numpy as np


import matplotlib.pyplot as plt
from np.random import choice

sys.path.insert(0, os.getcwd())

from qrec.utils import p


path = "Model_Semi_aware/"


def experiments(alpha, duration):
    """
    A list of 0 and 1 distributed according to
    p(x|alpha).

    alpha: float
        parameter of the distribution

    duration:
        the size of the list
    """
    prob_0 = p(alpha, 0)
    return choice(
        (
            0,
            1,
        ),
        size=duration,
        p=prob_0,
    )


def main():
    """Main loop"""

    alpha = 0.5  # This is not accessible in the experiment
    independent_experiments = 10
    for j in range(independent_experiments):
        print("Experiment j")
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

        x = np.linspace(tot_experiments - len(results), tot_experiments, len(results))

        plt.plot(x, results, label=f"Experiment {j}")
    plt.axhline(alpha, color="black")
    plt.ylabel("prediction")
    plt.xlabel("experiments used")
    plt.legend()
    plt.show()
    return -1


if __name__ == "__main__":
    exit(main())
