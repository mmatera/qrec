import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
import matplotlib.pyplot as plt

from qrec.utils import detection_state_probability

path = "Model_Semi_aware/"


def guess_intensity(alpha, duration, lambd=0.0):
    """
    Estimates alpha from the results of previous experiments.
    """
    observations = np.random.choice(
        [0, 1],
        duration,
        p=[
            detection_state_probability(alpha, 0, lambd, 0),
            detection_state_probability(alpha, 0, lambd, 1),
        ],
    )
    p0 = 0
    for i in range(duration):
        if observations[i] == 0:
            p0 += 1 / duration
    return np.sqrt(-np.log(p0))


if __name__ == "__main__":
    alpha = 0.5  # This is not accessible in the experiment
    results = []
    tot_experiments = 2000
    disperssion = np.array([alpha + 1 / np.sqrt(i) for i in range(tot_experiments)])
    for i in range(tot_experiments):
        print(i)
        results.append(guess_intensity(alpha, i))

    plt.plot(results, label=r"$\alpha^{guess}$")
    plt.plot(disperssion, label=r"$\frac{1}{\sqrt{N}}$")
    plt.axhline(alpha, color="black")
    plt.ylabel(r"$\alpha$")
    plt.xlabel(r"$N$")
    plt.legend()
    plt.show()
