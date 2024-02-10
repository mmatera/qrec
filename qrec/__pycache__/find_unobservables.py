import numpy as np
from scipy.optimize import dual_annealing
from utils import p
import matplotlib.pyplot as plt
def Probability(alpha, sign, beta, observations):
    prob = np.float64(1.0)
    for i in range(len(sign)):
        prob *= (p(alpha * sign[i] + beta[i], observations[i]) ** (1/len(sign)))
    return 1 - prob

def experiments(alpha, duration):
    observations = []
    betas = []
    states = []
    for i in range(duration):
        beta = np.random.uniform(0, 1)
        state = ((-1) ** np.random.randint(0, 2))

        if np.random.uniform(0, 1) < p(state * alpha + beta, 0):
            observations.append(0)
        else:
            observations.append(1)
        betas.append(beta)
        states.append(state)
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
        prediction = dual_annealing(Probability, bounds, args=(states, betas, observations))
        results.append(prediction.x[0])
    x = np.linspace(0, max_experiments, max_experiments)

    plt.scatter(x, results, label="predicted intensities", s=3)
    plt.axhline(alpha, color="black")
    plt.ylabel("prediction")
    plt.xlabel("experiments used")
    plt.legend()
    plt.show()