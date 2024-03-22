"""
Stage run

The objective of this script is to run the experiment,
track he hyperparameters, save the values of interest 
and make the decisions.

"""

import numpy as np

from .utils import (
    Perr_model,
    p_model,
    ep_greedy,
    give_outcome,
    calculate_mean_rew,
    model_aware_optimal,
    Psq,
    QLearningParms0,
    give_reward,
)

# from Model_Semi_aware.intensity_guess import guess_intensity
import time

"""
Hyperparameters:
[0] Epsilon0.
[1] how fast epsilon varies.
[2] Dispersion of random Gaussian.
[3] Learning rate reset.
"""

from collections import namedtuple

# Better explicit that implicit...
Hyperparameters = namedtuple(
    "Hyperparameter",
    [
        "epsilon0",
        "delta_epsilon",
        "delta_learning_rate",
        "dispersion_random",
        "temperature",
    ],
)



# How the q-learning parameters update.
def updates(indb, result, guess, reward, qlearning:QlearningParms, lr=0.001):
    """
    Dado que se observó el índice indb, con resultado n,
    y estimación previa g, se actualizan los valores
    con una recompensa r y y learning rate lr.

    Parameters
    ----------
    indb : TYPE
        index choice 
    result : TYPE
        resultado de la observación actual.
    guess : TYPE
        clasificación estimada.
    reward : TYPE
        DESCRIPTION.
    qlearning : QLarningParms
        current values of the learning parameters.
    lr : TYPE, optional
        Learning rate. The default is 0.001.

    Returns
    -------
    result : QLearningParms
        the new values of the learning parameters.

    """
    # TODO: check if it is possible to update 1-g too.
    n = result
    q0, q1, n0, n1 = qlearning
    q1[indb, n, g] += (1 / n1[indb, n, guess]) * (reward - q1[indb, n, guess])
    q0[indb] += (1 / n0[indb]) * np.max(
        [q1[indb, n, g] for g in [0, 1]] - q0[indb]
    )
    n0[indb] += 1
    n1[indb, n, guess] += 1
    return QLearningParms(q0, q1, n0, n1)


# How the learning rate changes when the environment changes.
# It could be interesting to change the reward with the mean_rew.
def Update_reload(n0, n1, mean_rew, restart_point, restart_epsilon):
    """
    Resetea n0 y n1 cuando el cambio es grande

    Parameters
    ----------
    n0 : TYPE
        parameter.
    n1 : TYPE
        DESCRIPTION.
    mean_rew : float
        current mean reward.
    restart_point : TYPE
        DESCRIPTION.
    restart_epsilon : float
        threshold value to restart.

    Returns
    -------
    n0 : TYPE
        DESCRIPTION.
    n1 : TYPE
        DESCRIPTION.
    epsilon : TYPE
        DESCRIPTION.

    """
    epsilon = restart_epsilon
    for i, n1_i in enumerate(n1):
        n0[i] = restart_point
        if n0[i] < 1:
            n0[i] = 1
        for j, n1_i_j in enumerate(n1_i):
            for k, n1_i_j_k in enumerate(n1_i_j):
                n1[i, j, k] = restart_point
                if n1_i_j_k < 1:
                    n1[i, j, k] = 1
    return n0, n1, epsilon


# Reload the Q-function with the model.
def reset_with_model(alpha, beta_grid, q0, q1):
    """
    Reload the Q-function assuming a model

    Parameters
    ----------
    alpha : TYPE
        DESCRIPTION.
    beta_grid : TYPE
        DESCRIPTION.
    q0 : TYPE
        DESCRIPTION.
    q1 : TYPE
        DESCRIPTION.

    Returns
    -------
    q0 : TYPE
        DESCRIPTION.
    q1 : TYPE
        DESCRIPTION.

    """
    for i in range(len(beta_grid)):
        q0[i] = 1 - Perr_model(beta_grid[i], alpha)

    for i in range(len(q1)):
        for j in range(len(q1[i])):
            for k in range(len(q1[i, j])):
                q1[i, j, k] = p_model(
                    (-1) ** (k + 1) * alpha, -beta_grid[i], j
                ) / (
                    p_model(-alpha, -beta_grid[i], j)
                    + p_model(alpha, -beta_grid[i], j)
                )

    return q0, q1


# Change in beta.
def Experiment_noise_1(q0, q1, betas_grid, hyperparam, epsilon, alpha, lambd):
    """Simulate the experiment changing beta"""
    hidden_phase = np.random.choice([0, 1])

    indb, b = ep_greedy(
        q0, betas_grid, hyperparam.delta_learning_rate, ep=epsilon
    )
    n = give_outcome(hidden_phase, b, alpha=alpha, lambd=lambd)
    indg, g = ep_greedy(
        q1[indb, n, :], [0, 1], hyperparam.delta_learning_rate, ep=epsilon
    )
    r = give_reward(g, hidden_phase)
    return indb, b, n, indg, g, r


# Change in the priors
def Experiment_noise_2(q0, q1, betas_grid, hyperparam, epsilon, alpha, lambd):
    """Simulate the experiment with modified priors"""
    hidden_phase = np.random.choice([0, 1], p=[0.5 - lambd, 0.5 + lambd])
    indb, b = ep_greedy(
        q0, betas_grid, hyperparam.delta_learning_rate, ep=epsilon
    )
    n = give_outcome(hidden_phase, b, alpha=alpha, lambd=0)
    indg, g = ep_greedy(
        q1[indb, n, :], [0, 1], hyperparam.delta_learning_rate, ep=epsilon
    )
    r = give_reward(g, hidden_phase)
    return indb, b, n, indg, g, r


# Exactly the same but checks with model to update initial parameters.
def Run_Experiment(
    details,
    N,
    alpha,
    hyperparam=(),
    delta1=1000,
    current=1.5,
    lambd=0.0,
    model=True,
    noise_type=None,
):
    """
    Makes the experiment and updates the decision of the agent.

    Parameters
    ----------
    details : dictionary
        All the information about the agent parameters and the previous experiments.
    N : int
        Amount of states use for the training.
    alpha : float
        Intensity of the state used in the experiment.
    hyperparam: Hyperparameters
        Hyperparameters used for the Q-learning algorithm.
    delta1: int
        Amount of rewards used to calculate the mean value.
    current: float.
        intensity predicted for the model previously.
    lambd: float
        Amount of unkwnown noise. 0.0 meaning that there is no extra noise in the channel.
    model: bool
        True if the agent can use a model to predict initial values, False otherwise.
    noise_type: int
        1 -> change in the expected displacement, 2 -> change in the prior.


    Returns
    -------
    details: dictionary.
        All the information about the agent parameters and the previous experiments.

    """
    start = time.time()
    mean_rew = float(details["mean_rewards"][-1])
    points = [mean_rew, float(details["mean_rewards"][-2])]
    means = details["means"]
    q0, q1, n0, n1 = details["tables"]
    betas_grid = details["betas"]

    guessed_intensity = current
    epsilon = float(details["ep"])
    checked = False

    for experiment in range(0, N):
        if epsilon > 0.05:
            epsilon -= hyperparam.delta_epsilon
        else:
            epsilon = 0.05
        if experiment % (N // 10) == 0:
            print(experiment)

        if noise_type == 0:
            pass
        if noise_type == 1:
            indb, b, n, indg, g, r = Experiment_noise_1(
                q0, q1, betas_grid, hyperparam, epsilon, alpha, lambd
            )
        if noise_type == 2:
            indb, b, n, indg, g, r = Experiment_noise_2(
                q0, q1, betas_grid, hyperparam, epsilon, alpha, lambd
            )

        means, mean_rew = calculate_mean_rew(means, mean_rew, r, delta1)

        # Check if the reward is smaller
        if experiment % delta1 == 0:
            points[0] = points[1]
            points[1] = mean_rew
            mean_deriv = points[1] - points[0]
            if mean_deriv <= -0.2 and n0[indb] > 300:
                n0, n1, epsilon = Update_reload(
                    n0,
                    n1,
                    mean_rew,
                    hyperparam.dispersion_random,
                    hyperparam.epsilon0,
                )
                checked = True

        # If it looks like changed, guess a new intensity to verify.
        if model and checked:
            guessed_intensity = guess_intensity(
                alpha, delta1 * 10, lambd=lambd
            )
            print(guessed_intensity)
            if np.abs(guessed_intensity - current) > 5 / np.sqrt(delta1 * 10):
                current = guessed_intensity
                q0, q1 = reset_with_model(current, betas_grid, q0, q1)
            checked = False

        q0, q1, n0, n1 = updates(indb, n, g, r, q0, q1, n0, n1)

        _, pstar, bstar = model_aware_optimal(
            betas_grid, alpha=alpha, lambd=lambd
        )

        details["mean_rewards"].append(mean_rew)
        details["means"] = means
        details["experience"].append([b, n, g, r / (1 - pstar)])
        details["Ps_greedy"].append(
            Psq(
                q0,
                q1,
                betas_grid,
                hyperparam.delta_learning_rate,
                alpha=alpha,
                lambd=lambd,
            )
        )
    details["tables"] = [q0, q1, n0, n1]
    end = time.time() - start
    details["total_time"] = end
    details["ep"] = f"{epsilon}"
    print(n0)
    return details
