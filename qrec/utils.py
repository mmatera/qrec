"""
Utils
"""

# import os
# import pickle
# import time

# import matplotlib.pyplot as plt
import numpy as np
# from numba import jit
from scipy.optimize import minimize
# from tqdm import tqdm


# Probability of observing 0 or 1.
def probability_distribution(alpha: float, n: int):
    """
    p(n|alpha), born rule
    """
    pr = np.exp(-((alpha) ** 2))
    return [pr, 1 - pr][n]


# Error probability.
def error_probability(beta, alpha=0.4):
    """
    Error probability given beta and alpha
    """
    ps = 0
    for n in range(2):
        ps += np.max([probability_distribution(sgn * alpha + beta, n) for sgn in [-1, 1]])
    return 1 - ps / 2


# Theoretical limit calculation.
def model_aware_optimal(betas_grid, alpha=0.4):
    """
    Find the optimal parameters that minimize
    `error_probability`
    using a model.

    Parameters
    ----------
    betas_grid : TYPE
        DESCRIPTION.
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.4.

    Returns
    -------
    mmin : TYPE
        minimum value for the error probability.
    p_star : TYPE
        p where the minimum is attained .
    beta_star : TYPE
        beta where the minimum is attained.

    """
    # Landscape inspection
    mmin = minimize(
        error_probability,
        x0=-alpha, args=(alpha), bounds=[(np.min(betas_grid), np.max(betas_grid))]
    )
    p_star = mmin.fun
    beta_star = mmin.x
    return mmin, p_star, beta_star


# Add a value for the mean reward using delta1 values.
def calculate_mean_rew(means: list, mean_rew: float, r: float, max_len: int):
    """
    Compute the mean reward from the previous values and
    the new reward.

    Parameters
    ----------
    means : list
        DESCRIPTION.
    mean_rew : float
        DESCRIPTION.
    r : float
        The new reward value.
    max_len : int
        The length of the memory of previous rewards.

    Returns
    -------
    means : list
        new list of means
    mean_rew : f
        DESCRIPTION.

    """
    # Este código parece querer hacer lo mismo que
    #     means_rew = np.average(means)
    # pero usando el promedio anterior para ahorrarse 
    # las sumas. ¿Realmente se gana tanto?
    
    means_sum = mean_rew *len(means) + r
    means.append(r)
    means_len = len(means)
    mean_rew = means_sum / means_len 
    
    if means_len > max_len:
        means = means[-max_len:]

    
    return means,  means_rew


# Q-Learning approach
def define_q(nbetas=10):
    """
    Generate the initial structures for the Q-learning
    approach.

    Parameters
    ----------
    nbetas : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    betas_grid : np.ndarray
        the range of intermediate values of beta
    list
        the other parameters.

    """

    betas_grid = np.linspace(-2, 0, nbetas)
    q0 = np.zeros(betas_grid.shape[0])  # Q(beta)
    q1 = np.zeros((betas_grid.shape[0], 2, 2))  # Q(beta,n; g)
    n0 = np.ones(betas_grid.shape[0])  # Q(beta)
    n1 = np.ones((betas_grid.shape[0], 2, 2))  # Q(beta,n; g)
    return betas_grid, [q0, q1, n0, n1]


# Choose the point with the bigger success probability.
def greedy(arr: np.ndarray):
    """
    Pick a random element from arr. If the element
    is the maximum value of arr, return 1. Otherwise,
    return 0.

    Parameters
    ----------
    arr : np.ndarray
        a list of numbers

    Returns
    -------
    int:
        1 if the choosen element is a maximum. 0 otherwize.
    """

    return np.random.choice(np.where(arr == np.max(arr))[0])


# Makes a Gaussian distribution over the values with
# the current biggest success probabilities.
def ProbabilityRandom(N, val, maximum, T, delta1):
    """
    Parameters
    ----------
    N : TYPE
        number of samples.
    val : float | np.ndarray
        value / values where the gaussian is evaluated.
    maximum : float
        Center of the gaussian
    T : float
        Initial variance related.
    delta1 : float
        final variance related.

    Returns
    -------
    flat | np.array()
        The gaussian function evaluated over val.

    """

    k = delta1 * np.abs(maximum - val) / (N * (1 + T))
    return np.exp(-(k**2))


# returns the index of the displacement choosen,
#  given the probability distribution.
def near_random(arr, ep, variation, temp_rel=10):
    """
    return an index according the probability 
    distribution described by arr

    Parameters
    ----------
    arr : TYPE
        DESCRIPTION.
    ep : TYPE
        DESCRIPTION.
    variation : TYPE
        DESCRIPTION.
    temp_rel : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    i : TYPE
        DESCRIPTION.

    """

    maximum = max(arr)
    arr_len = len(arr)
    prob = []
    T = ep * temp_rel
    
    # Comentario: Acá entiendo que querés una 
    # distribución de probabilidad
    # para arr de manera que sus valores se distribuyan como
    # una gaussiana centrada en el máximo de arr,
    # y con ancho  `variation`.
    # Fijate porque me parece que esto no está bien.
    
    for i,arr_i in enumerate(arr):
        prob.append(ProbabilityRandom(arr_len,
                                      arr_i, 
                                      maximum,
                                      T, 
                                      variation))
    
    # Cumulative Probability Distribution
    cum_distribution = np.cumsum(prob)
    cum_distribution /= cum_distribution[-1]
    
    # numpy tiene la función randomchoice,
    # a la que le podés pasar una distribución
    # de probabilidad...
    random = np.random.uniform(0, 1)
    for i in range(arr_len):
        if random <= prob[i]:
            return i
    return prob[-1]


# Decides if running a random displacement or the one with the bigger reward.
def ep_greedy(qvals, actions, variation, temp_rel, ep=1.0):
    """
    Decide if running a random
    displacement or the one with the biggest reward.

    policy(q1, betas_grid)
    policy(q1[1,0,:], [0,1])
    """
    if np.random.random() < ep:
        inda = near_random(qvals, ep, variation, temp_rel)
    else:
        inda = greedy(qvals)
    return inda, actions[inda]


# Experiment.
def give_outcome(hidden_phase, beta, alpha=0.4):
    """
    hidden_phase in {0,1}
    """
    return np.random.choice(
        np.array([0, 1]),
        [probability_distribution(alpha * (-1) ** hidden_phase +
                     beta, n) for n in [0, 1]]
    )


# reward. 1 -> correct. 0 -> incorrect
def give_reward(g, hidden_phase) -> int:
    """

    Parameters
    ----------
    g : TYPE
        DESCRIPTION.
    hidden_phase : TYPE
        DESCRIPTION.

    Returns
    -------
    result
        1. if g == hidden_phase, 0 otherwise
    """
    if int(g) == int(hidden_phase):
        return 1.0

    return 0.0


def Psq(q0, q1, betas_grid, variation, temp_rel, alpha=0.4):
    """

    Parameters
    ----------
    q0 : TYPE
        DESCRIPTION.
    q1 : TYPE
        DESCRIPTION.
    betas_grid : TYPE
        DESCRIPTION.
    variation : TYPE
        DESCRIPTION.
    temp_rel : TYPE
        DESCRIPTION.
    alpha : TYPE, optional
        DESCRIPTION. The default is 0.4.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    ps = 0
    indb, b = ep_greedy(q0, betas_grid, variation, temp_rel, ep=0)
    for n in range(2):
        indg, g = ep_greedy(q1[indb, n, :], [0, 1], variation, temp_rel, ep=0)
        ps += probability_distribution(alpha * (-1) ** g + b, n)
    return ps / 2
