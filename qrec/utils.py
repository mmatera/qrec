import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
import time
import pickle
import os
from numba import jit

# Probability of observing 0 or 1.
def p(alpha,n):
    """
    p(n|alpha), born rule
    """
    pr = np.exp(-(alpha)**2)
    return [pr, 1-pr][n]

# Error probability.
def Perr(beta,alpha=0.4):
    ps=0
    for n in range(2):
        ps+=np.max([p(sgn*alpha + beta,n) for sgn in [-1,1]])
    return 1-ps/2

# Theoretical limit calculation.
def model_aware_optimal(betas_grid, alpha=0.4):
    #### Landscape inspection
    mmin = minimize(Perr, x0=-alpha, args=(alpha),bounds = [(np.min(betas_grid), np.max(betas_grid))])
    p_star = mmin.fun
    beta_star = mmin.x
    return mmin, p_star, beta_star

# Add a value for the mean reward using delta1 values.
def calculate_mean_rew(means, mean_rew, r, delta1):
    means.append(r)
    # To be acurate this should be the lenght of current list, but it just adds noise, this points can be ignored.
    mean_rew += r / delta1
        
    if len(means) > delta1:
        mean_rew -= means[0] / delta1
        means.pop(0)
    return means, mean_rew

####   Q-Learning approach
def define_q(nbetas=10):
    betas_grid = np.linspace(-2, 0, nbetas)
    q0 = np.zeros(betas_grid.shape[0])  #Q(beta)
    q1 = np.zeros((betas_grid.shape[0],2,2)) # Q(beta,n; g)
    n0 = np.ones(betas_grid.shape[0])  #Q(beta)
    n1 = np.ones((betas_grid.shape[0],2,2)) # Q(beta,n; g)
    return betas_grid, [q0, q1,n0,n1]

# Choose the point with the bigger success probability.
def greedy(arr):
    return np.random.choice(np.where( arr == np.max(arr))[0])

# Makes a Gaussian distribution over the values with the current biggest success probabilities.
def ProbabilityRandom(N, val, maximum, T, delta1):
    # delta1 -> final variance related
    # T -> Initial variance related
    k = delta1 * np.abs(maximum-val) / (N * (1 + T))
    return np.exp(-(k**2))

# returns the index of the displacement choosed given the probability distribution. 
def near_random(arr, ep, variation, temp_rel=10):
    maximum = max(arr)
    prob = []
    T = ep * temp_rel

    for i in range(len(arr)):
        prob.append(ProbabilityRandom(len(arr), arr[i], maximum, T, variation))

    for i in range(1, len(prob)):
        prob[i] = prob[i-1] + prob[i]

    prob = np.array(prob)
    prob /= prob[-1]

    random = np.random.uniform(0, 1)
    for i in range(len(prob)):
        if random <= prob[i]:
            return i

# Decides if running a random displacement or the one with the bigger reward.
def ep_greedy(qvals, actions, variation, temp_rel, ep=1.):
    """
    policy(q1, betas_grid)
    policy(q1[1,0,:], [0,1])
    """
    if len(actions) == 2:
        inda = np.random.choice(np.array(range(len(actions))))
    elif np.random.random() < ep:
        inda = near_random(qvals, ep, variation, temp_rel)
    else:
        inda = greedy(qvals)
    return inda, actions[inda]

# Experiment.
def give_outcome(hidden_phase, beta, alpha=0.4):
    """
    hidden_phase in {0,1}
    """
    return np.random.choice(np.array([0,1]), p= [p(alpha*(-1)**hidden_phase + beta,n) for n in [0,1]])

# reward. 1 -> correct. 0 -> incorrect
def give_reward(g, hidden_phase):
    if int(g) == int(hidden_phase):
        return 1.
    else:
        return 0.


def Psq(q0,q1,betas_grid, variation, temp_rel, alpha=0.4):
    ps=0
    indb, b = ep_greedy(q0, betas_grid, variation, temp_rel, ep=0)
    for n in range(2):
        indg, g = ep_greedy(q1[indb,n,:], [0,1], variation, temp_rel, ep=0)
        ps+=p(alpha*(-1)**g + b,n)
    return ps/2
