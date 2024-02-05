import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tqdm import tqdm
import time
import pickle
import os


def p(alpha,n):
    """
    p(n|alpha), born rule
    """
    pr = np.exp(-(alpha)**2)
    return [pr, 1-pr][n]

def Perr(beta,alpha=0.4):
    ps=0
    for n in range(2):
        ps+=np.max([p(sgn*alpha + beta,n) for sgn in [-1,1]])
    return 1-ps/2

def model_aware_optimal(betas_grid, alpha=0.4):
    #### Landscape inspection
    mmin = minimize(Perr, x0=-alpha, args=(alpha),bounds = [(np.min(betas_grid), np.max(betas_grid))])
    p_star = mmin.fun
    beta_star = mmin.x
    return mmin, p_star, beta_star




####   Q-Learning approach
def define_q(nbetas=10):
    betas_grid = np.linspace(-2, 0, nbetas)
    q0 = np.zeros(betas_grid.shape[0])  #Q(beta)
    q1 = np.zeros((betas_grid.shape[0],2,2)) # Q(beta,n; g)
    n0 = np.ones(betas_grid.shape[0])  #Q(beta)
    n1 = np.ones((betas_grid.shape[0],2,2)) # Q(beta,n; g)
    return betas_grid, [q0, q1,n0,n1]

def greedy(arr):
    return np.random.choice(np.where( arr == np.max(arr))[0])


def ep_greedy(qvals, actions, ep=1.):
    """
    policy(q1, betas_grid)
    policy(q1[1,0,:], [0,1])
    """
    if np.random.random() < ep:
        inda = np.random.choice(np.array(range(len(actions))))
    else:
        inda = greedy(qvals)
    return inda,actions[inda]

def give_outcome(hidden_phase, beta, alpha=0.4):
    """
    hidden_phase in {0,1}
    """
    return np.random.choice(np.array([0,1]), p= [p(alpha*(-1)**hidden_phase + beta,n) for n in [0,1]])

def give_reward(g, hidden_phase):
    if int(g) == int(hidden_phase):
        return 1.
    else:
        return 0.


def Psq(q0,q1,betas_grid,alpha=0.4):
    ps=0
    indb, b = ep_greedy(q0, betas_grid, ep=0)
    for n in range(2):
        indg, g = ep_greedy(q1[indb,n,:], [0,1], ep=0)
        ps+=p(alpha*(-1)**g + b,n)
    return ps/2
