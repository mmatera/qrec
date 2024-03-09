import numpy as np
from scipy.optimize import minimize

# Probability of observing 0 or 1.
def p(alpha, beta, lambd, n):
    """
    p(n|alpha), born rule
    """
    pr = np.exp(-np.abs(alpha + (beta * (1 + lambd)))**2)
    return [pr, 1-pr][n]

def p_model(alpha, beta, n):
    """
    p(n|alpha), born rule
    """
    pr = np.exp(-np.abs(alpha + beta)**2)
    return [pr, 1-pr][n]

def Perr(beta,alpha=0.4, lambd=0.0, noise_type = 1):
    """
    Error probability given beta, alpha and noise lambd
    noise_type : int
                1 -> Change in the value of beta.  2 -> Change in the prior.
    """
    if noise_type == 1:
        ps=0
        p_sign = [0.5, 0.5]
        for n in range(2):
            ps+=np.max([p(sgn*alpha, beta, lambd, n) * p_sign[ind] for ind,sgn in enumerate([-1,1])])
    if noise_type == 2:
        ps=0
        p_sign = [0.5 + lambd, 0.5 - lambd]
        for n in range(2):
            ps+=np.max([p(sgn*alpha, beta, 0, n) * p_sign[ind] for ind,sgn in enumerate([-1,1])])
    return 1-ps

def Perr_model(beta,alpha=0.4):
    """
    Error probability given by the model used.
    """
    ps=0
    for n in range(2):
        ps+=np.max([p_model(sgn*alpha, beta, n) for ind,sgn in enumerate([-1,1])])
    return 1-ps/2


# Theoretical limit calculation.
def model_aware_optimal(betas_grid, alpha=0.4, lambd=0.0, noise_type=1):
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

    mmin = minimize(Perr, x0=-alpha, args=(alpha, lambd, noise_type),bounds = [(np.min(betas_grid), np.max(betas_grid))])
    p_star = mmin.fun
    beta_star = mmin.x
    return mmin, p_star, beta_star

# Add a value for the mean reward using delta1 values.
def calculate_mean_rew(means, mean_rew, r, max_len):
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
        
    means.append(r)
    mean_rew = np.average(means)
        
    if len(means) > max_len:
        means = means[-max_len:]
    return means, mean_rew

####   Q-Learning approach
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
    q0 = np.zeros(betas_grid.shape[0])  #Q(beta)
    q1 = np.zeros((betas_grid.shape[0],2,2)) # Q(beta,n; g)
    n0 = np.ones(betas_grid.shape[0])  #Q(beta)
    n1 = np.ones((betas_grid.shape[0],2,2)) # Q(beta,n; g)
    return betas_grid, [q0, q1,n0,n1]

# Choose the point with the bigger success probability.
def greedy(arr):
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

    return np.random.choice(np.where( arr == np.max(arr))[0])

# Makes a Gaussian distribution over the values with the current biggest success probabilities.
def ProbabilityRandom(val, maximum, delta1):
    """
    Parameters
    ----------
    N : TYPE
        number of samples.
    val : float | np.ndarray
        value / values where the gaussian is evaluated.
    maximum : float
        Center of the gaussian
    delta1 : float
        variance related.

    Returns
    -------
    flat | np.array()
        The gaussian function evaluated over val.

    """
    k = delta1 * (maximum-val)
    return np.exp(-(k**2))


def near_random(q0, delta1):
    """
    return an index according the probability 
    distribution described by arr

    Parameters
    ----------
    q0 : TYPE
        Experimentally calculated success probability for each point.
    delta1 : TYPE
        Dispersion for the random values.

    Returns
    -------
    i : TYPE
        index of the action taken.

    """
    maximum = max(q0)
    weight = np.array([ProbabilityRandom(q0[i], maximum, delta1) for i in range(len(q0))])
    weight[list(q0).index(maximum)] = 0
    weight /= np.cumsum(weight)[-1]

    return np.random.choice([i for i in range(len(q0))], p=weight)

# Decides if running a random displacement or the one with the bigger reward.
def ep_greedy(qvals, actions, delta1, ep=1.):
    """
    Decide if running a random
    displacement or the one with the biggest reward.

    policy(q1, betas_grid)
    policy(q1[1,0,:], [0,1])
    """
    if np.random.random() < ep:
        inda = near_random(qvals, delta1)
    else:
        inda = greedy(qvals)
    return inda, actions[inda]

# Experiment.
def give_outcome(hidden_phase, beta, alpha=0.4, lambd=0.0):
    """
    hidden_phase in {0,1}
    """
    return np.random.choice(np.array([0,1]), p= [p(alpha*(-1)**hidden_phase, beta, lambd, n) for n in [0,1]])

# reward. 1 -> correct. 0 -> incorrect
def give_reward(g, hidden_phase):
    """

    Parameters
    ----------
    g : int
        DESCRIPTION.
    hidden_phase : int
        DESCRIPTION.

    Returns
    -------
    result
        1. if g == hidden_phase, 0 otherwise
    """
    if int(g) == int(hidden_phase):
        return 1.
    else:
        return 0.


def Psq(q0,q1,betas_grid, delta1, alpha=0.4, lambd=0.0):
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
    ps=0
    indb, b = ep_greedy(q0, betas_grid, delta1, ep=0)
    for n in range(2):
        indg, g = ep_greedy(q1[indb,n,:], [0,1], delta1, ep=0)
        ps+=p(alpha*(-1)**g, b, lambd,n)
    return ps/2
