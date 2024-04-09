"""
Policies functions


This module contains the functions that implement the different
update policies for the parameters of the detector.

"""
import numpy as np
from numpy.random import choice, random

from qrec.device_simulation import detection_state_probability
from qrec.qlearning import Qlearning_parameters

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
    max_value = np.max(arr)
    candidates = np.where(arr == max_value)[0]
    return choice(candidates)


# Makes a Gaussian distribution over the values with
# the current biggest success probabilities.
def gaussian_values(val, maximum, delta):
    """
    (former ProbabilityRandom)
    Evaluates an (unnormalized) gaussian
    function center at maximum,
    and with inverse dispersion delta
    over the values`val`.

    Parameters
    ----------
    val : float | np.ndarray
        value / values where the gaussian is evaluated.
    maximum : float
        Center of the gaussian
    delta : float
        inverse of the related variance.

    Returns
    -------
    flat | np.array()
        The gaussian function evaluated over val.

    """
    k = delta * (maximum - val)
    return np.exp(-(k**2))


def near_random(q_0, delta):
    """
    Choose an element of q_0 according to the probability distribution
    P_i \\propto exp(-(q_i-q_max)^2 * delta_1^2) if q_i!=q_max
                                                    else 0

    Parameters
    ----------
    q_0 : ndarray
        Experimentally calculated success probability for each point.
    delta : float
        Dispersion for the random values.

    Returns
    -------
    indx : int
        index of the action taken.

    """
    q0_size = len(q_0)
    max_idx = np.argmax(q_0)
    if q0_size >= 2:
        maximum = q_0[max_idx]
        weights = gaussian_values(q_0, maximum, delta)
        # Esto es raro: por qué 2 es especial?
        if q0_size != 2:
            weights[max_idx] = 0
        weights /= np.sum(weights)
        return choice(len(q_0), p=weights)
    # if there are two possibilities, try with the other
    if q0_size > 1:
        return 1 - max_idx
    return 0



# Decides if running a random displacement or the one with the bigger reward.


def ep_greedy(rewards, actions, delta=0, near_prob=1.0):
    """
    Decide an action according to the estimated rewards.

    Parameters
    ----------
    rewards : float
        reward values associated to each action.
    actions : List
        Possible outputs
    delta : float
        The inverse dispersion used when near_random is used.
    near_prob : float, optional
        Probability of using `near_random` as the function
        used to choice the index. The default is 1., meaning
        that this function is always used.

    Returns
    -------
    action_ind : int
        index of the choosen action.
    action:
        The choosen element of `actions`.

    if near_probability == 1, use the `near_random` policy,
    which looks for actions with rewards close to the maximum value.
    How much close is controlled by delta: if delta=0, then
    any value different to the maximum value is chosen.
    For large values of delta, the choice is picked from
    actions with rewards close to the maximum value.

    if near_probability == 0, use the `greedy` policy,
    which looks for actions with maximum reward. `delta`
    is not used in this case.

    if 0<near_probability<1, the `near_random` policy is chosen
    with probability `near_probability`, and `greedy` with
    probability 1-`near_probability`
    """
    # A Random number is just used if near_prob is not 0 or 1.
    # This economizes both computing power and "randomness"
    use_near = (near_prob == 1) or (near_prob and random() < near_prob)
    inda = near_random(rewards, delta) if use_near else greedy(rewards)
    return inda, actions[inda]


