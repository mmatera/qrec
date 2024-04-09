"""
Utility functions

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
    NEW = True
    if NEW:
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

    maximum = max(q_0)
    weight = np.array([gaussian_values(q0_i, maximum, delta) for q0_i in q_0])
    if len(weight) != 2:
        weight[list(q_0).index(maximum)] = 0
    weight /= np.cumsum(weight)[-1]
    # print(weight)

    return choice(list(range(len(q_0))), p=weight)


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


def comm_success_prob(
    qlearning: Qlearning_parameters, dispersion, alpha=0.4, detuning=0.0
):
    """
    (Former Psq)
    Compute the success probability of a communication.


    Parameters
    ----------
    qlearning : Qlearning_parameters
        Qlearning_parameters.
    dispersion : float
        dispersion of the gaussian.
    alpha : TYPE, optional
        Offset of the signal. The default is 0.4.
    detuning: float, optional
        detuning parameter in the detector.

    Returns
    -------
    float
        DESCRIPTION.

    """
    q_0 = qlearning.q0
    q_1 = qlearning.q1
    betas_grid = qlearning.betas_grid
    # Always use "near_prob" to choice beta and alpha
    # With this choice, dispersion is not required.
    near_prob = 1
    # Pick beta from the beta grid
    indb, beta = ep_greedy(q_0, betas_grid, dispersion, near_prob=near_prob)
    alpha_phases = [alpha, -alpha]

    def alpha_from_experiment(out):
        """
        returns alpha or -alpha according
        to the parameters
        """
        return ep_greedy(
            q_1[indb, out, :], alpha_phases, dispersion, near_prob=near_prob
        )[1]

    return 0.5 * sum(
        detection_state_probability(
            # pick alpha  or -alpha according q1 and the parameters.
            alpha_from_experiment(outcome),
            beta,
            detuning,
            outcome,
        )
        for outcome in range(2)
    )
