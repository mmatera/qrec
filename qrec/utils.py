"""
Utility functions

"""
from collections import namedtuple

import numpy as np
from numpy.random import random
from scipy.optimize import minimize

# Probability of observing 0 or 1.
# former  "p"

# Better explicit that implicit...
Hyperparameters = namedtuple(
    "Hyperparameters",
    [
        "eps_0",  # Epsilon0
        "delta_epsilon",  # epsilon0 change rate
        "delta",  # Dispersion of the random gaussian
        "delta_learning_rate",  # Learning rate reset
    ],
)


Qlearning_parameters = namedtuple(
    "Qlearning_parameters", ["q0", "q1", "n0", "n1", "betas_grid"]
)


def detection_state_probability(alpha, beta, detunning, outcome):
    """
    (former `p`)

    Compute the probability of obtaining `n`
    in the detector, assuming the state |alpha>
    is received, and the offset in detector
    is set to beta.

    Parameters
    ----------
    alpha : complex
        displacement in the received state
    beta : complex
        detector offset
    detunning : float
        failure in the exact tunning of the parameter
        beta in the detector.
    outcome : TYPE
        the outcome.

    Returns
    -------
    float
        the probability of getting the outcome `n`.

    """

    pr_0 = np.exp(-np.abs(alpha + (beta * (1 + detunning))) ** 2)
    return 1 - pr_0 if outcome else pr_0


def p_model(alpha, beta, outcome):
    """
    Compute the probability of obtaining `n`
    in the detector, assuming the state |alpha>
    is received, and the offset in detector
    is set to beta.

    Parameters
    ----------
    alpha : complex
        displacement in the received state |alpha>
    beta : complex
        detector offset
    outcome : TYPE
        the outcome.

    Returns
    -------
    float
        the probability of getting the outcome `outcome`.

    """
    return detection_state_probability(alpha, beta, 0, outcome)


def bayes_decision_error_probability(beta, alpha=0.4, noise_val=0.0, noise_type=1):
    """
    (former Perr)
    Error probability given beta, alpha and noise lambd
    assuming the Bayes' decision criterion.

    The Bayes' decision criterion establish that the probability
    distribution of the source is the one which produces
    the higgest probabilities for the current observations.

    The criterion not always provides the right answer. This
    function computes the probability of choosing the wrong source.


    Parameters
    ----------
    beta : complex
        detector offset
    alpha : complex
        displacement in the received state. The default is 0.4.

    noise_val : TYPE, optional
        failure in the exact tunning of the parameter
        beta in the detector.
        The default is 0.0.
    noise_type : int
                1 -> Change in the value of beta.
                2 -> Change in the prior.
                The default value is 1
    Returns
    -------
    float
        the probability of choose the wrong source.

    """
    prob = detection_state_probability

    if noise_type == 1:
        priors = [0.5, 0.5]
        detune = noise_val
    elif noise_type == 2:
        priors = [0.5 + noise_val, 0.5 - noise_val]
        detune = 0
    return 1 - sum(
        max(
            prob(sgn * alpha, beta, detune, outcome) * prior
            for sgn, prior in zip([-1, 1], priors)
        )
        for outcome in range(2)
    )


def perr_model(beta, alpha=0.4):
    """
    Error probability given by the model used.
    Is the same that
    bayes_decision_error_probability(beta, alpha, 0, 1)
    """
    return bayes_decision_error_probability(beta, alpha)


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
    vals = np.array(
        [
            bayes_decision_error_probability(
                beta, alpha=alpha, noise_val=lambd, noise_type=noise_type
            )
            for i, beta in enumerate(betas_grid)
        ]
    )
    mmin = vals.min()
    p_star = vals.min()
    beta_star = betas_grid[list(vals).index(mmin)]
    return mmin, p_star, beta_star


# Add a value for the mean reward using delta1 values.


def update_buffer_and_compute_mean(outcomes_buffer, value, max_len):
    """
    Compute the mean reward from the previous values and
    the new reward. Append the new reward to the buffer.

    Parameters
    ----------
    outcomes_buffer : list
        DESCRIPTION.
    value : float
        The new reward value.
    max_len : int
        The maximum size of the rewards buffer.

    Returns
    -------
    outcomes_buffer: list
        updated buffer of rewards.
    mean_val : float
        current average reward.

    """
    if True:
        outcomes_buffer.append(value)
        mean_rew = np.average(outcomes_buffer)
        if len(outcomes_buffer) > max_len:
            outcomes_buffer = outcomes_buffer[-max_len:]
        return outcomes_buffer, mean_rew

    # Another possibility. Check why this does not produce
    # the same result.
    curr_size = len(outcomes_buffer)
    if curr_size == max_len:
        outcomes_buffer[:-1] = outcomes_buffer[1:]
        outcomes_buffer[-1] = value
    elif curr_size < max_len:
        outcomes_buffer.append(value)
    else:
        # Reduce the size of the buffer
        last = max_len - 1
        outcomes_buffer[:last] = outcomes_buffer[last:]
        outcomes_buffer[last] = value
        otcomes_buffer.resize(max_len)

    mean_val = np.average(outcomes_buffer)
    return outcomes_buffer, mean_val
    


# Q-Learning approach


def define_q(beta_steps=10, range=[-2, 0]):
    """
    Generate the initial structures for the Q-learning
    approach.

    Parameters
    ----------
    nbetas : int, optional
        The number steps in the grid used to estimate beta. The default is 10.

    range : list, option
        The range of values that beta can take. The default is [-2, 0]

    Returns
    -------
    qlearn : Qlearning_parameters
        the learning parameters.

    """

    betas_grid = np.linspace(range[0], range[1], beta_steps)
    qlearn = Qlearning_parameters(
        np.zeros(betas_grid.shape[0]),  # Q(beta)
        np.zeros((betas_grid.shape[0], 2, 2)),  # Q(beta,n; g)
        np.ones(betas_grid.shape[0]),  # Q(beta)
        np.ones((betas_grid.shape[0], 2, 2)),  # Q(beta,n; g)
        betas_grid,
    )
    return qlearn


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
    return np.random.choice(np.where(arr == max_value)[0])


# Makes a Gaussian distribution over the values with the current biggest success probabilities.
def gaussian_values(val, maximum, delta1):
    """
    (former ProbabilityRandom)
    Evaluates an (unnormalized) gaussian
    function center at maximum,
    and with dispersion delta1
    over the values`val`.

    Parameters
    ----------
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
    k = delta1 * (maximum - val)
    return np.exp(-(k**2))


def near_random(q_0, delta1):
    """
    return an index according the probability
    distribution described by arr

    Parameters
    ----------
    q_0 : ndarray
        Experimentally calculated success probability for each point.
    delta1 : TYPE
        Dispersion for the random values.

    Returns
    -------
    indx : int
        index of the action taken.

    """
    maximum = max(q_0)
    weight = np.array([gaussian_values(q0_i, maximum, delta1) for q0_i in q_0])
    if len(weight) != 2:
        weight[list(q_0).index(maximum)] = 0
    weight /= np.cumsum(weight)[-1]
    # print(weight)

    return np.random.choice(list(range(len(q_0))), p=weight)


# Decides if running a random displacement or the one with the bigger reward.


def ep_greedy(qvals, actions, delta1, nr_prob=1.0):
    """
    Decide if running a random
    displacement or the one with the biggest reward.

    policy(q1, betas_grid)
    policy(q1[1,0,:], [0,1])


    Parameters
    ----------
    qvals : TYPE
        DESCRIPTION.
    actions : List
        Possible outputs
    delta1 : float
        DESCRIPTION.
    nr_prob : float, optional
        Probability of using `near_random` as the function
        used to choice the index. The default is 1., meaning
        that this function is always used.

    Returns
    -------
    inda : TYPE
        index of the choosen action.
    TYPE
        The choosen element of `actions`.

    """
    if random() < nr_prob:
        inda = near_random(qvals, delta1)
    else:
        inda = greedy(qvals)
    return inda, actions[inda]


# Experiment.


def give_outcome(hidden_phase, beta, alpha=0.4, lambd=0.0):
    """

    Simulates the outcome of a detection assuming that
    the signal has an offset (-1)**hidden_phase * alpha.

    Parameters
    ----------
    hidden_phase : int
        hidden_phase in {0,1}.
    beta : TYPE
        detector tunning.
    alpha : TYPE, optional
        The offset of the signal. The default is 0.4.
    lambd : float, optional
        degree of the detunning in the detector. The default is 0.0.

    Returns
    -------
    outcome: int
        0 or 1 choosen randomly acoording to the detection probability.
    """

    sgn = (-1) ** hidden_phase
    values = np.array([0, 1])
    p_0 = detection_state_probability(alpha * sgn, beta, lambd, 0)
    return np.random.choice(values, p=[p_0, 1 - p_0])


# reward. 1 -> correct. 0 -> incorrect


def give_reward(guess, hidden_phase):
    """
    Compute the reward. If the `guess` matches
    the `hidden_phase`, returns 1., else 0.

    Parameters
    ----------
    guess : int
        The guessed phase.
    hidden_phase : int
        The current phase.

    Returns
    -------
    reward: float
        1. if guess == hidden_phase, 0 otherwise
    """
    return int(int(guess) == int(hidden_phase))


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
    nr_prob = 1  # Always use "greedy" to choice beta and alpha
    # Pick beta from the beta grid
    indb, beta = ep_greedy(q_0, betas_grid, dispersion, nr_prob=nr_prob)
    alpha_phases = [alpha, -alpha]

    def alpha_from_experiment(out):
        """
        returns alpha or -alpha according
        to the parameters
        """
        return ep_greedy(q_1[indb, out, :], alpha_phases, dispersion, nr_prob=nr_prob)[
            1
        ]

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
