"""
Reinforced learning.

This module defines the functions and structures for implementing the
reinforced learning.


In this module, we only have access to the parameter beta to be determined,
and the previous outcomes and rewards.

"""

from collections import namedtuple

import numpy as np

Qlearning_parameters = namedtuple(
    "Qlearning_parameters", ["q0", "q1", "n0", "n1", "betas_grid", "parms"]
)

# Better explicit that implicit...
Hyperparameters = namedtuple(
    "Hyperparameters",
    [
        "eps_0",  # Epsilon0
        "delta_epsilon",  # epsilon0 change rate
        "delta",  # Dispersion of the random gaussian
        "delta_learning_rate",  # Learning rate reset
        "check_jump_threshold",  # how much statistics is required
        # to start checking for jumps
        # in the reward mean derivative.
    ],
)


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
        np.ones(betas_grid.shape[0]),  # N(beta)
        np.ones((betas_grid.shape[0], 2, 2)),  # N(beta,n; g)
        betas_grid,
        {
            "epsilon": 1.0,  # epsilon
            "guessed_intensity": 1.5,  # guessed_intensity
        },
    )
    return qlearn


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



# How the q-learning parameters update.
def updates(
    beta_indx,
    outcome,
    guess,
    reward,
    qlearning: Qlearning_parameters,
    #    learning_rate=0.001,
):
    """
    Given that by setting beta=beta[indb] `outcome` was obtained,
    and the `guess`, update the qlearning params
    using `reward` and `the learning rate lr`.

    Parameters
    ----------
    beta_indx : int
        index of the beta parameter
    outcome: int
        the actual result of the measurement
    guess : TYPE
        estimated result.
    reward : float
        the reward obtained if guess and outcome match.
    qlearning : QLarningParms
        current values of the learning parameters.

    Returns
    -------
    result : QLearningParms
        the new values of the learning parameters.

    """

    q_0 = qlearning.q0
    q_1 = qlearning.q1
    n_0 = qlearning.n0
    n_1 = qlearning.n1

    n1_curr = n_1[beta_indx, outcome, guess]
    q_1_curr = q_1[beta_indx, outcome, guess]
    q_1[beta_indx, outcome, guess] += (reward - q_1_curr) / n1_curr

    n_0_curr = n_0[beta_indx]
    optimal_reward = np.max([q_1[beta_indx, outcome, g] for g in range(2)])
    q_0[beta_indx] += (optimal_reward - q_0[beta_indx]) / n_0_curr

    n_0[beta_indx] += 1
    n_1[beta_indx, outcome, guess] += 1
    return qlearning


# How the learning rate changes when the environment changes.
# It could be interesting to change the reward with the mean_rew.


def update_reload(qlearning: Qlearning_parameters, restart_point, restart_epsilon):
    """
    Reset n0 and n1 when the change is large.

    Parameters
    ----------
    qlearning : Qlearning_parameters
        DESCRIPTION.
    restart_point : TYPE
        DESCRIPTION.
    restart_epsilon : TYPE
        DESCRIPTION.
    """
    # TODO: check why restart_epsilon is a parameter of this function.
    # Also check why always returns epsilon=1
    epsilon = 1
    n_0 = qlearning.n0
    n_1 = qlearning.n1

    for i, n1_i in enumerate(qlearning.n1):
        n_0[i] = restart_point
        if n_0[i] < 1:
            n_0[i] = 1
        for j, n1_i_j in enumerate(n1_i):
            for k, n1_i_j_k in enumerate(n1_i_j):
                n_1[i, j, k] = restart_point
                if n1_i_j_k < 1:
                    n_1[i, j, k] = 1
    qlearning.parms["epsilon"] = epsilon


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
        outcomes_buffer.resize(max_len)

    mean_val = np.average(outcomes_buffer)
    return outcomes_buffer, mean_val
