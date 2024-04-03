"""
Stage run

The objective of this script is to run the experiment,
track he hyperparameters, save the values of interest 
and make the decisions.

"""

import time
import numpy as np

from typing import List, Tuple
from qrec.Model_Semi_aware.intensity_guess import guess_intensity
from qrec.utils import (
    Hyperparameters,
    Qlearning_parameters,
    calculate_mean_reward,
    comm_success_prob,
    ep_greedy,
    give_outcome,
    give_reward,
    model_aware_optimal,
    p_model,
    perr_model,
)


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

    q_1[beta_indx, outcome, guess] += (1 / n_1[beta_indx, outcome, guess]) * (
        reward - q_1[beta_indx, outcome, guess]
    )
    q_0[beta_indx] += (1 / n_0[beta_indx]) * np.max(
        [q_1[beta_indx, outcome, g] for g in [0, 1]] - q_0[beta_indx]
    )
    n_0[beta_indx] += 1
    n_1[beta_indx, outcome, guess] += 1
    return qlearning


# How the learning rate changes when the environment changes.
# It could be interesting to change the reward with the mean_rew.


def update_reload(
    qlearning: Qlearning_parameters, restart_point, restart_epsilon
):
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

    Returns
    -------
    n_0 : TYPE
        DESCRIPTION.
    n_1 : TYPE
        DESCRIPTION.
    epsilon : TYPE
        DESCRIPTION.

    """
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
    return epsilon


# Reload the Q-function with the model.
def reset_with_model(alpha, qlearning: Qlearning_parameters):
    """


    Parameters
    ----------
    alpha : TYPE
        the offset of the signal.
    qlearning : Qlearning_parameters
        DESCRIPTION.

    Returns
    -------
    qlearn : TYPE
        DESCRIPTION.

    """
    q_0 = qlearning.q0
    q_1 = qlearning.q1
    beta_grid = qlearning.betas_grid
    # set q_0 and q_1 with the sucess probabilities
    # from the surmised score function for the Bayes' decision rule
    for i, beta in enumerate(beta_grid):
        q_0[i] = 1 - perr_model(beta, alpha)

    for i, q1_i in enumerate(q_1):
        for outcome, q1_ij in enumerate(q1_i):
            for k in range(len(q1_ij)):
                beta = -beta_grid[i]
                prob = p_model((-1) ** (k + 1) * alpha, beta, outcome)

                # Marginal probability of the outcome
                # for unknown phase of alpha8
                total_prob = p_model(-alpha, beta, outcome) + p_model(
                    alpha, beta, outcome
                )

                q_1[i, outcome, k] = prob / total_prob

    return qlearning


class PhotonSource:
    def __init__(self, alpha=1.5, lambd=0, bias=0.0):
        self.beta = 0.0  # Offset in the detector.
        self.alpha = alpha  # The amplitude of the coherent state in the source
        self.lambd = lambd  # Amplitude fluctuations
        self.bias = bias  # prior bias in the signal.

    def signal(self, message: Tuple[int]) -> Tuple[int]:
        """
        Simulate the arrival of a train of photons to the detector,
        Phase of the photons codify message.

        Parameters
        ----------
        message : Tuple[int]
            A tuple of zeros and ones codifying the message

        Returns
        -------
        signal: Tuple[int]
            the sequence of the outcomes in the detector.
        """
        return tuple(
            give_outcome(
                msg_bit, self.beta, alpha=self.alpha, lambd=self.lambd
            )
            for msg_bit in message
        )

    def training_signal(self, size) -> Tuple[Tuple[int], Tuple[int]]:
        """
        Simulates a source that produces a stream of bits and
        a the signal from the detector assuming that the phases of the
        photons encode the stream.

        Parameters
        ----------

        size: int
           the length of the random message

        Returns
        -------
        message: Tuple[int]
            a random stream
        signal: Tuple[int]
            the output of the detector.

        """
        message = tuple(int(m) for m in np.random.rand(size) + 0.5 * self.bias)
        return message, self.signal(message)


def callibration(
    src: PhotonSource,
    size_training: int,
    epsilon: float,
    qlearning: Qlearning_parameters,
    hyperparam: Hyperparameters,
):
    """
    Perform a step of callibration by asking a training set to the source.
    """
    q_0 = qlearning.q0
    q_1 = qlearning.q1
    betas_grid = qlearning.betas_grid
    beta_indx, beta = ep_greedy(
        q_0, betas_grid, hyperparam.delta_learning_rate, nr_prob=epsilon
    )
    src.beta = beta
    hidden_phase, outcome = src.training_signal(size_training)
    rewards = []
    guesses = []
    beta_indxs = []

    for phase, out in zip(hidden_phase, outcome):
        guess_indx, guess = ep_greedy(
            q_1[beta_indx, out, :],
            [0, 1],
            hyperparam.delta_learning_rate,
            nr_prob=epsilon,
        )
        rewards.append(give_reward(guess, phase))
        guesses.append(guess)
        beta_indxs.append(beta_indx)

    return hidden_phase, outcome, beta_indxs, guesses, rewards


def run_experiment(
    details,
    training_size,
    alpha,
    hyperparam: Hyperparameters,
    buffer_size=1000,
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
        All the information about the agent parameters and the previous
        experiments.
    training_size : int
        Amount of states use for the training.
    alpha : float
        Intensity of the state used in the experiment.
    hyperparam: Hyperparameters
        Hyperparameters used for the Q-learning algorithm.
    buffer_size: int
        size of the rewards buffer.
    current: float.
        intensity predicted for the model previously.
    lambd: float
        Amount of 'unkwnown' noise. 0.0 meaning that there is no extra noise in
        the channel.
    model: bool
        True if the agent can use a model to predict initial values,
        False otherwise.
    noise_type: int
        1 -> change in the expected displacement, 2 -> change in the prior.


    Returns
    -------
    details: dictionary.
        All the information about the agent parameters and
        the previous experiments.

    """
    if noise_type == 0:
        source = PhotonSource(alpha, 0, 0)
    elif noise_type == 1:
        source = PhotonSource(alpha, lambd, 0)
    elif noise_type == 2:
        source = PhotonSource(alpha, 0, lambd)

    witness = float(details["witness"][-1])
    experience = details["experience"]
    points = [witness, float(details["witness"][-2])]
    rewards_buffer = details["means"]
    qlearning = details["tables"]
    betas_grid = qlearning.betas_grid
    reward = 0
    guessed_intensity = current
    epsilon = float(details["ep"])
    checked = False

    start = time.time()
    for experiment in range(0, training_size):
        if epsilon > hyperparam.eps_0:
            epsilon *= hyperparam.delta_epsilon
        else:
            epsilon = hyperparam.eps_0

        if experiment % (training_size // 10) == 0:
            print(experiment)

        (hidden_phases, outcomes, beta_indxs, guesses, rewards) = callibration(
            source, 1, epsilon, qlearning, hyperparam
        )
        # Update the buffer and the mean reward
        rewards_buffer.extend(rewards)
        rewards_buffer = rewards_buffer[-buffer_size:]
        witness = np.average(rewards_buffer)

        # Check if the reward is smaller
        if experiment % buffer_size == 0:
            points[0] = points[1]
            points[1] = witness
            mean_deriv = points[1] - points[0]
            # print(mean_deriv)
            if (
                mean_deriv >= 0.05
                and qlearning.n0[list(qlearning.q0).index(max(qlearning.q0))]
                > 3000
            ):
                epsilon = update_reload(
                    qlearning,
                    hyperparam.delta_learning_rate,
                    hyperparam.eps_0,
                )
                checked = True

        # If it looks like changed, guess a new intensity to verify.
        if model and checked:
            guessed_intensity = guess_intensity(
                alpha, buffer_size * 10, lambd=lambd
            )
            print(guessed_intensity)
            if np.abs(guessed_intensity - current) > 5 / np.sqrt(
                buffer_size * 10
            ):
                current = guessed_intensity
                reset_with_model(current, qlearning)
            checked = False

        for beta_indx, outcome, guess, reward in zip(
            beta_indxs, outcomes, guesses, rewards
        ):
            updates(beta_indx, outcome, guess, reward, qlearning)
            experience.append([betas_grid[beta_indx], outcome, guess, reward])

        # Extend the optimal beta using the current q0
        details["greed_beta"].append(
            betas_grid[list(qlearning.q0).index(max(qlearning.q0))]
        )

        # _, pstar, _ = model_aware_optimal(betas_grid,
        #                                   alpha=alpha, lambd=lambd)

        details["witness"].append(witness)
        details["means"] = rewards_buffer
        details["Ps_greedy"].append(
            comm_success_prob(
                qlearning,
                hyperparam.delta,
                alpha=alpha,
                detuning=lambd,
            )
        )
    details["tables"] = qlearning
    end = time.time() - start
    details["total_time"] = end
    details["ep"] = f"{epsilon}"
    print(qlearning.n0)
    return details
