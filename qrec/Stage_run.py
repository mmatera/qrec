"""
Stage run

The objective of this script is to run the experiment,
track he hyperparameters, save the values of interest
and make the decisions.

"""

import time
from typing import Tuple

import numpy as np

from qrec.utils import (
    Hyperparameters,
    Qlearning_parameters,
    comm_success_prob,
    ep_greedy,
    give_outcome,
    give_reward,
    #    model_aware_optimal,
    p_model,
    perr_model,
    update_buffer_and_compute_mean,
)


class PhotonSource:
    r"""
    A class that represents the optic bench, including a source of photons
    in gaussian states |\pm \alpha*(1+lambda)>, with signs distributed with probabilities
    p=.5+/-bias, and a detector with an offset parameter beta.
    """

    def __init__(self, alpha=1.5, lambd=0, bias=0.0, buffer_size=1000):
        self.beta = 0.0  # Offset in the detector.
        self.alpha = alpha  # The amplitude of the coherent state in the source
        self.lambd = lambd  # Amplitude fluctuations
        self.bias = bias  # prior bias in the signal.
        self.buffer_size = buffer_size

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
        bias = self.bias
        if bias == 0:
            message = tuple(np.random.choice([0, 1]) for i in range(size))
        else:
            message = tuple(
                np.random.choice([0, 1], [0.5 - bias, 0.5 + bias])
                for i in range(size)
            )
        # message = tuple(int(m+ 0.5 * self.bias) for m in np.random.random(size))
        return message, self.signal(message)

    def guess_intensity(self, duration):
        """
        Simulates the estimation of alpha from the outcomes obtained
        when beta is set to 0.

        PARAMETERS
        ==========

        duration: int
           size of the buffer

        RETURN
        ======
        estimated alpha: float
            the estimated amplitude of alpha

        To do the estimation, the beta parameter of the detector is set to 0. Then,
        it is asked to the source to send a stream of `duration` photons with
        random phases.
        Finally, beta is restorated, and the amplitude is estimated assuming a
        coherent state source.
        """
        # TODO: remove the next two lines
        use_new_code = False
        if not use_new_code:
            from qrec.Model_Semi_aware.intensity_guess import guess_intensity

            return guess_intensity(self.alpha, duration, self.lambd)

        old_beta = self.beta
        self.beta = 0
        _, outcomes = self.training_signal(duration)
        self.beta = old_beta
        return np.sqrt(-np.log(1 - np.average(outcomes)))


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

    q_1[beta_indx, outcome, guess] += (
        reward - q_1[beta_indx, outcome, guess]
    ) / n_1[beta_indx, outcome, guess]
    q_0[beta_indx] += (
        np.max([q_1[beta_indx, outcome, g] for g in range(2)]) - q_0[beta_indx]
    ) / n_0[beta_indx]
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
    return epsilon


# Reload the Q-function with the model.
def reset_with_model(
    guessed_intensity: float,
    source: PhotonSource,
    qlearning: Qlearning_parameters,
):
    """
    Parameters
    ----------
    guessed_intensity: float:
        the current guess of the amplitude.
    source : PhotonSource
        the offset of the signal.
    qlearning : Qlearning_parameters
        The qlearning structure.

    Returns
    -------
    guess_intensity : float
        The updated guess of the intensity

    """
    buffer_size = 10 * source.buffer_size
    new_guessed_intensity = source.guess_intensity(buffer_size)
    print("Guessed intensity:", new_guessed_intensity)
    if np.abs(new_guessed_intensity - guessed_intensity) <= 5 / np.sqrt(
        buffer_size
    ):
        return guessed_intensity
    guessed_intensity = new_guessed_intensity
    q_0 = qlearning.q0
    q_1 = qlearning.q1
    beta_grid = qlearning.betas_grid
    # set q_0 and q_1 with the sucess probabilities
    # from the surmised score function for the Bayes' decision rule
    for i, beta in enumerate(beta_grid):
        q_0[i] = 1 - perr_model(beta, guessed_intensity)

    for i, q1_i in enumerate(q_1):
        for outcome, q1_ij in enumerate(q1_i):
            for k in range(len(q1_ij)):
                beta = -beta_grid[i]
                prob = p_model(
                    (-1) ** (k + 1) * guessed_intensity, beta, outcome
                )

                # Marginal probability of the outcome
                # for unknown phase of alpha8
                total_prob = p_model(
                    -guessed_intensity, beta, outcome
                ) + p_model(guessed_intensity, beta, outcome)

                q_1[i, outcome, k] = prob / total_prob

    return new_guessed_intensity


def experiment_noise(source, qlearning, hyperparam, epsilon):
    """
    Performs a measurement to get an outcome, and compute the new estimation for beta,
    the guess and the reward.
    """
    alpha = source.alpha
    lambd = source.lambd
    bias = source.bias

    q_0 = qlearning.q0
    q_1 = qlearning.q1
    betas_grid = qlearning.betas_grid

    use_new_code = True
    if use_new_code:
        beta_indx, beta = ep_greedy(
            q_0, betas_grid, hyperparam.delta, near_prob=epsilon
        )
        source.beta = beta
        hidden_phases, outcomes = source.training_signal(1)
        hidden_phase = hidden_phases[0]
        outcome = outcomes[0]
    # TODO: Remove me when the tests are completed.
    else:
        # Both conditions should produce equivalent results. However,
        # internally it seems to produce different random sequences.
        if bias == 0:
            hidden_phase = np.random.choice([0, 1])
        else:
            hidden_phase = np.random.choice([0, 1], [0.5 - bias, 0.5 + bias])
        beta_indx, beta = ep_greedy(
            q_0, betas_grid, hyperparam.delta, near_prob=epsilon
        )
        source.beta = beta
        outcome = give_outcome(hidden_phase, beta, alpha=alpha, lambd=lambd)

    guess_indx, guess = ep_greedy(
        q_1[beta_indx, outcome, :],
        [0, 1],
        hyperparam.delta,
        near_prob=epsilon,
    )
    reward = give_reward(guess, hidden_phase)
    return beta_indx, beta, outcome, guess_indx, guess, reward


def run_experiment(
    details,
    training_size,
    alpha,
    hyperparam: Hyperparameters,
    buffer_size=1000,
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
        Size of the buffer of rewards.
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
    reward = 0
    guessed_intensity = 1.5

    epsilon = float(details["ep"])
    experience = details.get("experience", [])
    outcome_buffer = details["means"]
    ps_greedy = details.get("Ps_greedy", [])

    qlearning = details["tables"]
    betas_grid = qlearning.betas_grid

    witness_buffer = details["witness"]
    witness = float(witness_buffer[-1])
    points = [witness, float(witness_buffer[-2])]

    if noise_type == 1:
        bias = 0
    elif noise_type == 2:
        bias = lambd
        lambd = 0.0
    else:
        lambd = 0.0

    source = PhotonSource(alpha, lambd, bias, buffer_size)

    epoch_size = 10
    rounds = training_size // epoch_size
    start = time.time()
    for experiment in range(0, training_size):
        if True:
            if epsilon > hyperparam.eps_0:
                epsilon *= hyperparam.delta_epsilon
            else:
                epsilon = hyperparam.eps_0
    
            if experiment % (rounds) == 0:
                print(experiment)
    
            (
                beta_indx,
                beta,
                outcome,
                _,
                guess,
                reward,
            ) = experiment_noise(source, qlearning, hyperparam, epsilon)
            q0_max_idx = np.argmax(qlearning.q0)
            details["greed_beta"].append(betas_grid[q0_max_idx])
    
            # Update witness
            outcome_buffer, witness = update_buffer_and_compute_mean(
                outcome_buffer, outcome, buffer_size
            )
            witness_buffer.append(witness)
            updates(beta_indx, outcome, guess, reward, qlearning)
    
            # _, pstar, _ = model_aware_optimal(betas_grid, alpha=alpha, lambd=lambd)
    
            experience.append([beta, outcome, guess, reward])
            ps_greedy.append(
                comm_success_prob(
                    qlearning,
                    hyperparam.delta,
                    alpha=alpha,
                    detuning=lambd,
                )
            )
            # Each time the buffer is fully updated,
            # check if the reward is smaller
        if experiment % buffer_size == 0:
            points[0] = points[1]
            points[1] = witness
            mean_deriv = points[1] - points[0]
            # print(mean_deriv)
            if mean_deriv >= 0.05 and qlearning.n0[q0_max_idx] > 3000:
                epsilon = update_reload(
                    qlearning,
                    hyperparam.delta_learning_rate,
                    hyperparam.eps_0,
                )
                if model:
                    guessed_intensity = reset_with_model(
                        guessed_intensity, source, qlearning
                    )        
    end = time.time() - start
    details["Ps_greedy"] = ps_greedy
    details["ep"] = f"{epsilon}"
    details["experience"] = experience
    details["means"] = outcome_buffer
    details["tables"] = qlearning
    details["total_time"] = end
    details["witness"] = witness_buffer
    print(qlearning.n0)
    return details
