"""
Stage run

The objective of this script is to run the experiment,
track he hyperparameters, save the values of interest 
and make the decisions.

"""

import time

import numpy as np

from qrec.Model_Semi_aware.intensity_guess import guess_intensity
from qrec.utils import (
    Hyperparameters,
    Qlearning_parameters,
    comm_success_prob,
    ep_greedy,
    give_outcome,
    give_reward,
    model_aware_optimal,
    p_model,
    perr_model,
    update_buffer_and_compute_mean,
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
    q_0[beta_indx] += (
        np.max([q_1[beta_indx, outcome, g] for g in range(2)]) - q_0[beta_indx]
    ) / n_0[beta_indx]
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
        The qlearning structure.

    Returns
    -------
    qlearn : TYPE
        The updated qlearning structure.

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


# Change in beta.
def experiment_noise_1(
    qlearning: Qlearning_parameters,
    hyperparam: Hyperparameters,
    epsilon,
    alpha,
    lambd,
):
    """
    Run the experiment with type 1 noise

    Parameters
    ----------
    qlearn : Qlearning_parameters
        The current state of the qlearning.
    hyperparam : Hyperparameters
        The values of the hyperparameter.
    epsilon : TYPE
        probability of using a random policy in the beta update.
    alpha : TYPE
        The amplitude of the source coherent state.
    lambd : TYPE
        relative magnitude of the amplitude fluctuations.

    Returns
    -------
    beta_idx : int
        detector offset index.
    beta : float
        detector offset.
    outcome : TYPE
        actual experiment outcome.
    guess_indx : int
        guessed outcome index.
    guess : int
        guessed outcome.
    reward : TYPE
        reward obtained from the guess.

    """
    q_0 = qlearning.q0
    q_1 = qlearning.q1
    betas_grid = qlearning.betas_grid
    hidden_phase = np.random.choice([0, 1])

    beta_idx, beta = ep_greedy(q_0, betas_grid, hyperparam.delta, nr_prob=epsilon)
    outcome = give_outcome(hidden_phase, beta, alpha=alpha, lambd=lambd)
    guess_indx, guess = ep_greedy(
        q_1[beta_idx, outcome, :], [0, 1], hyperparam.delta, nr_prob=epsilon
    )
    reward = give_reward(guess, hidden_phase)
    return beta_idx, beta, outcome, guess_indx, guess, reward


# Change in the priors
def experiment_noise_2(
    qlearning: Qlearning_parameters,
    hyperparam: Hyperparameters,
    epsilon,
    alpha,
    lambd,
):
    """
    Simulate the experiment with modified priors


    Parameters
    ----------
    qlearn : Qlearning_parameters
        The current state of the qlearning.
    hyperparam : Hyperparameters
        The values of the hyperparameter.
    epsilon : TYPE
        probability of using a random policy in the beta update.
    alpha : TYPE
        The amplitude of the source coherent state.
    lambd : TYPE
        magnitude of the bias in the choice of the signal value.

    Returns
    -------
    beta_indx : TYPE
        index in betas_grid for the beta choosen.
    beta : TYPE
        detector offset.
    outcome : TYPE
        actual result of the measurement.
    guess_indx : TYPE
        index of the guessed value.
    guess : int
        guess value.
    reward : TYPE
        reward obtained from the experiment.

    """
    q_0 = qlearning.q0
    q_1 = qlearning.q1
    betas_grid = qlearning.betas_grid
    hidden_phase = np.random.choice([0, 1], p=[0.5 - lambd, 0.5 + lambd])
    beta_indx, beta = ep_greedy(
        q_0, betas_grid, hyperparam.delta_learning_rate, nr_prob=epsilon
    )
    outcome = give_outcome(hidden_phase, beta, alpha=alpha, lambd=0)
    guess_indx, guess = ep_greedy(
        q_1[beta_indx, outcome, :],
        [0, 1],
        hyperparam.delta_learning_rate,
        nr_prob=epsilon,
    )
    reward = give_reward(guess, hidden_phase)
    return beta_indx, beta, outcome, guess_indx, guess, reward


def experiment_noise_0(
    qlearning: Qlearning_parameters, hyperparam: Hyperparameters, epsilon, alpha, lamdb
):
    """Experiement without noise

    Parameters
    ----------
    qlearning : Qlearning_parameters
        DESCRIPTION.
    hyperparam : Hyperparameters
        DESCRIPTION.
    epsilon : TYPE
        DESCRIPTION.
    alpha : TYPE
        state displacement.
    lambd : TYPE
        noise parameter. Discarded.

    Returns
    -------
    beta_indx : TYPE
        index in betas_grid for the beta choosen.
    beta : TYPE
        detector offset.
    outcome : TYPE
        actual result of the measurement.
    guess_indx : TYPE
        index of the guessed value.
    guess : int
        guess value.
    reward : TYPE
        reward obtained from the experiment.
    """
    return experiment_noise_1(qlearning, hyperparam, epsilon, alpha, 0)


EXPERIMENT_NOISE_MODEL = {
    0: experiment_noise_0,
    1: experiment_noise_1,
    2: experiment_noise_2,
}


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
        Size of the buffer of rewards.
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
    start = time.time()
    witness = float(details["witness"][-1])
    points = [witness, float(details["witness"][-2])]
    outcome_buffer = details["means"]
    qlearning = details["tables"]
    betas_grid = qlearning.betas_grid
    reward = 0
    guessed_intensity = current
    epsilon = float(details["ep"])
    checked = False

    experiment_noise = EXPERIMENT_NOISE_MODEL.get(noise_type, experiment_noise_0)

    for experiment in range(0, training_size):
        if epsilon > hyperparam.eps_0:
            epsilon *= hyperparam.delta_epsilon
        else:
            epsilon = hyperparam.eps_0

        if experiment % (training_size // 10) == 0:
            print(experiment)

        (beta_indx, beta, outcome, guess_idx, guess, reward) = experiment_noise(
            qlearning, hyperparam, epsilon, alpha, lambd
        )

        outcome_buffer, witness = update_buffer_and_compute_mean(
            outcome_buffer, outcome, buffer_size
        )
        q0_max_idx = np.argmax(qlearning.q0)
        details["greed_beta"].append(betas_grid[q0_max_idx])
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
                checked = True

        # TODO: Check if this block should not be inside the previous if block
        # If it looks like changed, guess a new intensity to verify.
        if model and checked:
            guessed_intensity = guess_intensity(alpha, buffer_size * 10, lambd=lambd)
            print(guessed_intensity)
            if np.abs(guessed_intensity - current) > 5 / np.sqrt(buffer_size * 10):
                current = guessed_intensity
                reset_with_model(current, qlearning)
            checked = False

        updates(beta_indx, outcome, guess, reward, qlearning)

        # _, pstar, _ = model_aware_optimal(betas_grid, alpha=alpha, lambd=lambd)

        details["witness"].append(witness)
        details["means"] = outcome_buffer
        details["experience"].append([beta, outcome, guess, reward])
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
