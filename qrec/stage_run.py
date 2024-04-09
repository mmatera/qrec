"""
Stage run

The objective of this script is to run the experiment,
track he hyperparameters, save the values of interest
and make the decisions.

"""

import time

import numpy as np

from qrec.device_simulation import ExperimentResult, PhotonSource, give_outcome, p_model
from qrec.policies import comm_success_prob, ep_greedy
from qrec.qlearning import (
    Hyperparameters,
    Qlearning_parameters,
    give_reward,
    update_buffer_and_compute_mean,
    update_reload,
)
from qrec.utils import perr_model  # model_aware_optimal,


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


# Reload the Q-function with the model.
def reset_with_model(
    source: PhotonSource,
    qlearning: Qlearning_parameters,
):
    """
    Parameters
    ----------
    source : PhotonSource
        the offset of the signal.
    qlearning : Qlearning_parameters
        The qlearning structure.


    """
    guessed_intensity = qlearning.parms["guessed_intensity"]

    buffer_size = 10 * source.buffer_size
    new_guessed_intensity = source.guess_intensity(buffer_size)
    print("Guessed intensity:", new_guessed_intensity)
    if np.abs(new_guessed_intensity - guessed_intensity) <= 5 / np.sqrt(buffer_size):
        return
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
                prob = p_model((-1) ** (k + 1) * guessed_intensity, beta, outcome)

                # Marginal probability of the outcome
                # for unknown phase of alpha8
                total_prob = p_model(-guessed_intensity, beta, outcome) + p_model(
                    guessed_intensity, beta, outcome
                )

                q_1[i, outcome, k] = prob / total_prob

    qlearning.parms["guessed_intensity"] = guessed_intensity


def experiment_noise(source, qlearning, hyperparam):
    """
    Performs a measurement to get an outcome, and compute the new estimation for beta,
    the guess and the reward.
    """
    alpha = source.alpha
    lambd = source.lambd
    bias = source.bias
    epsilon = qlearning.parms["epsilon"]

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

    _, guess = ep_greedy(
        q_1[beta_indx, outcome, :],
        [0, 1],
        hyperparam.delta,
        near_prob=epsilon,
    )
    reward = give_reward(guess, hidden_phase)
    updates(beta_indx, outcome, guess, reward, qlearning)
    return ExperimentResult(beta_indx, outcome, guess, reward)


def check_mean_derivative(
    witness_buffer,
    source,
    qlearning,
    hyperparam,
):
    """
    Check for large changes in the mean derivative of the witness.
    """
    print("Check derivatives")
    epsilon = qlearning.parms["epsilon"]
    guessed_intensity = qlearning.parms["guessed_intensity"]
    model = qlearning.parms["use model"]
    points = qlearning.parms["points"]
    witness = witness_buffer[-1]
    points[0] = points[1]
    points[1] = witness
    print("   points:", points)
    print(
        "    witness     : ",
        [witness_buffer[-source.buffer_size], witness_buffer[-1]],
    )
    mean_deriv = points[1] - points[0]
    # print(mean_deriv)
    if mean_deriv >= 0.05:
        update_reload(
            qlearning,
            hyperparam.delta_learning_rate,
            hyperparam.eps_0,
        )
        if model:
            reset_with_model(source, qlearning)

        qlearning.parms["epsilon"] = epsilon
        qlearning.parms["guessed_intensity"] = guessed_intensity

    return epsilon, guessed_intensity


def evolve_epsilon(qlearning: Qlearning_parameters, hyperparam: Hyperparameters):
    """Evolve epsilon.

    PARAMETERS
    ==========

    epsilon: float
        the current value of epsilon
    hyperparm: Hyperparameters
        the hyperparameters
    """
    epsilon = qlearning.parms["epsilon"]
    if epsilon > hyperparam.eps_0:
        qlearning.parms["epsilon"] = epsilon * hyperparam.delta_epsilon
    else:
        qlearning.parms["epsilon"] = hyperparam.eps_0


def run_experiment(
    details,
    training_size,
    alpha,
    hyperparam: Hyperparameters,
    buffer_size=1000,
    lambd=0.0,
    use_model=True,
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
    use_model: bool
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
    # epsilon = float(details["ep"])
    experience = details.get("experience", [])
    outcomes_buffer = details["means"]
    ps_greedy = details.get("Ps_greedy", [])

    qlearning = details["tables"]
    betas_grid = qlearning.betas_grid
    witness_buffer = details["witness"]

    # Setup the detector:
    if noise_type == 1:
        bias = 0
    elif noise_type == 2:
        bias = lambd
        lambd = 0.0
    else:
        lambd = 0.0

    qlearning.parms["use model"] = use_model
    qlearning.parms["points"] = [float(witness_buffer[-1]), float(witness_buffer[-2])]
    source = PhotonSource(betas_grid, alpha, lambd, bias, buffer_size)

    # Trainig loop
    start = time.time()
    for experiment in range(training_size // buffer_size):
        print("  experiment:", experiment)
        for __ in range(buffer_size):
            evolve_epsilon(qlearning, hyperparam)
            result = experiment_noise(source, qlearning, hyperparam)
            new_beta_idx = np.argmax(qlearning.q0)
            details["greed_beta"].append(source.set_beta_idx(new_beta_idx))

            # Update witness
            outcomes_buffer, witness = update_buffer_and_compute_mean(
                outcomes_buffer, result.outcome, buffer_size
            )
            witness_buffer.append(witness)

            # _, pstar, _ = model_aware_optimal(betas_grid, alpha=alpha,
            # lambd=lambd)

            experience.append(result)
            ps_greedy.append(
                comm_success_prob(
                    qlearning,
                    hyperparam.delta,
                    alpha=alpha,
                    detuning=lambd,
                )
            )

        # Check for jumps in the parameters.
        if qlearning.n0[new_beta_idx] > hyperparam.check_jump_threshold:
            check_mean_derivative(
                witness_buffer,
                source,
                qlearning,
                hyperparam,
            )

    end = time.time() - start
    details["Ps_greedy"] = ps_greedy
    details["ep"] = f'{qlearning.parms["epsilon"]}'
    details["experience"] = experience
    details["means"] = outcomes_buffer
    details["tables"] = qlearning
    details["total_time"] = end
    details["witness"] = witness_buffer
    print(qlearning.n0)
    return details
