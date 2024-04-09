"""
Utility functions

"""


import numpy as np

from qrec.device_simulation import detection_state_probability

# Probability of observing 0 or 1.
# former  "p"


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


# Experiment.
