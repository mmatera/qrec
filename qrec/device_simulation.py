"""
Device simulation

"""
from collections import namedtuple
from typing import Tuple

import numpy as np
from numpy.random import choice

ExperimentResult = namedtuple(
    "ExperimentResult", ["beta_indx", "outcome", "guess", "reward"]
)


class PhotonSource:
    r"""
    A class that represents the optic bench, including a source of photons
    in gaussian states |\pm \alpha*(1+lambda)>, with signs distributed with probabilities
    p=.5+/-bias, and a detector with an offset parameter beta.
    """

    def __init__(self, beta_grid, alpha=1.5, lambd=0, bias=0.0, buffer_size=1000):
        self.beta = 0.0  # Offset in the detector.
        self.alpha = alpha  # The amplitude of the coherent state in the source
        self.lambd = lambd  # Amplitude fluctuations
        self.bias = bias  # prior bias in the signal.
        self.buffer_size = buffer_size
        self.beta_grid = beta_grid

    def set_beta_idx(self, beta_indx):
        """
        Set the value of the offset beta in the detector.

        """
        self.beta = self.beta_grid[beta_indx]
        return self.beta

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
            give_outcome(msg_bit, self.beta, alpha=self.alpha, lambd=self.lambd)
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
                np.random.choice([0, 1], [0.5 - bias, 0.5 + bias]) for i in range(size)
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
    assert 0 <= pr_0 <= 1, [alpha, beta, detunning]
    return 1 - pr_0 if outcome else pr_0


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

    sgn = -1 if hidden_phase else 1
    values = np.array([0, 1])
    p_0 = detection_state_probability(alpha * sgn, beta, lambd, 0)
    return choice(values, p=[p_0, 1 - p_0])


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
