"""Tests of form Accuracy vs. N, where N is the number of binomial trials in
the sampling scheme that generates ~A"""
import scipy
import numpy as np

def acc_vs_N(
    A:scipy.sparse.sparray,
    rng:np.random.Generator,
    Ns:np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """The 'score' (2-norm) of ~A^TA vs A^TA, over variable N (expectation that
    increasing N increases accuracy)

    Args:
        A (scipy.sparse.sparray): The matrix under consideration
        rng (np.random.Generator): For repeatable randomization
        Ns (np.ndarray): The number of bernoulli trials in generating an
        approximation for A^TA

    Returns:
        xs (np.ndarray): The same as Ns
        ys (np.ndarray): The accuracy for each N approximation
        lbl (str): string representation of this test
	"""
    pass


def percentile_vs_N (
    A:scipy.sparse.sparray,
    delta:float,
    Ns:np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """The 'score' (2-norm) of ~A^TA vs A^TA, over variable N (expectation that
    increasing N increases accuracy)

    Args:
        A (scipy.sparse.sparray): The matrix under consideration
        delta (float): The percentile being visualized, i.e. "with probability
        at least delta, ||A~v|| is within some epsilon of ||Av||", where v is
        the top eigenvector of A^TA and ~v is the top eigenvector of ~A^TA
        Ns (np.ndarray): The number of bernoulli trials in generating an
        approximation for A^TA

    Returns:
        xs (np.ndarray): The same as Ns
        ys (np.ndarray): The accuracy for each N approximation
        lbl (str): string representation of this test
	"""
    pass
