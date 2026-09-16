"""For generating ammount of work to compute an average of approximations of 
A^TA"""
import scipy

def work_naive(
    A:scipy.sparse.coo_array,
    num_tirals:int,
) -> int:
    """Calculate ammount of work (scalar mults) involved in computing this 
    average in a naive way

    Args:
        A (scipy.sparse.coo_array): The matrix to approximate
        num_tirals (int): The ammount of approximations to average over

    Returns:
        int: amount of scalar mults
    """
    pass

def work_fast(
        A:scipy.sparse.coo_array,
        num_trials:int,
        is_distributed:bool,
) -> int:
    """Calculate ammount of work required to do the parallelizable calculation 
    of an averaged approximation of A^TA

    Args:
        A (scipy.sparse.coo_array): The matrix in question
        num_trials (int): The ammount of trials to average over
        is_distributed (bool): True -> taking minimum, False -> summing up work 
        per row

    Returns:
        int: Ammount of work per thread
    """
    pass