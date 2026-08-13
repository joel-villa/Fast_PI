"""Some functions which come directly from theoretical results on the accuracy
bounds of the sampling scheme"""

# Math defaults to base e
from math import log, exp, sqrt

from .util.meta_data import get_power_consts, get_n_norm, get_var_proxy
from .util.BoundsType import BoundsType

def param_check(
       params:list,
) -> None:
    """Are all the parameters positive?

    Args:
        params: list of parameters to check 

    Raises:
        ValueError: If any parameters are non-positive
    """

    if (any([val < 0 for val in params])):
        # Something is negative, get upset
        raise ValueError(f"Not all parameters are positive: {params}")

def prob_of_success( 
        epsilon: float,
        A_norm: float,
        n: int,
        var_proxy:float,
) -> float:
    """Get the probability of success given the necessary data

    Args:
        epsilon (float): Amount of acceptable error
        A_norm (float): The operator norm of A
        n (int): Number of rows of A
        var_proxy (float): Some measure of how much the random variable varies

    Returns:
        float: probability that ||~A||^2 is within plus or minus epsilon^2 of 
        ||A||^2
    """
    numerator = 3 * (A_norm ** 4)
    denominator = (2 * (A_norm * A_norm)) + ((6 * var_proxy) / (epsilon * epsilon))
    fraction = numerator / denominator
    print(f"log(n) = {log(n)}, (epsilon ** 2) * fraction = {(epsilon ** 2) * fraction}")
    exponent = log(n) - ((epsilon * epsilon) * fraction)

    probability = 1 - 2 * exp(exponent)

    if probability > 1:
        raise ValueError(f"Invalid probability of success: {probability}")

    return probability


def error_bounds(A_norm:float, epsilon:float) -> tuple[float, float]:
    """With prob_of_success()...
    (1 - epsilon^2)||A||^2 <= ||~A||^2 <= (1 + epsilon^2)||A||^2

    Args:
        A_norm (float): operator norm of the matrix in question
        epsilon (float): Some amount of acceptable error

    Returns:
        tuple[float, float]: (lower bound, upper bound)
    """
    lb = (1 - (epsilon ** 2)) * (A_norm ** 2)
    ub = (1 + (epsilon ** 2)) * (A_norm ** 2)

    return lb, ub

def prob_in_bounds(
        epsilon: float,
        A_norm: float,
        n: int,
        var_proxy:float,
) -> tuple[float, tuple[float, float], str]:
    """The probability that ||~A||^2 is in some bounds

    Args:
        epsilon (float): Amount of acceptable error
        A_norm (float): The operator norm of A
        n (int): Number of rows of A
        var_proxy (float): Some measure of how much the random variable varies

    Returns:
        tuple[float, tuple[float, float], str]: [
        float: probability of success,
        tuple[float, float]: [lowerbound, upperbound]
        str: string representation of info
        ]
    """
    param_check([epsilon, A_norm, n, var_proxy])

    prob_success = prob_of_success(
        epsilon=epsilon,
        A_norm=A_norm,
        n=n,
        var_proxy=var_proxy,
    )

    lb, ub = error_bounds(A_norm=A_norm, epsilon=epsilon)

    latex_str = fr"$||\tilde A||^2 \in [{lb}, {ub}]$"
    string_rep = f"With probability {prob_success}, {latex_str}"

    return prob_success, (lb, ub), string_rep

def get_power_var_proxy(
    c:float,
    kappa:float,
    n:int,
) -> float:
    """Get the variance proxy for a power-law distribution

    Args:
        c (float): _description_
        kappa (float): _description_
        n (int): _description_

    Returns:
        float: _description_
    """
    param_check([c, kappa, n])

    kappa_inverse = 1 / kappa
    n_to_the_neg_kappa = 1 / (n ** kappa)
    return (c ** 3) * (1 + kappa_inverse * (1 - n_to_the_neg_kappa))

def var_from_type(
        bounds_type: BoundsType,
        mat_name: str,
        n: int,
) -> float:
    """Get the appropriate variance proxy metric

    Args:
        bounds_type (BoundsType): Some type of variance proxy metric
        mat_name (str): SuiteSparse matrix name
        n (int): number of rows in the matrix

    Raises:
        NotImplementedError: In case bounds are made in the future that aren't
        currently handled

    Returns:
        float: the variance proxy for the bounding type
    """
    match bounds_type:
        # Set variane proxy based on bounds type
        case BoundsType.POWER:
            c, kappa = get_power_consts(mat_name)
            return get_power_var_proxy(
                c=c,
                kappa=kappa,
                n=n,
            )
        case BoundsType.STRICT:
            return get_var_proxy(mat_name)
        case _:
            raise NotImplementedError(f"Unexpected BoundsType: {bounds_type}")

def get_expectation_bounds(
        mat_name: str,
        epsilon: float,
        bounds_type: BoundsType,
) -> tuple[float, tuple[float, float], str]:
    """Get the string defining the proability of success and the bounds

    Args:
        mat_name (str): the suite-sparse matrix name
        epsilon (float): ammount of allowable error

    Returns:
        tuple[float, tuple[float, float], str]: [
        float: probability of success,
        tuple[float, float]: [lowerbound, upperbound]
        str: string representation of info
        ]
    """
    n, op_norm = get_n_norm(mat_name)
    
    var_proxy = var_from_type(
        bounds_type=bounds_type,
        mat_name=mat_name,
        n=n,
    )
    
    return prob_in_bounds(
        epsilon=epsilon,
        A_norm=op_norm,
        n=n,
        var_proxy=var_proxy,
    )

"""The following bounds take in the probability of success, and return an 
epsilon value"""


def epsilon_from_delta(
        delta:float,
        n: int,
        A_norm:float,
        var_proxy:float,
) -> float:
    """Get epsilon from the metadata + some probability of success

    Args:
        delta (float): Proability of success
        n (int): number of rows in A
        A_norm (float): The operator norm of A
        var_proxy (float): The variance proxy for this matrix

    Returns:
        float: epsilon
    """
    
    log_term = log((2 * n) / (1 - delta))

    p_numerator = log_term + sqrt(log_term * (log_term + 18 * var_proxy))
    n_numerator = log_term - sqrt(log_term * (log_term + 18 * var_proxy))
    numerator = n_numerator # Numerator is smaller of the two options
    if n_numerator < 0:
        numerator = p_numerator

    denominator = 3 * (A_norm * A_norm)

    epsilon = sqrt(numerator / denominator)

    return epsilon

def valid_epsilon(
        delta: float,
        n: int,
        A_norm: float,
        var_proxy:float,
) -> tuple[float, tuple[float, float], str]:
    """The epsilon bounds for ||~A||^2

    Args:
        delta (float): Proability of success
        n (int): number of rows in A
        A_norm (float): The operator norm of A
        var_proxy (float): The variance proxy for this matrix

    Returns:
        tuple[float, tuple[float, float], str]: [
        float: epsilon,
        tuple[float, float]: [lowerbound, upperbound]
        str: string representation of info
        ]
    """
    param_check([delta, A_norm, var_proxy])

    epsilon = epsilon_from_delta(
        delta=delta,
        n=n,
        A_norm=A_norm,
        var_proxy=var_proxy,
    )

    lb, ub = error_bounds(A_norm=A_norm, epsilon=epsilon)

    latex_str = fr"$||\tilde A||^2 \in [{lb}, {ub}]$"
    string_rep = f"With probability {delta}, {latex_str}"

    return epsilon, (lb, ub), string_rep

def get_epsilon_bounds(
        mat_name: str,
        delta: float,
        bounds_type: BoundsType,
) -> tuple[float, tuple[float, float], str]:
    """Get epsilon, the epsilon bounds, plus a string representation of things

    Args:
        mat_name (str): the suite-sparse matrix name
        delta (float): the probability of success

    Returns:
        tuple[float, tuple[float, float], str]: [
        float: epsilon,
        tuple[float, float]: [lowerbound, upperbound]
        str: string representation of info
        ]
    """
    n, op_norm = get_n_norm(mat_name)

    var_proxy = var_from_type(
        bounds_type=bounds_type,
        mat_name=mat_name,
        n=n,
    )
    
    return valid_epsilon(
        delta=delta,
        n=n,
        A_norm=op_norm, 
        var_proxy=var_proxy,
    )