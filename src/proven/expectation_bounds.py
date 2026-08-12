"""Some functions which come directly from theoretical results on the accuracy
bounds of the sampling scheme"""

from math import log, exp #base e

from .util.meta_data import get_n_c_kappa_norm_var
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

def f_of_eps_n(
        A_norm: float,
        c: float,
        epsilon: float,
        kappa:float,
        n: int,
) -> float:
    """The function 'f' in the theoretical results, which describes the bounds
    on the accuracy of the sampling scheme in regards to operator norm 
    preservation

    f = \frac{3||A||^4 }{2||A||^2 + \frac{6c^3}{epsilon^2}(1 + 1/kappa(1 - 1/n^kappa))}

    Args:
        A_norm (float): The operator norm of A
        c (float): The coefficient of the row-norm-distribution
        epsilon (float): Amount of acceptable error
        kappa (float): 3k - 1, where k is the power of the power-law row-norm 
        distribution
        n (int): Number of rows of A

    Returns:
        float: The function evaluation of the given parameters
    """
    param_check([A_norm, c, epsilon, kappa, n])

    numerator = 3 * (A_norm ** 4)

    kappa_funct = 1 + ((1 / kappa) * (1 - (1 / (n ** kappa))))

    if kappa_funct < 0:
        # Uh-oh, n^kappa < 1
        raise ValueError(f"n to the kappa less than one: {n ** kappa}")

    denom_fraction = (6 * (c ** 3)) / (epsilon ** 2)

    denominator = (2 * (A_norm ** 2)) + (denom_fraction * kappa_funct)

    return numerator / denominator

def prob_of_success_power_law(
        A_norm: float,
        c: float,
        epsilon: float,
        kappa:float,
        n: int,
) -> float:
    """The probability that ||\tildeA||^2 is in bounds

    Args:
        A_norm (float): The operator norm of A
        c (float): The coefficient of the row-norm-distribution
        epsilon (float): Amount of acceptable error
        kappa (float): 3k - 1, where k is the power of the power-law row-norm 
        distribution
        n (int): Number of rows of A
        bounds_type: the type of bounds

    Returns:
        float: The probability that ||\tilde A|| is in plus or minus epsilon
        error
    """
    f_val = f_of_eps_n(
        A_norm=A_norm,
        c=c,
        epsilon=epsilon,
        kappa=kappa,
        n=n
    )
    print(f"log(n) = {log(n)}, (epsilon ** 2) * f_val = {(epsilon ** 2) * f_val}")
    exponent = log(n) - ((epsilon ** 2) * f_val)

    probability = 1 - 2 * exp(exponent)

    if probability > 1:
        raise ValueError(f"Invalid probability of success: {probability}")

    # if probability < 0:
    #     probability = 0

    return probability

def prob_of_success_strict(
        A_norm:float,
        epsilon:float,
        n:int,
        var_proxy:float,
) -> float:
    """The probability that ||~A||^2 is in (1 plus/minus epsilon)||A||^2, given
    the variance proxy: ||sum_i (||a_i|| - ||a_i||^2) a_i^Ta_i||

    Args:
        A_norm (float): The operator norm of the matrix
        epsilon (float): Some amount of allowed error
        n (int): The number of rows in A
        var_proxy (float): The matrix bernstein's variance proxy

    Returns:
        float: The probability that ||~A||^2 is within some epsilon of ||A||
    """
    numerator = 3 * (A_norm ** 4)
    denominator = (2 * (A_norm ** 2)) + ((6 * var_proxy) / (epsilon ** 2))
    fraction = numerator / denominator
    print(f"log(n) = {log(n)}, (epsilon ** 2) * fraction = {(epsilon ** 2) * fraction}")
    exponent = log(n) - ((epsilon ** 2) * fraction)

    probability = 1 - 2 * exp(exponent)

    if probability > 1:
        raise ValueError(f"Invalid probability of success: {probability}")

    return probability

def prob_of_success( #TODO
        A_norm: float,
        c: float,
        epsilon: float,
        kappa:float,
        n: int,
        var_proxy:float,
        bounds_type: BoundsType,
) -> float:
    match bounds_type:
        case BoundsType.POWER_TYPE:
            return prob_of_success_power_law(
                A_norm=A_norm,
                c=c,
                epsilon=epsilon,
                kappa=kappa,
                n=n,
            )
        case BoundsType.TIGHT_TYPE:
            return prob_of_success_strict(
                A_norm=A_norm,
                epsilon=epsilon,
                n=n,
                var_proxy=var_proxy, #TODO
            )
        case _:
            raise NotImplementedError()


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
        A_norm: float,
        c: float,
        epsilon: float,
        kappa:float,
        n: int,
        var_proxy:float,
        bounds_type: BoundsType,
) -> tuple[float, tuple[float, float], str]:
    """The probability that ||~A||^2 is in some bounds

    Args:
        A_norm (float): The operator norm of A
        c (float): The coefficient of the row-norm-distribution
        epsilon (float): Amount of acceptable error
        kappa (float): 3k - 1, where k is the power of the power-law row-norm 
        distribution
        n (int): Number of rows of A
        bounds_type: The type of bounds

    Returns:
        tuple[float, tuple[float, float], str]: [
        float: probability of success,
        tuple[float, float]: [lowerbound, upperbound]
        str: string representation of info
        ]
    """
    param_check([epsilon, A_norm, n, c, kappa])
    prob_success = prob_of_success(
        A_norm=A_norm,
        c=c,
        epsilon=epsilon,
        kappa=kappa,
        n=n,
        var_proxy=var_proxy,
        bounds_type=bounds_type,
    )
    lb, ub = error_bounds(A_norm=A_norm, epsilon=epsilon)

    latex_str = fr"$||\tilde A||^2 \in [{lb}, {ub}]$"

    string_rep = f"With probability {prob_success}, {latex_str}"

    return prob_success, (lb, ub), string_rep

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
    n, c, kappa, op_norm, var_proxy = get_n_c_kappa_norm_var(mat_name)
    return prob_in_bounds(
        A_norm=op_norm, 
        c=c, 
        epsilon=epsilon,
        kappa=kappa,
        n=n,
        var_proxy=var_proxy,
        bounds_type=bounds_type,
    )