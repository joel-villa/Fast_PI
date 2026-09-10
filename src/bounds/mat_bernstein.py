"""Ooh-wee, finally some good results, how exciting:

With probability at least 1 - exp(ln(2n) - N * ||Aw||^2 * epsilon^2 / 4):
    ||A~w|| is in [(1 - epsilon) * ||Aw||, ||Aw||]
"""

import numpy as np
from .util.meta_data import get_n_norm

def get_min_epsilon(
        n: int,
        N: int,
        lambda_max: float,
) -> float | None:
    """Get the minimum valid epsilon, s.t. an epsilon bound exists

    Args:
        n (int): number of rows in the matrix
        N (int): number of samples of ~A
        lambda_max (float): ||Aw||

    Returns:
        float | None: if a minimum exists return it, if this minimum is larger 
        than one, return None
    """
    sqrt_term = np.sqrt(np.log(2 * n) / N)
    epsilon_min = (2 / lambda_max) * sqrt_term

    if epsilon_min > 1:
        return None
    else:
        return epsilon_min

def delta_from_epsilon(
        n: int, 
        N: int, 
        lambda_max: float, 
        epsilon: float,
) -> float:
    """Get the theoretical probability of good events

    With probability at least 1 - exp(ln(2n) - N * ||Aw||^2 * epsilon^2 / 4):
    ||A~w|| is in [(1 - epsilon) * ||Aw||, ||Aw||]

    Args:
        n (int): number of rows in the matrix
        N (int): number of iid samples of the A approximation
        lambda_max (float): ||Aw||
        epsilon (float): ammount of allowed error

    Returns:
        float: The theoretical probability of success
    """
    assert epsilon >= 0 and epsilon <= 1, f"epsilon={epsilon}"
    assert N > 0, f"N={N}"

    val_a = np.log(2 * n)

    epsilon_sq = epsilon ** 2
    lambda_sq = lambda_max ** 2
    val_b = (1/4) * N * lambda_sq * epsilon_sq

    # assert val_b > val_a, f"np.log(2 * n) = {val_a},    (1/4) * N * lambda_sq * epsilon_sq = {val_b}"

    delta = 1 - np.exp(val_a - val_b)

    assert delta <= 1 and delta >= 0, f"delta={delta}"

    return delta

def epsilon_from_delta(
        n: int, 
        N: int, 
        lambda_max: float, 
        delta: float,
) -> float:
    """Get the theoretical epsilon

    with probability at least delta:
        ||A~w|| is in [(1 - epsilon) * ||Aw||, ||Aw||]
    where epsilon = 2/||Aw|| * sqrt(1/N * ln(2n/(1 - delta)))

    Args:
        n (int): number of rows in the matrix
        N (int): number of iid samples of the A approximation
        lambda_max (float): ||Aw||
        delta (float): desired probability of success

    Returns:
        float: The theoretical epsilon
    """
    assert delta >= 0 and delta < 1, f"delta={delta}"
    assert N > 0, f"N={N}"
    assert lambda_max > 0, f"lambda_max={lambda_max}"

    log_term = np.log((2 * n) / (1 - delta))
    sqrt_term = np.sqrt((1 / N) * log_term)
    epsilon = (2 / lambda_max) * sqrt_term

    assert epsilon >= 0 and epsilon <= 1, f"epsilon={epsilon}"

    return epsilon
    

def get_n_and_norm(mat_name:str) -> tuple[int, float]:
    """Get the operator norm and the number of rows of the matrix

    Args:
        mat_name (str): Suite Sparse matrix name

    Returns:
        tuple[int, float]: 
        int: number of rows of A
        float: operator norm of A
    """
    # For ||Av|| = ||A||, A must be hermitian
    # assert mat_name is hermetian #TODO

    return get_n_norm(matrix_name=mat_name)

def mat_min_epsilon(
        mat_name:str,
        N: int
) -> float | None:
    """Get the minimum epsilon of the given matrix

    Args:
        mat_name (str): Suite Sparse matrix name
        N (int): Number of samples of ~A

    Returns:
        float | None: if minimum epsilon <= 1, that minimum epsilon, else None
    """
    n, op_norm = get_n_and_norm(mat_name=mat_name)

    return get_min_epsilon(
        n=n,
        N=N,
        lambda_max=op_norm,
    )

def get_mat_delta(
        mat_name: str,
        N: int,
        epsilon: float,
) -> float:
    """Get the theoretical probability of success

    Args:
        mat_name (str): The matrix to do fast power iteration on
        N (int): Number of iid samples of approximately A
        epsilon (float): Ammount of allowable error

    Returns:
        float: probability of success
    """

    n, op_norm = get_n_and_norm(mat_name)

    return delta_from_epsilon(
        n=n,
        N=N,
        lambda_max=op_norm,
        epsilon=epsilon,
    )

def get_mat_epsilon(
        mat_name: str,
        N: int,
        delta: float,
) -> float:
    """Get the theoretical ammount of error

    Args:
        mat_name (str): Matrix name
        N (int): Number of iid samples of approximately A
        delta (float): Desired probability of success

    Returns:
        float: Theoretical maximum ammount of error
    """

    n, op_norm = get_n_norm(matrix_name=mat_name)

    return epsilon_from_delta(
        n=n,
        N=N,
        lambda_max=op_norm,
        delta=delta,
    )



if __name__ == '__main__':
    """Main for testing purposes
    """
    mats = [
        "1138_bus",
        "494_bus",
        "Harvard500",
        "bcspwr06",
        "bcsstk07",
        "bcsstk08",
        "bcsstk19",
        "bcsstk34",
        "bcsstm07",
        "blckhole",
        "cage7",
        "can_229",
        "dwt_193",
        "eris1176",
        "ex2",
        "fs_541_1",
        "gre_1107",
        "gre_343",
        "hor_131",
        "lshp1561",
        "msc00726",
        "nasa1824",
        "nos3",
        "tomography",
        # "nopoly",
    ]

    mats = sorted(mats) #Alphabetical order
    COUNT = 5

    for mat in mats:
        print(mat)
        num_rows, _ = get_n_and_norm(mat_name=mat)

        Ns = np.linspace(
            start=np.ceil(np.log(num_rows)), 
            stop=num_rows // 2, 
            num=COUNT,
        )

        for N in Ns:
            print(f"N={N}")
            min_epsilon = mat_min_epsilon(
                mat_name=mat,
                N=N,
            )

            if min_epsilon is None or min_epsilon > 0.75:
                print(f"skipping N={N}, min epsilon > 0.75")
            else:
                print(f"min_epsilon={min_epsilon}")
                epsilons = np.linspace(start=min_epsilon + 0.00001, stop=0.75, num=COUNT)

                for epsilon in epsilons:
                    delta = get_mat_delta(
                        mat_name=mat,
                        N=N,
                        epsilon=epsilon,
                    )
                    print(f"With probability at least {delta}, "
                          fr"$||A \tilde w|| \in [(1 - {epsilon}) ||Aw||, ||Aw||]$")
