"""Ooh-wee, finally some good results, how exciting:

With probability at least 1 - exp(ln(2n) - N * ||Aw||^2 * epsilon^2 / 4):
    ||A~w|| is in [(1 - epsilon) * ||Aw||, ||Aw||]
"""

import numpy as np
from .util.meta_data import get_n_norm

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

    assert val_b > val_a, f"np.log(2 * n) = {val_a},    (1/4) * N * lambda_sq * epsilon_sq = {val_b}"

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
    Ns = [10, 100, 1000, 10000]
    eps = [0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625]
    delts = [0.5, 0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.9921875]
    for mat in mats:
        print(mat)

        for N in Ns:
            # print(f"Epsilon tests for N={N}")
            # for epsilon in eps:
            #     try:
            #         delta=get_mat_delta(
            #             mat_name=mat,
            #             N=N,
            #             epsilon=epsilon,
            #         )
            #         print(f"With probability at least {delta}, "
            #               fr"$||A \tilde w|| \in [(1 - {epsilon}) ||Aw||, ||Aw||]$")
            #     except Exception as e:
            #          print(f"skipping N={N}, epsilon={epsilon}, error: {e}")
                

            print(f"Delta tests for N={N}")
            for delta in delts:
                try:
                    epsilon=get_mat_epsilon(
                        mat_name=mat,
                        N=N,
                        delta=delta,
                    )
                    print(f"With probability at least {delta}, "
                          fr"$||A \tilde w|| \in [(1 - {epsilon}) ||Aw||, ||Aw||]$")
                except Exception as e:
                    print(f"skipping N={N}, delta={delta}, error: {e}")