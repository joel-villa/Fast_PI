"""Contains the newest sets of tests (has provable closeness results)"""

import scipy
import numpy as np

from .util import comp as comp
from .util.approx import get_next_A_tilde, naive_A_tilde_sq, binomial_A_tilde_sq
from .util import work as work
from ..util.power import power, rayleigh_quotient


def baseline(
        A: scipy.sparse.sparray,
        v0: np.ndarray,
        max_iter: int,
        tol: float,
) -> tuple [np.ndarray, np.ndarray, str]:
    """Run the default power iteration on this matrix

    Args:
        A: the original matrix
        v0: initial guess for top eigenvector
        max_iter: maximum number of iterations to do power iteration
        tol (float): how much precision before terminating power

    Returns:
        tuple [np.ndarray, np.ndarray, str]:
        np.ndarray: the x-values (the ammount of work done),
        np.ndarray: the y-values (the score of the vector),
        str: the string representation of this test
    """
    v=v0
    scores, _, iter = comp.init_test(
        A=A,
        v0=v0,
        max_iter=max_iter,
    )
    while iter < max_iter:
        lam, v = power(
            A = A,
            v0=v,
            num_iter=1,
        )
        scores[iter] = lam
        if (comp.rel_error(scores[iter], scores[iter - 1]) < tol):
            iter += 1
            break
        iter += 1

    scores = scores[0:iter]
    power_wrk = work.power_work(matrix=A, num_iter=iter)
    lbl = "baseline"
    return power_wrk, scores, lbl

def baseline_pays(
        A: scipy.sparse.sparray,
        v0: np.ndarray,
        max_iter: int,
        tol: float,
        is_distributed:bool,
) -> tuple [np.ndarray, np.ndarray, str]:
    """Run the default power iteration on this matrix, charging it for the
    initial cost of computing A^TA

    Args:
        A: the original matrix
        v0: initial guess for top eigenvector
        max_iter: maximum number of iterations to do power iteration
        tol (float): how much precision before terminating power
        is_distributed (bool): Changes ammount of work that goes into computing
        A^TA

    Returns:
        tuple [np.ndarray, np.ndarray, str]:
        np.ndarray: the x-values (the ammount of work done),
        np.ndarray: the y-values (the score of the vector),
        str: the string representation of this test
    """
    xs, ys, lbl = baseline(
        A=A.transpose() @ A,
        v0=v0,
        max_iter=max_iter,
        tol=tol
    )

    # Add initial work
    init_work = work.work_baseline(
        A=A,
        is_distributed=is_distributed,
    )
    xs += init_work

    return xs, ys, f"{lbl} (is_distributed={is_distributed})"


def test(
        A_sq: scipy.sparse.sparray,
        tilde_A_sq: scipy.sparse.sparray,
        v0: np.ndarray,
        max_iter: int,
        tol: float,
        init_work: int,
) -> tuple[np.ndarray, np.ndarray]:
    """A generic test of the accuracy of tilde_A_sq, given A_sq and other
    informaiton for power iteration

    Args:
        A_sq (scipy.sparse.sparray): A^TA
        tilde_A_sq (scipy.sparse.sparray): Approximation of
        v0 (np.ndarray): initial guess for top eigenvector
        max_iter (int): max number of iterations
        tol (float): tolerance of power iteration
        init_work (int): to be added to the work calculations

    Returns:
        tuple[np.ndarray, np.ndarray]:
        np.ndarray: work done per iteration of power
        np.ndarray: score of top eigenvector approxation per iteration of power
    """
    scores, _, i = comp.init_test(
        A=A_sq,
        v0=v0,
        max_iter=max_iter,
    )
    v=v0

    while i < max_iter:
        _, v = power(
            A=tilde_A_sq,
            v0=v,
            num_iter=1,
        )
        scores[i] = rayleigh_quotient( # Score of approximation based on actual
            x=v,
            A=A_sq,
        )
        if (comp.rel_error(scores[i], scores[i - 1]) < tol): #TODO: this was copied and pasted, bad coding practice, maybe add a helper function for rel error check
            i += 1
            break
        i += 1

    scores = scores[0:i]
    pwr_work = work.power_work(matrix=tilde_A_sq, num_iter=i)
    pwr_work += init_work

    return pwr_work, scores #TODO: untested


def naive_test(
        A: scipy.sparse.sparray,
        v0: np.ndarray,
        max_iter: int,
        tol: float,
        num_trials:int,
        seed:int,
        is_distributed:bool
) -> tuple [np.ndarray, np.ndarray, str]:
    """Run the less-parallelizable approach, return results

    Args:
        A (scipy.sparse.sparray): Matrix in question
        v0 (np.ndarray): Initial guess for top eigenvector
        max_iter (int): Max number of iterations for power
        tol (float): Tolerance of power iteration
        num_trials (int): Number of things to average
        seed (int): For repeatable randomization
        is_distributed (bool): Changes ammount of work that goes into computing
        an approximation for A^TA

    Returns:
        tuple [np.ndarray, np.ndarray, str]:
        np.ndarray: work per iteration
        np.ndarray: score of approximate eigenvector per iteration
        str: label of this test
    """
    rng = np.random.default_rng(seed=seed)
    tilde_A_sq = naive_A_tilde_sq(
        A=A,
        num_trials=num_trials,
        rng=rng,
    )
    init_work = work.work_naive_pessimistic(
        A=A,
        num_tirals=num_trials,
        is_distributed=is_distributed
    )

    xs, ys = test(
        A_sq=A.transpose() @ A,
        tilde_A_sq=tilde_A_sq,
        v0=v0,
        max_iter=max_iter,
        tol=tol,
        init_work=init_work,
    )

    lbl = f"naive (distributed={is_distributed}), {num_trials} trials"

    return xs, ys, lbl


def binomial_test(
        A: scipy.sparse.sparray,
        v0: np.ndarray,
        max_iter: int,
        tol: float,
        num_trials:int,
        seed:int,
        is_distributed:bool
) -> tuple [np.ndarray, np.ndarray, str]:
    """Run the parallelizable approach, return results

    Args:
        A (scipy.sparse.sparray): Matrix in question
        v0 (np.ndarray): Initial guess for top eigenvector
        max_iter (int): Max number of iterations for power
        tol (float): Tolerance of power iteration
        num_trials (int): Number of things to average
        seed (int): For repeatable randomization
        is_distributed (bool): Changes ammount of work that goes into computing
        an approximation for A^TA

    Returns:
        tuple [np.ndarray, np.ndarray, str]:
        np.ndarray: work per iteration
        np.ndarray: score of approximate eigenvector per iteration
        str: label of this test
    """
    rng = np.random.default_rng(seed=seed)
    tilde_A_sq = binomial_A_tilde_sq(
        A=A,
        num_trials=num_trials,
        rng=rng,
    )
    init_work = work.work_binomial(
        A=A,
        is_distributed=is_distributed
    )

    xs, ys = test(
        A_sq=A.transpose() @ A,
        tilde_A_sq=tilde_A_sq,
        v0=v0,
        max_iter=max_iter,
        tol=tol,
        init_work=init_work,
    )

    lbl = f"binomial (distributed={is_distributed}), {num_trials} trials"

    return xs, ys, lbl

def main(): #TODO: scale things from zero to one
    """For testing purposes
    """
    import matplotlib
    matplotlib.use("QtAgg")

    from ..bounds.preprocess import preprocess
    import matplotlib.pyplot as plt
    from .util.comp import rel_score


    print(plt.get_backend())

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
    ]

    mats = sorted(mats) #Alphabetical order
    max_iter = 30
    tol=1/64
    SEED = 7

    # HYPER PARAMS:
    dists = [
        True,
        # False,
    ]
    Ns = [
        1,
        4,
        8,
        16,
        32,
        64,
    ]
    funcs = [
        # naive_test,
        binomial_test,
    ]

    for mat in mats:
        print(mat)
        A, _ = preprocess(mat_name=mat)
        A_sqr = A.transpose() @ A
        two_norm = scipy.sparse.linalg.norm(
            x=A_sqr,
            ord=2,
        )
        rng = np.random.default_rng(seed=5334)
        rand_vect = rng.normal(loc=0.0, scale=0.0625, size=A.shape[0])
        normalized_vect = rand_vect / np.linalg.norm(rand_vect, ord=2)
        print(f"initial ray_quot = {comp.rayleigh_quotient(normalized_vect, A_sqr)}")

        baseline_work = []
        work_offsets = []

        for i, is_dist in enumerate(dists):
            xs, ys, lbl = baseline_pays(
                A=A,
                v0=normalized_vect,
                max_iter=max_iter,
                tol=tol,
                is_distributed=is_dist,
            )

            work_offsets.append(xs[0])
            baseline_work.append(xs[-1]) # Track final work

            # Start at zero
            xs = xs - work_offsets[i]

            xs = rel_score(
                max=baseline_work[i],
                xs=xs,
                check_max=True,
            )
            ys = rel_score(
                max=two_norm,
                xs=ys,
                check_max=True,
            )
            print(f"xs:{xs[:5]}, ys:{ys[:5]}, lbl:{lbl}")
            plt.plot(xs,ys, label=lbl)
        for func in funcs:
            for i, is_dist in enumerate(dists):
                for N in Ns:
                    xs, ys, lbl = func(
                        A=A,
                        v0=normalized_vect,
                        max_iter=max_iter,
                        tol=tol,
                        num_trials=N,
                        seed=SEED,
                        is_distributed=is_dist,
                    )
                    # Scale to be between zero and one
                    xs = xs - work_offsets[i]
                    xs = rel_score(
                        max=baseline_work[i],
                        xs=xs,
                        check_max=False,
                    )
                    ys = rel_score(
                        max=two_norm,
                        xs=ys,
                        check_max=True,
                    )
                    plt.plot(xs, ys, label=lbl)

        plt.title(f"Work vs. Accuracy of Top Eigenvector ({mat})")
        plt.xlabel(f"Approximate Proportion of Scalar Mults")
        plt.ylabel(r"$\frac{|A^TA\tilde v_1|}{|A^TA|}$", rotation=0)
        plt.legend()
        plt.show()

if __name__ == '__main__':
    """For testing purporses"""
    main()
