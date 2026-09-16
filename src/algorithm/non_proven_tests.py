"""All functions here have the same return type (xs, ys, lbl)"""

#TODO: max work
import scipy
import numpy as np

from .util.approx import get_next_A_tilde
from .util.work import power_work
from .util import comp as comp
from .tests import baseline


def test_averaging(
        A:scipy.sparse,
        v0:np.ndarray,
        # lam_star:float,
        max_iter:int,
        seed:int,
        tol:float,
        num_samples:int,
        is_max: bool,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Test an averaging approach, generate N, approximations of A, use them 
    to independently get N seperate approximations for v_star, average them, 
    hopefully getting better results

    Args:
        A (scipy.sparse): the original matrix
        v0 (np.ndarray): initial guess for top eigenvector
        max_iter (max_iter): maximum number of iterations to do power iteration
        seed (int): for repeatable randomness (scikit does not have repeatable 
              randomness)
        tol (float): how much precision before terminating power
        num_samples (int): How many approximations of A?
        is_max (bool): If True then get the maximum work done by any given 
        parallel power iteration, else get the total work across all

    Returns:
        tuple[np.ndarray, np.ndarray, str]: 
        np.ndarray: the x-values (the ammount of work done),
        np.ndarray: the y-values (the score of the vector),
        str: the string representation of this test
    """
    #TODO: THIS BOY
    rng = np.random.default_rng(seed=seed)

    avg_vs = np.zeros(shape=(max_iter, A.shape[0]))
    all_work = np.zeros(shape=(max_iter, ))
    act_max_iter = 0

    for i in range(num_samples):
        A_tilde = get_next_A_tilde(A, rng=rng)
        _, vects, num_iter = comp.test_A_tilde(
            A_tilde=A_tilde,
            v0=v0,
            max_iter=max_iter,
            tol=tol,
        )
        assert avg_vs.shape == vects.shape, f"avg_vs.shape = {avg_vs.shape} != {vects.shape} = vects.shape" #TODO can delete this after testing
        avg_vs = comp.online_avg(
            old_avg=avg_vs,
            new_val=vects,
            new_total=i + 1,
        )
        work = np.zeros_like(all_work)
        work[0:num_iter] = power_work(
            matrix=A_tilde,
            num_iter=num_iter,
        )
        work[num_iter:] = work[num_iter - 1]

        if act_max_iter < num_iter:
            # New maximum!
            act_max_iter = num_iter

        if is_max:
            # Get the maximum work per index
            all_work = np.maximum(all_work, work)
        else:
            # Sum the work!
            all_work += work
    all_work = all_work[0:act_max_iter]
    avg_vs = avg_vs[0:act_max_iter]
    scores = np.asarray([comp.rayleigh_quotient(v, A) for v in avg_vs])
    lbl = f"avg ({num_samples} trials)"

    # Check work is increasing:
    assert (np.diff(all_work) >= 0).all(), f"np.diff(all_work): {all_work}"

    return all_work, scores, lbl


if __name__ == '__main__':
    """Yeahhh
    """
    from ..bounds.preprocess import preprocess
    import matplotlib.pyplot as plt

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
    for mat in mats:
        print(mat)
        A, _ = preprocess(mat_name=mat)
        rng = np.random.default_rng(seed=5334)
        rand_vect = rng.normal(loc=0.0, scale=0.0625, size=A.shape[0])
        print(f"ray_quot = {comp.rayleigh_quotient(rand_vect, A)}")
        xs, ys, lbl = baseline(
            A=A,
            v0=rand_vect,
            max_iter=max_iter,
            tol=tol,
        )
        plt.plot(xs,ys, label=lbl)

        for N in [1, 2, 4, 8, 16]:
            xs, ys, lbl = test_averaging(
                A=A,
                v0=rand_vect,
                max_iter=max_iter,
                seed=7,
                tol=tol,
                num_samples=N,
                is_max=True,
            )
            plt.plot(xs, ys, label=lbl)

        plt.legend()
        plt.show()
