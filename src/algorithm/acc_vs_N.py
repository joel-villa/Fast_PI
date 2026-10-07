"""Tests of form Accuracy vs. N, where N is the number of binomial trials in
the sampling scheme that generates ~A"""
import scipy
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse.linalg import norm

from .util.approx import binomial_A_tilde_sq

def acc_vs_N(
    A:scipy.sparse.sparray,
    rng:np.random.Generator,
    Ns:np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """The 'score' (2-norm) of ~A^TA vs A^TA, over variable N (expectation that
    increasing N increases accuracy)

    Resources:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigs.html

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
    ys = np.zeros_like(Ns)

    for i, N in enumerate(Ns):
        tilde_A_snd_moment = binomial_A_tilde_sq(
            A=A,
            num_trials=N,
            rng=rng,
        )

        # The approximation for v
        _, v_tilde = eigs(
            A=tilde_A_snd_moment,
            k=1,
            which="LM",
        )
        v_tilde = v_tilde[:,0]

        # The approximate score
        ys[i] = norm(A @ v_tilde)

    return Ns, ys, "empirical results"


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


def main():
    """For testing purposes
	"""
    on_easley = False

    import matplotlib
    if (not on_easley):
        matplotlib.use("QtAgg")

    from ..bounds.preprocess import preprocess
    import matplotlib.pyplot as plt
    from .util.comp import rel_value, rel_residue

    num_avg = 32
    Ns = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    run_length_type = 2
    log_y = True
    log_x = True
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

    for mat in mats:
        fig, ax = plt.subplots(
            figsize=(12.8, 14.4),
            dpi=256,
        )

        print(mat)
        A, _ = preprocess(mat_name=mat)
        tilde_A_snd_moment = A.transpose() @ A
        # The approximation for v
        _, v_tilde = eigs(
            A=tilde_A_snd_moment,
            k=1,
            which="LM",
        )
        v_tilde = v_tilde[:,0]
        top_score = norm(
            x=A @ v_tilde,
            ord=2,
        )

        rng = np.random.default_rng(seed=5334)

        baseline_work = []
        work_offsets = []

        ys = np.zeros_like(Ns)
        xs = np.zeros_like(Ns)
        ys_i = np.full((num_avg, ys.shape[0]), np.nan) # for averaging
        run_iters = np.zeros(num_avg)
        lbl = ""
        for i in range(num_avg):
            xs_temp, ys_temp, lbl = acc_vs_N(
                A=A,
                rng=rng,
                Ns=Ns,
            )

            ys_temp = rel_residue(
                max=top_score,
                xs=ys_temp,
                check_max=True,
            )

            # Iteration tracking
            curr_iter = ys_temp.shape[0]
            ys_i[i, : curr_iter] = ys_temp
            run_iters[i] = curr_iter
            if (run_iters[i] == np.max(run_iters)):
                # Track longest!
                xs = xs_temp

        ys = np.nanmean(ys_i, axis=0)

        least_iter = np.min(run_iters)
        most_iter = np.max(run_iters)
        median_iter = round(np.median(run_iters))

        if (run_length_type == 1):
            xs = xs[:least_iter]
            ys = ys[:least_iter]
        elif (run_length_type == 2):
            xs = xs[:median_iter]
            ys = ys[:median_iter]
        else:
            xs = xs[:most_iter]
            ys = ys[:most_iter]

        print(f"xs.shape = {xs.shape}")
        print(f"ys.shape = {ys.shape}")

        ax.plot(xs, ys, label=f"{lbl}, avg of {num_avg}")

        ax.set_title(f"Accuracy vs. N  ({mat}: {A.shape[0]}x{A.shape[1]}, {A.nnz} nnz)")
        ax.set_xlabel("N")
        ax.set_ylabel(r"$\frac{|Av| - |A\tilde v_1|}{|Av|}$", rotation=0)
        if (log_x):
            ax.set_xscale('log')
        if (log_y):
            ax.set_yscale('log')

        ax.legend()
        fig.canvas.draw()
        fig.savefig(f"plots/{mat}.png", dpi=plt.figure().dpi)
        if (not on_easley):
            plt.show()
        plt.close(fig)

if __name__ == '__main__':
    """For testing purposes
	"""
    main()
