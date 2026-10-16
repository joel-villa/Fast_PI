"""This is the runnable file!"""
import scipy
import numpy as np

from tqdm import tqdm

from . import acc_vs_work
from .util import comp as comp
from .util import work as work

from ..bounds.preprocess import preprocess
from ..results.Results import Results

def main(
    log_y:bool,
    run_length_type:int,
    num_avg:int,
    mats:list[str],
    on_easley:bool,
    max_iter:int,
    tol:float,
    seed:int,
    Ns:list[int],
):
    """Generate performance plots for Fast-PI on those matrices

    Args:
        log_y (bool): Y-axis of plot log scaled?
        run_length_type (int): When averaging variable length sequences, must choose
        to average over all or part of them.
            1 -> average over those that exist for all trials
            2 -> average over the median num trials
            3 -> average over all (the outlier longest will be weighted 100%)
        num_avg (int): The number averaged in the result (to reduce noise) # TODO
        mats (list[str]): The matrices to test
        on_easley (bool): on_easley ->  don't display, not on_easly -> display!
        max_iter (int): max # of iterations of power
        tol (float): Stopping tolerance of power iteration
        seed (int): for repeated randomness
        Ns (list[int]): Number of bernoulli trials
    """
#    import matplotlib
#    if (not on_easley):
#        matplotlib.use("QtAgg")

#    import matplotlib.pyplot as plt
#    from .util.comp import rel_value, rel_residue


#    print(plt.get_backend())

    # HYPER PARAMS:
#    dists = [
#        True,
#        # False,
#    ]
#    funcs = [
#        # naive_test,
#        acc_vs_work.binomial_test,
#    ]
    #TODO: clean up this function don't need all of this complexity anymore..

    for mat in mats:
        print(mat)
        A, _ = preprocess(mat_name=mat)
        snd_moment_A = A.transpose() @ A
        two_norm = scipy.sparse.linalg.norm(
            x=snd_moment_A,
            ord=2,
        )
        rng = np.random.default_rng(seed=5334)
        rand_vect = rng.normal(loc=0.0, scale=0.0625, size=A.shape[0])
        normalized_vect = rand_vect / np.linalg.norm(rand_vect, ord=2)
        print(f"initial ray_quot = {comp.rayleigh_quotient(normalized_vect, snd_moment_A)}")

        work_baseline, acc_baseline, _ = acc_vs_work.baseline_pays(
            A=A,
            v0=normalized_vect,
            max_iter=max_iter,
            tol=tol,
            is_distributed=True, # Always assuming distributed -> less work
        )

#        work_offsets.append(work_baseline[0])
#        baseline_work.append(work_baseline[-1]) # Track final work
#
#        # Start at zero
#        work_baseline = work_baseline - work_offsets[i]
#
#        work_baseline = rel_value(
#            max=baseline_work[i],
#            xs=work_baseline,
#            check_max=True,
#        )
#        ys = rel_residue(
#            max=two_norm,
#            xs=ys,
#            check_max=True,
#        )
#        print(f"xs_temp.shape: {xs_temp.shape}, ys_temp.shape: {ys.shape}")
#        print(f"xs_temp[-8:]:{xs_temp[-8:]}, ys[-8:]:{ys[-8:]}, lbl:{lbl}")
#        ax.plot(xs_temp,ys, label=f"{lbl}, nnz = {snd_moment_A.nnz}")

        lbls = []
        num_Ns_tested = len(Ns)
        test_run_iters = np.zeros((num_Ns_tested, num_avg), int)
        nnzs = np.zeros((num_Ns_tested, num_avg))
        work = np.zeros((num_Ns_tested, max_iter))
        accuracies = np.full((num_Ns_tested, num_avg, max_iter), np.nan) # NaNs for that good averaging

        for i, N in tqdm(enumerate(Ns)):
#            ys = np.zeros(max_iter)
#            xs = np.zeros(max_iter)
#            ys_j = np.full((num_avg, max_iter), np.nan) # for averaging
            lbl = ""
            for j in range(num_avg):
                xs_temp, ys_temp, nnzs[i, j], lbl = acc_vs_work.binomial_test(
                    A=A,
                    v0=normalized_vect,
                    max_iter=max_iter,
                    tol=tol,
                    num_trials=N,
                    seed=seed,
                    is_distributed=True, # Always distributed => less work
                )
                # Iteration tracking
                curr_iter = ys_temp.shape[0]
#                ys_j[j, : curr_iter] = ys_temp
                accuracies[i, j, : curr_iter] = ys_temp
                test_run_iters[i, j] = curr_iter
                if (test_run_iters[i, j] == np.max(test_run_iters[i])):
                    # Track longest!
#                    xs = xs_temp
                    work[i, :test_run_iters[i,j]] = xs_temp

            lbls.append(lbl)

#            ys = np.nanmean(ys_j, axis=0)
#            nnz = round(np.mean(nnzs[k]))
#
#            least_iter = np.min(test_run_iters[k])
#            most_iter = np.max(test_run_iters[k])
#            median_iter = round(np.median(test_run_iters[k]))
#
#            if (run_length_type == 1):
#                xs = xs[:least_iter]
#                ys = ys[:least_iter]
#            elif (run_length_type == 2):
#                xs = xs[:median_iter]
#                ys = ys[:median_iter]
#            else:
#                xs = xs[:most_iter]
#                ys = ys[:most_iter]
#
#            print(f"xs.shape = {xs.shape}")
#            print(f"ys.shape = {ys.shape}")
#
#            ax.plot(xs, ys, label=f"{lbl}, nnz = {nnz}")
#
#        ax.set_title(f"Work vs. Accuracy of Top Eigenvector ({mat}: {A.shape[0]}x{A.shape[1]}, avg of {num_avg})")
#        ax.set_xlabel("Approximate Proportion of Scalar Mults")
#        ax.set_ylabel(r"$\frac{|A^TA| - |A^TA\tilde v_1|}{|A^TA|}$", rotation=0)
#        if (log_y):
#            ax.set_yscale('log')
#        ax.legend()
#        fig.canvas.draw()
#        fig.savefig(f"plots/{mat}.png", dpi=fig.dpi) # Goal: high-resolution
#        if (not on_easley):
#            plt.show()
#        plt.close(fig)
        """
        mat_name: str | None = None,
        mat_shape: tuple[int, int] | None = None,
        is_sparse: bool | None = None,
        snd_moment_norm: float | None = None,
        snd_moment_nnz: int | None = None,
        max_iter: int | None = None,
        tol: float | None = None,
        seed: int | None = None,
        num_avg: int | None = None,
        run_length_type: int | None = None,
        n_binom_trials: list[int] | None = None,
        lbls: list[str] | None = None,
        notes: str | None = None,
        test_run_iters: np.ndarray | None = None,
        nnzs: np.ndarray | None = None,
        baseline_work: np.ndarray | None = None,
        baseline_accuracies: np.ndarray | None = None,
        work: np.ndarray | None = None,
        accuracies: np.ndarray | None = None,
        """
        results = Results(
            mat_name = mat,
            mat_shape = A.shape,
            is_sparse = True,
            snd_moment_norm = two_norm,
            snd_moment_nnz = snd_moment_A.nnz,
            max_iter = max_iter,
            tol = tol,
            seed = seed,
            num_avg = num_avg,
            run_length_type = run_length_type,
            n_binom_trials = Ns,
            lbls = lbls,
            notes = None,
            test_run_iters = test_run_iters,
            nnzs = nnzs,
            baseline_work = work_baseline,
            baseline_accuracies = acc_baseline,
            work = work,
            accuracies = accuracies,
        )
        file_id = results.save_results()
        if not on_easley:
            results = Results(file_id=file_id)
            results.disp_results(log_y=True)


if __name__ == '__main__':
    """For testing purporses"""
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
#    mats = [
#        "ct20stif",
#        "finan512",
#        "nasasrb",
#    ]

    mats = sorted(mats) #Alphabetical order
    max_iter = 2 ** 17 # 131, 072
    tol=0 # tolerance of zero -> machine percision
    seed = 7

    Ns = [
        1,
        2,
        4,
        8,
        16,
        32,
        64,
    ]

    main(
        num_avg=32,
        run_length_type=2,
        log_y=True,
        mats=mats,
        on_easley=True,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
        Ns=Ns,
    )
