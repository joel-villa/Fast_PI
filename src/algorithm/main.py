"""This is the runnable file!"""
import scipy
import numpy as np

from . import acc_vs_work
from .util import comp as comp
from .util import work as work

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
    import matplotlib
    if (not on_easley):
        matplotlib.use("QtAgg")

    from ..bounds.preprocess import preprocess
    import matplotlib.pyplot as plt
    from .util.comp import rel_value, rel_residue


    print(plt.get_backend())

    # HYPER PARAMS:
    dists = [
        True,
        # False,
    ]
    funcs = [
        # naive_test,
        acc_vs_work.binomial_test,
    ]
    #TODO: clean up this function don't need all of this complexity anymore..

    for mat in mats:
        fig, ax = plt.subplots(
            figsize=(12.8, 14.4),
            dpi=256,
        )

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
            xs_temp, ys, lbl = acc_vs_work.baseline_pays(
                A=A,
                v0=normalized_vect,
                max_iter=max_iter,
                tol=tol,
                is_distributed=is_dist,
            )

            work_offsets.append(xs_temp[0])
            baseline_work.append(xs_temp[-1]) # Track final work

            # Start at zero
            xs_temp = xs_temp - work_offsets[i]

            xs_temp = rel_value(
                max=baseline_work[i],
                xs=xs_temp,
                check_max=True,
            )
            ys = rel_residue(
                max=two_norm,
                xs=ys,
                check_max=True,
            )
            print(f"xs:{xs_temp[:5]}, ys:{ys[:5]}, lbl:{lbl}")
            ax.plot(xs_temp,ys, label=lbl)
        for func in funcs:
            for i, is_dist in enumerate(dists):
                for N in Ns:
                    ys = np.zeros(max_iter)
                    xs = np.zeros(max_iter)
                    ys_j = np.full((num_avg, max_iter), np.nan) # for averaging
                    run_iters = np.zeros(num_avg)
                    lbl = ""
                    for j in range(num_avg):
                        xs_temp, ys_temp, lbl = func(
                            A=A,
                            v0=normalized_vect,
                            max_iter=max_iter,
                            tol=tol,
                            num_trials=N,
                            seed=seed,
                            is_distributed=is_dist,
                        )
#                        print(f"xs={xs[:15]}")
                        # Scale to be between zero and one
                        xs_temp = xs_temp - work_offsets[i]
                        xs_temp = rel_value(
                            max=baseline_work[i],
                            xs=xs_temp,
                            check_max=False,
                        )
                        ys_temp = rel_residue(
                            max=two_norm,
                            xs=ys_temp,
                            check_max=True,
                        )

                        # Iteration tracking
                        curr_iter = ys_temp.shape[0]
                        ys_j[j, : curr_iter] = ys_temp
                        run_iters[j] = curr_iter
                        if (run_iters[j] == np.max(run_iters)):
                            # Track longest!
                            xs = xs_temp

                    ys = np.nanmean(ys_j, axis=0)

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

        ax.set_title(f"Work vs. Accuracy of Top Eigenvector ({mat}: {A.shape[0]}x{A.shape[1]}, {A.nnz} nnz)")
        ax.set_xlabel("Approximate Proportion of Scalar Mults")
        ax.set_ylabel(r"$\frac{|A^TA| - |A^TA\tilde v_1|}{|A^TA|}$", rotation=0)
        if (log_y):
            ax.set_yscale('log')
        ax.legend()
        fig.canvas.draw()
        fig.savefig(f"plots/{mat}.png", dpi=plt.figure().dpi)
        if (not on_easley):
            plt.show()
        plt.close(fig)


if __name__ == '__main__':
    """For testing purporses"""
    # mats = [
    #     "1138_bus",
    #     "494_bus",
    #     "Harvard500",
    #     "bcspwr06",
    #     "bcsstk07",
    #     "bcsstk08",
    #     "bcsstk19",
    #     "bcsstk34",
    #     "bcsstm07",
    #     "blckhole",
    #     "cage7",
    #     "can_229",
    #     "dwt_193",
    #     "eris1176",
    #     "ex2",
    #     "fs_541_1",
    #     "gre_1107",
    #     "gre_343",
    #     "hor_131",
    #     "lshp1561",
    #     "msc00726",
    #     "nasa1824",
    #     "nos3",
    #     "tomography",
    # ]
    mats = [
        "ct20stif",
#        "finan512",
#        "nasasrb",
    ]

    mats = sorted(mats) #Alphabetical order
    max_iter = 2048
    tol=1/8192
    seed = 7

    Ns = [
        1,
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
