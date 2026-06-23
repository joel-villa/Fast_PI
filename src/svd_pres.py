"""
Main Testing suite for svd preservation tests, that is how well are the top-k 
left-singular vectors preserved? 
"""

import numpy as np

from scipy.sparse.linalg import svds

from .tests import svd_pres as tst
from .util.svd_pres import sort_svd_output

from Sparsification_Research.src.Plotter import Plotter
from Sparsification_Research.src.SSGetter import SSGetter

def init(mat_name):
    """ Get intial information for tests

    Args: 
        mat_name: the name of the matrix in the SuiteSparse Matrix Collection

    Return:
        A: the matrix in CSR format
    """
    ss_getter = SSGetter(in_csr=False)
    A = ss_getter.get(mat_name)

    return A

if __name__ == '__main__':
    pows = np.linspace(
        start=0,
        stop=0.25,
        num=16,
    )


    ps = np.linspace(
        start=0,
        stop=100,
        num=32,
    )
    seed = 10
    k = 8

    proven_types = (
        # "one-norm", 
        "two-norm",
        "inf-norm",
        "uniform",
    )

    sub_types = (
        # "random", 
        # "1-norm", 
        # "2-norm", 
        # "2-norm2", 
        # "nystrom",
        # "scale",
        # "row",
        # "row_choice",
        # "row_ind",
        # "const-dim",
        "det-2-norm",
    ) 

    spr_types = (
        # "Generic",
        "SD",
        # "CD",
        # "MB", #TODO test this
    )

    gamma = 0.9 #For nystrom sampling
    
    mats = [
        "494_bus", # Invalid for mag-based sparsifification
        "bp_0",
        "bcsstk07", 
        "bcsstk08", 
        "bcsstk19", #TODO: this has odd behavior w/ proven tests (not starting at zero)
        "bcsstm07", 
        "bcspwr06",
        # "impcol_d", # NOT POSITIVE DEFINITE, doesn't work with nystrom sampling
        # "bibd_13_6", # RECTANGULAR: Doesn't work with Nystrom
    ]

    plotter = Plotter(save_fig=False, show_fig=True, fig_size=(12, 6))

    for mat_name in mats:
        # TODO: fix labeling, allow for looping through hyper-params
        plotter.init_plot(
            title=mat_name, 
            x_label="",
            y_label=r"$\frac{|v_k - v_k^*|_2}{|v_k^*|_2}$", 
            save_name=f"{mat_name}_svd_pres",
            grid_on=True,
        ) 
         
        A = init(mat_name=mat_name)

        rng = np.random.default_rng(seed=seed)
        u0 = rng.normal(0, 1, A.shape[0])
        v0 = rng.normal(0, 1, A.shape[1])

        u, s, vh = svds(
            A=A,
            k=k,
            # maxiter=256, #TODO: do we need this? Plus v0
            rng=seed,
        )

        u_star, s, v_star = sort_svd_output(u, s, vh)

        xs, ys, lbl = tst.pi_inc_pow(
            A=A,
            v0=v0,
            v_star=v_star,
            k=k,
            seed=seed,
            pows=pows,
            gen_type=proven_types[0],
            sf_kwargs={},
        )

        for i in range(ys.shape[1]):
            plotter.add_to_plot(xs, ys[:, i], label=f"k = {i + 1} ({lbl})")

        # xs, ys, lbl = tst.subset_p(
        #     A=A, 
        #     u_0=u0, 
        #     u_star=u_star, 
        #     k=k, 
        #     seed=seed, 
        #     ps=ps, 
        #     type=sub_types[0], 
        #     gamma=gamma,
        # )

        # for i in range(ys.shape[1]):
        #     plotter.add_to_plot(xs, ys[:, i], label=f"k = {i + 1} ({lbl})")

        # xs, ys = tst.sparse_p(
        #     A=A, 
        #     u_0=u0, 
        #     u_star=u_star, 
        #     k=k, 
        #     seed=seed, 
        #     ps=ps, 
        #     type=spr_types[sub_i], 
        # )

        # for i in range(ys.shape[1]):
        #     plotter.add_to_plot(xs, ys[:, i], label=f"k = {i + 1} ({spr_types[sub_i]})")

        plotter.finish()


