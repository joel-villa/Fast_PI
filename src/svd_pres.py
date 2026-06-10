"""
Main Testing suite for svd preservation tests, that is how well are the top-k 
left-singular vectors preserved? 
"""

import numpy as np

from scipy.sparse.linalg import svds

from .tests.svd_pres import subset_p

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
    seed = 10
    k = 3
    sub_types = (
        "random", 
        "1-norm", 
        "2-norm", 
        "nystrom",
    )
    ps = range(0, 100, 10)
    gamma = 0.9
    
    mats = [
        "494_bus", # Invalid for mag-based sparsifification
        # "bcsstk07", 
        # "bcsstk08", 
        # "bcsstk19", 
        # "bcsstm07", 
        # "bcspwr06",
        # "impcol_d", # NOT POSITIVE DEFINITE, doesn't work with nystrom sampling
        # "bibd_13_6", # RECTANGULAR: Doesn't work with Nystrom
    ]

    plotter = Plotter(save_fig=False, show_fig=True, fig_size=(12, 6))

    for mat_name in mats:
        plotter.init_plot(
            title=None, 
            x_label="ps",
            y_label=None, 
            save_name=f"{mat_name}_svd_pres",
            grid_on=True,
        ) 
         
        A = init(mat_name=mat_name)

        rng = np.random.default_rng(seed=seed)
        u0_shape = (A.shape[0], k)
        u0 = rng.normal(0, 1, u0_shape)

        u_star, s, vh = svds(
            A=A,
            k=k,
            # maxiter=256, #TODO: do we need this? Plus v0
            rng=seed,
        )

        xs, ys = subset_p(
            A=A, 
            u_0=u0, 
            u_star=u_star, 
            k=k, 
            seed=seed, 
            ps=ps, 
            type=sub_types[0], 
            gamma=gamma,
        )

        for i in range(ys.shape[0]):
            plotter.add_to_plot(xs, ys[i], label=f"{i}")

        plotter.finish()


