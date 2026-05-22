"""
For plotting tests which take into account work required to get a 
top-eigenvector approximation, with some score

Where work is number of scalar mults and "score" is closeness of singular value
"""
from Sparsification_Research.src.Plotter import Plotter
from .plot_util.plot import init

import numpy as np

from src.tests import score as tst

def test(funct, plotter, num_avg, A, num_iter, seed, kwargs={}):
    """Test the given function from src/tests
    
    Args:
        funct: a fucntion which takes a matrix, some x vals, and a seed, and 
               returns xs and ys to be plotted
        plotter: a Plotter() object
        num_avg: average of how many tests?
        A: a CSR sparse matrix
        num_iter: how many iterations?
        seed: for randomized reproducability
        kwargs: input for funct
    Return: 
        The number of iterations ran (important for tests which terminate after
        converged)
    """
    ys = np.zeros(num_iter)
    xs = np.zeros(num_iter)
    ys_i = np.zeros(num_iter)

    label = ""
    
    for i in range(num_avg):
        seed_i = seed + i * 3 

        xs, ys_i, label = funct(A=A, max_iter=num_iter, seed=seed_i, **kwargs)
        ys += ys_i
    
    ys = ys / num_avg

    plotter.add_to_plot(xs, ys, label=label)

    # Remove trailing zeros only ('b' for back)
    xs = np.trim_zeros(xs, trim='b')
    ys = np.trim_zeros(ys, trim='b')

    return np.shape(xs)[0]



if __name__ == '__main__':
    seed = 10
    max_iter = 64
    p = 70
    num_tests = 1

    # SOME MATS THAT SHOW GOOD BEHVIOR: ["bcsstk07", "bcsstk19", "bcsstm07", "impcol_d"]
    mats = ["494_bus", "bcsstk07", "bcsstk08", "bcsstk19", "bcsstm07", "impcol_d"]

    """These mats seem to imperically have this in common: small spectral gap,
      and large eigenvalues
      TODO: prove why this may be the case? 
      TODO: an itterative approach which will use theses matrix approximations 
            to converge faster"""


    types = ["jl_gaussian", "jl_sparse"]

    plotter = Plotter(save_fig=False, show_fig=True, fig_size=(12, 6))

    for mat_name in mats:
        A, kwargs = init(mat_name=mat_name, seed=seed)

        

        plotter.init_plot(title=f"Power Convergence of {mat_name}", 
                          x_label="number of iterations",
                          y_label="residual", 
                          save_name=f"{mat_name}_subset_no_swap_{p}",
                          grid_on=True) 

        # Baseline test
        num_iter = test(funct=tst.baseline,
                        plotter=plotter,
                        num_avg=1,
                        num_iter=max_iter,
                        A=A, 
                        seed=seed,
                        kwargs=kwargs | {"tol": 1e-7})

        plotter.finish()