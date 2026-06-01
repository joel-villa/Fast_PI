"""
Has some runnable methods, which plot the convergence of a JL-enhanced power 
iteration method
"""

from scipy.linalg import norm # 2-norm by default

import numpy as np

from Sparsification_Research.src.Plotter import Plotter
from Sparsification_Research.src.SSGetter import SSGetter

from .util.eig_functs import top_left
from .tests import power as pwr
from .tests import sparse_power as spwr
from .tests import subset_power as sub

def test(funct, plotter, num_avg, A, num_iter, seed, kwargs={}):
    """Test the given function from tests.py 
    
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

        xs, ys_i, label = funct(A=A, num_iter=num_iter, seed=seed_i, **kwargs)
        ys += ys_i
    
    ys = ys / num_avg

    # Remove trailing zeros only ('b' for back)
    xs = np.trim_zeros(xs, trim='b')
    ys = np.trim_zeros(ys, trim='b')

    plotter.add_to_plot(xs, ys, label=label)

    return np.shape(xs)[0]

def init(mat_name, seed):
    """ Get intial information for tests

    Args: 
        mat_name: the name of the matrix in the SuiteSparse Matrix Collection
        seed: for generating initial guess for top left eigenvector

    Return:
        A: the matrix in CSR format
        kwargs: a dictionary containing the initial guess for u, as wall as the 
                solution u
    """
    ss_getter = SSGetter(in_csr=False)
    A = ss_getter.get(mat_name)
        
    u_star =  top_left(A)

    rng = np.random.default_rng(seed=seed)
    u0 = rng.normal(0, 1, A.shape[0])
    u0 = u0 / norm(u0)

    print(f"Testing {mat_name}")

    kwargs = {"u_star": u_star,
              "u_0": u0,
              }

    return A, kwargs

def main_swap():
    """
    For testing swap behavior 
    """

    # mats    = ["494_bus"]
    seed    = 10
    num_avg = 1
    max_iter = 64


    # SOME MATS THAT SHOW GOOD BEHVIOR: ["bcsstk07", "bcsstk19", "bcsstm07", "impcol_d"]
    mats = ["bcsstk07", "bcsstk19", "bcsstm07", "impcol_d"]
    mats = ["bcsstk19", "bcsstm07"]
    """These mats seem to imperically have this in common: small spectral gap,
      and large eigenvalues
      TODO: prove why this may be the case? 
      TODO: an itterative approach which will use theses matrix approximations 
            to converge faster"""


    types = ["jl_gaussian", "jl_sparse"]
    ps = [80]
    step_size = 8

    plotter = Plotter(save_fig=False, show_fig=True, fig_size=(12, 6))

    for mat_name in mats:
        A, kwargs = init(mat_name=mat_name, seed=seed)

        plotter.init_plot(title=f"Power Convergence of {mat_name}", 
                          x_label="number of iterations",
                          y_label="residual", 
                          save_name=f"{mat_name}_{ps[0]}_swap",
                          grid_on=True,
                          ) 

        # Baseline test
        num_iter = test(funct=pwr.baseline_pwr_tolerance_termination,
                        plotter=plotter,
                        num_avg=1,
                        num_iter=max_iter,
                        A=A, 
                        seed=seed,
                        kwargs=kwargs | {"tol": 1e-7})
        
        for p in ps:
            for type in types:
                test(funct=pwr.jl_percent_reduced, 
                     plotter=plotter, 
                     num_avg=num_avg,
                     A=A, 
                     num_iter=num_iter,
                     seed=seed,
                     # '|' is dictionary union
                     kwargs=kwargs | {"p": p, "type" : type}, 
                     )
                test(funct=pwr.multi_jl_p_reduce, 
                     plotter=plotter, 
                     num_avg=num_avg,
                     A=A,
                     num_iter=num_iter,
                     seed=seed,
                     kwargs=kwargs | {"p": p, "step_size": step_size, "type" : type},
                     )
            
            test(funct=sub.percent_subset_pow, 
                 plotter=plotter, 
                 num_avg=num_avg,
                 A=A, 
                 num_iter=num_iter, 
                 seed=seed, 
                 kwargs=kwargs | {"p": p},
                 )
            
            if (mat_name == "impcol_d"):
                # impcol_d gets "infs" with percent_subset_pow_swap, skipping it for now
                continue 
            
            test(funct=sub.percent_subset_pow_swap, 
                 plotter=plotter, 
                 num_avg=num_avg,
                 A=A, 
                 num_iter=num_iter,
                 seed=seed,
                 kwargs=kwargs | {"p": p, "step_size" : step_size},
                 )

        plotter.finish()


def main_no_swap():
    """
    For testing behavior of several approaches inluding:
    - jl reduction
    - column subsets
    - sparsification
    """
    seed    = 10
    num_avg = 1
    max_iter = 64
    p = 70
    num_tests = 2


    # SOME MATS THAT SHOW GOOD BEHVIOR: ["bcsstk07", "bcsstk19", "bcsstm07", "impcol_d"]
    mats = ["494_bus", "bcsstk07", "bcsstk08", "bcsstk19", "bcsstm07"]
    # mats = ["impcol_d"] # Not positive definite: doesn't work for nystrom sampling

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
        num_iter = test(funct=pwr.baseline_pwr_tolerance_termination,
                        plotter=plotter,
                        num_avg=1,
                        num_iter=max_iter,
                        A=A, 
                        seed=seed,
                        kwargs=kwargs | {"tol": 1e-7})
        
        # for type in types:
        #     for i in range(num_tests):
        #         test(funct=pwr.jl_percent_reduced, 
        #              plotter=plotter, 
        #              num_avg=num_avg, 
        #              A=A, 
        #              num_iter=num_iter,
        #              seed=seed * i + i, 
        #              kwargs=kwargs | {"p": p, "type" : type},
        #              )
            
        for i in range(num_tests):
            for type in ("random", "1-norm", "2-norm", "nystrom"):
                test(
                    funct=sub.percent_subset_pow, 
                    plotter=plotter, 
                    num_avg=num_avg,
                    A=A,
                    num_iter=num_iter,
                    seed=seed * i + i, 
                    kwargs=kwargs | {"p": p, "type": type, "gamma": 0.5},
                )

        # for i in range(num_tests):
        #     test(funct=spwr.percent_sparse_pwr,
        #          plotter=plotter,
        #          num_avg=num_avg,
        #          A=A,
        #          num_iter= num_iter, 
        #          seed=seed * i + i,
        #          kwargs=kwargs | {"p": 4, "type": "Generic", "tol": None},
        #          )

        plotter.finish()

def main_sparsification():
    """
    For testing behavior of sparsification
    """
    seed    = 10
    """NOTE: when doing sparse-technique, label will only show last 'new zeros', 
    not an of all runs """
    num_avg = 5
    max_iter = 64

    # Expected percent of sparsification
    ps = [0.01, 0.1, 1, 10]
    xs = [1, 4, 16, 64]


    mats = ["494_bus", 
            "bcsstk07", 
            "bcsstk08", 
            "bcsstk19", 
            "bcsstm07", 
            "impcol_d",
            ]

    plotter = Plotter(save_fig=False, show_fig=True, fig_size=(12, 6))


    for mat_name in mats:
        A, kwargs = init(mat_name=mat_name, seed=seed)
        
        plotter.init_plot(title=f"Power Convergence of {mat_name}", 
                          x_label="number of iterations",
                          y_label="residual", 
                          save_name=f"{mat_name}_sparsification",
                          grid_on=True) 

        # Baseline test
        num_iter = test(funct=pwr.baseline_pwr_tolerance_termination,
                        plotter=plotter,
                        num_avg=1,
                        num_iter=max_iter,
                        A=A, 
                        seed=seed,
                        kwargs=kwargs | {"tol": 1e-7})
        
        tol = None

        # Percentage tests (maintain diagonal)
        for i, p in enumerate(ps):
            test(funct=spwr.percent_sparse_pwr,
                 plotter=plotter,
                 A=A,
                 seed=2 + i * 3,
                 num_avg=num_avg,
                 num_iter= num_iter, 
                 kwargs= kwargs | {"p": p, "type": "MD","tol": tol},
                 )
        
        # Percentage tests (sparsify diagonal)
        for i, p in enumerate(ps):
            test(funct=spwr.percent_sparse_pwr,
                 plotter=plotter,
                 A=A,
                 seed=2 + i * 3,
                 num_avg=num_avg,
                 num_iter= num_iter, 
                 kwargs= kwargs | {"p": p, "type": "GENERIC","tol": tol},
                 )
        
        # Expected new zeros tests
        # for i, x in enumerate(xs):
        #     test(funct=spwr.expected_sparse_pwr,
        #          plotter=plotter,
        #          mat_name=mat,
        #          seed=2 + i * 3,
        #          num_avg=num_avg,
        #          num_iter= num_iter, 
        #          kwargs= {"x": x, "tol": tol},
        #          )
        plotter.finish()

if __name__ == '__main__':
    main_no_swap()
    # main_swap()
    # main_sparsification()