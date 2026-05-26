"""
For plotting tests which take into account work required to get a 
top-eigenvector approximation, with some score

Where work is number of scalar mults and "score" is closeness of singular value
"""

from scipy.linalg import norm # 2-norm by default
from scipy.sparse.linalg import svds

import numpy as np

from Sparsification_Research.src.Plotter import Plotter
from Sparsification_Research.src.SSGetter import SSGetter

from src.tests import score as tst
from src.tests.power_lazy import lazy_percent

def test(funct, plotter, num_avg, A, num_iter, kwargs={}):
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

    trim_size_prev = None
    
    for i in range(num_avg):
        #TODO: averaging doesn't work :(
        if "seed" in kwargs:
            # If seed in kwargs, update it for variable tests to average
            kwargs["seed"] = kwargs["seed"] * (i + 1) + i * 3

        xs, ys_i, label = funct(A=A, max_iter=num_iter, **kwargs)
        ys += ys_i

        trim_size = np.trim_zeros(xs).shape[0]

        if (trim_size_prev is not None) and (trim_size_prev != trim_size):
            print("WARNING: getting variable length convergence")
        
        trim_size_prev = trim_size
    
    ys = ys / num_avg

    # print(f"{label}: xs[0] = {xs[0]}, ys[0] = {ys[0]}")

    # Remove trailing zeros only ('b' for back)
    xs = np.trim_zeros(xs, trim='b')
    ys = np.trim_zeros(ys, trim='b')

    #UNCOMMENT FOLLOWING LINE FOR ITTERATIVE VIEW
    # xs = np.arange(xs.shape[0])

    """For plotting initial values (since using log-scale-x, need to add 
    perturbation to see x=0 values)"""
    xs = xs + 1e-2

    plotter.add_to_plot(xs, ys, label=label)

def init(mat_name, seed, tol):
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
        
    _, s_star, _ =  svds(A, k=1)

    s_star = s_star[0]

    rng = np.random.default_rng(seed=seed)
    u0 = rng.normal(0, 1, A.shape[0])
    u0 = u0 / norm(u0)

    print(f"Testing {mat_name}")

    kwargs = {"s_star": s_star,
              "u_0": u0,
              "tol": tol,
              }

    return A, kwargs

if __name__ == '__main__':
    seed = 10
    max_iter = 64
    num_avg = 5
    jl_ps = (80, 97)
    ps = (25, 50, 97)
    types = ("simple", "gaussian", "sparse")
    tol = 1e-7
    epsilon = 0.98
    
    # SOME MATS THAT SHOW GOOD BEHVIOR: ["bcsstk07", "bcsstk19", "bcsstm07", "impcol_d"]
    mats = ["494_bus", "bcsstk07", "bcsstk08", "bcsstk19", "bcsstm07", "impcol_d"]

    plotter = Plotter(save_fig=False, show_fig=True, fig_size=(12, 6))

    for mat_name in mats:
        A, funct_args = init(mat_name=mat_name, seed=seed, tol=tol)

        plotter.init_plot(title=f"Power Convergence of {mat_name}", 
                          x_label="work (approximate number of scalar multiplications)",
                          y_label=r"Error Rate $\left(\frac{|\lambda^* - \lambda|}{|\lambda^*|}\right)$", 
                          save_name=f"{mat_name}_error_v_work",
                          grid_on=True) 
        
        kwargs = {"plotter": plotter,
                  "num_iter": max_iter,
                  "A": A,
                  }
        
        # Baseline test
        test(funct=tst.baseline,
             num_avg=1,
             kwargs=funct_args,
             **kwargs,
             )
        
        kwargs = kwargs | {"num_avg" : 1} #TODO: make num_avg != 1 work
        funct_args = funct_args | {"seed": seed}

        # JL-reduction tests
        jl_args = funct_args | {"eps": epsilon}
        # for p in jl_ps:
        #     for type in types:
        #         p_args = jl_args | {"type": type, "p": p}
        #         test(funct=tst.jl_percent,
        #              kwargs=p_args,
        #              **kwargs)
                
        # JL lazy
        for p in ps:
            p_args = jl_args | {"p": p}
            test(funct=lazy_percent,
                kwargs=p_args,
                **kwargs)
                
        # Row Sampling
        for p in ps: 
            p_args = funct_args | {"p": p}
            test(funct=tst.row_sample_p,
                 kwargs=p_args,
                 **kwargs)
            
        # Sparsification
        for p in (2, 16):
            for type in ("MD", "Generic"):
                p_args = funct_args | {"p": p, "type": type}
                test(
                    funct=tst.percent_sparse,
                    kwargs=p_args,
                    **kwargs,
                )


        plotter.finish(
            # xscale="log"
            )