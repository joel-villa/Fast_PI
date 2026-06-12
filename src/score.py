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
from src.tests import score_sample as smpl
from src.tests import power_lazy as lzy

def test(funct, plotter, num_avg, A, num_iter, kwargs={}, show_more=False):
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
    ys_i = np.full((num_avg, num_iter), np.nan) # for averaging

    label = ""

    max_iter = 0
    min_iter = num_iter
    
    for i in range(num_avg):
        if "seed" in kwargs:
            # If seed in kwargs, update it for variable tests to average
            kwargs["seed"] = kwargs["seed"] * (i + 1) + i * 3

        xs_temp, ys_i[i], label = funct(A=A, max_iter=num_iter, **kwargs)

        curr_iter = np.trim_zeros(xs_temp).shape[0]

        if (not show_more) and (min_iter > curr_iter):
            # Smaller x value, store this one for consistent averaging
            xs = xs_temp
            min_iter = curr_iter
        elif show_more and (max_iter < curr_iter):
            # Larger x value, store this one for consistent averaging
            xs = xs_temp
            max_iter = curr_iter

    # Calculate average
    ys = np.nanmean(ys_i, axis=0)

    # Remove trailing zeros only ('b' for back)
    if not show_more:
        xs = xs[:min_iter]
        ys = ys[:min_iter]
    else:
        xs = xs[:max_iter]
        ys = ys[:max_iter]

    #UNCOMMENT FOLLOWING LINE FOR ITTERATIVE VIEW
    # xs = np.arange(xs.shape[0])

    """For plotting initial values (since using log-scale-x, need to add 
    perturbation to see x=0 values)"""
    # xs = xs + 1e-2

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

def jl_reduction(funct_args, kwargs, ps):
    """Test the naive JL enhancement to power iteration
    
    Args:
        funct_args: a dictionary of arguments for the function jl_percent()
                    (except for type, and p)
        kwargs: a dictionary of arguments for the test function
        ps: the percent reduction ammounts
    
    Return: None
    """
    types = ("simple", "gaussian", "sparse")

    for p in ps:
        for type in types:
            p_args = funct_args | {"type": type, "p": p}
            test(funct=tst.jl_percent,
                 kwargs=p_args,
                 **kwargs)

def jl_lazy(funct_args, ps, kwargs):
    """Test the JL enhancement to power iteration, with lazy multiplication
    
    Args:
        funct_args: dictionary of arguments for the function lazy_percent() 
                    (except for p)
        ps: the percent reduction ammounts
        kwargs: a dictionary of arguments for the test function
        
    Return: None
    """
    for p in ps:
        p_args = funct_args | {"p": p}
        test(funct=lzy.lazy_percent,
            kwargs=p_args,
            **kwargs)
        
def col_sample(ps, funct_args, kwargs, types):
    """Test column sampling enhancement to power iteration

    Args: 
        ps: the percent reduction ammounts
        funct_args: dictionary of arguments for the function col_p_sample() 
                    (except for p and type)
        kwargs: a dictionary of arguments for the test function
        types: a list of string representations of the types of column sampling

    Return: None
    """
    # gammas = (
    #     0.8,
    #     0.85,
    #     0.9,
    #     0.95,
    # ) # for nystrom sampling
    for p in ps: 
        for type in types:
            # if type == "nystrom":
            #     for gamma in gammas:
            #         p_args = funct_args | {"p": p, "type": type, "gamma": gamma}
            #         test(funct=smpl.col_sample_p,
            #              kwargs=p_args,
            #              **kwargs)
            # else:
            p_args = funct_args | {"p": p, "type": type}
            test(funct=smpl.col_sample_p,
                 kwargs=p_args,
                 **kwargs)
            
def sparsification(funct_args, kwargs):
    """Test sparsification enhancement to power iteration
    
    Args: 
        funct_args: a dictionary of arguments for the function sparsification() 
                    (except for p and type)
        kwargs: a dictionary of arguments for the test function
    
    Return: None
    """
    ps = (
        # 0.1, 
        4, 
        16,
    )
    types = (
        "Generic",
        "SD",
        "CD",
        "MB", #TODO test this
    )

    for p in ps:
        for type in types:
            p_args = funct_args | {"p": p, "type": type}
            test(
                funct=tst.percent_sparse,
                kwargs=p_args,
                **kwargs,
            )

def col_sample_dec(funct_args, kwargs, types):
    """Test column sampling enhancement to power iteration

    Args: 
        funct_args: dictionary of arguments for the function col_p_sample() 
                    (except for p and type)
        kwargs: a dictionary of arguments for the test function

    Return: None
    """
    min_iter = 4

    # init_ps = (99, 98, 96, 92, 84, 68, 36) 
    # init_ps = (98, 97,96, 95, 94) 
    # init_ps = (97, ) # TODO: 97 found imeprically (how to generalize?)
    # init_ps = (96, ) # TODO: 96 found imeprically (how to generalize?) 494_bus

    # dec_functs = (lambda p: p - 1, lambda p: p - 2, lambda p: p - 4,lambda p: p - 8, lambda p: p - 16, lambda p: p - 32)
    # dec_functs = (lambda p: p - 4,lambda p: p // 2, lambda p: p // 1.5, lambda p: p // 1.25, lambda p: p // 1.125, lambda p: p // 1.0625, lambda p: p // 1.03125, lambda p: p // 1.015625,) 
    # dec_functs = (lambda p: p // 1.03125, lambda p: p - 2, lambda p: math.log2(p), lambda p: math.log(p), lambda p: math.log10(p)) 
    # dec_functs = (lambda p: p // 1.25, lambda p: p - 2) # TODO: found imeprically (how to generalize?)
    # dec_functs = (lambda p: p // 1.25,) # TODO: found imeprically (how to generalize?)
    # dec_functs = (lambda p: p - 4, lambda p: p // 1.03125) # TODO: found imeprically (how to generalize?) 494_bus
    # dec_functs = (lambda p: p // 1.03125,) # TODO: found imeprically (how to generalize?) 494_bus
   
    # swap_tols = (0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5)
    # swap_tols = (0.0008, 0.0009, 0.0010, 0.0011, 0.0012,)
    # swap_tols = (0.001,) #TODO: found imperically
    # swap_tols = (0.005,) #TODO: found imperically 494_bus    

    #BEST PERFORMING PARAMETERS FOUND EMPIRICALLY:
    init_ps = (96, )
    init_ps = (99, ) #BETTER FOR SCALING ALG? 
    dec_functs = (lambda p: p // 1.25,)
    swap_tols = (0.005,)

    funct_args = funct_args | {
        "min_iter": min_iter, 
    }

    for init_p in init_ps:
        for dec_funct in dec_functs:
            for swap_tol in swap_tols:
                for type in types:
                    test_args = funct_args | {
                        "p0": init_p, 
                        "dec_funct": dec_funct,
                        "swap_tol": swap_tol,
                        "type": type,
                    }
                    test(funct=smpl.col_sample_dec_p,
                         kwargs=test_args,
                         **kwargs)
                
def sample_lazy_combo(ps, types, funct_args, kwargs):
    for p in ps: 
        for type in types:
            p_args = funct_args | {"p": p, "type": type}
            test(funct=lzy.sample_lazy,
                 kwargs=p_args,
                 **kwargs)

def sparse_jl_combo(funct_args, kwargs): #TODO: make this better? is there a way to do this lazily? Prob not 
    funct_args = funct_args | {
        "p_sparse": 3,
        "type_sparse": "Generic",
        "p_jl": 97,
        "type_jl": "Simple",
    }
    test(
                funct=tst.sparse_jl_combo,
                kwargs=funct_args,
                **kwargs,
            )

if __name__ == '__main__':
    seed = 10
    max_iter = 128
    num_avg = 5
    ps = (25, 50, 75, 99)
    jl_ps = (97, None)
    tol = 1e-5
    epsilon = 0.5
    # epsilon = 0.24

    sample_types = (
        # "random", 
        # "1-norm", 
        # "2-norm", 
        # "nystrom",
        "scale",
        "row",
    ) 
    
    mats = [
        "494_bus", # Invalid for mag-based sparsifification
        "bcsstk07", 
        "bcsstk08", 
        "bcsstk19", 
        "bcsstm07", 
        "bcspwr06",
        # "impcol_d", # NOT POSITIVE DEFINITE, doesn't work with nystrom sampling
        "bibd_13_6", # RECTANGULAR: Doesn't work with Nystrom
    ]

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
        
        # BASELINE TEST
        test(funct=tst.baseline,
             num_avg=1,
             kwargs=funct_args,
             **kwargs,
             )
        
        # Arguments for randomized tests
        kwargs = kwargs | {"num_avg" : num_avg} 
        funct_args = funct_args | {"seed": seed}

        # Args for JL reduction tests
        jl_args = funct_args | {"eps": epsilon}

        # JL REDUCTION TEST
        # jl_reduction(funct_args=jl_args, kwargs=kwargs, ps=jl_ps)

        # JL LAZY TEST
        # jl_lazy(funct_args=jl_args, ps=jl_ps, kwargs=kwargs)

        # ARGS for column sampling tests
        sample_args = funct_args | {"gamma": 0.9} # for nystrom sampling

        # COLUMN SAMPLING TEST
        col_sample(
            ps=ps, 
            funct_args=sample_args, 
            kwargs=kwargs, 
            types=sample_types
        )

        # SPARSIFICATION TEST
        # sparsification(funct_args=funct_args, kwargs=kwargs)
        
        # COLUMN SAMPLING WITH DECREASING P TEST
        col_sample_dec(
            funct_args=sample_args, 
            kwargs=kwargs,
            types=sample_types,
        )

        # COMBINATION TESTS
        # sample_lazy_combo(ps=(30, 50, 70), 
        #                   types=sample_types, 
        #                   funct_args=jl_args, 
        #                   kwargs=kwargs,
        #                   )

        # sparse_jl_combo(funct_args=jl_args, kwargs=kwargs)

        plotter.finish(
            # xscale="log"
            )