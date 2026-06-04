"""
Helper functions for sparsification tests
"""

from Sparsification_Research.src.CDSparsifier import CDSparsifier
from Sparsification_Research.src.SDSparsifier import SDSparsifier
from Sparsification_Research.src.Sparsifier import Sparsifier
from ..util.Sparsifier import Sparsifier as MagSparsifier
from Sparsification_Research.src.SGenerator import SGenerator

def p_sparse(A, p, type, seed):
    """ Sparsifies the matrix A by p-percent

    Args: 
        A: A scipy.sparse matrix which will be sparsified in-place
        p: expected percent of new zeroes in the sparsified A (not counting
           diagonal entries)
        type: type of sparsification
              'mb' -> magnitude based sparsification (variable per entry)
              'sd' -> scale diagonal sparsification
              'cd' -> constant diagonal sparsification
              'generic' -> default behavior
        seed: for duplicatable randomness

    Returns: A sparsified version of A
    """
    sparse_A = A.copy()
    
    if type.lower() == "mb":
        # P-Based Sparsifier
        sparsifier = MagSparsifier(seed=seed)
        sparsifier.sparsify(sparse_A, p=p)
        
    else:
        # S-Based Sparsifiers
        s_generator = SGenerator(A.shape[0], A.nnz)
        include_diags = True
        match type.lower():
            case "sd":
                sparsifier = SDSparsifier(seed=seed)

                # s based off of off-diagonal non-zeroes
                include_diags = False

            case "cd":
                sparsifier = CDSparsifier(seed=seed)
                # s based off of off-diagonal non-zeroes
                include_diags = False

            case "generic":
                sparsifier = Sparsifier(seed=seed)

                # s based off of nnzs
                include_diags = True
            case _:
                raise TypeError(f"Invalid sparisifier type {type}")
        # get s associated with an expected percent sparsification p
        expected_proportion = p / 100
        s = s_generator.proportion_sparse_s(p=expected_proportion, 
                                            include_diags=include_diags)

        # Sparsify A with s
        sparsifier.sparsify(sparse_A, s)
    
    return sparse_A