"""Some helper functions for sparse arrays"""
import scipy
import numpy as np

#TODO: delete this file
def check_equal_dense(
        A:scipy.sparse.sparray,
        B:scipy.sparse.sparray,
) -> bool:
    """Grossly checking equality by converting to dense

    Args:
        A (scipy.sparse.sparray): _description_
        B (scipy.sparse.sparray): _description_

    Returns:
        bool: _description_
    """
    return np.array_equal(A.todense(), B.todense())

def check_equal(
        A:scipy.sparse.sparray,
        B:scipy.sparse.sparray,
) -> bool:
    """Check if two scipy sparse matrices are equivalent

    Args:
        A (scipy.sparse.sparray): One of the sparse matrices
        B (scipy.sparse.sparray): Another sparse matrix

    Returns:
        bool: True if A == B, False otw
    """
    A.sort_indices()
    B.sort_indices()

    fst_equality = bool(np.all(A.indices == B.indices))
    snd_equality = bool(np.all(A.indptr == B.indptr))
    thd_equality = bool(np.allclose(A.data, B.data))

    return fst_equality and snd_equality and thd_equality

if __name__ == '__main__':
    """Testing sparse equality implementation
    """
    A = scipy.sparse.csr_array(([1, 2], [0, 1], [0, 0, 2]))
    B = scipy.sparse.csr_array(([2, 1], [1, 0], [0, 0, 2]))
    print(check_equal(A, B))
    print(check_equal_dense(A, B))