"""
For sparsifying a Dense matrix via Thm 8.2.2 in 
https://users.cs.utah.edu/~jeffp/teaching/cs7931-S15/cs7931/8-sparsification.pdf
"""

from Sparsification_Reasearch.src.Sparsifier import Sparsifier
from scipy.sparse import coo_array

class MDSparsifier(Sparsifier):
    """
    Mantain Diagonal Sparsifier, i.e. sparsify as usual, except garuntee 
    diagonal is kept and scaled
    """
    def __init__(self, seed=42):
        super().__init__(seed)

    
    def sparsify(self, A, s):
        """
        Based on Theorem 8.2.2 in Sparsification Algorithms Paper:
        https://users.cs.utah.edu/~jeffp/teaching/cs7931-S15/cs7931/8-sparsification.pdf

        Sparsify a matrix A given some value s
        A - the matrix to sparsify (assumed to be 2d numpy array)
        s - factor of sparsification (Increasing s => make more sparse)

        Psuedocode: 
        (1) sparsify entries of A' with probability 1 - 1/s 
        (2) if not sparsified, scale up by factor of s
        
        TODO: tests w/ this may be useful
        """
        A = A.copy()
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                A[i][j] = self.sparse_entry(x=A[i][j], s=s)

        return coo_array(A)