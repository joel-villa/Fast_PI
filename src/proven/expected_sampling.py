"""A file for generating the expected sampling of a matrix A, given the sampling
scheme which was proven to be theoretically perfomant"""

def expected_sampling(
    matrix: str,
) -> float:
    """ Get the expected sampling of the matrix, under the assumption that the 
    sampling scheme is to pick the i'th row with probability pi, and to scale 
    the i'th row by 1/sqrt(pi).

    Args:
        matrix (str): The name of a Suite-Sparse matrix

    Returns:
        float: expected rows kept of that matrix
    """
    