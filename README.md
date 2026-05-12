# JL_Analysis
We dimensionally reduce

## To load Sparsification_Research Repository Run
`git submodule update --init --recursive`

## To update Sparsification_Research Directory
`cd Sparsification_Research`
`git pull origin main`

## To run main:
python -m src.main

## A Note on ssgetpy

The ssgetpy library will download matrices onto your machine, at the root in the .ssgetpy directory

## TODO

- A proof that JL reductions approximately preserve eigenvectors.

- Does the sparse JL reduction result in dense matrices? What is the density of a JL reduction via the sparse matrix? It is sparse matrix sparse matrix multiplication, so this may not be the case, and a less drastic dimensionality reduction may be valid due to this.

- Implementation of a JL reduction which takes into account the density of the original matrix (with the goal of preserving number of non-zeroes). As mentioned in Section 4.1, distributed arithmetic with dense matrices is more efficient than with sparse matrices with the same number of non-zeroes due to latency due to non-patterned messaging. 

- A JL-enhanced SVD algorithm which uses a guess and check approach, re-reducing the matrix until one which preserves the top eigenvector is found (this can be verified by running some iterations with the original matrix), and continuing with the standard SVD after that JL-reduction has converged on a sufficiently accurate "initial guess" for the top eigenvector. 

- Counting number of scalar multiplications in original SVD vs. a JL-enhanced SVD.

- Proof on the lowerbound of dimensionality reduction in order to preserve top eigenvectors of a matrix: $d = \frac{C \log m}{\epsilon}$?

- Timing convergence of original SVD vs. a JL-enhaced SVD.

- A JL-enhanced SVD algorithm which takes averages of guesses for the top eigenvector. 

- Does probability of preserving top eigenvectors depend on spectral gap? To what extent? 

- How does spectral gap impact the ability of a JL reduction to preserve tpo eigenvectors. 

- How does JL preserve top k eigenvectors? How does JL preserve top k eigenvectors when difference in singular values is relatively small?
  

## Things we Could Feasibly Hope to Prove
- JL preservation of top eigenvectors (given some spectral gap maybe?)
- Expected number of JL reductions before getting one that preserves top eigenvectors
- JL swap algorithm empirically seems to be working best on those matrices with the 
  largest singular values: does this imply scaling up the matrices may be of worth? 
