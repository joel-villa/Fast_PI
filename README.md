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

- A proof that JL reductions approximately preserve eigenvectors (not in expectation).

- Sparsify JL, reduction, cause why not :D

- Do parameter sweep for Chan approach on 494_bus, see if it's results generalize better

- Change lbl for sparsification to expected # of new zeros?

- Parallelize things: GET EASLEY ACCESS

- Proof on the lowerbound of dimensionality reduction in order to preserve top eigenvectors of a matrix: $d = \frac{C \log m}{\epsilon}$?

- Timing convergence of original SVD vs. a JL-enhaced SVD.

- A JL-enhanced SVD algorithm which takes averages of guesses for the top eigenvector. 

- Some interesting behavior: convergence independent of dec_funct, when 
  decrementing

## Things we Could Feasibly Hope to Prove
- JL preservation of top eigenvectors (given some spectral gap maybe?)
- Expected number of JL reductions before getting one that preserves top eigenvectors 
