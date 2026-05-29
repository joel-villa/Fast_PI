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

- Nystrom APPROACH + PERFORMANCE: https://arxiv.org/pdf/2604.20077
  - 7 & 8 are of importance

- Volume Sampling Approach:
  - https://d1wqtxts1xzle7.cloudfront.net/32456103/v002a012-libre.pdf?1391557305=&response-content-disposition=inline%3B+filename%3DMatrix_approximation_and_projective_clus.pdf&Expires=1780071764&Signature=IoUR4r2G~3-89YYE15LLvbQaZ-5QULDJ5ZbzJKWwkQcvSZYh9sLaTdNKOI5v7FsCo8DF8tjivO-bx~4hGEriLK3anXKCXk1MfWSU3lmYkQckcWbG29x4gNMp6ZR-RF3CKzui~Y2czH4AHh6FyaAJZma3ixeUNSix-snRe2nCxtbWTGlIhBM86Bisp81IJ2qfHDDoRSt2Uq68ygFRglYaRIkyvfNjj5hsvGjkyGIx15Jn2JE15kJKYZbgnQEnFAuwHZw8yL4OsW1hXZSxlsXp7-gihCz7mf~7cjYdcio3AgIC0JYrXSTaEzT14kwRfQ8I0G6YeYV4ZX69vc6vZi51bQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA#page=2&zoom=100,106,750
  - Thm 1.1, and 2.2 Algorithm are of importance


- A proof that JL reductions approximately preserve eigenvectors (not in expectation).

- Sparsify JL, reduction, cause why not :D

- Empirically find better swap tolerance? 

- Change lbl for sparsification to expected # of new zeros?

- Parallelize things: GET EASLEY ACCESS

- Some Datasets:
  - https://arxiv.org/pdf/1303.4207
    - Enron: https://www.cs.cmu.edu/~enron/
    - Dexter
    - Farm Ads
    - Gisette

- Proof on the lowerbound of dimensionality reduction in order to preserve top eigenvectors of a matrix: $d = \frac{C \log m}{\epsilon}$?

- Timing convergence of original Power vs. a JL-enhaced Power.

- A JL-enhanced Power algorithm which takes averages of guesses for the top eigenvector. 

## Things we Could Feasibly Hope to Prove
- How to choose apt hyperparameters for the Upscaling approach? Other than guess and check? 
- JL preservation of top eigenvectors (given some spectral gap maybe?)
- Expected number of JL reductions before getting one that preserves top eigenvectors 
