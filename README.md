# FAST_PI

An approach to speeding up Power Iteration via row sampling based on row-magnitudes

## To load Sparsification_Research Repository Run
```(linux)
git submodule update --init --recursive
```

## To update Sparsification_Research Directory
```(linux)
cd Sparsification_Research
git pull origin main
```

## To run main:
python -m src.main

## A Note on ssgetpy

The ssgetpy library will download matrices onto your machine, at the root in the .ssgetpy directory

## TODO

### Empirical

- compute new metadata (eig-info) for all matrices in question
- CODE: delta given epsilon, epsilon given delta stuffs
- Clean up README runnables
- Convergence vs. Work graph + line for bounds on $(1 \pm \epsilon)||A||$
- Make code strongly typed and redo doc comments
- Move two_norm.npz handling into proven/util/npz_wrapper.py

### Theory

- Take advantage of eigen-gap? Other routes forward? 
- Main theorem of the form: Let $\tilde x$ be the vector returned by FAST Power-iteration (PI on teh projected matrix. Then... $||Ax|| \ge (1 - \dots) \sigma_1(A)$ with probability of error $\le $ $\dots$. Want dots to be small
- FAST-PI requires $O(\dots)$ scalar mults, vs. $O(\dots)$ of the standard power-iteration...
- Bounds on Expected number of rows sampled?
