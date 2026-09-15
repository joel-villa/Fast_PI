# FAST_PI

An approach to speeding up Power Iteration via row sampling based on row-magnitudes. For the theoretical results see the docs/ directory.

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

- Score vs. Work Graph
    - Total &/or Max Work
- Weighted Averaging of eigenvectors
- compute $\frac{1}{N} \sum_{i=1}^N \tilde A_i^T \tilde A_i^T A$ prior to power 
  iteration (this is what I definitely have thoeretical results for)
- New parrallelizable idea: $\tilde A^T \tilde A = \sum_{j=1}^n \frac{a_j^Ta_j}{N||a_j||} \cdot Binom(N, ||a_j||)$
- N vs. error plot
- Clean up README runnables
- Convergence vs. Work graph + line for bounds on $(1 \pm \epsilon)||A||$
- Make code strongly typed and redo doc comments
- Move two_norm.npz handling into proven/util/npz_wrapper.py

### Theory

- ~A^T~A = ? => ~A = ?
- min delta for theoretical results to exist
- Relate $w_i$'s to $v$ in some cool manner
  - Saia's advice: weighted average
- Take advantage of eigen-gap? Other routes forward?
- Change $w$ to $v$, more sensible
- FAST-PI requires $O(\dots)$ scalar mults, vs. $O(\dots)$ of the standard power-iteration...
- Bounds on Expected number of rows sampled?
