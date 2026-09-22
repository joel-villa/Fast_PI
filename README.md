# FAST_PI

An approach to speeding up Power Iteration via row sampling based on row-
magnitudes. For the theoretical results see the docs/ directory.

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

On UNIX based machines, the ssgetpy library will download matrices onto your 
machine, at the root in the .ssgetpy directory

## TODO

### Empirical

- Collection test sweep averaging
- N vs. error plot
- Clean up README runnables
- Convergence vs. Work graph + line for bounds on $(1 \pm \epsilon)||A||$
- Make code strongly typed and redo doc comments
- Move two_norm.npz handling into proven/util/npz_wrapper.py

### Theory

- Binomial expected work: (1 - ||a_j||)^N
- Stochastic equivalence of binomial approach to 'naive' approach
- min delta for theoretical results to exist
- Take advantage of eigen-gap? Other routes forward?
- Change $w$ to $v$, more sensible
- FAST-PI requires $O(\dots)$ scalar mults, vs. $O(\dots)$ of the standard 
  power-iteration...
