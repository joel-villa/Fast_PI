# FAST_PI

An approach to speeding up Power Iteration via row sampling based on row-
magnitudes. For the theoretical results see the docs/ directory.

## Generating Plots

```(linux)
python -m src.algorithm.tests
```

## To load Sparsification_Research Repository Run

```(linux)
git submodule update --init --recursive
```

## To update Sparsification_Research Directory

```(linux)
cd Sparsification_Research
git pull origin main
```

## A Note on ssgetpy

On UNIX based machines, the ssgetpy library will download matrices onto your 
machine, at the root in the .ssgetpy directory

## TODO

### Empirical

- [ ] Do norms impact, performance, why are some not seeing gains?
- [ ] Run on CARC:
  - [ ] Collection test sweep averaging (Ask Saia about this: is it possible if 
    we have variable number of iteratiosn?)
  - [ ] Run on large matrices!
- [ ] Longer runs -> Stronger results
- [ ] N vs. error plot
  - [ ] Plus line for epsilon bounds
- [ ] Clean up README runnables
- [ ] Convergence vs. Work graph + line for bounds on $(1 \pm \epsilon)||A||$
- [ ] Make code strongly typed and redo doc comments
- [ ] Move two_norm.npz handling into proven/util/npz_wrapper.py

### Theory

- [ ] Expected amount of reduced work per iteration (expected new nnzs)
- [ ] Binomial expected work: (1 - ||a_j||)^N
- [ ] Take advantage of eigen-gap? Other routes forward?
- [ ] FAST-PI requires $O(\dots)$ scalar mults, vs. $O(\dots)$ of the standard 
  power-iteration...
