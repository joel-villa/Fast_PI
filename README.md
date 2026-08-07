# FAST_PI

An approach to speeding up Power Iteration via row sampling based on row-magnitudes

## To load Sparsification_Research Repository Run
```(linux)
git submodule update --init --recursiv
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

- Clean up README runnables
- Write code for calculating $f$ in $\Pr( \left|||\tilde A||^2 - ||A||^2\right| \ge ||A|| \epsilon^2) \le 2 \exp(\ln (n) - \epsilon^2 f)$
- Convergence vs. Work graph + line for bounds on $(1 \pm \epsilon)||A||$
- Expected amount of rows sampled vs. actual? 

### Theory

- Prove something about the bounds of $\left|||\tilde A||^2 - ||A||^2\right|$
- Main theorem of the form: Let $\tilde x$ be the vector returned by FAST Power-iteration (PI on teh projected matrix. Then... $||Ax|| \ge (1 - \dots) \sigma_1(A)$ with probability of error $\le $ $\dots$. Want dots to be small
- FAST-PI requires $O(\dots)$ scalar mults, vs. $O(\dots)$ of the standard power-iteration...
- Bounds on Expected number of rows sampled?
