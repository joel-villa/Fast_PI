#!/bin/bash
#SBATCH --job-name=Work_vs_Accuracy
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=test.out
#SBATCH --error=test.err

module load python

python -m venv .venv
source .venv/bin/activate
pip install scipy
pip install scikit-learn
pip install matplotlib
pip install ssgetpy
pip install PyQt6

python -m src.algorithm.main

