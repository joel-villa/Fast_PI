#!/bin/bash
#SBATCH --job-name=Work_vs_Accuracy
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=test.out
#SBATCH --error=test.err

module load python

python -m venv .venv
source .venv/bin/activate
pip install scipy
pip install scikit-learn
pip install matplotlib
pip install ssgetpy

python -m src.algorithm.main

