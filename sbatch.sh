#!/bin/bash
#SBATCH --job-name=Work_vs_Accuracy
#SBATCH --account 2016579
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=test.out
#SBATCH --error=test.err

module load python

python -m venv .venv
pip install scipy
pip install scikit-learn
pip install ssgetpy
pip isntall PyQt6

python -m src.algorithm.tests

