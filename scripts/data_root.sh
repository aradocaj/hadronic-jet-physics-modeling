#!/bin/bash

#SBATCH --job-name=steve

#SBATCH --output=output.out

#SBATCH --error=err.err

#SBATCH --time=00:03:00

#SBATCH --mem=50G

#SBATCH --partition=short

#SBATCH --cpus-per-task=10

python data_root.py
