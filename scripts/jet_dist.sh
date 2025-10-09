#!/bin/bash

#SBATCH --job-name=steve_job

#SBATCH --output=output.out

#SBATCH --error=err.err

#SBATCH --time=12:00:00

#SBATCH --mem=50G

#SBATCH --partition=day

#SBATCH --cpus-per-task=10

python jet_dist.py highpt
