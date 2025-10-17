#!/bin/bash

#SBATCH --job-name=job_try

#SBATCH --output=try.out

#SBATCH --error=try.err

#SBATCH --time=00:00:30

#SBATCH --mem=100M

#SBATCH --partition=short

#SBATCH --cpus-per-task=4

python try1.py
