#!/bin/bash

#SBATCH --job-name=job_ms

#SBATCH --output=ms.out

#SBATCH --error=ms.err

#SBATCH --time=00:10:00

#SBATCH --mem=2G

#SBATCH --partition=short

#SBATCH --cpus-per-task=1

python ms.py
