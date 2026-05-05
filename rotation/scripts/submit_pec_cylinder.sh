#!/bin/bash
#SBATCH --job-name=dielectric_k${k0}_Om${Omega_tag}_n${n}
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mem-per-cpu=5G

./pec_cylinder