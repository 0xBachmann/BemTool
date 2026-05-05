#!/usr/bin/env bash
mkdir -p n1
mkdir -p n1.5

# Sweep 1:
# k = 1,2,3
# Omega = -0.02 -0.04 -0.06 -0.08 0.02 0.04 0.06 0.08
# n = 1, 1.5

for n in 1 1.5; do
  for k0 in 1 2 3; do
    for Omega in -0.02 -0.04 -0.06 -0.08 0.02 0.04 0.06 0.08; do

      Omega_tag=$(echo "${Omega}" | sed 's/-/m/g; s/\./p/g')

      sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=dielectric_k${k0}_Om${Omega_tag}_n${n}
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mem-per-cpu=5G

./dielectric_cylinder ${k0} ${Omega} ${n}
EOF

    done
  done
done


# Sweep 2:
# k = 1,2,3,4,5
# Omega = -1e-7 -1e-6 -1e-5 -1e-4 -1e-3 -1e-2 -1e-1 0 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7
# n = 1, 1.5

for n in 1 1.5; do
  for k0 in 1 2 3 4 5; do
    for Omega in -1e-7 -1e-6 -1e-5 -1e-4 -1e-3 -1e-2 -1e-1 0 1e-1 1e-2 1e-3 1e-4 1e-5 1e-6 1e-7; do

      Omega_tag=$(echo "${Omega}" | sed 's/-/m/g; s/\./p/g; s/+//g')

      sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=dielectric_k${k0}_Om${Omega_tag}_n${n}
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --mem-per-cpu=5G

./dielectric_cylinder ${k0} ${Omega} ${n}
EOF

    done
  done
done
