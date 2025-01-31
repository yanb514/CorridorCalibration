#!/bin/bash
#SBATCH --job-name=sumo_calib
#SBATCH -N 1
#SBATCH -c 16
#SBATCH -t 3-00:00:00 
#SBATCH --ntasks-per-node=2         # Match 16-core NUMA nodes
#SBATCH -p general
#SBATCH -q public
#SBATCH -o _log/%x_%j.out                 # Simplified output naming
#SBATCH -e _log/%x_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=%u@asu.edu
#SBATCH --export=NONE


# Load modules with AMD optimization
module purge
module load mamba/latest
module load sumo-1.19.0-gcc-12.1.0

# Activate environment with NUMA awareness
# source activate sumo-env
source activate mypytorch-1.8.2

# Set SUMO_HOME environment variable
# export SUMO_HOME=/packages/apps/spack/21/opt/spack/linux-rocky8-zen3/gcc-12.1.0/sumo-1.19.0-hkbmitts4svgguaerh3osctddszoeu4m

# Run with explicit core binding
echo "Starting job at $(date)"
python onramp_cGAN.py 
echo "Job completed at $(date)"