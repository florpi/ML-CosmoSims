#!/bin/bash -l
# This script runs the python files to create one hdf5 file of the SubFind
# in an ordere according to the DHhalo merger tree outputs with added
# nodeIndex key to identify the halos & subhalos

# SBATCH -L /bin/bash
#SBATCH -t 1:00:00
#SBATCH -J F63_DensityMap 
#SBATCH -o F63_DensityMap.err
#SBATCH -e F63_DensityMap.out
#SBATCH -p cosma6
#SBATCH -A dp004
#SBATCH --exclusive

# Load Module
module purge
module load python/3.6.0 gnu_comp/c4/4.7.2 platform_mpi


# Execute script
python3 ./main.py \
