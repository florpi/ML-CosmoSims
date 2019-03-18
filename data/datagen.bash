#!/bin/bash -l
# Create dataset
#
#SBATCH -n 1
#SBATCH -t 04:00:00
#SBATCH -J DMDG 
#SBATCH -o DMDG.out
#SBATCH -e DMDG.err
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH --exclusive

# Load Module
module purge
module load gnu_comp/7.3.0 openmpi python/3.6.5

simtype=dark_matter_only   #[dark_matter_only, full_physics]
simdir=/cosma6/data/dp004/dc-arno1/SZ_project/${simtype}/L62_N512_GR/
outdir=/cosma5/data/dp004/dc-beck3/Dark2Light/data/${simtype}
nsnap=0     #[0...45]
nvoxel=1024 # number of voxels per box edge

# Execute script
python3 ./data_generate.py $simdir $outdir $nsnap $nvoxel

