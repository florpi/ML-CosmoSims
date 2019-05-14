#!/bin/bash -l
# Create dataset
#
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -J DMDG 
#SBATCH -o DMDG.out
#SBATCH -e DMDG.err
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH --exclusive

# Load Module
module purge
module load gnu_comp/7.3.0 openmpi python/3.6.5

#simtype=dark_matter_only   #[dark_matter_only, full_physics]
simtype=full_physics #[dark_matter_only, full_physics]
#simdir=/cosma6/data/dp004/dc-arno1/SZ_project/${simtype}/L62_N512_GR/
simdir=/cosma7/data/TNG/TNG300-1/
outdir=/cosma7/data/dp004/dc-beck3/Dark2Light/data/${simtype}/
nsnap=(99)
#(45 44 43 42 41 40 39 38 37 36 35 34 33 32 31 30 29 28 27 26 25 24 23 22)
nvoxel=256 # number of voxels per box edge
Lbox=205  #[Mpc/h]

# Execute script
python3 ./data_generate.py $simdir $outdir $simtype "$(echo ${nsnap[@]})" $nvoxel $Lbox

