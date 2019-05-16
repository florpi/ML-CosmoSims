#!/bin/bash -l
# Create dataset
#
#SBATCH -n 1
#SBATCH -t 01:30:00
#SBATCH -J TNG3001 
#SBATCH -o TNG3001.out
#SBATCH -e TNG3001.err
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH --exclusive

# Load Module
module purge
module load gnu_comp/7.3.0 openmpi python/3.6.5

#simtype=dark_matter_only   #[dark_matter_only, full_physics]
simtype=dark_matter_only #[dark_matter_only, full_physics]
#simdir=/cosma6/data/dp004/dc-arno1/SZ_project/${simtype}/L62_N512_GR/
simdir=/cosma7/data/TNG/TNG300-1-Dark/
outdir=/cosma7/data/dp004/dc-beck3/Dark2Light/data/${simtype}/
nsnap=(99)
num_child_voxel=1024 # number of child-voxels per box edge
num_parent_voxel=32 # number of parent-voxels per box edge
Lbox=205  #[Mpc/h]

# Execute script
python3 ./data_generate.py $simdir $outdir $simtype "$(echo ${nsnap[@]})" $num_child_voxel $num_parent_voxel $Lbox

