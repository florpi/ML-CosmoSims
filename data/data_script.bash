#!/bin/bash -l
# Create dataset
#
#SBATCH -n 1
#SBATCH -t 05:00:00
#SBATCH -J TNG3001 
#SBATCH -o ./logs/TNG300_%j.out
#SBATCH -e ./logs/TNG300_%j.err
#SBATCH -p cosma7
#SBATCH -A dp004
#SBATCH --exclusive

# Load Module
module unload python
module load python/3.6.5

simtype=full_physics #[dark_matter_only, full_physics]
simdir=/cosma7/data/TNG/TNG300-1/
outdir=/cosma7/data/dp004/dc-beck3/Dark2Light/data/${simtype}/
nsnap=(99)
num_child_voxel=1024 # number of child-voxels per box edge
Lbox=205  #[Mpc/h]

# Execute script
python3 ./data_init_1.py $simdir $outdir $simtype "$(echo ${nsnap[@]})" $num_child_voxel $Lbox

indir=/cosma7/data/dp004/dc-beck3/Dark2Light/data/${simtype}/
num_parent_voxel=32 # number of parent-voxels per box edge
train_perc=60
valid_perc=20
test_perc=20

python3 ./data_crea_2.py $indir $outdir $simtype "$(echo ${nsnap[@]})" $num_child_voxel $num_parent_voxel $train_perc $valid_perc $test_perc



# Catalogue of Simulations
# ------------------------
# simdir=/cosma6/data/dp004/dc-arno1/SZ_project/${simtype}/L62_N512_GR/
# simdir=/cosma7/data/TNG/TNG100-1/
# simdir=/cosma7/data/TNG/TNG300-1/



