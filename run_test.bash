#!/bin/bash -l

# Parameters for Dataset
datapath='/cosma7/data/dp004/dc-beck3/Dark2Light/data/dark_matter_only/'
labelpath='/cosma7/data/dp004/dc-beck3/Dark2Light/data/full_physics/'
nsnap=99
num_child_voxel=1024 # number of child-voxels per box edge
num_parent_voxel=32 # number of parent-voxels per box edge

# Parameters for ML-algorithm
batch_size=2
learning_rate=0.001
max_epoch=2
lossweight=160
normalize=0
modelidx=2
target_class=0

python3 main.py $datapath $labelpath $nsnap $num_child_voxel
