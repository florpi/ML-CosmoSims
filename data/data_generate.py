# Create dataset
#
from __future__ import division
import os, sys
import numpy as np
import pandas as pd
import read_hdf5
import h5py
from itertools import product

args = {}
args["simdir"]   = sys.argv[1]
args["outdir"]   = sys.argv[2]
args["simtype"]  = sys.argv[3]
args["nsnap"]    = sys.argv[4].split(' ')
args["nvoxel"]   = int(sys.argv[5])

# Run through snapshots
print(args["nsnap"])
for ii in range(len(args["nsnap"])):
    args["nsnap"][ii] = int(args["nsnap"][ii])
    print(args["nsnap"][ii])
    s = read_hdf5.snapshot(args["nsnap"][ii], args["simdir"]) 
    s.read(["Coordinates"], parttype=[1])

    dm_pos = pd.DataFrame(s.data['Coordinates']['dm'])*s.header.hubble
    dm_pos.columns = ['x','y','z']
    dm_pos['x_b'] = (np.floor(dm_pos['x']/(62/args["nvoxel"]))).astype(int)
    dm_pos['y_b'] = (np.floor(dm_pos['y']/(62/args["nvoxel"]))).astype(int)
    dm_pos['z_b'] = (np.floor(dm_pos['z']/(62/args["nvoxel"]))).astype(int)
    dm_pos.drop(['x', 'y', 'z'], axis=1)
    dm_pos['c'] = 1
    dm_pos = dm_pos.groupby(
            ['x_b','y_b','z_b'])['c'].count().reset_index(name="count")
    dataset_shape = (args["nvoxel"], args["nvoxel"], args["nvoxel"])
    ids = np.ravel_multi_index(
        dm_pos[['x_b','y_b','z_b']].values.T,
        dataset_shape
    )
    arr = np.bincount(
        ids,
        dm_pos['count'].values,
        minlength=np.prod(dataset_shape),
    ).reshape(dataset_shape)
    
    filename = args["outdir"]+"%s_s%d_v%d" % (
            args["simtype"], args["nsnap"][ii], args["nvoxel"])
    np.save(filename, arr)
