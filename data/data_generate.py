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

# Functions --------------------------------------------------------------------
def voxeling(positions):
    "Put particles in voxels"
    positions['x_b'] = (np.floor(positions['x']/(62/args["nvoxel"]))).astype(int)
    positions['y_b'] = (np.floor(positions['y']/(62/args["nvoxel"]))).astype(int)
    positions['z_b'] = (np.floor(positions['z']/(62/args["nvoxel"]))).astype(int)
    positions.drop(['x', 'y', 'z'], axis=1)
    return positions


def counting(positions):
    "Count particles in a voxel"
    positions['c'] = 1
    positions = positions.groupby(
            ['x_b','y_b','z_b'])['c'].count().reset_index(name="count")
    return positions

def structuring(positions):
    "Create 3D data structure for output"
    dataset_shape = (args["nvoxel"], args["nvoxel"], args["nvoxel"])
    ids = np.ravel_multi_index(
        positions[['x_b','y_b','z_b']].values.T,
        dataset_shape)
    arr = np.bincount(
        ids,
        positions['count'].values,
        minlength=np.prod(dataset_shape)).reshape(dataset_shape)
    return arr

# ------------------------------------------------------------------------------

# Run through snapshots
for ii in range(len(args["nsnap"])):
    args["nsnap"][ii] = int(args["nsnap"][ii])
    s = read_hdf5.snapshot(args["nsnap"][ii], args["simdir"])

    # Read subhalos
    s.group_catalog(["SubhaloPos"])
    # Write subhalos
    pos = pd.DataFrame(s.cat['SubhaloPos'])*s.header.hubble
    pos.columns = ['x','y','z']
    pos = voxeling(pos)
    pos = counting(pos)
    arr = structuring(pos)
    filename = args["outdir"]+"%s_s%d_v%d_%s" % (
            args["simtype"], args["nsnap"][ii], args["nvoxel"], 'sh')
    np.save(filename, arr)

    # Read particles
    if args["simtype"] == "dark_matter_only":
        s.read(["Coordinates"], parttype=[1])
    else:
        s.read(["Coordinates"], parttype=[1, 4])
   
    # Write dark-matter
    pos = pd.DataFrame(s.data['Coordinates']['dm'])*s.header.hubble
    pos.columns = ['x','y','z']
    pos = voxeling(pos)
    pos = counting(pos)
    arr = structuring(pos)
    filename = args["outdir"]+"%s_s%d_v%d_%s" % (
            args["simtype"], args["nsnap"][ii], args["nvoxel"], 'dm')
    np.save(filename, arr)
    
    # Write stars
    pos = pd.DataFrame(s.data['Coordinates']['stars'])*s.header.hubble
    pos.columns = ['x','y','z']
    pos = voxeling(pos)
    pos = counting(pos)
    arr = structuring(pos)
    filename = args["outdir"]+"%s_s%d_v%d_%s" % (
            args["simtype"], args["nsnap"][ii], args["nvoxel"], 'st')
    np.save(filename, arr)
    
