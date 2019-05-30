from __future__ import division
import os, gc, sys
import numpy as np
import pandas as pd
import read_hdf5
import h5py
from itertools import product
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def voxeling(positions, args):
    print("Put %d particles in voxels" % len(positions[:, 0]))
    scale = args["Lbox"] / args["num_child_voxel"]
    positions[:, 0] = np.floor(positions[:, 0] / scale)
    positions[:, 1] = np.floor(positions[:, 1] / scale)
    positions[:, 2] = np.floor(positions[:, 2] / scale)
    return positions


def counting(positions):
    print("Counting particles in a voxel")
    positions["c"] = pd.Series(1, dtype=np.uint64, index=positions.index)
    positions = (
        positions.groupby(["x", "y", "z"])["c"].sum().reset_index(name="c")
    )
    print("Counting %d in file" % positions["c"].sum())
    #positions["x"][positions["x"] < 0.0] = int(0)
    #positions["y"][positions["y"] < 0.0] = int(0)
    #positions["z"][positions["z"] < 0.0] = int(0)
    return positions


def structuring(positions, args):
    print("Create 3D data structure for output")
    dataset_shape = (
        args["num_child_voxel"],
        args["num_child_voxel"],
        args["num_child_voxel"],
    )
    ids = np.ravel_multi_index(positions[["x", "y", "z"]].values.T, dataset_shape)
    arr = np.bincount(
        ids, positions["c"].values, minlength=np.prod(dataset_shape)
    ).reshape(dataset_shape)
    return arr


def saving(array, fname):
    print("Saving %s" % fname)
    hf = h5py.File(fname, "w")
    hf.create_dataset("child_voxels", data=array)
    hf.close()

