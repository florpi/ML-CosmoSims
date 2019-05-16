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
args["simdir"] = sys.argv[1]
args["outdir"] = sys.argv[2]
args["simtype"] = sys.argv[3]
args["nsnap"] = sys.argv[4].split(" ")
args["num_child_voxel"] = int(sys.argv[5])
args["num_parent_voxel"] = int(sys.argv[6])
args["Lbox"] = float(sys.argv[7])

# Functions --------------------------------------------------------------------
def voxeling(positions):
    print("Put particles in voxels")
    unit_change = lambda x: np.floor(x/(args["Lbox"]/args["num_child_voxel"])).astype(int)
    positions["x"] = positions["x"].apply(unit_change)
    positions["y"] = positions["y"].apply(unit_change)
    positions["z"] = positions["z"].apply(unit_change)
    return positions


def counting(positions):
    print("Count particles in a voxel")
    positions["c"] = 1
    positions = (
        positions.groupby(["x", "y", "z"])["c"].count().reset_index(name="count")
    )
    positions["x"][positions["x"] < 0.0] = int(0)
    positions["y"][positions["y"] < 0.0] = int(0)
    positions["z"][positions["z"] < 0.0] = int(0)
    return positions


def structuring(positions):
    print("Create 3D data structure for output")
    dataset_shape = (
        args["num_child_voxel"],
        args["num_child_voxel"],
        args["num_child_voxel"],
    )
    print("## 1 ##", positions["x"].values.min(), positions["x"].values.max())
    ids = np.ravel_multi_index(positions[["x", "y", "z"]].values.T, dataset_shape)
    arr = np.bincount(
        ids, positions["count"].values, minlength=np.prod(dataset_shape)
    ).reshape(dataset_shape)
    return arr


def grouping_and_saving(array, fname):
    print("Create families by sorting child-voxels to their parent-voxels")
    hf = h5py.File(fname, "w")
    # Group along x-axis
    x = np.asarray(np.split(array, args["num_parent_voxel"], axis=0))
    # Group Groups along y-axis
    y = [
        np.asarray(np.split(x[ii], args["num_parent_voxel"], axis=1))
        for ii in range(args["num_parent_voxel"])
    ]
    # Group Groups Groups along z-axis
    for ii in range(args["num_parent_voxel"]):  # run through x-axis
        for jj in range(args["num_parent_voxel"]):  # run through y-axis
            arr = np.asarray(
                np.split(y[ii][jj], args["num_parent_voxel"], axis=2)
            )
            for kk in range(args["num_parent_voxel"]):  # run through z-axis
                parent_name = "parent_x%d_y%d_z%d" % (ii, jj, kk)
                hf.create_dataset(parent_name, data=arr[kk])
    hf.close()
    return array


def saving(array, fname):
    print("Save")
    hf = h5py.File(fname, "w")
    hf.create_dataset("child_voxels", data=array)
    hf.close()


# ------------------------------------------------------------------------------

# Run through snapshots
for ii in range(len(args["nsnap"])):
    args["nsnap"][ii] = int(args["nsnap"][ii])
    print("load df", args["nsnap"][ii], args["simdir"])
    s = read_hdf5.snapshot(
        args["nsnap"][ii], args["simdir"], check_total_particle_number=True
    )
    bname = args["outdir"] + "%s_s%d_v%d" % (
        args["simtype"],
        args["nsnap"][ii],
        args["num_child_voxel"],
    )
    args["Lbox"] = args["Lbox"] * 1e3 / s.header.hubble

    # Read subhalos
    s.group_catalog(["SubhaloPos"])
    # Write subhalos
    pos = pd.DataFrame(s.cat["SubhaloPos"])  # [kpc]
    pos.columns = ["x", "y", "z"]
    pos = voxeling(pos)
    pos = counting(pos)
    arr = structuring(pos)
    saving(arr, bname+"_sh.h5")
    #grouping_and_saving(arr, bname+"_sh.h5")
    print("1")
    if args["simtype"] == "dark_matter_only":
        print("2")
        # Read particles
        s.read(["Coordinates"], parttype=[1])

        # Write dark-matter
        pos = pd.DataFrame(s.data["Coordinates"]["dm"])  #[kpc]
        print("3")
        pos.columns = ["x", "y", "z"]
        pos = voxeling(pos)
        print("4")
        pos = counting(pos)
        print("5")
        arr = structuring(pos)
        print("6")
        saving(arr, bname+"_dm.h5")
        print("7")
        #arr = grouping_and_saving(arr, bname+"_dm.h5")

    else:
        # Read particles
        s.read(["Coordinates"], parttype=[1, 4])
        
        # Write stars
        pos = pd.DataFrame(s.data["Coordinates"]["stars"])  #[kpc]
        pos.columns = ["x", "y", "z"]
        pos = voxeling(pos)
        pos = counting(pos)
        arr = structuring(pos)
        arr = grouping_and_saving(arr, bname+"_st.h5")

        # Write dark-matter
        pos = pd.DataFrame(s.data["Coordinates"]["dm"])  #[kpc]
        pos.columns = ["x", "y", "z"]
        pos = voxeling(pos)
        pos = counting(pos)
        arr = structuring(pos)
        arr = grouping_and_saving(arr, bname+"_dm.h5")
