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
    "Put particles in voxels"
    positions["x_b"] = (
        np.floor(positions["x"] / (args["Lbox"] / args["num_child_voxel"]))
    ).astype(int)
    positions["y_b"] = (
        np.floor(positions["y"] / (args["Lbox"] / args["num_child_voxel"]))
    ).astype(int)
    positions["z_b"] = (
        np.floor(positions["z"] / (args["Lbox"] / args["num_child_voxel"]))
    ).astype(int)
    positions.drop(["x", "y", "z"], axis=1)
    return positions


def counting(positions):
    "Count particles in a voxel"
    positions["c"] = 1
    positions = (
        positions.groupby(["x_b", "y_b", "z_b"])["c"].count().reset_index(name="count")
    )
    positions["x_b"][positions["x_b"] < 0.0] = int(0)
    positions["y_b"][positions["y_b"] < 0.0] = int(0)
    positions["z_b"][positions["z_b"] < 0.0] = int(0)
    return positions


def structuring(positions):
    "Create 3D data structure for output"
    dataset_shape = (
        args["num_child_voxel"],
        args["num_child_voxel"],
        args["num_child_voxel"],
    )
    print("## 1 ##", positions["x_b"].values.min(), positions["x_b"].values.max())
    ids = np.ravel_multi_index(positions[["x_b", "y_b", "z_b"]].values.T, dataset_shape)
    arr = np.bincount(
        ids, positions["count"].values, minlength=np.prod(dataset_shape)
    ).reshape(dataset_shape)
    return arr


def grouping(array):
    "Create families by sorting child-voxels to their parent-voxels"
    print(" test1", array.shape)
    x = np.asarray(np.split(array, args["num_parent_voxel"], axis=0))
    y = [
        np.asarray(np.split(x[ii], args["num_parent_voxel"], axis=1))
        for ii in range(args["num_parent_voxel"])
    ]
    z = []
    for ii in range(args["num_parent_voxel"]):
        for jj in range(args["num_parent_voxel"]):
            z.append(np.asarray(np.split(y[ii][jj], args["num_parent_voxel"], axis=2)))
    init = 1
    for ii in range(args["num_parent_voxel"] * args["num_parent_voxel"] - 1):
        if init == 1:
            array = np.concatenate((z[ii], z[ii + 1]), axis=0)
            init = 0
        else:
            array = np.concatenate((array, z[ii + 1]), axis=0)
    print(" test2", array.shape)
    return array


# ------------------------------------------------------------------------------

# Run through snapshots
for ii in range(len(args["nsnap"])):
    args["nsnap"][ii] = int(args["nsnap"][ii])
    print("load df", args["nsnap"][ii], args["simdir"])
    s = read_hdf5.snapshot(
        args["nsnap"][ii], args["simdir"], check_total_particle_number=True
    )

    # Read subhalos
    s.group_catalog(["SubhaloPos"])
    # Write subhalos
    pos = pd.DataFrame(s.cat["SubhaloPos"]) * s.header.hubble / 1e3  # [Mpc/h]
    pos.columns = ["x", "y", "z"]
    pos = voxeling(pos)
    pos = counting(pos)
    arr = structuring(pos)
    arr = grouping(arr)
    filename = args["outdir"] + "%s_s%d_v%d_%s" % (
        args["simtype"],
        args["nsnap"][ii],
        args["num_child_voxel"],
        "sh",
    )
    hf = h5py.File(filename, "w")
    hf.create_dataset("snapnum", data=lc["snapnum_box"])
    hf.close()

    if args["simtype"] == "dark_matter_only":
        # Read particles
        s.read(["Coordinates"], parttype=[1])

        # Write dark-matter
        pos = pd.DataFrame(s.data["Coordinates"]["dm"]) * s.header.hubble / 1e3
        pos.columns = ["x", "y", "z"]
        pos = voxeling(pos)
        pos = counting(pos)
        arr = structuring(pos)
        arr = grouping(arr)
        filename = args["outdir"] + "%s_s%d_v%d_%s" % (
            args["simtype"],
            args["nsnap"][ii],
            args["num_child_voxel"],
            "dm",
        )
        np.save(filename, arr)

    else:
        # Read particles
        s.read(["Coordinates"], parttype=[1, 4])

        # Write stars
        pos = pd.DataFrame(s.data["Coordinates"]["stars"]) * s.header.hubble / 1e3
        pos.columns = ["x", "y", "z"]
        pos = voxeling(pos)
        pos = counting(pos)
        arr = structuring(pos)
        arr = grouping(arr)
        filename = args["outdir"] + "%s_s%d_v%d_%s" % (
            args["simtype"],
            args["nsnap"][ii],
            args["num_child_voxel"],
            "st",
        )
        np.save(filename, arr)

        # Write dark-matter
        pos = pd.DataFrame(s.data["Coordinates"]["dm"]) * s.header.hubble / 1e3
        pos.columns = ["x", "y", "z"]
        pos = voxeling(pos)
        pos = counting(pos)
        arr = structuring(pos)
        arr = grouping(arr)
        filename = args["outdir"] + "%s_s%d_v%d_%s" % (
            args["simtype"],
            args["nsnap"][ii],
            args["num_child_voxel"],
            "dm",
        )
        hf = h5py.File(filename, "w")
        hf.create_dataset("parent_voxel", arr)
        hf.close()
