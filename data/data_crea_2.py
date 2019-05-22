# Create dataset for neural-network by deviding the initialized vocels of the
# whole simulation box into trainig, validation, and test samples
from __future__ import division
import os, gc, sys
import numpy as np
import pandas as pd
import read_hdf5
import h5py
import data_crea_2_func as dcf

args = {}
args["indir"] = sys.argv[1]
args["outdir"] = sys.argv[2]
args["simtype"] = sys.argv[3]
args["nsnap"] = sys.argv[4].split(" ")
args["num_child_voxel"] = int(sys.argv[5])
args["num_parent_voxel"] = int(sys.argv[6])
args["train_percentage"] = int(sys.argv[7])
args["valid_percentage"] = int(sys.argv[8])
args["test_percentage"] = int(sys.argv[9])

if args["simtype"] == "dark_matter_only":
    # Read dark-matter
    bname = args["outdir"] + "%s_s%d_v%d" % (
        args["simtype"],
        int(args["nsnap"][0]),
        args["num_child_voxel"],
    )
    cvoxels = h5py.File(bname + "_sh.h5", "r")["child_voxels"]
    pvoxels = dcf.grouping(cvoxels, args)
    print("##1 ##", len(pvoxels.keys()))
    train_pvoxels, valid_pvoxels, test_pvoxels = dcf.partitioning(pvoxels, args)
    dcf.saving(train_pvoxels, bname + "_sh_train.h5")
    dcf.saving(valid_pvoxels, bname + "_sh_valid.h5")
    dcf.saving(test_pvoxels, bname + "_sh_test.h5")
else:
    # Read dark-matter
    bname = args["outdir"] + "%s_s%d_v%d" % (
        args["simtype"],
        args["nsnap"][ii],
        args["num_child_voxel"],
    )
    f = h5py.File(bname + "_dm.h5", "r")


