# Create initial dataset by deviding the whole simulation box into 
# int(num_child_voxel) voxels containing the number of particles in each.
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

args = {}
args["simdir"] = sys.argv[1]
args["outdir"] = sys.argv[2]
args["simtype"] = sys.argv[3]
args["nsnap"] = sys.argv[4].split(" ")
args["num_child_voxel"] = int(sys.argv[5])
args["Lbox"] = float(sys.argv[6])
memory_cutoff = 1e8


# Run through snapshots
for ii in range(len(args["nsnap"])):
    args["nsnap"][ii] = int(args["nsnap"][ii])
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
    pos = voxeling(s.cat["SubhaloPos"])
    pos = pd.DataFrame(pos, dtype=np.int32)
    pos.columns = ["x", "y", "z"]
    pos = counting(pos)
    arr = structuring(pos)
    saving(arr, bname+"_sh.h5")

    # Read particles
    if args["simtype"] == "dark_matter_only":
        print("----> Write dark-matter <----")
        if s.header.num_total[1] >= memory_cutoff: 
            pnum = 10

            for pp in range(pnum):
                s.read(["Coordinates"], parttype=[1], partition=[pnum, pp])
                pos_pp = voxeling(s.data["Coordinates"]["dm"])
                pos_pp = pd.DataFrame(pos_pp, dtype=np.int32)
                pos_pp.columns = ["x", "y", "z"]
                pos_pp = counting(pos_pp)

                if pp == 0:
                    pos = pos_pp
                else:
                    pos = pd.concat([pos, pos_pp])
                    pos = (
                        pos.groupby(["x", "y", "z"])["c"]
                        .count()
                        .reset_index(name="c")
                    )
                print("dataframes", pos_pp.shape, pos.shape, pos.columns)
        else:
            s.read(["Coordinates"], parttype=[1], partition=[1, 0])
            pos = voxeling(s.data["Coordinates"]["dm"])
            del s
            gc.collect()
            pos = pd.DataFrame(pos, dtype=np.int32)
            pos.columns = ["x", "y", "z"]
            pos = counting(pos)

        arr = structuring(pos)
        saving(arr, bname + "_dm.h5")

    elif args["simtype"] == "full_physics":
        print("----> Write stars <----")
        if s.header.num_total[4] >= memory_cutoff:
            pnum = 10

            for pp in range(pnum):
                s.read(["Coordinates"], parttype=[4], partition=[pnum, pp])
                pos_pp = voxeling(s.data["Coordinates"]["stars"])
                pos_pp = pd.DataFrame(pos_pp, dtype=np.int32)
                pos_pp.columns = ["x", "y", "z"]
                pos_pp = counting(pos_pp)

                if pp == 0:
                    pos = pos_pp
                else:
                    pos = pd.concat([pos, pos_pp])
                    pos = (
                        pos.groupby(["x", "y", "z"])["c"]
                        .count()
                        .reset_index(name="c")
                    )
        else:
            s.read(["Coordinates"], parttype=[1], partition=[1, 0])
            pos = voxeling(s.data["Coordinates"]["dm"])
            pos = pd.DataFrame(pos, dtype=np.int32)
            pos.columns = ["x", "y", "z"]
            pos = counting(pos)

        arr = structuring(pos)
        saving(arr, bname + "_st.h5")
        
        print("----> Write dark-matter <----")
        if s.header.num_total[1] >= memory_cutoff:
            pnum = 10

            for pp in range(pnum):
                s.read(["Coordinates"], parttype=[1], partition=[pnum, pp])
                pos_pp = voxeling(s.data["Coordinates"]["dm"])
                pos_pp = pd.DataFrame(pos_pp, dtype=np.int32)
                pos_pp.columns = ["x", "y", "z"]
                pos_pp = counting(pos_pp)

                if pp == 0:
                    pos = pos_pp
                else:
                    pos = pd.concat([pos, pos_pp])
                    pos = (
                        pos.groupby(["x", "y", "z"])["c"]
                        .count()
                        .reset_index(name="c")
                    )
                print("dataframes", pos_pp.shape, pos.shape, pos.columns)
        else:
            s.read(["Coordinates"], parttype=[1], partition=[1, 0])
            pos = voxeling(s.data["Coordinates"]["dm"])
            del s
            gc.collect()
            pos = pd.DataFrame(pos, dtype=np.int32)
            pos.columns = ["x", "y", "z"]
            pos = counting(pos)

        arr = structuring(pos)
        saving(arr, bname + "_dm.h5")
