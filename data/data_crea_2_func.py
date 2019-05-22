import h5py
import numpy as np
from itertools import islice


def grouping(array, args):
    """
    Create families by sorting child-voxels to their parent-voxels
    
    Parameter
    ---------
    cvoxels : np.ndarray
        3D-array of child-voxels where elements represent # of particles
    args : np.dict
        Dictionary of flags from .bash script

    Return
    ------
    pvoxels : np.dict
        Dictionary[parent-voxels id][child-voxel coord][# of particles]
    """
    pvoxels = {}

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
            arr = np.asarray(np.split(y[ii][jj], args["num_parent_voxel"], axis=2))
            for kk in range(args["num_parent_voxel"]):  # run through z-axis
                parent_name = "parent_x%d_y%d_z%d" % (ii, jj, kk)
                pvoxels[parent_name] = arr[kk]

    return pvoxels


def partitioning(pvoxels, args):
    """
    Create families by sorting child-voxels to their parent-voxels
    
    Parameter
    ---------
    pvoxels : np.dict
        Dictionary[parent-voxels id][child-voxel coord][# of particles]
    args : np.dict
        Dictionary of flags from .bash script

    Return
    ------
    train_pvoxels : np.dict
    valid_pvoxels : np.dict
    test_pvoxels : np.dict
    """
    tot_num_pvoxel = args["num_parent_voxel"] ** 3

    train_num_pvoxel = int((tot_num_pvoxel * args["train_percentage"]/100))
    valid_num_pvoxel = int((tot_num_pvoxel * args["valid_percentage"]/100))
    test_num_pvoxel = int((tot_num_pvoxel * args["test_percentage"]/100))
    print(len(pvoxels.keys()), tot_num_pvoxel, train_num_pvoxel, valid_num_pvoxel, test_num_pvoxel)
    iter_pvoxels = iter(pvoxels)
    
    train_pvoxels = {
        k: pvoxels[k]
        for k in islice(
            iter_pvoxels,
            0,
            train_num_pvoxel
        )
    }
    valid_pvoxels = {
        k: pvoxels[k]
        for k in islice(
            iter_pvoxels,
            train_num_pvoxel,
            train_num_pvoxel + valid_num_pvoxel
        )
    }
    test_pvoxels = {
        k: pvoxels[k]
        for k in islice(
            iter_pvoxels,
            train_num_pvoxel + valid_num_pvoxel,
            train_num_pvoxel + valid_num_pvoxel + test_num_pvoxel,
        )
    }
    return train_pvoxels, valid_pvoxels, test_pvoxels


def saving(dicts, fname):
    print("Saving %s" % fname)
    hf = h5py.File(fname, "w")
    for k, v in dicts.items():
        hf.create_dataset(k, data=v)
    hf.close()
